"""
Task 1 — Text Infilling Evaluation
────────────────────────────────────
Given a prefix and a suffix, generate the missing middle span.

  Input:  "<prefix tokens> [MASK] … [MASK] <suffix tokens>"
  Output: fill in the masked middle span with coherent text

Why this task:
  • Diffusion models see prefix + suffix via bidirectional attention — infilling
    is their native operation.
  • Autoregressive GPT-2 can only use the prefix; the suffix is ignored
    (unless the model was trained with Fill-In-the-Middle, which standard
    gpt2 was not).
  • The architectural gap is largest and most measurable here.

Metrics:
  - BLEU-4 of generated span vs. reference span
  - Span perplexity: GPT-2 score of the full reconstructed sentence
  - Generative PPL: GPT-2-XL evaluates the entire generated sequence

Usage:
  python -m tasks.infilling --model_path checkpoints/bert_moe_diffusion/final_model.pt \
                            --num_samples 500 --num_steps 1000
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from sacrebleu.metrics import BLEU
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np

from model import BertMoEDiffusion, LogLinearNoiseSchedule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─── Dataset preparation ──────────────────────────────────────────────────────

def build_infilling_examples(
    tokenizer,
    split: str = "test",
    num_examples: int = 500,
    seq_len: int = 128,
    mask_frac: float = 0.3,
    seed: int = 42,
) -> List[dict]:
    """Sample LM1B test sentences and carve out a middle span to fill.

    Returns a list of dicts:
      {
        "full_ids":    LongTensor (seq_len,) — full ground-truth sequence
        "prefix_ids":  LongTensor (prefix_len,)
        "suffix_ids":  LongTensor (suffix_len,)
        "span_ids":    LongTensor (span_len,)  — ground-truth middle span
        "span_start":  int
        "span_end":    int
        "full_text":   str
      }
    """
    torch.manual_seed(seed)
    dataset = load_dataset("lm1b", split=split, streaming=True, trust_remote_code=True)
    examples = []

    pbar = tqdm(total=num_examples, desc="Building examples", unit="ex")
    for item in dataset:
        if len(examples) >= num_examples:
            break
        text = item["text"].strip()
        enc = tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)   # (seq_len,)
        real_len = enc["attention_mask"].squeeze(0).sum().item()
        if real_len < 20:                   # skip very short sentences
            continue

        # Carve out the middle ``mask_frac`` fraction
        span_len = max(1, int(real_len * mask_frac))
        margin = (real_len - span_len) // 2
        span_start = margin
        span_end = margin + span_len

        prefix_ids = ids[:span_start].clone()
        suffix_ids = ids[span_end:real_len].clone()
        span_ids = ids[span_start:span_end].clone()

        examples.append({
            "full_ids": ids,
            "prefix_ids": prefix_ids,
            "suffix_ids": suffix_ids,
            "span_ids": span_ids,
            "span_start": span_start,
            "span_end": span_end,
            "real_len": int(real_len),
            "full_text": text,
        })
        pbar.update(1)
    pbar.close()

    return examples


# ─── Diffusion infilling ──────────────────────────────────────────────────────

def diffusion_infill(
    model: BertMoEDiffusion,
    examples: List[dict],
    noise_schedule: LogLinearNoiseSchedule,
    tokenizer,
    num_steps: int,
    device: torch.device,
    batch_size: int = 16,
) -> List[str]:
    """Generate infilled sequences using the diffusion model.

    Pins prefix and suffix positions; runs reverse diffusion only on the
    middle span.  Returns a list of decoded string sequences (full sentences).
    """
    model.eval()
    mask_id = tokenizer.mask_token_id
    model.set_mask_token_id(mask_id)

    results = []
    for i in tqdm(
        range(0, len(examples), batch_size),
        desc="Diffusion infilling",
        unit="batch",
    ):
        batch_ex = examples[i : i + batch_size]
        B = len(batch_ex)

        # Determine maximum seq_len across batch
        seq_len = max(ex["full_ids"].shape[0] for ex in batch_ex)

        # Build starting z: all masks except prefix/suffix
        z = torch.full((B, seq_len), mask_id, dtype=torch.long, device=device)
        fixed_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

        for j, ex in enumerate(batch_ex):
            ps = ex["span_start"]
            pe = ex["span_end"]
            rl = ex["real_len"]
            # Pin prefix
            z[j, :ps] = ex["prefix_ids"].to(device)
            fixed_mask[j, :ps] = True
            # Pin suffix (everything after the span up to real_len)
            z[j, pe:rl] = ex["suffix_ids"].to(device)
            fixed_mask[j, pe:rl] = True

        # Run reverse diffusion
        gen_ids = model.sample(
            batch_size=B,
            seq_len=seq_len,
            num_steps=num_steps,
            noise_schedule=noise_schedule,
            tokenizer=tokenizer,
            device=device,
            fixed_token_mask=fixed_mask,
            init_z=z,
        )

        for j in range(B):
            rl = batch_ex[j]["real_len"]
            tokens = gen_ids[j, :rl].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(text)

    return results


# ─── GPT-2 infilling (prefix-only baseline) ──────────────────────────────────

def gpt2_infill(
    gpt2_model: GPT2LMHeadModel,
    gpt2_tokenizer,
    examples: List[dict],
    bert_tokenizer,
    device: torch.device,
    max_new_tokens: int = 40,
) -> List[str]:
    """GPT-2 infilling: generate continuation from prefix only.

    Standard GPT-2 has no access to the suffix — this is the architectural
    weakness we want to expose.  We generate ``max_new_tokens`` tokens
    after the prefix, then append the suffix to form the full sentence.
    """
    gpt2_model.eval()
    results = []

    for ex in tqdm(examples, desc="GPT-2 infilling", unit="ex"):
        prefix_text = bert_tokenizer.decode(
            ex["prefix_ids"].tolist(), skip_special_tokens=True
        )
        suffix_text = bert_tokenizer.decode(
            ex["suffix_ids"].tolist(), skip_special_tokens=True
        )

        # Encode with GPT-2 tokenizer
        enc = gpt2_tokenizer(prefix_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = gpt2_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,       # greedy for reproducibility
                pad_token_id=gpt2_tokenizer.eos_token_id,
            )
        # Decode only the newly generated portion
        n_prefix_tokens = enc["input_ids"].shape[1]
        generated_tokens = out[0, n_prefix_tokens:].tolist()
        generated_text = gpt2_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Reconstruct full sentence
        full_text = prefix_text.strip() + " " + generated_text.strip() + " " + suffix_text.strip()
        results.append(full_text.strip())

    return results


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Corpus-level BLEU-4 score."""
    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score(hypotheses, [references])
    return result.score


# Module-level scorer cache: (model_name, device_str) → (tokenizer, model).
# The scorer is reused across every call to compute_generative_ppl so we don't
# re-download the weights or re-move them to GPU for every model we evaluate.
_SCORER_CACHE: dict = {}


def _get_scorer(scorer_model_name: str, device: torch.device):
    """Load-and-cache a GPT-2 scorer model + tokenizer for gen-PPL evaluation."""
    key = (scorer_model_name, str(device))
    if key not in _SCORER_CACHE:
        tok = AutoTokenizer.from_pretrained(scorer_model_name)
        tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(scorer_model_name).to(device)
        mdl.eval()
        _SCORER_CACHE[key] = (tok, mdl)
    return _SCORER_CACHE[key]


def compute_generative_ppl(
    texts: List[str],
    scorer_model_name: str = "gpt2",
    device: torch.device = torch.device("cpu"),
    batch_size: int = 8,
) -> float:
    """Compute mean perplexity of generated texts under a GPT-2 scorer.

    The scorer is cached at module level (see ``_SCORER_CACHE``); repeated calls
    with the same ``scorer_model_name`` + ``device`` reuse the loaded weights.
    """
    scorer_tok, scorer_model = _get_scorer(scorer_model_name, device)

    all_ppls = []
    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Scoring gen PPL",
        unit="batch",
    ):
        batch_texts = texts[i : i + batch_size]
        enc = scorer_tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = scorer_model(**enc, labels=enc["input_ids"])
            nll = outputs.loss.item()

        all_ppls.append(np.exp(nll))

    return float(np.mean(all_ppls))


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_infilling_evaluation(
    diffusion_model_path: Optional[str],
    cfg: Optional[dict],
    num_samples: int = 200,
    num_steps: int = 1000,
    seq_len: int = 128,
    output_path: str = "results/infilling_results.json",
    device_str: str = "cuda",
) -> dict:
    """Run full infilling benchmark.

    Evaluates:
      - BertMoEDiffusion (ours, fine-tuned)
      - GPT-2 (117M)
      - GPT-2 Medium (345M)

    Returns dict of results tables.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── BERT tokenizer ────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # ── Build infilling examples ───────────────────────────────────────────────
    logger.info("Building infilling examples from LM1B test set …")
    examples = build_infilling_examples(bert_tokenizer, num_examples=num_samples, seq_len=seq_len)
    reference_texts = [ex["full_text"] for ex in examples]
    logger.info(f"  → {len(examples)} examples ready")

    results = {}
    noise_schedule = LogLinearNoiseSchedule()

    # ── 1. Diffusion model (BertMoEDiffusion) ─────────────────────────────────
    if diffusion_model_path and Path(diffusion_model_path).exists():
        logger.info("Loading BertMoEDiffusion …")
        ckpt = torch.load(diffusion_model_path, map_location=device)
        saved_cfg = ckpt.get("config", cfg or {})
        moe_cfg = saved_cfg.get("model", {}).get("moe", {})
        diff_model = BertMoEDiffusion(
            bert_model_name=saved_cfg.get("model", {}).get("backbone", "bert-base-uncased"),
            moe_layers=moe_cfg.get("moe_layers", [3, 5, 7, 9, 11]),
            num_experts=moe_cfg.get("num_experts", 8),
            num_experts_per_token=moe_cfg.get("num_experts_per_token", 2),
        )
        diff_model.load_state_dict(ckpt["model"])
        diff_model.to(device)

        logger.info(f"Running diffusion infilling (T={num_steps}) …")
        diff_texts = diffusion_infill(
            diff_model, examples, noise_schedule, bert_tokenizer, num_steps, device
        )
        bleu = compute_bleu(diff_texts, reference_texts)
        gen_ppl = compute_generative_ppl(diff_texts, device=device)
        results["BertMoEDiffusion"] = {"bleu4": bleu, "generative_ppl": gen_ppl}
        logger.info(f"  BertMoEDiffusion — BLEU-4: {bleu:.2f} | Gen PPL: {gen_ppl:.2f}")
    else:
        logger.warning("Diffusion model path not found; skipping BertMoEDiffusion eval.")

    # ── 2. GPT-2 (117M) ───────────────────────────────────────────────────────
    logger.info("Loading GPT-2 (117M) …")
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_texts = gpt2_infill(gpt2_mdl, gpt2_tok, examples, bert_tokenizer, device)
    bleu = compute_bleu(gpt2_texts, reference_texts)
    gen_ppl = compute_generative_ppl(gpt2_texts, device=device)
    results["GPT-2 (117M)"] = {"bleu4": bleu, "generative_ppl": gen_ppl}
    logger.info(f"  GPT-2 (117M) — BLEU-4: {bleu:.2f} | Gen PPL: {gen_ppl:.2f}")

    # ── 3. GPT-2 Medium (345M) ────────────────────────────────────────────────
    logger.info("Loading GPT-2 Medium (345M) …")
    gpt2m_tok = AutoTokenizer.from_pretrained("gpt2-medium")
    gpt2m_tok.pad_token = gpt2m_tok.eos_token
    gpt2m_mdl = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    gpt2m_texts = gpt2_infill(gpt2m_mdl, gpt2m_tok, examples, bert_tokenizer, device)
    bleu = compute_bleu(gpt2m_texts, reference_texts)
    gen_ppl = compute_generative_ppl(gpt2m_texts, device=device)
    results["GPT-2 Medium (345M)"] = {"bleu4": bleu, "generative_ppl": gen_ppl}
    logger.info(f"  GPT-2 Medium — BLEU-4: {bleu:.2f} | Gen PPL: {gen_ppl:.2f}")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infilling benchmark")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--output", default="results/infilling_results.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_infilling_evaluation(
        diffusion_model_path=args.model_path,
        cfg=None,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        output_path=args.output,
        device_str=args.device,
    )
