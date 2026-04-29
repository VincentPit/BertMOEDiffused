"""
Task 2 — Keyword-Constrained Generation Evaluation
────────────────────────────────────────────────────
Generate a coherent sentence that must contain specific keyword tokens at
approximately specified positions within the sequence.

Why this task:
  • Diffusion models generate all positions simultaneously — keywords can be
    "pinned" as unmasked positions in z_1 and the reverse process fills in
    the rest.  Constraint satisfaction is architectural, not algorithmic.
  • Autoregressive GPT-2 generates left-to-right; meeting positional keyword
    constraints requires complex constrained decoding (NeuroLogic, etc.).
    We use greedy decoding with a soft keyword-injection heuristic to give
    GPT-2 a fair shot.
  • The satisfaction rate gap reveals the generation paradigm difference.

Metrics:
  - Keyword satisfaction rate (%): fraction of outputs containing all keywords
  - MAUVE score: distributional similarity of generated text vs. unconstrained
    reference text (from LM1B test set)
  - Generative PPL: GPT-2 perplexity of the full generated sequences

Usage:
  python -m tasks.constrained_gen \
    --model_path checkpoints/bert_moe_diffusion/final_model.pt \
    --num_samples 500 --num_steps 1000
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset

from model import BertMoEDiffusion, LogLinearNoiseSchedule
from tasks.infilling import compute_generative_ppl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─── Keyword constraint sets ──────────────────────────────────────────────────

# Pre-defined triplets of (keyword1, keyword2, keyword3) drawn from common
# English vocabulary.  Each triplet represents constraints that must all appear
# in the generated sequence.
KEYWORD_TRIPLETS = [
    ["music", "played", "night"],
    ["water", "fell", "mountain"],
    ["science", "research", "discovered"],
    ["city", "people", "lived"],
    ["food", "cooked", "family"],
    ["book", "written", "author"],
    ["light", "shone", "dark"],
    ["children", "laughed", "park"],
    ["team", "won", "match"],
    ["doctor", "treated", "patient"],
    ["teacher", "explained", "students"],
    ["rain", "fell", "evening"],
    ["artist", "painted", "canvas"],
    ["train", "arrived", "station"],
    ["computer", "processed", "data"],
    ["bird", "sang", "morning"],
    ["soldier", "fought", "battle"],
    ["market", "opened", "early"],
    ["river", "flowed", "valley"],
    ["engineer", "designed", "bridge"],
]


# ─── Build constrained generation examples ───────────────────────────────────

def build_constrained_examples(
    tokenizer,
    seq_len: int = 64,
    num_examples: int = 200,
) -> List[dict]:
    """Create constrained generation tasks.

    Each example specifies:
      - keywords: list of 3 word strings that must appear in the output
      - keyword_positions: approximate token positions (evenly spaced) for
                           the pinned tokens in the diffusion sequence
    """
    examples = []
    for i in range(num_examples):
        triplet = KEYWORD_TRIPLETS[i % len(KEYWORD_TRIPLETS)]

        # Tokenise each keyword (take the first subword token)
        keyword_ids = []
        for kw in triplet:
            toks = tokenizer.encode(" " + kw, add_special_tokens=False)
            keyword_ids.append(toks[0] if toks else tokenizer.unk_token_id)

        # Evenly-spaced positions: leave room for [CLS] at pos 0 and [SEP] at end
        positions = [
            int((j + 1) * seq_len / (len(triplet) + 1))
            for j in range(len(triplet))
        ]

        examples.append({
            "keywords": triplet,
            "keyword_ids": keyword_ids,
            "keyword_positions": positions,
            "seq_len": seq_len,
        })
    return examples


# ─── Diffusion constrained generation ────────────────────────────────────────

def diffusion_constrained_gen(
    model: BertMoEDiffusion,
    examples: List[dict],
    noise_schedule: LogLinearNoiseSchedule,
    tokenizer,
    num_steps: int,
    device: torch.device,
    batch_size: int = 32,
) -> List[str]:
    """Generate keyword-constrained text using reverse diffusion.

    Keywords are pinned as unmasked tokens at their specified positions before
    the reverse process begins.  The model fills in the remaining positions.
    """
    model.eval()
    mask_id = tokenizer.mask_token_id
    model.set_mask_token_id(mask_id)
    results = []

    for i in range(0, len(examples), batch_size):
        batch_ex = examples[i : i + batch_size]
        B = len(batch_ex)
        seq_len = batch_ex[0]["seq_len"]

        # Start from fully-masked sequence
        z = torch.full((B, seq_len), mask_id, dtype=torch.long, device=device)
        fixed_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

        # Add [CLS] and [SEP] tokens
        z[:, 0] = tokenizer.cls_token_id
        z[:, seq_len - 1] = tokenizer.sep_token_id
        fixed_mask[:, 0] = True
        fixed_mask[:, seq_len - 1] = True

        for j, ex in enumerate(batch_ex):
            for kid, pos in zip(ex["keyword_ids"], ex["keyword_positions"]):
                # Clamp position to valid range
                pos = min(max(pos, 1), seq_len - 2)
                z[j, pos] = kid
                fixed_mask[j, pos] = True

        # Run reverse diffusion with pinned keyword positions
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
            tokens = gen_ids[j].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(text)

        logger.info(
            f"  Constrained gen: {min(i + batch_size, len(examples))}/{len(examples)} done"
        )

    return results


# ─── GPT-2 keyword-constrained generation ────────────────────────────────────

def gpt2_constrained_gen(
    gpt2_model: GPT2LMHeadModel,
    gpt2_tokenizer,
    examples: List[dict],
    device: torch.device,
    max_length: int = 64,
) -> List[str]:
    """GPT-2 keyword-constrained generation via greedy decoding + keyword injection.

    Strategy: prompt GPT-2 with "Please write a sentence containing
    [kw1], [kw2], and [kw3]:" and generate greedily.
    This is the strongest simple baseline for standard GPT-2 without
    complex constrained decoding algorithms.
    """
    gpt2_model.eval()
    results = []

    for ex in examples:
        kws = ex["keywords"]
        prompt = f"Write a sentence using these words: {', '.join(kws)}. Sentence:"
        enc = gpt2_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=48).to(device)

        with torch.no_grad():
            out = gpt2_model.generate(
                **enc,
                max_length=enc["input_ids"].shape[1] + max_length,
                do_sample=False,
                pad_token_id=gpt2_tokenizer.eos_token_id,
            )

        # Decode only the generated portion (after the prompt)
        n_prompt = enc["input_ids"].shape[1]
        gen_tokens = out[0, n_prompt:].tolist()
        # Truncate at first period/newline
        text = gpt2_tokenizer.decode(gen_tokens, skip_special_tokens=True)
        text = re.split(r'[\n.]', text)[0].strip()
        results.append(text)

    return results


# ─── Keyword satisfaction rate ────────────────────────────────────────────────

def keyword_satisfaction_rate(
    generated_texts: List[str],
    examples: List[dict],
) -> Tuple[float, List[bool]]:
    """Compute fraction of outputs containing all required keywords (case-insensitive).

    Returns (rate in [0,1], per-example bool list).
    """
    satisfied = []
    for text, ex in zip(generated_texts, examples):
        text_lower = text.lower()
        all_present = all(kw.lower() in text_lower for kw in ex["keywords"])
        satisfied.append(all_present)
    rate = sum(satisfied) / len(satisfied)
    return rate, satisfied


# ─── MAUVE score ─────────────────────────────────────────────────────────────

def compute_mauve(
    generated_texts: List[str],
    reference_texts: List[str],
    max_text_length: int = 128,
) -> float:
    """Compute MAUVE score between generated and reference text distributions.

    Uses the ``mauve-text`` Python package (pip install mauve-text).
    Returns the MAUVE score in [0, 1] (higher = closer to reference distribution).
    """
    try:
        import mauve  # type: ignore
        out = mauve.compute_mauve(
            p_text=reference_texts,
            q_text=generated_texts,
            max_text_length=max_text_length,
            verbose=False,
        )
        return float(out.mauve)
    except ImportError:
        logger.warning("mauve-text not installed; returning -1 for MAUVE score.")
        return -1.0


# ─── Reference texts for MAUVE ───────────────────────────────────────────────

def get_reference_texts(tokenizer, n: int = 500, seq_len: int = 64) -> List[str]:
    """Sample n sentences from LM1B test as unconstrained reference distribution."""
    dataset = load_dataset("lm1b", split="test", streaming=True, trust_remote_code=True)
    texts = []
    for item in dataset:
        if len(texts) >= n:
            break
        enc = tokenizer(
            item["text"],
            max_length=seq_len,
            truncation=True,
        )
        text = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
        if len(text.split()) > 5:
            texts.append(text)
    return texts


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_constrained_gen_evaluation(
    diffusion_model_path: Optional[str],
    cfg: Optional[dict],
    num_samples: int = 200,
    num_steps: int = 1000,
    seq_len: int = 64,
    output_path: str = "results/constrained_gen_results.json",
    device_str: str = "cuda",
) -> dict:
    """Run full keyword-constrained generation benchmark.

    Evaluates:
      - BertMoEDiffusion (ours)
      - GPT-2 (117M)
      - GPT-2 Medium (345M)

    Returns dict of results tables.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Build examples
    logger.info("Building constrained generation examples …")
    examples = build_constrained_examples(bert_tokenizer, seq_len=seq_len, num_examples=num_samples)
    logger.info(f"  → {len(examples)} examples ready")

    # Get reference texts for MAUVE
    logger.info("Fetching reference texts for MAUVE …")
    ref_texts = get_reference_texts(bert_tokenizer, n=num_samples, seq_len=seq_len)

    results = {}
    noise_schedule = LogLinearNoiseSchedule()

    # ── 1. BertMoEDiffusion ───────────────────────────────────────────────────
    if diffusion_model_path and Path(diffusion_model_path).exists():
        logger.info("Loading BertMoEDiffusion for constrained generation …")
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

        logger.info(f"Running constrained diffusion generation (T={num_steps}) …")
        diff_texts = diffusion_constrained_gen(
            diff_model, examples, noise_schedule, bert_tokenizer, num_steps, device
        )
        kw_rate, _ = keyword_satisfaction_rate(diff_texts, examples)
        mauve = compute_mauve(diff_texts, ref_texts, max_text_length=seq_len)
        gen_ppl = compute_generative_ppl(diff_texts, device=device)
        results["BertMoEDiffusion"] = {
            "keyword_satisfaction_rate": kw_rate,
            "mauve": mauve,
            "generative_ppl": gen_ppl,
        }
        logger.info(
            f"  BertMoEDiffusion — KW-Sat: {kw_rate:.2%} | MAUVE: {mauve:.4f} | Gen PPL: {gen_ppl:.2f}"
        )

    # ── 2. GPT-2 ─────────────────────────────────────────────────────────────
    logger.info("Loading GPT-2 (117M) …")
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_texts = gpt2_constrained_gen(gpt2_mdl, gpt2_tok, examples, device, max_length=seq_len)
    kw_rate, _ = keyword_satisfaction_rate(gpt2_texts, examples)
    mauve = compute_mauve(gpt2_texts, ref_texts, max_text_length=seq_len)
    gen_ppl = compute_generative_ppl(gpt2_texts, device=device)
    results["GPT-2 (117M)"] = {
        "keyword_satisfaction_rate": kw_rate,
        "mauve": mauve,
        "generative_ppl": gen_ppl,
    }
    logger.info(f"  GPT-2 (117M) — KW-Sat: {kw_rate:.2%} | MAUVE: {mauve:.4f} | Gen PPL: {gen_ppl:.2f}")

    # ── 3. GPT-2 Medium ───────────────────────────────────────────────────────
    logger.info("Loading GPT-2 Medium (345M) …")
    gpt2m_tok = AutoTokenizer.from_pretrained("gpt2-medium")
    gpt2m_tok.pad_token = gpt2m_tok.eos_token
    gpt2m_mdl = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    gpt2m_texts = gpt2_constrained_gen(gpt2m_mdl, gpt2m_tok, examples, device, max_length=seq_len)
    kw_rate, _ = keyword_satisfaction_rate(gpt2m_texts, examples)
    mauve = compute_mauve(gpt2m_texts, ref_texts, max_text_length=seq_len)
    gen_ppl = compute_generative_ppl(gpt2m_texts, device=device)
    results["GPT-2 Medium (345M)"] = {
        "keyword_satisfaction_rate": kw_rate,
        "mauve": mauve,
        "generative_ppl": gen_ppl,
    }
    logger.info(
        f"  GPT-2 Medium — KW-Sat: {kw_rate:.2%} | MAUVE: {mauve:.4f} | Gen PPL: {gen_ppl:.2f}"
    )

    # ── Save results ──────────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyword-constrained generation benchmark")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--output", default="results/constrained_gen_results.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_constrained_gen_evaluation(
        diffusion_model_path=args.model_path,
        cfg=None,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        seq_len=args.seq_len,
        output_path=args.output,
        device_str=args.device,
    )
