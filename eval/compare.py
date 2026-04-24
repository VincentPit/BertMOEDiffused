"""
Comprehensive evaluation & comparison script.

Runs all four models on both tasks plus unconditional perplexity, then
produces a consolidated results table and plots.

Usage:
  python eval/compare.py \
    --diffusion_ckpt checkpoints/bert_moe_diffusion/final_model.pt \
    --num_samples 200 \
    --device cuda
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

from tasks.infilling import run_infilling_evaluation
from tasks.constrained_gen import run_constrained_gen_evaluation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─── Unconditional PPL sweep (diffusion steps) ────────────────────────────────

def evaluate_unconditional_ppl(
    diffusion_model_path: Optional[str],
    num_samples: int = 100,
    steps_list: list = [10, 100, 1000],
    seq_len: int = 128,
    device_str: str = "cuda",
) -> dict:
    """Sweep diffusion steps T and measure generative PPL.

    Compares BertMoEDiffusion at multiple T values vs. GPT-2 baselines
    (GPT-2 does not depend on T, so its PPL is constant).

    Returns dict: model_name → list of (T, gen_ppl) tuples.
    """
    from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
    from model import BertMoEDiffusion, LogLinearNoiseSchedule
    from tasks.infilling import compute_generative_ppl

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    noise_schedule = LogLinearNoiseSchedule()

    results = {}

    # ── GPT-2 unconditional generation ────────────────────────────────────────
    for model_id, model_name in [("gpt2", "GPT-2 (117M)"), ("gpt2-medium", "GPT-2 Medium (345M)")]:
        gpt2_tok = AutoTokenizer.from_pretrained(model_id)
        gpt2_tok.pad_token = gpt2_tok.eos_token
        gpt2_mdl = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        gpt2_mdl.eval()

        generated_texts = []
        with torch.no_grad():
            batch_size = 16
            for _ in range(0, num_samples, batch_size):
                n = min(batch_size, num_samples - len(generated_texts))
                prompt_ids = torch.tensor([[gpt2_tok.bos_token_id]] * n, device=device)
                out = gpt2_mdl.generate(
                    prompt_ids,
                    max_length=seq_len,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    pad_token_id=gpt2_tok.eos_token_id,
                )
                for row in out:
                    text = gpt2_tok.decode(row.tolist(), skip_special_tokens=True)
                    generated_texts.append(text)

        gen_ppl = compute_generative_ppl(generated_texts, device=device)
        # T is not applicable for AR models; use a single entry
        results[model_name] = [(None, gen_ppl)]
        logger.info(f"  {model_name} unconditional Gen PPL: {gen_ppl:.2f}")

    # ── BertMoEDiffusion sweep ────────────────────────────────────────────────
    if diffusion_model_path and Path(diffusion_model_path).exists():
        ckpt = torch.load(diffusion_model_path, map_location=device)
        saved_cfg = ckpt.get("config", {})
        moe_cfg = saved_cfg.get("model", {}).get("moe", {})
        diff_model = BertMoEDiffusion(
            bert_model_name=saved_cfg.get("model", {}).get("backbone", "bert-base-uncased"),
            moe_layers=moe_cfg.get("moe_layers", [3, 5, 7, 9, 11]),
            num_experts=moe_cfg.get("num_experts", 8),
            num_experts_per_token=moe_cfg.get("num_experts_per_token", 2),
        )
        diff_model.load_state_dict(ckpt["model"])
        diff_model.to(device)

        mask_id = bert_tokenizer.mask_token_id
        diff_model.set_mask_token_id(mask_id)

        sweep_results = []
        for T in steps_list:
            logger.info(f"  BertMoEDiffusion unconditional gen, T={T} …")
            gen_ids = diff_model.sample(
                batch_size=num_samples,
                seq_len=seq_len,
                num_steps=T,
                noise_schedule=noise_schedule,
                tokenizer=bert_tokenizer,
                device=device,
            )
            texts = [
                bert_tokenizer.decode(row.tolist(), skip_special_tokens=True)
                for row in gen_ids
            ]
            gen_ppl = compute_generative_ppl(texts, device=device)
            sweep_results.append((T, gen_ppl))
            logger.info(f"    T={T:5d} → Gen PPL: {gen_ppl:.2f}")

        results["BertMoEDiffusion"] = sweep_results

    return results


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def plot_diffusion_steps_vs_ppl(
    unconditional_results: dict,
    output_path: str = "results/plots/diffusion_steps_ppl.png",
) -> None:
    """Line chart: diffusion steps T vs. Generative PPL for all models."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    for model_name, entries in unconditional_results.items():
        if entries[0][0] is None:
            # AR model — horizontal line
            _, ppl = entries[0]
            ax.axhline(ppl, linestyle="--", label=model_name)
        else:
            T_vals = [e[0] for e in entries]
            ppl_vals = [e[1] for e in entries]
            ax.plot(T_vals, ppl_vals, marker="o", label=model_name)

    ax.set_xscale("log")
    ax.set_xlabel("Diffusion Steps T (log scale)", fontsize=12)
    ax.set_ylabel("Generative PPL (↓ better)", fontsize=12)
    ax.set_title("Unconditional Generation Quality vs. Diffusion Steps", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Plot saved → {output_path}")


def plot_infilling_comparison(
    infilling_results: dict,
    output_path: str = "results/plots/infilling_comparison.png",
) -> None:
    """Bar chart comparing BLEU-4 and Generative PPL for infilling task."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    models = list(infilling_results.keys())
    bleu_vals = [infilling_results[m]["bleu4"] for m in models]
    ppl_vals = [infilling_results[m]["generative_ppl"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    bars1 = ax1.bar(x, bleu_vals, width, color=["#2196F3", "#FF5722", "#4CAF50"])
    ax1.set_xlabel("Model", fontsize=11)
    ax1.set_ylabel("BLEU-4 (↑ better)", fontsize=11)
    ax1.set_title("Text Infilling — BLEU-4", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(x, ppl_vals, width, color=["#2196F3", "#FF5722", "#4CAF50"])
    ax2.set_xlabel("Model", fontsize=11)
    ax2.set_ylabel("Generative PPL (↓ better)", fontsize=11)
    ax2.set_title("Text Infilling — Generative PPL", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Plot saved → {output_path}")


def plot_constrained_gen_comparison(
    cg_results: dict,
    output_path: str = "results/plots/constrained_gen_comparison.png",
) -> None:
    """Bar chart comparing keyword satisfaction rate, MAUVE, Gen PPL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    models = list(cg_results.keys())
    kw_vals = [cg_results[m]["keyword_satisfaction_rate"] * 100 for m in models]
    mauve_vals = [cg_results[m]["mauve"] for m in models]
    ppl_vals = [cg_results[m]["generative_ppl"] for m in models]

    x = np.arange(len(models))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, vals, ylabel, title in zip(
        axes,
        [kw_vals, mauve_vals, ppl_vals],
        ["Keyword Satisfaction (%)", "MAUVE (↑ better)", "Generative PPL (↓ better)"],
        ["Keyword Satisfaction Rate", "MAUVE Score", "Generative PPL"],
    ):
        ax.bar(x, vals, width, color=colors)
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Keyword-Constrained Generation Benchmark", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {output_path}")


def print_results_table(
    infilling_results: dict,
    cg_results: dict,
    unconditional_results: dict,
) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\n── Task 1: Text Infilling ──────────────────────────────────────────────")
    print(f"{'Model':<30} {'BLEU-4':>10} {'Gen PPL':>10}")
    print("-" * 55)
    for model, metrics in infilling_results.items():
        print(f"{model:<30} {metrics['bleu4']:>10.2f} {metrics['generative_ppl']:>10.2f}")

    print("\n── Task 2: Keyword-Constrained Generation ──────────────────────────────")
    print(f"{'Model':<30} {'KW-Sat (%)':>12} {'MAUVE':>8} {'Gen PPL':>10}")
    print("-" * 65)
    for model, metrics in cg_results.items():
        kw = metrics["keyword_satisfaction_rate"] * 100
        mv = metrics["mauve"]
        ppl = metrics["generative_ppl"]
        print(f"{model:<30} {kw:>12.1f} {mv:>8.4f} {ppl:>10.2f}")

    print("\n── Unconditional Generation (Generative PPL) ───────────────────────────")
    print(f"{'Model':<30} {'T':>8} {'Gen PPL':>10}")
    print("-" * 52)
    for model, entries in unconditional_results.items():
        for T, ppl in entries:
            t_str = str(T) if T is not None else "N/A (AR)"
            print(f"{model:<30} {t_str:>8} {ppl:>10.2f}")

    print("=" * 80 + "\n")


# ─── CSV writers ──────────────────────────────────────────────────────────────

def write_task1_csv(infilling_results: dict, output_path: str) -> None:
    """Emit the Task-1 results table (model × {BLEU-4, Gen PPL}) as CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "bleu4", "generative_ppl"])
        for model, metrics in infilling_results.items():
            w.writerow([model, metrics["bleu4"], metrics["generative_ppl"]])
    logger.info(f"CSV saved → {output_path}")


def write_task2_csv(cg_results: dict, output_path: str) -> None:
    """Emit the Task-2 results table (model × {KW-Sat, MAUVE, Gen PPL}) as CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "keyword_satisfaction_rate", "mauve", "generative_ppl"])
        for model, metrics in cg_results.items():
            w.writerow([
                model,
                metrics["keyword_satisfaction_rate"],
                metrics["mauve"],
                metrics["generative_ppl"],
            ])
    logger.info(f"CSV saved → {output_path}")


def write_unconditional_csv(unconditional_results: dict, output_path: str) -> None:
    """Emit the unconditional-PPL sweep (model × T × Gen PPL) as CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "diffusion_steps", "generative_ppl"])
        for model, entries in unconditional_results.items():
            for T, ppl in entries:
                w.writerow([model, T if T is not None else "N/A", ppl])
    logger.info(f"CSV saved → {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full evaluation & comparison")
    parser.add_argument("--diffusion_ckpt", default=None, help="Path to BertMoEDiffusion checkpoint")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--steps_sweep", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BertDiffused Full Evaluation")
    logger.info("=" * 60)

    # ── Task 1: Infilling ─────────────────────────────────────────────────────
    logger.info("\n[1/3] Running Text Infilling benchmark …")
    infilling_results = run_infilling_evaluation(
        diffusion_model_path=args.diffusion_ckpt,
        cfg=None,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        output_path=f"{args.output_dir}/infilling_results.json",
        device_str=args.device,
    )

    # ── Task 2: Keyword-Constrained Generation ────────────────────────────────
    logger.info("\n[2/3] Running Keyword-Constrained Generation benchmark …")
    cg_results = run_constrained_gen_evaluation(
        diffusion_model_path=args.diffusion_ckpt,
        cfg=None,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        output_path=f"{args.output_dir}/constrained_gen_results.json",
        device_str=args.device,
    )

    # ── Unconditional PPL sweep ───────────────────────────────────────────────
    logger.info("\n[3/3] Running Unconditional PPL sweep …")
    unconditional_results = evaluate_unconditional_ppl(
        diffusion_model_path=args.diffusion_ckpt,
        num_samples=min(args.num_samples, 100),
        steps_list=args.steps_sweep,
        device_str=args.device,
    )

    # ── Save consolidated results ─────────────────────────────────────────────
    all_results = {
        "infilling": infilling_results,
        "constrained_gen": cg_results,
        "unconditional_ppl_sweep": {
            k: [(str(t), p) for t, p in v]
            for k, v in unconditional_results.items()
        },
    }
    with open(f"{args.output_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── CSV tables ────────────────────────────────────────────────────────────
    write_task1_csv(infilling_results, f"{args.output_dir}/task1_table.csv")
    write_task2_csv(cg_results, f"{args.output_dir}/task2_table.csv")
    write_unconditional_csv(
        unconditional_results,
        f"{args.output_dir}/unconditional_ppl_table.csv",
    )

    # ── Print table ───────────────────────────────────────────────────────────
    print_results_table(infilling_results, cg_results, unconditional_results)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_diffusion_steps_vs_ppl(
        unconditional_results,
        output_path=f"{args.output_dir}/plots/diffusion_steps_ppl.png",
    )
    plot_infilling_comparison(
        infilling_results,
        output_path=f"{args.output_dir}/plots/infilling_comparison.png",
    )
    plot_constrained_gen_comparison(
        cg_results,
        output_path=f"{args.output_dir}/plots/constrained_gen_comparison.png",
    )

    logger.info("All evaluations complete. Results in results/")


if __name__ == "__main__":
    main()
