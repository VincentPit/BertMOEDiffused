"""
MDLM training loop for BertMoEDiffusion.

Training objective
──────────────────
  L_total = L_ELBO + lambda_moe * L_moe_aux

  L_ELBO = E_{t~U[0,1], z_t~q(z_t|x)} [ w(t) * CE(x_theta(z_t, t), x) ]
         = E [ (1/t) * sum_{l: z_t^l = MASK} CE(x_theta^l, x^l) ]

  w(t) = 1/t  for the log-linear schedule alpha(t) = 1 - t.
  (MDLM Eq. 8; only masked positions contribute — carry-over unmasking
   zeros out the unmasked positions' gradient automatically via SUBS.)

  L_moe_aux = sum over MoE layers of (z_loss + load_balancing_loss)
             (collected from moe_ffn.aux_loss in each BertLayer)

Usage
─────
  python train.py                          # uses configs/config.yaml
  python train.py training.max_steps=5000  # override with Hydra-style args
"""

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from model import BertMoEDiffusion, LogLinearNoiseSchedule
from data.lm1b_dataset import LM1BDataset
from data.etl import ProcessedParquetDataset

logger = logging.getLogger(__name__)


# ─── Utility helpers ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_cfg(cfg: dict, prefix: str = "") -> dict:
    """Flatten nested dict with dot-separated keys."""
    flat = {}
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_cfg(v, full_key))
        else:
            flat[full_key] = v
    return flat


# ─── MDLM loss ────────────────────────────────────────────────────────────────

def compute_mdlm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    time_eps: float = 1e-4,
) -> torch.Tensor:
    """Compute the MDLM continuous-time ELBO loss for a batch.

    Only masked positions in z_t contribute to the loss (carry-over unmasking
    means unmasked positions are reproduced exactly, so their CE is 0 after SUBS).
    The per-sequence loss is weighted by w(t) = 1/t (log-linear schedule).

    Args:
        logits:        (B, L, V) — model output (SUBS already applied).
        input_ids:     (B, L)   — clean ground-truth token ids.
        z_t:           (B, L)   — noised input token ids.
        t:             (B,)     — timesteps used to produce z_t.
        mask_token_id: Integer id of [MASK] token.
        time_eps:      Minimum t value to avoid 1/t singularity.

    Returns:
        loss: scalar — mean over the batch.
    """
    B, L, V = logits.shape

    # Weight w(t) = 1/t  (clamped to avoid infinity near t=0)
    weights = 1.0 / t.clamp(min=time_eps)  # (B,)

    # Identify masked positions
    is_masked = (z_t == mask_token_id)     # (B, L) bool

    # Cross-entropy loss at all positions; shape (B, L)
    # We use reduction='none' and manually mask non-masked positions.
    ce = F.cross_entropy(
        logits.reshape(B * L, V),
        input_ids.reshape(B * L),
        reduction='none',
    ).reshape(B, L)                        # (B, L)

    # Zero out unmasked positions (SUBS ensures these are already correct)
    ce = ce * is_masked.float()            # (B, L)

    # Sum over positions, then weight by 1/t, then average over batch
    loss_per_seq = ce.sum(-1)              # (B,)
    loss = (weights * loss_per_seq).mean()
    return loss


# ─── Evaluation: bits-per-dimension (BPD) ────────────────────────────────────

@torch.no_grad()
def evaluate_bpd(
    model: BertMoEDiffusion,
    dataloader: DataLoader,
    noise_schedule: LogLinearNoiseSchedule,
    mask_token_id: int,
    device: torch.device,
    num_eval_steps: int = 200,
    time_eps: float = 1e-4,
) -> float:
    """Estimate the NELBO in bits-per-dimension on a held-out set.

    Uses Monte-Carlo integration over t with ``num_eval_steps`` timesteps.
    BPD = NELBO / (L * log 2).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        B, L = input_ids.shape

        # Average over multiple timesteps per batch for lower-variance estimate
        batch_loss = 0.0
        for _ in range(num_eval_steps):
            t = noise_schedule.sample_t(B, device, low_discrepancy=True)
            z_t = noise_schedule.noise_sequence(input_ids, t, mask_token_id)
            logits = model(z_t, t, attention_mask=attention_mask)
            loss = compute_mdlm_loss(logits, input_ids, z_t, t, mask_token_id, time_eps)
            batch_loss += loss.item()

        total_loss += batch_loss / num_eval_steps
        total_tokens += B * L
        n_batches += 1

        if n_batches >= 50:   # Cap at 50 batches for speed
            break

    avg_nll = total_loss / n_batches   # nats per sequence (summed over L)
    # Convert to BPD: nll_per_token / log(2)
    bpd = avg_nll / (math.log(2) * (total_tokens / (n_batches * B)))
    return bpd


# ─── Main training function ───────────────────────────────────────────────────

def train(cfg: dict) -> None:
    # ── Setup ──────────────────────────────────────────────────────────────────
    set_seed(cfg["training"]["seed"])
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log"),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── MLflow setup ───────────────────────────────────────────────────────────
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        mlflow_cfg.get("tracking_uri", "mlruns"),
    )
    experiment_name = mlflow_cfg.get("experiment_name", "BertMoEDiffusion")
    run_name = mlflow_cfg.get("run_name", None)
    registry_model_name = mlflow_cfg.get("registered_model_name", "BertMoEDiffusion")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Enable system metrics logging (CPU/GPU/memory)
    mlflow.enable_system_metrics_logging()

    mlflow_run = mlflow.start_run(run_name=run_name)
    logger.info(f"MLflow run started: {mlflow_run.info.run_id}")

    # Log all config parameters (flattened)
    mlflow.log_params(flatten_cfg(cfg))

    # Log device info as tags
    mlflow.set_tags({
        "device": str(device),
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    })

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    mask_token_id: int = tokenizer.mask_token_id

    # ── Dataset ────────────────────────────────────────────────────────────────
    # Prefer preprocessed Parquet shards from ETL pipeline; fall back to raw
    etl_output_dir = Path(cfg.get("etl", {}).get("output_dir", "data/processed"))
    train_etl_dir = etl_output_dir / cfg["training"]["dataset_split_train"]
    eval_etl_dir = etl_output_dir / cfg["training"]["dataset_split_eval"]

    if train_etl_dir.exists() and any(train_etl_dir.glob("shard-*.parquet")):
        logger.info(f"Using preprocessed ETL data from {train_etl_dir}")
        train_dataset = ProcessedParquetDataset(train_etl_dir)
        mlflow.set_tag("data_source", "etl_parquet")
    else:
        logger.info("No ETL shards found — using raw on-the-fly tokenization")
        train_dataset = LM1BDataset(
            split=cfg["training"]["dataset_split_train"],
            tokenizer=tokenizer,
            max_seq_len=cfg["model"]["max_seq_len"],
        )
        mlflow.set_tag("data_source", "raw_hf")

    if eval_etl_dir.exists() and any(eval_etl_dir.glob("shard-*.parquet")):
        eval_dataset = ProcessedParquetDataset(eval_etl_dir)
    else:
        eval_dataset = LM1BDataset(
            split=cfg["training"]["dataset_split_eval"],
            tokenizer=tokenizer,
            max_seq_len=cfg["model"]["max_seq_len"],
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg["evaluation"]["eval_batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    moe_cfg = cfg["model"]["moe"]
    lora_cfg = cfg["model"].get("lora", {})
    lora_enabled = lora_cfg.get("enabled", False)

    model = BertMoEDiffusion(
        bert_model_name=cfg["model"]["backbone"],
        moe_layers=moe_cfg["moe_layers"],
        num_experts=moe_cfg["num_experts"],
        num_experts_per_token=moe_cfg["num_experts_per_token"],
        expert_hidden_multiplier=moe_cfg["expert_hidden_multiplier"],
        router_jitter=moe_cfg["router_jitter"],
        router_z_loss_coef=moe_cfg["router_z_loss_coef"],
        router_aux_loss_coef=moe_cfg["router_aux_loss_coef"],
        time_embed_dim=cfg["model"]["time_embed_dim"],
        use_time_conditioning=cfg["model"]["use_time_conditioning"],
        dropout=cfg["model"]["dropout"],
        lora_enabled=lora_enabled,
        lora_rank=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 16.0),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        lora_target_modules=lora_cfg.get("target_modules", None),
    )
    model.set_mask_token_id(mask_token_id)
    model.to(device)

    # Parameter breakdown (LoRA-aware)
    param_summary = model.trainable_parameters_summary()
    logger.info(
        f"Model parameters: {param_summary['total']:,} total, "
        f"{param_summary['trainable']:,} trainable ({param_summary['trainable_pct']:.1f}%), "
        f"{param_summary['frozen']:,} frozen"
    )
    logger.info(f"MoE parameters: {param_summary['moe']:,} (across {len(model.moe_layers_list)} MoE layers)")
    if lora_enabled:
        logger.info(
            f"LoRA parameters: {param_summary['lora']:,} "
            f"(rank={lora_cfg.get('rank', 8)}, alpha={lora_cfg.get('alpha', 16.0)})"
        )

    # Log model architecture stats to MLflow
    mlflow.log_metrics({
        "model/total_params": param_summary["total"],
        "model/trainable_params": param_summary["trainable"],
        "model/frozen_params": param_summary["frozen"],
        "model/trainable_pct": param_summary["trainable_pct"],
        "model/moe_params": param_summary["moe"],
        "model/lora_params": param_summary["lora"],
        "model/num_moe_layers": len(model.moe_layers_list),
    })
    mlflow.set_tag("lora_enabled", str(lora_enabled))

    # ── Noise schedule ─────────────────────────────────────────────────────────
    noise_schedule = LogLinearNoiseSchedule()

    # ── Optimizer & scheduler ──────────────────────────────────────────────────
    # Only pass trainable parameters to the optimizer (critical when using LoRA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["training"]["learning_rate"],
        betas=(cfg["training"]["adam_beta1"], cfg["training"]["adam_beta2"]),
        eps=cfg["training"]["adam_epsilon"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["training"]["warmup_steps"],
        num_training_steps=cfg["training"]["max_steps"],
    )

    # ── Mixed precision ────────────────────────────────────────────────────────
    use_fp16 = cfg["training"]["fp16"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    global_step = 0
    resume_path = cfg["training"].get("resume_from_checkpoint")
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["global_step"]

    # ── Training loop ──────────────────────────────────────────────────────────
    accum_steps = cfg["training"]["gradient_accumulation_steps"]
    max_steps = cfg["training"]["max_steps"]
    log_steps = cfg["training"]["log_steps"]
    eval_steps = cfg["training"]["eval_steps"]
    save_steps = cfg["training"]["save_steps"]
    time_eps = cfg["diffusion"]["time_eps"]

    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    running_moe_loss = 0.0

    train_iter = iter(train_loader)

    logger.info(f"Starting training — max_steps={max_steps}, device={device}")

    while global_step < max_steps:
        # ── Get next batch (cycle through the dataset) ─────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        B, L = input_ids.shape

        # ── Sample timesteps ───────────────────────────────────────────────────
        t = noise_schedule.sample_t(B, device, low_discrepancy=True)

        # ── Apply forward (noising) process ───────────────────────────────────
        z_t = noise_schedule.noise_sequence(input_ids, t, mask_token_id)

        # ── Forward pass ───────────────────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=use_fp16):
            logits = model(z_t, t, attention_mask=attention_mask)  # (B, L, V)

            # MDLM ELBO loss
            elbo_loss = compute_mdlm_loss(
                logits, input_ids, z_t, t, mask_token_id, time_eps
            )

            # MoE auxiliary loss
            moe_aux = model.moe_aux_loss
            total_loss = elbo_loss + moe_aux

            # Normalise for gradient accumulation
            total_loss = total_loss / accum_steps

        # ── Backward pass ──────────────────────────────────────────────────────
        scaler.scale(total_loss).backward()
        running_loss += elbo_loss.item()
        running_moe_loss += moe_aux.item() if isinstance(moe_aux, torch.Tensor) else moe_aux

        # ── Optimizer step (every accum_steps mini-batches) ────────────────────
        if (global_step + 1) % accum_steps == 0 or global_step == max_steps - 1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["max_grad_norm"]
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        # ── Logging ────────────────────────────────────────────────────────────
        if global_step % log_steps == 0:
            avg_loss = running_loss / log_steps
            avg_moe = running_moe_loss / log_steps
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Step {global_step:6d} | ELBO loss {avg_loss:.4f} | "
                f"MoE aux {avg_moe:.4f} | LR {lr:.2e}"
            )
            mlflow.log_metrics(
                {
                    "train/elbo_loss": avg_loss,
                    "train/moe_aux_loss": avg_moe,
                    "train/learning_rate": lr,
                    "train/epoch": global_step / len(train_loader),
                },
                step=global_step,
            )
            running_loss = 0.0
            running_moe_loss = 0.0

        # ── Evaluation ─────────────────────────────────────────────────────────
        if global_step % eval_steps == 0:
            bpd = evaluate_bpd(
                model, eval_loader, noise_schedule, mask_token_id, device, time_eps=time_eps
            )
            logger.info(f"Step {global_step:6d} | Eval BPD: {bpd:.4f}")
            mlflow.log_metric("eval/bpd", bpd, step=global_step)
            model.train()

        # ── Checkpoint ─────────────────────────────────────────────────────────
        if global_step % save_steps == 0:
            ckpt_path = output_dir / f"checkpoint-{global_step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint → {ckpt_path}")
            mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

    # ── Final save ─────────────────────────────────────────────────────────────
    # Save LoRA adapters separately (lightweight checkpoint for sharing)
    if lora_enabled and model.lora_enabled:
        lora_path = output_dir / "lora_adapters.pt"
        lora_state = {
            k: v for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k
        }
        torch.save({"lora_state_dict": lora_state, "config": cfg}, lora_path)
        logger.info(f"LoRA adapters saved → {lora_path}")
        mlflow.log_artifact(str(lora_path), artifact_path="lora_adapters")

        # Merge LoRA into base weights for efficient inference
        model.merge_lora()
        logger.info("LoRA weights merged into base model for inference.")

    final_path = output_dir / "final_model.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, final_path)
    logger.info(f"Training complete. Final model saved → {final_path}")

    # ── MLflow: log final model to Model Registry ─────────────────────────────
    # Log the model with a custom PyFunc wrapper for serving
    mlflow.log_artifact(str(final_path), artifact_path="final_model")
    mlflow.log_artifact("configs/config.yaml", artifact_path="config")

    # Register model via MLflow PyTorch flavor
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=registry_model_name,
        pip_requirements=[
            "torch>=2.2.0",
            "transformers>=4.40.0",
            "mlflow>=2.14.0",
        ],
    )
    logger.info(f"Model registered in MLflow as '{registry_model_name}'")

    # End MLflow run
    mlflow.end_run()
    logger.info("MLflow run ended.")


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BertMoEDiffusion")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    # Allow dot-notation overrides: e.g. training.max_steps=5000
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    for override in args.overrides:
        key, _, value = override.partition("=")
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        # Attempt type coercion
        existing = node.get(keys[-1])
        if isinstance(existing, bool):
            node[keys[-1]] = value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            node[keys[-1]] = int(value)
        elif isinstance(existing, float):
            node[keys[-1]] = float(value)
        else:
            node[keys[-1]] = value

    train(cfg)
