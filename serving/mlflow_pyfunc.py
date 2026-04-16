"""
Custom MLflow PyFunc model for BertMoEDiffusion serving.

Wraps the model so MLflow model serving (`mlflow models serve`) can
produce text via the standard predict() interface.

Supports two modes:
  1. "infill"   — given a prompt with [MASK] tokens, infill them.
  2. "generate" — unconditional generation from all [MASK] tokens.

Input schema (pd.DataFrame or dict):
  - prompt:  str or list[str]        — input text(s) with optional [MASK]
  - mode:    "infill" | "generate"   — generation mode (default: "generate")
  - steps:   int                     — diffusion steps  (default: 100)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from model import BertMoEDiffusion, LogLinearNoiseSchedule

logger = logging.getLogger(__name__)


class BertDiffusionPyFunc(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper for BertMoEDiffusion inference."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load model artifacts (called once when the model is loaded for serving)."""
        import yaml

        # Load config
        config_path = context.artifacts.get("config", None)
        if config_path:
            with open(config_path) as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {}

        # Load the PyTorch model from the logged artifact
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = context.artifacts["model"]
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        model_cfg = ckpt.get("config", self.cfg)
        moe_cfg = model_cfg.get("model", {}).get("moe", {})

        self.model = BertMoEDiffusion(
            bert_model_name=model_cfg.get("model", {}).get("backbone", "bert-base-uncased"),
            moe_layers=moe_cfg.get("moe_layers", [3, 5, 7, 9, 11]),
            num_experts=moe_cfg.get("num_experts", 8),
            num_experts_per_token=moe_cfg.get("num_experts_per_token", 2),
            expert_hidden_multiplier=moe_cfg.get("expert_hidden_multiplier", 4),
            router_jitter=0.0,  # no jitter at inference
            router_z_loss_coef=moe_cfg.get("router_z_loss_coef", 1e-3),
            router_aux_loss_coef=moe_cfg.get("router_aux_loss_coef", 1e-2),
            time_embed_dim=model_cfg.get("model", {}).get("time_embed_dim", 128),
            use_time_conditioning=model_cfg.get("model", {}).get("use_time_conditioning", True),
            dropout=0.0,
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

        # Tokenizer
        backbone = model_cfg.get("model", {}).get("backbone", "bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.model.set_mask_token_id(self.mask_token_id)
        self.noise_schedule = LogLinearNoiseSchedule()

        logger.info("BertDiffusionPyFunc model loaded successfully.")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Run inference.

        Args:
            model_input: DataFrame with columns: prompt, mode (optional), steps (optional)
            params:      Optional dict with override params.

        Returns:
            DataFrame with column 'generated_text'.
        """
        prompts = model_input["prompt"].tolist()
        mode = model_input.get("mode", pd.Series(["generate"] * len(prompts))).tolist()
        steps_col = model_input.get("steps", pd.Series([100] * len(prompts)))
        num_steps = int(steps_col.iloc[0])

        results = []
        for prompt_text, gen_mode in zip(prompts, mode):
            text = self._run_diffusion(prompt_text, gen_mode, num_steps)
            results.append(text)

        return pd.DataFrame({"generated_text": results})

    @torch.no_grad()
    def _run_diffusion(self, prompt: str, mode: str, num_steps: int) -> str:
        """Run reverse diffusion to generate or infill text."""
        max_len = self.cfg.get("model", {}).get("max_seq_len", 128)

        if mode == "infill":
            encoded = self.tokenizer(
                prompt, return_tensors="pt", max_length=max_len,
                padding="max_length", truncation=True
            )
            z_t = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
        else:
            # Full generation: start from all [MASK]
            z_t = torch.full((1, max_len), self.mask_token_id, device=self.device)
            attention_mask = torch.ones(1, max_len, device=self.device)

        # Iterative denoising (MDLM-style reverse process)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)

        for i in range(num_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = t_now.unsqueeze(0)

            logits = self.model(z_t, t_batch, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)

            # Unmask probability: fraction of masks to reveal this step
            is_masked = (z_t == self.mask_token_id)
            if not is_masked.any():
                break

            # Sample tokens for masked positions
            sampled = torch.multinomial(
                probs.view(-1, probs.size(-1)), num_samples=1
            ).view(z_t.shape)

            # Determine which masks to reveal (proportional to schedule)
            unmask_prob = (t_now - t_next) / t_now if t_now > 0 else 1.0
            unmask_mask = torch.bernoulli(
                torch.full_like(z_t, unmask_prob, dtype=torch.float)
            ).bool() & is_masked

            z_t = torch.where(unmask_mask, sampled, z_t)

        decoded = self.tokenizer.decode(z_t[0], skip_special_tokens=True)
        return decoded
