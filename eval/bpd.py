"""
Test-set likelihood utilities for comparing diffusion and autoregressive models.

Two comparable scalars are exposed:

  • ``bpd_diffusion``        — MDLM NELBO per token on held-out sequences,
                               averaged over several noise levels.  Units: nats/token.
  • ``bpd_autoregressive``   — Standard cross-entropy of a causal LM on the
                               same sequences.  Units: nats/token.

Both are reported as bits-per-dim (nats × log2(e)) so a diffusion model and a
GPT-2 baseline can be placed on the same axis even though their training
objectives differ.

This module does no masking/tokenization of its own — callers pass in an
already-tokenized batch of input_ids tensors.  The caller is responsible for
picking a tokenizer that matches the scored model.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


_NATS_TO_BITS = 1.0 / math.log(2.0)


@torch.no_grad()
def diffusion_nelbo_per_token(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    noise_schedule,
    mask_token_id: int,
    num_t_samples: int = 8,
    device: Optional[torch.device] = None,
) -> float:
    """Monte-Carlo estimate of NELBO (nats/token) on a batch of clean sequences.

    For each sampled t ∈ (0, 1), noise the sequence, run the denoiser, then
    compute the MDLM loss 1/t · CE(x_theta, x) over positions that were masked.
    Average across ``num_t_samples`` t values and all real tokens to get a
    single scalar.

    Args:
        model:              BertMoEDiffusion instance (already .eval()).
        input_ids:          (B, L) clean token ids.
        attention_mask:     (B, L) 1 = real token, 0 = padding.  Used to exclude
                            padding from both masking and the average.
        noise_schedule:     LogLinearNoiseSchedule instance.
        mask_token_id:      [MASK] token id.
        num_t_samples:      Number of Monte-Carlo t draws to average.
        device:             If None, uses input_ids.device.

    Returns:
        Mean NELBO in nats per real token (float).
    """
    if device is None:
        device = input_ids.device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    B, L = input_ids.shape
    if attention_mask is None:
        attention_mask = torch.ones(B, L, dtype=torch.long, device=device)

    per_sample_losses: List[float] = []
    for _ in range(num_t_samples):
        t = noise_schedule.sample_t(B, device)
        z_t = noise_schedule.noise_sequence(input_ids, t, mask_token_id)

        # Keep padding positions clean and excluded from the loss.
        pad = attention_mask == 0
        z_t[pad] = input_ids[pad]

        logits = model(z_t, t, attention_mask=attention_mask)  # (B, L, V), SUBS applied

        # CE only on positions that were masked AND are real tokens.
        masked = (z_t == mask_token_id) & (~pad)
        if masked.sum() == 0:
            continue

        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            input_ids.reshape(-1),
            reduction="none",
        ).reshape(B, L)

        # MDLM time weight = 1 / t (applied per sequence, broadcast to positions).
        w = noise_schedule.time_weight(t).unsqueeze(1)  # (B, 1)
        per_token_loss = (ce * masked.float() * w).sum() / masked.float().sum().clamp(min=1)
        per_sample_losses.append(per_token_loss.item())

    return float(np.mean(per_sample_losses)) if per_sample_losses else float("nan")


@torch.no_grad()
def ar_cross_entropy_per_token(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> float:
    """Mean cross-entropy (nats/token) of a causal LM on a list of texts.

    Uses the model's own tokenizer; counts only non-padding positions when
    averaging.  Comparable to ``diffusion_nelbo_per_token`` as both are in
    nats/token.
    """
    if device is None:
        device = next(model.parameters()).device
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        logits = model(**enc).logits  # (B, L, V)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = enc["input_ids"][:, 1:].contiguous()
        shift_mask = enc["attention_mask"][:, 1:].contiguous().bool()

        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(shift_labels.shape)

        total_nll += (ce * shift_mask.float()).sum().item()
        total_tokens += shift_mask.sum().item()

    return total_nll / max(total_tokens, 1)


def nats_to_bits(nats: float) -> float:
    """Convert nats/token to bits/token for reporting."""
    return nats * _NATS_TO_BITS
