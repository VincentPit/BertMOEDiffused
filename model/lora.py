"""
LoRA (Low-Rank Adaptation) layer for parameter-efficient fine-tuning.

Implements LoRA as described in:
  Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)

Given a pretrained weight matrix W ∈ R^{d_out × d_in}, LoRA freezes W and
learns a low-rank update: W' = W + (α/r) · B @ A, where:
  - A ∈ R^{r × d_in}  (initialized from N(0, σ²))
  - B ∈ R^{d_out × r}  (initialized to zeros → ΔW = 0 at init)
  - r = rank (typically 4–64)
  - α = scaling factor

Only A and B are trained — the original W stays frozen.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank trainable adapter.

    The original weight is kept frozen. Only the LoRA matrices A, B are trained.

    Args:
        original_linear: The pretrained nn.Linear module to wrap.
        rank:            Rank r of the low-rank decomposition.
        alpha:           LoRA scaling factor α. Effective scale = α/r.
        dropout:         Dropout applied to input before the LoRA branch.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming uniform (same as nn.Linear)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B starts at zero → ΔW = 0 at initialization

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path
        base_out = self.original(x)
        # LoRA path: x @ A^T @ B^T * scaling
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into the original linear layer for inference.

        Returns a single nn.Linear with the merged weights (no overhead at
        inference time).
        """
        with torch.no_grad():
            merged = nn.Linear(
                self.original.in_features,
                self.original.out_features,
                bias=self.original.bias is not None,
            )
            merged.weight.copy_(
                self.original.weight + self.scaling * (self.lora_B @ self.lora_A)
            )
            if self.original.bias is not None:
                merged.bias.copy_(self.original.bias)
        return merged


def apply_lora_to_module(
    module: nn.Module,
    target_modules: Optional[list[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> dict[str, LoRALinear]:
    """Apply LoRA adapters to specific named Linear layers in a module.

    Args:
        module:          The parent nn.Module to modify in-place.
        target_modules:  List of submodule name fragments to match.
                         Default targets BERT's Q, K, V projection layers.
        rank:            LoRA rank.
        alpha:           LoRA alpha.
        dropout:         LoRA dropout.

    Returns:
        Dict mapping full parameter path → LoRALinear instance.
    """
    if target_modules is None:
        target_modules = ["query", "key", "value"]

    lora_layers: dict[str, LoRALinear] = {}

    for name, submodule in list(module.named_modules()):
        if not isinstance(submodule, nn.Linear):
            continue
        if not any(target in name for target in target_modules):
            continue

        # Navigate to the parent and replace the child
        parts = name.split(".")
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]

        lora_linear = LoRALinear(submodule, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_linear)
        lora_layers[name] = lora_linear

    return lora_layers


def merge_lora_weights(module: nn.Module) -> None:
    """Merge all LoRA layers in-place for zero-overhead inference."""
    for name, submodule in list(module.named_modules()):
        if isinstance(submodule, LoRALinear):
            parts = name.split(".")
            parent = module
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], submodule.merge_weights())
