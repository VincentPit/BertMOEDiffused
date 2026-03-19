"""
Mixture-of-Experts (MoE) feed-forward layer.

Architecture:
  - Token-choice sparse routing: each token is processed by top-k experts.
  - Router: linear projection to num_experts logits + softmax.
  - Auxiliary losses:
      * Load-balancing loss (Switch Transformer style): encourages uniform
        expert utilization across the batch.
      * Z-loss (ST-MoE style): penalises large router logits, stabilises training.

References:
  [1] Fedus et al. "Switch Transformers" (2021)
  [2] Zoph et al. "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (2022)
  [3] Jiang et al. "Mixtral of Experts" (2024)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class MoERouter(nn.Module):
    """Linear router that produces per-token expert probabilities.

    Args:
        hidden_size: Dimensionality of input token representations.
        num_experts: Total number of experts.
        top_k: Number of experts each token routes to.
        router_jitter: Uniform noise in [0, jitter) added to logits during
                       training for exploration (Lepikhin et al., GShard).
        z_loss_coef: Weight for the router z-loss auxiliary term.
        aux_loss_coef: Weight for the load-balancing auxiliary loss.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        router_jitter: float = 0.01,
        z_loss_coef: float = 1e-3,
        aux_loss_coef: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter = router_jitter
        self.z_loss_coef = z_loss_coef
        self.aux_loss_coef = aux_loss_coef

        # Weights initialised with small values (Lepikhin et al.)
        self.weight = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.weight.weight, mean=0.0, std=0.02)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            router_weights: (batch, seq_len, top_k) — softmax weights for
                            selected experts, used to combine expert outputs.
            expert_indices: (batch, seq_len, top_k) — indices of selected experts.
            aux_loss:       scalar — sum of load-balancing and z-losses.
        """
        B, L, H = hidden_states.shape

        # Flatten to (B*L, H) for routing
        flat = hidden_states.reshape(-1, H)   # (B*L, H)

        # Router logits with optional jitter noise during training
        logits = self.weight(flat)             # (B*L, num_experts)
        if self.training and self.router_jitter > 0:
            noise = torch.empty_like(logits).uniform_(0, self.router_jitter)
            logits = logits + noise

        # ---------- Z-loss (stabilises large logits) ----------
        # L_z = (1/B*L) * sum_i log^2( sum_e exp(logit_{i,e}) )
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        z_loss = self.z_loss_coef * z_loss

        # ---------- Top-k routing ----------
        router_probs = F.softmax(logits, dim=-1)  # (B*L, num_experts)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalise the top-k weights so they sum to 1 per token
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # ---------- Load-balancing auxiliary loss (Switch Transformer) ----------
        # f_i = fraction of tokens routed to expert i (hard assignment of argmax)
        # P_i = mean router probability for expert i across tokens
        # L_aux = num_experts * sum_i f_i * P_i
        with torch.no_grad():
            top1_indices = top_k_indices[:, 0]                  # (B*L,)
            # one-hot mask for each expert's token assignment
            expert_mask = F.one_hot(top1_indices, self.num_experts).float()  # (B*L, E)
            f = expert_mask.mean(0)                              # (E,) fraction per expert
        P = router_probs.mean(0)                                 # (E,) mean prob per expert
        aux_loss = self.aux_loss_coef * self.num_experts * (f * P).sum()

        total_aux_loss = z_loss + aux_loss

        # Reshape back to (B, L, top_k)
        router_weights = top_k_probs.view(B, L, self.top_k)
        expert_indices = top_k_indices.view(B, L, self.top_k)

        return router_weights, expert_indices, total_aux_loss


# ---------------------------------------------------------------------------
# Single expert (standard FFN block)
# ---------------------------------------------------------------------------

class ExpertFFN(nn.Module):
    """Single expert — a standard 2-layer FFN with GELU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


# ---------------------------------------------------------------------------
# MoE Feed-Forward Layer
# ---------------------------------------------------------------------------

class MoEFeedForward(nn.Module):
    """Sparse MoE drop-in replacement for a standard BERT FFN sub-layer.

    Each token is routed to ``top_k`` out of ``num_experts`` expert FFNs.
    Outputs are a weighted combination of the selected experts' outputs.

    The module collects and exposes auxiliary losses via ``self.aux_loss``
    after each forward pass, so the training loop can add them to the main
    diffusion ELBO loss.

    Args:
        hidden_size: Model hidden dimension (768 for BERT-base).
        intermediate_size: Inner FFN dimension for each expert.
        num_experts: Number of parallel expert FFNs.
        top_k: Number of experts each token uses.
        dropout: Dropout probability inside each expert.
        router_jitter: Noise added to router logits during training.
        z_loss_coef: Z-loss coefficient for routing stability.
        aux_loss_coef: Load-balancing loss coefficient.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        router_jitter: float = 0.01,
        z_loss_coef: float = 1e-3,
        aux_loss_coef: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = MoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            router_jitter=router_jitter,
            z_loss_coef=z_loss_coef,
            aux_loss_coef=aux_loss_coef,
        )
        self.experts = nn.ModuleList(
            [ExpertFFN(hidden_size, intermediate_size, dropout) for _ in range(num_experts)]
        )

        # Populated after each forward pass; training loop reads this.
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            output: (batch, seq_len, hidden_size)
        """
        B, L, H = hidden_states.shape

        router_weights, expert_indices, aux_loss = self.router(hidden_states)
        self.aux_loss = aux_loss

        # Flatten tokens for expert dispatch
        flat = hidden_states.reshape(B * L, H)          # (N, H)  N = B*L
        flat_indices = expert_indices.reshape(B * L, self.top_k)   # (N, top_k)
        flat_weights = router_weights.reshape(B * L, self.top_k)   # (N, top_k)

        output = torch.zeros_like(flat)                  # (N, H)

        # Token-choice dispatch: for each expert, gather its assigned tokens,
        # run them through the expert, then scatter the weighted result back.
        for expert_id, expert in enumerate(self.experts):
            # Find all (token, position) pairs where this expert is selected
            token_mask, k_pos = torch.where(flat_indices == expert_id)
            # token_mask: (M,) — which tokens selected this expert
            # k_pos:      (M,) — at which top-k slot (0 or 1, etc.)
            if token_mask.numel() == 0:
                continue

            expert_input = flat[token_mask]              # (M, H)
            expert_out = expert(expert_input)            # (M, H)
            weights = flat_weights[token_mask, k_pos].unsqueeze(-1)  # (M, 1)

            output.index_add_(0, token_mask, weights * expert_out)

        return output.reshape(B, L, H)
