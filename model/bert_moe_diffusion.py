"""
BertMoEDiffusion: BERT-base with selective Mixture-of-Experts FFN layers,
fine-tuned as a masked diffusion language model denoiser.

Architecture overview
─────────────────────
  Input: noised token sequence z_t  (some tokens replaced by [MASK])
         + scalar timestep t ∈ [0, 1]

  1. Token embeddings   (BertEmbeddings — unchanged from bert-base-uncased)
  2. Time embedding     Sinusoidal(t) → Linear → hidden_size
                        Added to every token's embedding at the input layer.
  3. Transformer layers 0 … 11 (BertLayer):
       • Layers NOT in moe_layers: standard BERT FFN (BertIntermediate+BertOutput)
       • Layers IN moe_layers:     FFN replaced by MoEFeedForward
  4. MLM prediction head (BertOnlyMLMHead — unchanged)
  5. SUBS post-processing (applied after the head, NOT a learned layer):
       a. Zero masking probabilities: logits[:,:,MASK_ID] = -inf
       b. Carry-over unmasking:       for unmasked input tokens, override
                                      output logits with a one-hot at that token.

SUBS parameterisation ensures:
  - The model never predicts [MASK] as the clean token.
  - Unmasked positions are always reproduced exactly (no gradient wasted).

MoE auxiliary losses:
  - Collected from every MoE layer after each forward pass.
  - Exposed via self.moe_aux_loss (scalar) for the training loop to add to
    the main MDLM ELBO loss.

Reference: Sahoo et al., "Simple and Effective Masked Diffusion Language Models"
           (NeurIPS 2024), Section 3 + Appendix D.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM

from .moe_layer import MoEFeedForward
from .lora import LoRALinear, apply_lora_to_module, merge_lora_weights


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for a scalar timestep t ∈ [0, 1].

    Maps t → R^{embed_dim} via sin/cos at geometrically spaced frequencies,
    then projects with a 2-layer MLP to hidden_size so it can be added
    directly to token embeddings.
    """

    def __init__(self, embed_dim: int, hidden_size: int) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for sinusoidal embedding"
        self.embed_dim = embed_dim
        # Two-layer MLP projection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) float tensor with values in [0, 1].
        Returns:
            emb: (B, hidden_size) time embeddings.
        """
        half = self.embed_dim // 2
        # Frequencies: geometric sequence from 1 to 10000
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )                                       # (half,)
        args = t.unsqueeze(-1) * freq           # (B, half)  — outer product
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)
        return self.proj(emb)                   # (B, hidden_size)


# ---------------------------------------------------------------------------
# BertMoEDiffusion
# ---------------------------------------------------------------------------

class BertMoEDiffusion(nn.Module):
    """BERT-base denoiser with MoE FFN layers for masked diffusion LM.

    Args:
        bert_model_name:        HuggingFace checkpoint name or path.
        moe_layers:             Indices of transformer layers whose FFN is
                                replaced by MoE (0-indexed, up to 11 for BERT-base).
        num_experts:            Total experts per MoE layer.
        num_experts_per_token:  top-k routing.
        expert_hidden_multiplier: Each expert's inner dim = hidden * multiplier.
        router_jitter:          Training-time router noise.
        router_z_loss_coef:     Z-loss weight.
        router_aux_loss_coef:   Load-balancing loss weight.
        time_embed_dim:         Sinusoidal embedding dimension (projected to hidden_size).
        use_time_conditioning:  Whether to inject the time embedding into tokens.
        dropout:                Dropout in expert FFNs.
        lora_enabled:           Whether to apply LoRA adapters to attention layers.
        lora_rank:              Rank of LoRA low-rank decomposition.
        lora_alpha:             LoRA scaling factor (effective scale = alpha/rank).
        lora_dropout:           Dropout in LoRA branch.
        lora_target_modules:    Which linear layers to apply LoRA to.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        moe_layers: Optional[List[int]] = None,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_hidden_multiplier: int = 4,
        router_jitter: float = 0.01,
        router_z_loss_coef: float = 1e-3,
        router_aux_loss_coef: float = 1e-2,
        time_embed_dim: int = 128,
        use_time_conditioning: bool = True,
        dropout: float = 0.1,
        lora_enabled: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        if moe_layers is None:
            moe_layers = [3, 5, 7, 9, 11]   # every other upper layer gets MoE

        self.moe_layer_indices = set(moe_layers)
        self.use_time_conditioning = use_time_conditioning

        # ── Load pretrained BERT ──────────────────────────────────────────────
        self.bert = BertForMaskedLM.from_pretrained(bert_model_name)
        cfg: BertConfig = self.bert.config
        hidden_size: int = cfg.hidden_size
        intermediate_size: int = cfg.intermediate_size
        self.mask_token_id: int = cfg.vocab_size - 1   # will be overwritten by tokenizer
        self.vocab_size: int = cfg.vocab_size

        # ── Time conditioning ─────────────────────────────────────────────────
        if use_time_conditioning:
            self.time_embed = SinusoidalTimeEmbedding(time_embed_dim, hidden_size)
        else:
            self.time_embed = None

        # ── Swap selected FFN layers with MoE ─────────────────────────────────
        # BERT's transformer layers live at:
        #   self.bert.bert.encoder.layer[i]   → BertLayer
        #     .intermediate                   → BertIntermediate (fc1 + GELU)
        #     .output                         → BertOutput (fc2 + LN + dropout)
        #
        # Strategy: replace each targeted BertLayer's FFN sub-layers with a
        # MoEFeedForward module wrapped in a thin adapter that mirrors the
        # BertOutput interface (returns hidden_states and attention tensors).

        self.moe_layers_list: nn.ModuleList = nn.ModuleList()
        moe_layer_map: dict = {}   # layer_index → MoEFeedForward

        encoder = self.bert.bert.encoder
        for idx, bert_layer in enumerate(encoder.layer):
            if idx in self.moe_layer_indices:
                moe_ffn = MoEFeedForward(
                    hidden_size=hidden_size,
                    intermediate_size=hidden_size * expert_hidden_multiplier,
                    num_experts=num_experts,
                    top_k=num_experts_per_token,
                    dropout=dropout,
                    router_jitter=router_jitter,
                    z_loss_coef=router_z_loss_coef,
                    aux_loss_coef=router_aux_loss_coef,
                )
                # Replace the BERT layer's FFN sub-modules
                _patch_bert_layer_with_moe(bert_layer, moe_ffn, hidden_size, dropout)
                moe_layer_map[idx] = moe_ffn
                self.moe_layers_list.append(moe_ffn)

        self._moe_layer_map = moe_layer_map

        # Accumulated MoE auxiliary loss from the last forward pass
        self.moe_aux_loss: torch.Tensor = torch.tensor(0.0)

        # ── LoRA adapters on attention projections ────────────────────────────
        self.lora_enabled = lora_enabled
        self.lora_layers: dict = {}
        if lora_enabled:
            if lora_target_modules is None:
                lora_target_modules = ["query", "key", "value"]
            self.lora_layers = apply_lora_to_module(
                self.bert,
                target_modules=lora_target_modules,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            # Freeze all BERT parameters except LoRA, MoE, and time embedding
            self._freeze_base_bert()

    # ── Freeze / unfreeze helpers ────────────────────────────────────────────

    def _freeze_base_bert(self) -> None:
        """Freeze all base BERT parameters. Only LoRA, MoE, and time embedding remain trainable."""
        # First, freeze everything in self.bert
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters
        for lora_layer in self.lora_layers.values():
            lora_layer.lora_A.requires_grad = True
            lora_layer.lora_B.requires_grad = True

        # Unfreeze MoE layers
        for moe_ffn in self.moe_layers_list:
            for param in moe_ffn.parameters():
                param.requires_grad = True

        # Unfreeze MoE wrapper LayerNorms
        for layer_idx in self.moe_layer_indices:
            bert_layer = self.bert.bert.encoder.layer[layer_idx]
            if hasattr(bert_layer.output, "layer_norm"):
                for param in bert_layer.output.layer_norm.parameters():
                    param.requires_grad = True

        # Unfreeze time embedding (new parameters, not pretrained)
        if self.time_embed is not None:
            for param in self.time_embed.parameters():
                param.requires_grad = True

    def merge_lora(self) -> None:
        """Merge LoRA weights into base BERT for zero-overhead inference."""
        if self.lora_enabled and self.lora_layers:
            merge_lora_weights(self.bert)
            self.lora_layers = {}
            self.lora_enabled = False

    def trainable_parameters_summary(self) -> dict:
        """Return a dict summarizing total vs. trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        lora_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "lora_" in name and p.requires_grad
        )
        moe_params = sum(
            p.numel() for moe_ffn in self.moe_layers_list
            for p in moe_ffn.parameters() if p.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "lora": lora_params,
            "moe": moe_params,
            "trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
        }

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise a partially-masked sequence at noise level t.

        Args:
            z_t:            (B, L) token ids — noised input (contains [MASK] tokens).
            t:              (B,) float in [0, 1] — noise level for each sequence.
            attention_mask: (B, L) optional mask (1 = attend, 0 = ignore).

        Returns:
            logits: (B, L, vocab_size) — predicted logits for the clean tokens,
                    after SUBS post-processing.
        """
        B, L = z_t.shape

        # ── 1. Base BERT forward ──────────────────────────────────────────────
        # We intercept after the embedding layer to inject the time signal.
        embeddings = self.bert.bert.embeddings(z_t)  # (B, L, H)

        # ── 2. Inject time conditioning ───────────────────────────────────────
        if self.use_time_conditioning and self.time_embed is not None:
            t_emb = self.time_embed(t)               # (B, H)
            embeddings = embeddings + t_emb.unsqueeze(1)  # broadcast over L

        # ── 3. Transformer encoder ────────────────────────────────────────────
        encoder_output = self.bert.bert.encoder(
            embeddings,
            attention_mask=_make_extended_attention_mask(attention_mask, embeddings),
        )
        hidden_states = encoder_output.last_hidden_state  # (B, L, H)

        # ── 4. Pooler is not needed; skip directly to the MLM head ───────────
        # BertOnlyMLMHead = BertPredictionHeadTransform + linear
        sequence_output = self.bert.bert.pooler if False else hidden_states
        logits = self.bert.cls(hidden_states)   # (B, L, vocab_size)

        # ── 5. Collect MoE auxiliary losses ───────────────────────────────────
        total_aux = torch.tensor(0.0, device=z_t.device)
        for moe_ffn in self.moe_layers_list:
            total_aux = total_aux + moe_ffn.aux_loss
        self.moe_aux_loss = total_aux

        # ── 6. SUBS post-processing ───────────────────────────────────────────
        logits = self._apply_subs(logits, z_t)

        return logits

    def _apply_subs(self, logits: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Apply the SUBS (SUBStitution) parameterisation.

        (a) Zero masking probabilities: set logits for [MASK] token to -inf
            so the model never predicts [MASK] as the original token.
        (b) Carry-over unmasking: for positions that were NOT masked in z_t,
            override the model's logits with a sharp one-hot at the input token
            (the denoiser must reproduce unmasked tokens exactly).

        Args:
            logits: (B, L, V) — raw output of BERT's MLM head.
            z_t:    (B, L)    — noised input token ids.
        Returns:
            logits: (B, L, V) — post-processed logits.
        """
        # (a) Zero masking probabilities
        logits = logits.clone()
        logits[:, :, self.mask_token_id] = float('-inf')

        # (b) Carry-over: positions that are NOT [MASK] in z_t
        is_unmasked = (z_t != self.mask_token_id)          # (B, L) bool
        if is_unmasked.any():
            # Build one-hot logits: large positive at input token, -inf elsewhere
            one_hot = torch.full_like(logits, float('-inf'))
            idx = z_t[is_unmasked].unsqueeze(-1)           # (M, 1)
            fill_val = 6.0e4 if logits.dtype == torch.float16 else 1e9
            one_hot[is_unmasked] = one_hot[is_unmasked].scatter(-1, idx, fill_val)
            logits[is_unmasked] = one_hot[is_unmasked]

        return logits

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int,
        noise_schedule,
        tokenizer,
        device: torch.device,
        prefix_ids: Optional[torch.Tensor] = None,
        suffix_ids: Optional[torch.Tensor] = None,
        fixed_token_mask: Optional[torch.Tensor] = None,
        init_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ancestral sampling from the reverse diffusion process.

        Starts from a fully-masked sequence z_1 = [MASK, MASK, …, MASK]
        and runs ``num_steps`` reverse steps to produce clean token sequences.

        Constrained generation: if ``fixed_token_mask`` is provided (bool tensor,
        True = position is fixed/constrained), those positions are pinned to
        their values in ``init_z`` (or ``prefix_ids``/``suffix_ids``) and never
        updated.

        Args:
            batch_size:       Number of sequences to generate.
            seq_len:          Sequence length L.
            num_steps:        Number of reverse diffusion steps T.
            noise_schedule:   LogLinearNoiseSchedule instance.
            tokenizer:        HuggingFace tokenizer (provides mask_token_id).
            device:           Target device.
            prefix_ids:       (B, prefix_len) or None — for infilling.
            suffix_ids:       (B, suffix_len) or None — for infilling.
            fixed_token_mask: (B, L) bool or None — True = position is fixed.
            init_z:           (B, L) or None — starting sequence with caller-pinned
                              tokens already in place (used for per-example variable
                              prefix/suffix or keyword positions). If None, starts
                              fully masked.

        Returns:
            generated_ids: (B, L) integer token ids of the generated sequences.
        """
        self.eval()
        mask_id = tokenizer.mask_token_id
        self.mask_token_id = mask_id

        # Start from caller-supplied z if given, else a fully-masked sequence
        if init_z is not None:
            z = init_z.to(device).clone()
        else:
            z = torch.full((batch_size, seq_len), mask_id, dtype=torch.long, device=device)

        # If infilling: place prefix and suffix, mark them as fixed
        if prefix_ids is not None or suffix_ids is not None:
            if fixed_token_mask is None:
                fixed_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            if prefix_ids is not None:
                pl = prefix_ids.shape[1]
                z[:, :pl] = prefix_ids.to(device)
                fixed_token_mask[:, :pl] = True
            if suffix_ids is not None:
                sl = suffix_ids.shape[1]
                z[:, -sl:] = suffix_ids.to(device)
                fixed_token_mask[:, -sl:] = True

        # Reverse diffusion: t from 1 → 0 in num_steps steps
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)  # descending

        for i in range(num_steps):
            t_val = timesteps[i]
            s_val = timesteps[i + 1]

            t_batch = torch.full((batch_size,), t_val, device=device)
            s_batch = torch.full((batch_size,), s_val, device=device)

            # Model predicts logits for clean x_0
            logits = self.forward(z, t_batch)   # (B, L, V) — SUBS applied

            # Compute posterior logits q(z_s | z_t, x_theta)
            posterior = noise_schedule.posterior_logits(
                logits_x0=logits,
                z_t=z,
                t=t_batch,
                s=s_batch,
                mask_token_id=mask_id,
            )                                   # (B, L, V)

            # Sample from the posterior
            z_s = torch.distributions.Categorical(logits=posterior).sample()  # (B, L)

            # Enforce carry-over: already-unmasked positions never change
            carry = (z != mask_id)
            z_s[carry] = z[carry]

            # Enforce fixed constraints (prefix, suffix, keywords)
            if fixed_token_mask is not None:
                z_s[fixed_token_mask] = z[fixed_token_mask]

            z = z_s

        return z

    def set_mask_token_id(self, mask_token_id: int) -> None:
        """Set the mask token id (call after initialising the tokenizer)."""
        self.mask_token_id = mask_token_id


# ---------------------------------------------------------------------------
# Helper: patch a BertLayer's FFN sub-modules with a MoEFeedForward
# ---------------------------------------------------------------------------

class _MoEBertOutput(nn.Module):
    """Thin wrapper that gives MoEFeedForward the same interface as BertOutput.

    BertLayer calls:  hidden_states = self.output(intermediate_output, input_tensor)
    where input_tensor is the residual (attention output before FFN).
    We replicate: output = LayerNorm(MoEFFN(input_tensor) + input_tensor)
    (skipping the intermediate layer entirely for MoE layers).
    """

    def __init__(self, moe_ffn: MoEFeedForward, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.moe_ffn = moe_ffn
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # hidden_states here is the output of BertIntermediate (ignored for MoE)
        # We re-run the MoE FFN on input_tensor (the pre-FFN residual)
        moe_out = self.moe_ffn(input_tensor)          # (B, L, H)
        moe_out = self.dropout(moe_out)
        return self.layer_norm(moe_out + input_tensor)


class _IdentityIntermediate(nn.Module):
    """Pass-through module to replace BertIntermediate in MoE layers.

    BertLayer always calls intermediate then output. Since _MoEBertOutput
    will ignore the intermediate output and use input_tensor directly,
    this intermediate just returns a zero tensor of the correct shape.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states  # will be ignored by _MoEBertOutput


def _patch_bert_layer_with_moe(
    bert_layer: nn.Module,
    moe_ffn: MoEFeedForward,
    hidden_size: int,
    dropout: float,
    upcycle_noise: float = 0.01,
) -> None:
    """Replace a BertLayer's FFN sub-modules in-place with MoE equivalents.

    Sparse upcycling (Komatsuzaki et al., 2023): each expert is initialised
    from the pretrained BERT FFN weights + small Gaussian noise for
    diversification, rather than random init.  The pretrained LayerNorm is
    also copied into the MoE output wrapper.
    """
    # ── 1. Extract pretrained weights BEFORE replacing sub-modules ────────────
    fc1_w = bert_layer.intermediate.dense.weight.data.clone()  # (intermediate, hidden)
    fc1_b = bert_layer.intermediate.dense.bias.data.clone()    # (intermediate,)
    fc2_w = bert_layer.output.dense.weight.data.clone()        # (hidden, intermediate)
    fc2_b = bert_layer.output.dense.bias.data.clone()          # (hidden,)
    ln_w  = bert_layer.output.LayerNorm.weight.data.clone()    # (hidden,)
    ln_b  = bert_layer.output.LayerNorm.bias.data.clone()      # (hidden,)

    # ── 2. Copy into each expert with small diversification noise ─────────────
    # Only upcycle if the expert intermediate_size matches the pretrained FFN.
    # (They match when expert_hidden_multiplier == bert_config.intermediate_size / hidden_size.)
    can_upcycle = (
        moe_ffn.experts[0].fc1.weight.shape == fc1_w.shape and
        moe_ffn.experts[0].fc2.weight.shape == fc2_w.shape
    )
    if can_upcycle:
        for expert in moe_ffn.experts:
            expert.fc1.weight.data.copy_(fc1_w)
            expert.fc1.weight.data += torch.randn_like(fc1_w) * upcycle_noise
            expert.fc1.bias.data.copy_(fc1_b)

            expert.fc2.weight.data.copy_(fc2_w)
            expert.fc2.weight.data += torch.randn_like(fc2_w) * upcycle_noise
            expert.fc2.bias.data.copy_(fc2_b)

    # ── 3. Replace the BERT layer sub-modules ─────────────────────────────────
    bert_layer.intermediate = _IdentityIntermediate()
    moe_output = _MoEBertOutput(moe_ffn, hidden_size, dropout)

    # Copy pretrained LayerNorm weights into the MoE wrapper
    moe_output.layer_norm.weight.data.copy_(ln_w)
    moe_output.layer_norm.bias.data.copy_(ln_b)

    bert_layer.output = moe_output


# ---------------------------------------------------------------------------
# Helper: extended attention mask (BERT-style, passed to encoder)
# ---------------------------------------------------------------------------

def _make_extended_attention_mask(
    attention_mask: Optional[torch.Tensor],
    embeddings: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Convert a 2D (B, L) attention mask to BERT's 4D extended format.

    Returns None if no mask is provided (all positions attend to all).
    """
    if attention_mask is None:
        return None
    B, L = attention_mask.shape
    # (B, 1, 1, L) → broadcasted inside multi-head attention
    extended = attention_mask[:, None, None, :]  # (B, 1, 1, L)
    # Convert 0/1 mask to large negative / 0 additive mask
    extended = (1.0 - extended.float()) * -10000.0
    return extended
