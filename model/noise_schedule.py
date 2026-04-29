"""
Log-linear noise schedule for masked (absorbing state) diffusion.

Forward process:
    alpha(t) = 1 - t,  t in [0, 1]
    q(z_t | x) = Cat(z_t; alpha(t) * x + (1 - alpha(t)) * m)

    i.e., each token is masked independently with probability (1 - alpha(t)) = t.

Time-weighting for MDLM ELBO:
    w(t) = -alpha'(t) / (1 - alpha(t)) = 1 / t   (log-linear schedule)

Reference: Sahoo et al., "Simple and Effective Masked Diffusion Language Models"
           (NeurIPS 2024), Eq. (8) and Appendix A.
"""

import torch


class LogLinearNoiseSchedule:
    """Log-linear noise schedule: alpha(t) = 1 - t.

    All methods operate on a batch of timesteps t (float tensor in [0, 1]).
    """

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Probability that a token *remains unmasked* at time t.

        alpha(t) = 1 - t

        Args:
            t: Tensor of shape (...,) with values in [0, 1].
        Returns:
            Tensor of the same shape, values in [0, 1].
        """
        return 1.0 - t

    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative of alpha(t): d alpha / dt = -1 (constant)."""
        return torch.full_like(t, -1.0)

    def time_weight(self, t: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """MDLM ELBO time weight: w(t) = -alpha'(t) / (1 - alpha(t)) = 1/t.

        Clamps t from below by ``eps`` to avoid division by zero near t=0.

        Args:
            t:   Tensor of shape (...,).
            eps: Minimum value for t to avoid singularity.
        Returns:
            Tensor of the same shape (all positive).
        """
        t_clamped = t.clamp(min=eps)
        return 1.0 / t_clamped

    def sample_t(
        self,
        batch_size: int,
        device: torch.device,
        low_discrepancy: bool = True,
    ) -> torch.Tensor:
        """Sample one timestep per sequence in the batch.

        Args:
            batch_size:       Number of sequences.
            device:           Target device.
            low_discrepancy:  If True, use a shifted uniform grid (stratified
                              sampling) to reduce variance — ~30% reduction
                              observed in MDLM experiments.
        Returns:
            t: Tensor of shape (batch_size,) with values in (0, 1).
        """
        if low_discrepancy:
            # Stratified: divide [0,1] into batch_size equal intervals,
            # then sample uniformly within each interval.
            offset = torch.rand(1, device=device)
            t = (torch.arange(batch_size, device=device) + offset) / batch_size
            t = t % 1.0                       # wrap into [0, 1)
        else:
            t = torch.rand(batch_size, device=device)
        # Avoid t≈0 (unbounded 1/t weight → fp16 overflow) and t≈1 (zero gradient).
        # 1e-3 matches the ε recommended by Sahoo et al. (MDLM, NeurIPS 2024).
        t = t.clamp(1e-3, 1.0 - 1e-3)
        return t

    def noise_sequence(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        mask_token_id: int,
    ) -> torch.Tensor:
        """Apply the absorbing forward process to a batch of token sequences.

        For each token in each sequence, independently replace it with
        [MASK] (id = mask_token_id) with probability (1 - alpha(t)).

        Args:
            input_ids:     (B, L) integer token ids — the *clean* sequences.
            t:             (B,) timesteps for each sequence.
            mask_token_id: Integer id of the [MASK] token.

        Returns:
            z_t: (B, L) noised sequences with some tokens replaced by [MASK].
        """
        B, L = input_ids.shape
        alpha_t = self.alpha(t)                     # (B,)
        mask_prob = 1.0 - alpha_t                   # (B,) probability of masking each token
        mask_prob = mask_prob.unsqueeze(1).expand(B, L)  # (B, L)

        # Bernoulli draw: 1 means "mask this token"
        should_mask = torch.bernoulli(mask_prob).bool()     # (B, L)
        z_t = input_ids.clone()
        z_t[should_mask] = mask_token_id
        return z_t

    def posterior_logits(
        self,
        logits_x0: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        mask_token_id: int,
    ) -> torch.Tensor:
        """Compute the reverse posterior q(z_s | z_t, x_theta) logits.

        For positions that are *already unmasked* at time t:  q(z_s | z_t, x) = delta(z_s, z_t)
        For positions that are *masked* at time t, the posterior over z_s is:
            p_mask   = (1 - alpha_s) / (1 - alpha_t)   <- stay masked
            p_unmask = (alpha_s - alpha_t) / (1 - alpha_t)  <- unmask to x

        We return log-space weights for stable sampling.

        Args:
            logits_x0:     (B, L, V) — model's predicted logits for the clean token x_0.
            z_t:           (B, L)    — current noised sequence.
            t:             (B,)      — current time.
            s:             (B,)      — previous (smaller) time.
            mask_token_id: Integer id for [MASK].

        Returns:
            posterior_logits: (B, L, V) log-unnormalised weights for sampling z_s.
        """
        B, L, V = logits_x0.shape
        alpha_t = self.alpha(t)[:, None]    # (B, 1)
        alpha_s = self.alpha(s)[:, None]    # (B, 1)

        # Probability weights for the [MASK] position in the posterior
        denom = (1.0 - alpha_t).clamp(min=1e-8)
        p_mask_weight = (1.0 - alpha_s) / denom          # (B, 1) — weight for staying masked
        p_token_weight = (alpha_s - alpha_t) / denom     # (B, 1) — weight for unmasking

        # For unmasked positions: copy current token (carry-over, delta distribution)
        is_masked = (z_t == mask_token_id).unsqueeze(-1)  # (B, L, 1)

        # Build posterior logits:
        # log_softmax normalises x_theta to a proper distribution before mixing
        # with the mask weight; without it Categorical(logits=...) under-weights
        # [MASK] by a factor of sum_v exp(logits_x0[v]) and the chain unmasks
        # too aggressively (so num_steps stops mattering).
        log_p_x0 = torch.log_softmax(logits_x0, dim=-1)                                  # (B,L,V)
        posterior = log_p_x0 + torch.log(p_token_weight.unsqueeze(-1).clamp(min=1e-8))   # (B,L,V)
        posterior[:, :, mask_token_id] = torch.log(p_mask_weight.expand(B, L).clamp(min=1e-8))

        # For unmasked positions: one-hot at current token (carry-over)
        carry_over = torch.full_like(posterior, float('-inf'))
        carry_over.scatter_(-1, z_t.unsqueeze(-1), 0.0)   # 0 in log-space = weight 1

        # Mix based on masking status
        out = torch.where(is_masked, posterior, carry_over)
        return out
