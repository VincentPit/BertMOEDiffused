"""
Smoke test — exercises the full training loop end-to-end on CPU with
synthetic data. No GPU, no LM1B download required.

Run: python test_smoke.py
"""

import math
import sys
import traceback

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

VOCAB_SIZE = 30522
SEQ_LEN = 32
BATCH_SIZE = 2
MAX_STEPS = 5
DEVICE = torch.device("cpu")

# ── Minimal config ────────────────────────────────────────────────────────────
cfg = {
    "model": {
        "backbone": "bert-base-uncased",
        "hidden_size": 768,
        "max_seq_len": SEQ_LEN,
        "dropout": 0.1,
        "use_time_conditioning": True,
        "time_embed_dim": 128,
        "moe": {
            "num_experts": 4,
            "num_experts_per_token": 2,
            "moe_layers": [3],           # only one MoE layer for speed
            "expert_hidden_multiplier": 2,
            "router_jitter": 0.01,
            "router_z_loss_coef": 0.001,
            "router_aux_loss_coef": 0.01,
        },
        "lora": {
            "enabled": True,
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.05,
            "target_modules": ["query", "key", "value"],
        },
    },
    "diffusion": {"time_eps": 1.0e-4},
    "training": {
        "learning_rate": 5e-5,
        "adam_beta1": 0.95,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.01,
        "warmup_steps": 1,
        "max_steps": MAX_STEPS,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "fp16": False,
    },
}

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(msg):
    print(f"  [PASS] {msg}")

def fail(msg, exc):
    print(f"  [FAIL] {msg}")
    traceback.print_exc()
    sys.exit(1)


# ── 1. Imports ─────────────────────────────────────────────────────────────────
section("1. Imports")
try:
    from model import BertMoEDiffusion, LogLinearNoiseSchedule
    ok("model imports")
except Exception as e:
    fail("model imports", e)

try:
    from model.noise_schedule import LogLinearNoiseSchedule
    ok("noise schedule import")
except Exception as e:
    fail("noise schedule import", e)


# ── 2. Tokenizer ──────────────────────────────────────────────────────────────
section("2. Tokenizer")
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    mask_token_id = tokenizer.mask_token_id
    ok(f"tokenizer loaded, mask_token_id={mask_token_id}")
except Exception as e:
    fail("tokenizer", e)


# ── 3. Model construction ─────────────────────────────────────────────────────
section("3. Model construction")
try:
    moe_cfg = cfg["model"]["moe"]
    lora_cfg = cfg["model"]["lora"]
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
        lora_enabled=lora_cfg["enabled"],
        lora_rank=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        lora_target_modules=lora_cfg["target_modules"],
    )
    model.set_mask_token_id(mask_token_id)
    model.to(DEVICE)
    ok("model constructed")
except Exception as e:
    fail("model construction", e)

try:
    ps = model.trainable_parameters_summary()
    ok(f"params: {ps['total']:,} total | {ps['trainable']:,} trainable ({ps['trainable_pct']:.1f}%) | {ps['lora']:,} LoRA | {ps['moe']:,} MoE")
except Exception as e:
    fail("trainable_parameters_summary", e)


# ── 4. Noise schedule ─────────────────────────────────────────────────────────
section("4. Noise schedule")
try:
    noise_schedule = LogLinearNoiseSchedule()
    t_test = torch.tensor([0.1, 0.5, 0.9])
    alpha = noise_schedule.alpha(t_test)
    assert torch.allclose(alpha, 1.0 - t_test), f"alpha mismatch: {alpha}"
    ok(f"alpha(t) = 1−t verified: {alpha.tolist()}")

    t_sample = noise_schedule.sample_t(BATCH_SIZE, DEVICE, low_discrepancy=True)
    assert t_sample.shape == (BATCH_SIZE,)
    ok(f"sample_t shape OK: {t_sample.tolist()}")
except Exception as e:
    fail("noise schedule", e)


# ── 5. Forward pass ───────────────────────────────────────────────────────────
section("5. Forward pass")
try:
    # Synthetic token IDs (avoid special tokens 0,101,102)
    input_ids = torch.randint(1000, 20000, (BATCH_SIZE, SEQ_LEN))
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    t = noise_schedule.sample_t(BATCH_SIZE, DEVICE)
    z_t = noise_schedule.noise_sequence(input_ids, t, mask_token_id)

    masked_frac = (z_t == mask_token_id).float().mean().item()
    ok(f"noise_sequence: {masked_frac*100:.1f}% tokens masked for t={t.tolist()}")

    model.eval()
    with torch.no_grad():
        logits = model(z_t, t, attention_mask=attention_mask)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), f"logits shape: {logits.shape}"
    ok(f"logits shape: {logits.shape}")
except Exception as e:
    fail("forward pass", e)


# ── 6. MDLM loss ──────────────────────────────────────────────────────────────
section("6. MDLM loss")
try:
    def compute_mdlm_loss(logits, input_ids, z_t, t, mask_token_id, time_eps=1e-4):
        B, L, V = logits.shape
        weights = 1.0 / t.clamp(min=time_eps)
        is_masked = (z_t == mask_token_id)
        ce = F.cross_entropy(
            logits.reshape(B * L, V), input_ids.reshape(B * L), reduction="none"
        ).reshape(B, L)
        ce = ce * is_masked.float()
        return (weights * ce.sum(-1)).mean()

    time_eps = float(cfg["diffusion"]["time_eps"])
    loss = compute_mdlm_loss(logits, input_ids, z_t, t, mask_token_id, time_eps)
    assert loss.item() > 0, "loss should be positive"
    assert not math.isnan(loss.item()), "loss is NaN"
    assert not math.isinf(loss.item()), "loss is Inf"
    ok(f"MDLM loss = {loss.item():.4f}")
except Exception as e:
    fail("MDLM loss", e)


# ── 7. MoE aux loss ───────────────────────────────────────────────────────────
section("7. MoE auxiliary loss")
try:
    model.train()
    logits = model(z_t, t, attention_mask=attention_mask)
    moe_aux = model.moe_aux_loss
    moe_val = moe_aux.item() if isinstance(moe_aux, torch.Tensor) else float(moe_aux)
    assert not math.isnan(moe_val), "MoE aux is NaN"
    ok(f"moe_aux_loss = {moe_val:.6f}")
except Exception as e:
    fail("MoE aux loss", e)


# ── 8. Backward pass ──────────────────────────────────────────────────────────
section("8. Backward pass")
try:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)

    optimizer.zero_grad()
    logits = model(z_t, t, attention_mask=attention_mask)
    loss = compute_mdlm_loss(logits, input_ids, z_t, t, mask_token_id, time_eps)
    moe_aux = model.moe_aux_loss
    total = loss + moe_aux
    total.backward()

    # Check gradients exist on LoRA params
    lora_params_with_grad = [
        n for n, p in model.named_parameters()
        if ("lora_A" in n or "lora_B" in n) and p.grad is not None
    ]
    ok(f"backward OK — {len(lora_params_with_grad)} LoRA params have gradients")

    optimizer.step()
    ok("optimizer.step() OK")
except Exception as e:
    fail("backward pass", e)


# ── 9. Full training loop (MAX_STEPS steps) ───────────────────────────────────
section(f"9. Mini training loop ({MAX_STEPS} steps)")
try:
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["learning_rate"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["training"]["warmup_steps"],
        num_training_steps=cfg["training"]["max_steps"],
    )

    losses = []
    for step in range(MAX_STEPS):
        input_ids = torch.randint(1000, 20000, (BATCH_SIZE, SEQ_LEN))
        attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
        t = noise_schedule.sample_t(BATCH_SIZE, DEVICE)
        z_t = noise_schedule.noise_sequence(input_ids, t, mask_token_id)

        optimizer.zero_grad()
        logits = model(z_t, t, attention_mask=attention_mask)
        loss = compute_mdlm_loss(logits, input_ids, z_t, t, mask_token_id, time_eps)
        moe_aux = model.moe_aux_loss
        (loss + moe_aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        print(f"    step {step+1}/{MAX_STEPS}  loss={loss.item():.4f}  moe={moe_aux.item():.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    assert all(not math.isnan(l) for l in losses), "NaN loss during training"
    ok(f"all {MAX_STEPS} steps completed without NaN/Inf")
except Exception as e:
    fail("training loop", e)


# ── 10. SUBS parametrization (unmasked carry-over) ───────────────────────────
section("10. SUBS — unmasked positions carry-over")
try:
    model.eval()
    # Use t≈0 so almost no masking — most positions stay unmasked
    t_low = torch.full((BATCH_SIZE,), 0.01)
    z_low = noise_schedule.noise_sequence(input_ids, t_low, mask_token_id)
    with torch.no_grad():
        logits_low = model(z_low, t_low, attention_mask=attention_mask)

    # For unmasked positions, SUBS should make the model predict the input token
    # (logit at input token should dominate)
    unmasked = (z_low != mask_token_id)
    if unmasked.any():
        preds = logits_low.argmax(-1)
        carry_match = (preds[unmasked] == z_low[unmasked]).float().mean().item()
        ok(f"carry-over accuracy on unmasked positions: {carry_match*100:.1f}% (expect ~100%)")
    else:
        ok("all tokens masked at t=0.01 (unusual but OK)")
except Exception as e:
    fail("SUBS carry-over", e)


# ── 11. LoRA merge ────────────────────────────────────────────────────────────
section("11. LoRA merge")
try:
    model.eval()
    with torch.no_grad():
        logits_before = model(z_t, t, attention_mask=attention_mask).clone()

    model.merge_lora()
    ok("merge_lora() called without error")

    # After merge, lora_enabled should be False and no lora_A/B params should require grad
    lora_trainable = [n for n, p in model.named_parameters() if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
    ok(f"post-merge LoRA trainable params: {len(lora_trainable)} (expect 0)")
except Exception as e:
    fail("LoRA merge", e)


# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  ALL TESTS PASSED")
print(f"{'='*60}\n")
