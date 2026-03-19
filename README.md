# BertDiffused

**MoE-BERT as a Masked Diffusion Language Model**

Fine-tunes `bert-base-uncased` augmented with sparse Mixture-of-Experts (MoE)
feed-forward layers as the denoiser in a Masked Diffusion LM (MDLM).
Quantitatively compares diffusion vs. autoregressive generation on two tasks
where bidirectional context gives a structural advantage.

---

## Overview

| Paradigm | Models | Generation |
|---|---|---|
| Masked Diffusion | **BertDiffused (ours)**, MDLM-OWT | Whole sequence denoised in parallel |
| Autoregressive | GPT-2, GPT-2 Medium | Left-to-right token by token |

**Why MoE?** The diffusion denoiser samples $t \sim \mathcal{U}[0,1]$ during
training, creating two qualitatively different sub-problems:
- High $t$ (~100% masked) → global language modelling
- Low $t$ (~10% masked) → local refinement

A single dense FFN handles both with one set of weights. Replacing FFNs in
layers {3, 5, 7, 9, 11} with sparse MoE blocks (8 experts, top-2 routing)
lets different experts specialize per noise level.

---

## Project Structure

```
BertDiffused/
├── model/
│   ├── noise_schedule.py       # log-linear α(t)=1−t, masking, posterior logits
│   ├── moe_layer.py            # MoERouter + ExpertFFN + MoEFeedForward
│   └── bert_moe_diffusion.py   # full model: BERT + time embed + MoE + SUBS
├── data/
│   └── lm1b_dataset.py         # LM1B dataset wrappers (map-style + streaming)
├── tasks/
│   ├── infilling.py            # Task 1: text infilling benchmark
│   └── constrained_gen.py      # Task 2: keyword-constrained generation
├── eval/
│   └── compare.py              # 4-model comparison + plots
├── proposal/
│   └── proposal.tex            # LaTeX proposal
├── configs/
│   └── config.yaml             # all hyperparameters
├── train.py                    # MDLM training loop with MoE aux loss
├── requirements.txt
└── README.md
```

---

## Installation

### Option A — Docker (recommended)

**Prerequisites**: Docker, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
git clone https://github.com/stephenlee/BertDiffused.git
cd BertDiffused

# 1. Copy and fill in environment variables
cp .env.example .env
#    edit .env: set HF_TOKEN, WANDB_API_KEY, NVIDIA_VISIBLE_DEVICES

# 2. Build the image
docker compose build

# 3. Train
docker compose run --rm train

# 4. Evaluate (infilling)
docker compose run --rm eval_infilling

# 5. Evaluate (constrained generation)
docker compose run --rm eval_constrained

# 6. Full comparison + plots
docker compose run --rm eval_compare

# Interactive shell
docker compose run --rm shell
```

HuggingFace model weights and checkpoints are persisted across runs via
named volumes (`hf_cache`) and bind mounts (`./checkpoints`, `./results`).

### Option B — Local

**Requirements**: Python 3.10+, PyTorch 2.2+, CUDA 11.8+

```bash
git clone https://github.com/stephenlee/BertDiffused.git
cd BertDiffused
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: set HF_TOKEN, WANDB_API_KEY
```

---

## Quickstart

### 1. Train BertDiffused

```bash
python train.py \
  --config configs/config.yaml \
  --output_dir checkpoints/bertdiffused
```

Key hyperparameters (`configs/config.yaml`):

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Dataset | LM1B, seq len 128 |
| Steps | 100K |
| Batch size | 512 |
| Learning rate | 5e-5 |
| MoE layers | {3, 5, 7, 9, 11} |
| Experts / top-k | 8 / 2 |
| Aux loss weights | lb=1e-2, z=1e-3 |

### 2. Run Task 1 — Text Infilling

```bash
python tasks/infilling.py \
  --diffusion_ckpt checkpoints/bertdiffused \
  --steps 100 \
  --n_samples 500
```

Reports BLEU-4, MAUVE, and Generative PPL across all four models.

### 3. Run Task 2 — Keyword-Constrained Generation

```bash
python tasks/constrained_gen.py \
  --diffusion_ckpt checkpoints/bertdiffused \
  --steps 100 \
  --n_samples 500
```

Reports KW-Sat %, MAUVE, and Generative PPL across all four models.

### 4. Full Comparison + Plots

```bash
python eval/compare.py \
  --diffusion_ckpt checkpoints/bertdiffused \
  --output_dir results/
```

Produces:
- `results/task1_table.csv` / `results/task2_table.csv`
- `results/steps_vs_ppl.png` — quality-speed curve for T ∈ {10, 100, 1000}
- `results/expert_routing_heatmap.png` — expert utilization vs. noise level t
- `results/moe_ablation.png` — dense vs. MoE at matched steps

---

## Method

### Masked Diffusion (MDLM)

Forward process independently masks each token at time $t \in [0,1]$:

$$q(\mathbf{z}_t \mid \mathbf{x}) = \prod_\ell \mathrm{Cat}(z_t^\ell;\;\alpha(t)\,x^\ell + (1-\alpha(t))\,[\texttt{MASK}])$$

with log-linear schedule $\alpha(t) = 1 - t$. Training objective:

$$\mathcal{L}_\text{NELBO} = \mathbb{E}\!\left[\frac{1}{t} \sum_{\ell:\,z_t^\ell=[\texttt{MASK}]} \mathrm{CE}\!\left(\mathbf{x}_\theta^\ell(\mathbf{z}_t,t),\,x^\ell\right)\right]$$

Generation starts from a fully-masked sequence $\mathbf{z}_1$ and runs $T$
reverse denoising steps to produce $\mathbf{z}_0 = \mathbf{x}$.

### MoE Block

Each MoE layer routes every token to 2 of 8 experts:

$$\mathrm{MoE}(h_i) = \sum_{e \in \mathrm{Top\text{-}2}(s_i)} p_{i,e}\,f_e(h_i), \qquad s_i = \mathrm{softmax}(W_r h_i)$$

where $W_r \in \mathbb{R}^{8 \times 768}$ is the learned router. Experts 
$f_e$ are independent 2-layer FFNs (768→3072→768). Only 2 of 8 experts 
activate per token — same FLOPs as a single dense FFN, but 8× the capacity.

Total training loss with routing regularization:

$$\mathcal{L} = \mathcal{L}_\text{NELBO} + \lambda_\text{lb}\,\mathcal{L}_\text{lb} + \lambda_z\,\mathcal{L}_z$$

| Term | Purpose |
|---|---|
| $\mathcal{L}_\text{lb}$ | Load-balancing — prevents expert collapse |
| $\mathcal{L}_z$ | Z-loss — stabilizes router logit scale |

### SUBS Post-processing

Applied to raw MLM head logits after every forward pass:
1. Set `[MASK]` logit to $-\infty$ — model never outputs a mask token as clean text
2. For already-unmasked positions, set a one-hot logit — carries over unchanged

---

## Evaluation Tasks

### Task 1 — Text Infilling

| | Diffusion | GPT-2 |
|---|---|---|
| **How** | Pin prefix + suffix as unmasked; reverse-diffuse the gap | Generate left-to-right from prefix only |
| **Context** | Full bidirectional (prefix + suffix seen simultaneously) | Prefix only (no FIM training) |

- **Setup**: 500 LM1B test sentences; central 30% of tokens masked as the target span
- **Metrics**: BLEU-4 ↑, MAUVE ↑ *(primary)*, Gen-PPL ↓

### Task 2 — Keyword-Constrained Generation

| | Diffusion | GPT-2 |
|---|---|---|
| **How** | Pin keywords as unmasked in $\mathbf{z}_1$; denoise rest freely | NeuroLogic constrained beam search |
| **Constraint handling** | Architectural (exact pinning) | Algorithmic (approximate) |

- **Setup**: 200 keyword triplets; generate a coherent 64-token sentence containing all three
- **Metrics**: KW-Sat % ↑ *(primary)*, MAUVE ↑, Gen-PPL ↓

> **Note on Gen-PPL**: scored by a held-out GPT-2 applied to all models' outputs equally.
> This scorer is independent of the GPT-2 generation baseline but may mildly favour
> AR-style fluency. MAUVE is therefore the primary quality metric as it is model-agnostic.

---

## Models Compared

| Model | Type | Params | Checkpoint |
|---|---|---|---|
| **BertDiffused (ours)** | Masked Diffusion + MoE | 132M active / 275M total | trained here |
| MDLM-OWT | Masked Diffusion | 130M | `kuleshov-group/mdlm-owt` |
| GPT-2 | Autoregressive | 117M | `openai-community/gpt2` |
| GPT-2 Medium | Autoregressive | 345M | `openai-community/gpt2-medium` |

---

## Hypotheses

- **(H1)** BertDiffused > GPT-2 on infilling BLEU-4 — bidirectional context
- **(H2)** BertDiffused > GPT-2 on KW-Sat — architectural constraint pinning
- **(H3)** BertDiffused-MoE < dense-BERT-diffusion in Gen-PPL — noise-level expert specialization
- **(H4)** GPT-2 Medium achieves lower Gen-PPL on unconstrained generation at T ≤ 100

---

## References

- Sahoo et al. (2024) — [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) *(MDLM)*
- Devlin et al. (2019) — [BERT](https://arxiv.org/abs/1810.04805)
- Fedus et al. (2021) — [Switch Transformers](https://arxiv.org/abs/2101.03961)
- Zoph et al. (2022) — [ST-MoE](https://arxiv.org/abs/2202.08906)
- He et al. (2022) — [DiffusionBERT](https://arxiv.org/abs/2211.15029)
- Li et al. (2022) — [Diffusion-LM](https://arxiv.org/abs/2205.14217)
- Pillutla et al. (2021) — [MAUVE](https://arxiv.org/abs/2102.01454)
- Lu et al. (2021) — [NeuroLogic Decoding](https://arxiv.org/abs/2010.12884)
