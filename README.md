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
│   ├── lora.py                 # LoRA adapter: LoRALinear, apply/merge utilities
│   └── bert_moe_diffusion.py   # full model: BERT + time embed + MoE + LoRA + SUBS
├── data/
│   ├── lm1b_dataset.py         # LM1B dataset wrappers (map-style + streaming)
│   └── etl.py                  # ETL pipeline: extract → clean → dedup → tokenize → Parquet
├── serving/
│   ├── mlflow_pyfunc.py        # MLflow PyFunc model wrapper for production serving
│   └── inference.py            # Model registry loading + text generation API
├── monitoring/
│   └── __init__.py             # ModelMonitor: prediction metrics, data drift, validation
├── notebooks/
│   ├── BertDiffused_Colab.ipynb # Google Colab notebook (ETL + training + monitoring)
│   └── BertDiffused_Inference.ipynb # Inference demo (generation, infilling, analysis)
├── tasks/
│   ├── infilling.py            # Task 1: text infilling benchmark
│   └── constrained_gen.py      # Task 2: keyword-constrained generation
├── eval/
│   └── compare.py              # 4-model comparison + plots
├── proposal/
│   └── proposal.tex            # LaTeX proposal
├── configs/
│   └── config.yaml             # all hyperparameters (model, MoE, LoRA, ETL, MLflow)
├── train.py                    # MDLM training loop with MLflow tracking + LoRA
├── docker-compose.yml          # MLflow server, DB, training, ETL, serving services
├── Dockerfile
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
#    edit .env: set HF_TOKEN, NVIDIA_VISIBLE_DEVICES

# 2. Start MLflow server + Postgres backend
docker compose up -d mlflow-db mlflow-server

# 3. Run ETL pipeline (downloads LM1B, cleans, deduplicates, shards to Parquet)
docker compose run --rm etl

# 4. Train (LoRA + MoE, tracked by MLflow)
docker compose run --rm train

# 5. Evaluate
docker compose run --rm eval_infilling
docker compose run --rm eval_constrained
docker compose run --rm eval_compare

# 6. Serve model via MLflow
docker compose up -d model-serving
# API available at http://localhost:8080/invocations
```

MLflow UI is available at **http://localhost:5000** after starting the server.

HuggingFace model weights and checkpoints are persisted across runs via
named volumes (`hf_cache`, `mlflow_db_data`, `mlflow_artifacts`).

### Option B — Local

**Requirements**: Python 3.10+, PyTorch 2.2+, CUDA 11.8+

```bash
git clone https://github.com/stephenlee/BertDiffused.git
cd BertDiffused
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart

### 1. ETL Pipeline

Download and preprocess LM1B into tokenized Parquet shards:

```bash
python -m data.etl --config configs/config.yaml
```

Output: `data/processed/{train,test}/shard-*.parquet`

### 2. Train BertDiffused

```bash
python train.py \
  --config configs/config.yaml \
  --output_dir checkpoints/bertdiffused
```

Training is tracked by **MLflow** (local file store by default, or point
`mlflow.tracking_uri` in config to a remote server). LoRA adapters are
saved separately at each checkpoint (~2.5 MB each).

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
| LoRA rank / alpha | 8 / 16 (Q, K, V) |

### 3. Run Task 1 — Text Infilling

```bash
python tasks/infilling.py \
  --diffusion_ckpt checkpoints/bertdiffused \
  --steps 100 \
  --n_samples 500
```

Reports BLEU-4, MAUVE, and Generative PPL across all four models.

### 4. Run Task 2 — Keyword-Constrained Generation

```bash
python tasks/constrained_gen.py \
  --diffusion_ckpt checkpoints/bertdiffused \
  --steps 100 \
  --n_samples 500
```

Reports KW-Sat %, MAUVE, and Generative PPL across all four models.

### 5. Full Comparison + Plots

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

---

## LoRA (Parameter-Efficient Training)

Base BERT weights are frozen; only LoRA adapters (rank-8 on Q/K/V) and MoE
parameters are trained. This reduces trainable parameters from ~117M to ~51M.

Configure in `configs/config.yaml`:

```yaml
model:
  lora:
    enabled: true
    rank: 8
    alpha: 16.0
    dropout: 0.05
    target_modules: ["query", "key", "value"]
```

After training, adapters can be merged into base weights for zero-overhead
inference via `model.merge_lora()`.

---

## MLflow Integration

MLflow provides experiment tracking, model registry, and serving.

- **Tracking**: training metrics (ELBO, MoE aux loss, BPD), hyperparameters,
  system metrics, and dataset lineage are logged automatically.
- **Model Registry**: final models are registered with versioning.
- **Serving**: production inference via `mlflow models serve` or the Docker
  `model-serving` container (port 8080).
- **Monitoring**: `monitoring.ModelMonitor` tracks prediction distributions
  and detects data drift (Jensen-Shannon divergence).

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000
```

---

## ETL Pipeline

The ETL pipeline (`data/etl.py`) preprocesses raw HuggingFace datasets
into tokenized Parquet shards for efficient training:

1. **Extract** — Download LM1B splits from HuggingFace Hub
2. **Transform** — Unicode normalize, quality filter, SHA-256 deduplicate
3. **Load** — Batch tokenize, write zstd-compressed Parquet shards

```bash
python -m data.etl --config configs/config.yaml
```

The processed data is used automatically by `train.py` when available.

---

## Google Colab

A ready-to-run notebook is provided at
[`notebooks/BertDiffused_Colab.ipynb`](notebooks/BertDiffused_Colab.ipynb).

The notebook handles:
- GPU verification, Drive mount, dependency installation
- Full ETL pipeline (configurable sample count)
- LoRA + MoE training with MLflow tracking
- Live loss/BPD plots
- **Space-efficient checkpoints**: only LoRA adapters (~2.5 MB) are synced
  to Drive during training; one final merged model is saved at the end
- Resume support (from VM checkpoint or Drive LoRA adapters)

### Inference Demo

[`notebooks/BertDiffused_Inference.ipynb`](notebooks/BertDiffused_Inference.ipynb) showcases
the trained model's capabilities:

- **Unconditional generation** — text from fully masked sequences
- **Text infilling** — fill missing spans using bidirectional context
- **Keyword-constrained generation** — guaranteed keyword satisfaction
- **Denoising visualisation** — step-by-step reverse diffusion with colour-coded tokens
- **MoE expert routing analysis** — expert specialisation across noise levels
- **Steps vs quality** — quality/speed trade-off across T ∈ {10, 25, 50, 100, 200, 500}

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
