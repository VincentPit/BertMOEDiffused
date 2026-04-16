# ── Base image ────────────────────────────────────────────────────────────────
# PyTorch 2.2 + CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        vim \
        build-essential \
        libssl-dev \
        libffi-dev \
        texlive-latex-extra \
        texlive-fonts-recommended \
        dvipng \
        cm-super \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# ── HuggingFace cache ─────────────────────────────────────────────────────────
# Mount a volume here to persist downloaded model weights across runs
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# ── Experiment output dirs ────────────────────────────────────────────────────
RUN mkdir -p /workspace/checkpoints \
             /workspace/results \
             /workspace/.cache/huggingface \
             /workspace/mlruns

# ── Expose MLflow serving port ────────────────────────────────────────────────
EXPOSE 8080

# ── Default command ───────────────────────────────────────────────────────────
CMD ["python", "train.py"]
