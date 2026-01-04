### RunPod REAP Runbook (`pod.md`)

This is a copy/paste checklist for starting a **new RunPod GPU pod** and getting `reap.prune` working reliably.

---

### Assumptions

- You are running inside a pod/container with NVIDIA GPUs available.
- REAP repo lives at **`/opt/reap`** (adjust paths if yours differs).
- You want to run with **`trust_remote_code=True`** (default in REAP) for models like Solar-Open.

---

### 1) Quick GPU sanity check

```bash
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count()); print('name0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

### 2) Get the latest REAP code

If the repo already exists:

```bash
cd /opt/reap
git fetch --all
git pull
```

If the repo does not exist:

```bash
cd /opt
git clone --recurse-submodules https://github.com/Akicuo/reap.git
cd /opt/reap
```

---

### 3) Create/activate the venv

If the venv exists:

```bash
cd /opt/reap
source .venv/bin/activate
python -V
```

If the venv does not exist:

```bash
cd /opt/reap
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

### 4) Install REAP + dependencies (critical)

This ensures you have **`bitsandbytes`** (needed for `--load_in_4bit true`) and all REAP deps pinned by the repo.

```bash
cd /opt/reap
source .venv/bin/activate
pip install -e .
```

Verify key imports:

```bash
python -c "import transformers; import accelerate; import datasets; import safetensors; print('transformers', transformers.__version__)"
python -c "import bitsandbytes as bnb; print('bnb ok', bnb.__version__)"
```

If `bitsandbytes` import fails, install it explicitly:

```bash
pip install 'bitsandbytes>=0.49.0'
```

---

### 5) Set environment variables (recommended)

#### HuggingFace auth (if the model is gated/private)

```bash
export HF_TOKEN='YOUR_TOKEN_HERE'
```

#### Local-only mode (if you already pre-downloaded model files)

REAP reads `REAP_LOCAL_FILES_ONLY`.

```bash
export REAP_LOCAL_FILES_ONLY=0   # allow downloads
# export REAP_LOCAL_FILES_ONLY=1 # forbid downloads
```

Optional: point HF cache to a large disk (recommended on pods):

```bash
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"
```

---

### 6) Recommended “first run” sanity check (tiny)

Use a tiny run to validate downloads + forward pass + observer hook:

```bash
cd /opt/reap
source .venv/bin/activate

python -u -m reap.prune \
  --model_name moonshotai/Kimi-K2-Thinking \
  --dataset_name theblackcat102/evol-codealpaca-v1 \
  --samples_per_category 48 \
  --model_max_length 8 \
  --prune_method reap \
  --compression_ratio 0.25 \
  --distance_measure cosine \
  --load_in_4bit true \
  --no_plot_clusters \
  --do_eval false \
  --run_lm_eval false \
  --run_evalplus false \
  --run_livecodebench false \
  --run_math false \
  --run_wildbench false
```

If this works, scale up your real run.

---

### 7) “Real run” example (you can edit these)

```bash
cd /opt/reap
source .venv/bin/activate

python -u -m reap.prune \
  --model_name upstage/Solar-Open-100B \
  --dataset_name theblackcat102/evol-codealpaca-v1 \
  --samples_per_category 48 \
  --model_max_length 8 \
  --prune_method reap \
  --compression_ratio 0.35 \
  --distance_measure cosine \
  --load_in_4bit true \
  --no_plot_clusters \
  --do_eval false \
  --run_lm_eval false \
  --run_evalplus false \
  --run_livecodebench false \
  --run_math false \
  --run_wildbench false
```

Artifacts are written under:

- `artifacts/<ModelName>/<DatasetName>/...`

---

### 8) Common failures & fixes

#### A) `PackageNotFoundError: bitsandbytes`

Cause: you requested `--load_in_4bit true`, but `bitsandbytes` isn’t installed.

Fix:

```bash
source /opt/reap/.venv/bin/activate
pip install 'bitsandbytes>=0.49.0'
```

#### B) `TypeError: check_model_inputs() missing 1 required positional argument: 'func'`

Cause: **Solar-Open remote code** uses `@check_model_inputs()` but your installed Transformers exposes it as `check_model_inputs(func)` in that version.

Fix options (try in order):
1. Update core dependencies (proven fix per user testing)
```bash
cd /opt/reap
source .venv/bin/activate
pip install -U transformers accelerate
```
2. Apply repo-specific runtime shim (fallback)
REAP applies a runtime shim in `src/reap/transformers_compat.py`.
What you must do on a new pod:
```bash
cd /opt/reap
git pull
source .venv/bin/activate
pip install -e .
```

#### C) Model downloads new remote code every run

Pin a revision by adding `--revision <sha-or-tag>` (if/when REAP exposes it) or set HF cache on persistent storage (see env vars above).

#### D) Out of disk

Move HF cache to `/workspace` (see env vars above) and prune old caches:

```bash
du -sh ~/.cache/huggingface/* 2>/dev/null || true
```

#### E) Smoke Test Failures (attention mask/Triton errors)

Cause:
1. Tokenizer pad token = eos token (configuration issue)
2. Missing Triton autotune directory (environment setup issue)

These do NOT indicate poor model generation quality.

Fix:
```bash
# Create missing Triton directory
mkdir -p /root/.triton/autotune

# Fix tokenizer pad token issue (persists across runs)
cd /opt/reap
source .venv/bin/activate
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('upstage/Solar-Open-100B'); tokenizer.pad_token = tokenizer.eos_token; tokenizer.save_pretrained('/workspace/.cache/huggingface/tokenizers/fixed_solar_open')"
```

#### F) `TRANSFORMERS_CACHE` Deprecation Warning

Cause: Using deprecated `TRANSFORMERS_CACHE` environment variable (will be removed in Transformers v5)

Fix (permanent system-wide):
```bash
# Unset deprecated variable and set HF_HOME
unset TRANSFORMERS_CACHE
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
mkdir -p $HF_HOME
# Save to bashrc for future sessions
echo "unset TRANSFORMERS_CACHE
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME" >> ~/.bashrc
```

#### G) Tokenizer Regex Pattern Error

Cause: Incorrect regex pattern in tokenizer configuration (linked to Mistral model lineage)

Fix (pruned model directory-specific):
```bash
# Run from your pruned model directory
cd /opt/reap/artifacts/Solar-Open-100B/evol-codealpaca-v1/pruned_models/reap-seed_42-0.50
# Fix regex + pad token and save to pruned model dir
python -c "from transformers import AutoTokenizer; import os; tokenizer = AutoTokenizer.from_pretrained(os.getcwd(), fix_mistral_regex=True); tokenizer.pad_token = tokenizer.eos_token; tokenizer.save_pretrained(os.getcwd())"
```

---

### 9) If you use a Docker image

If your pod launches from a prebuilt image, you still have two options:

- **No rebuild needed (quick)**: `git pull` + `pip install -e .` inside the running container.
- **Rebuild needed (baked-in)**: rebuild your image to bake the latest repo + deps into it (recommended if you want “launch and run”).


