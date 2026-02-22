# Router-weighted Expert Activation Pruning (REAP)

> **This is a fork of [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) from the `main` branch.**

## Summary

This repository contains code for REAP (Router-weighted Expert Activation Pruning), a method for compressing Mixture-of-Experts (MoE) LLMs by pruning less useful experts. REAP considers both router gate-values and expert activation norms to select experts that contribute minimally to layer output.

Paper: [REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression](https://arxiv.org/abs/2510.13999)

---

## <img src="./fig/hf-transparent.png" alt="Sponsor" width='20'>  Sponsor This Work

Running REAP on large MoE models requires significant GPU resources. I rent RunPod pods to prune these models and make them available to the community.

If you find this work useful, consider [buying me a coffee](https://www.buymeacoffee.com/Akicou) to help cover GPU rental costs. Your support enables more pruned models to be released!

---

## Model Releases

Pruned models are available in two formats:

- **Safetensors** - Standard HuggingFace format for vLLM and transformers
  <!-- BEGIN-SFTNS -->
  - [Akicou/GLM-4.7-Flash-REAP-09](https://huggingface.co/Akicou/GLM-4.7-Flash-REAP-09)
  - [Akicou/GLM-4.7-Flash-REAP-19](https://huggingface.co/Akicou/GLM-4.7-Flash-REAP-19)
  - [Akicou/GLM-4.7-Flash-REAP-39](https://huggingface.co/Akicou/GLM-4.7-Flash-REAP-39)
  - [Akicou/GLM-4.7-Flash-REAP-50](https://huggingface.co/Akicou/GLM-4.7-Flash-REAP-50)
  - [Akicou/INTELLECT-3-REAP-50-FP8-Dynamic](https://huggingface.co/Akicou/INTELLECT-3-REAP-50-FP8-Dynamic)
  - [Akicou/MiniMax-M2-5-REAP-19](https://huggingface.co/Akicou/MiniMax-M2-5-REAP-19)
  - [Akicou/MiniMax-M2-5-REAP-29](https://huggingface.co/Akicou/MiniMax-M2-5-REAP-29)
  - [Akicou/MiniMax-M2-5-REAP-39](https://huggingface.co/Akicou/MiniMax-M2-5-REAP-39)
  - [Akicou/MiniMax-M2-5-REAP-50](https://huggingface.co/Akicou/MiniMax-M2-5-REAP-50)
  - [Akicou/Solar-Open-69B-REAP](https://huggingface.co/Akicou/Solar-Open-69B-REAP)
  <!-- END-SFTNS -->

- **GGUF** - For llama.cpp and compatible frontends
  <!-- BEGIN-GGUF -->
  - [Akicou/GLM-4.7-Flash-REAP-09-GGUF](https://huggingface.co/Akicou/GLM-4.7-Flash-REAP-09-GGUF)
  - [Akicou/GLM-4.7-Flash-REAP-19-GGUF](https://huggingface.co/Akicou/GLM-4.7-Flash-REAP-19-GGUF)
  - [Akicou/GLM-4.7-Flash-REAP-39-GGUF](https://huggingface.co/Akicou/GLM-4.7-Flash-REAP-39-GGUF)
  - [Akicou/GLM-4.7-Flash-REAP-50-GGUF](https://huggingface.co/Akicou/GLM-4.7-Flash-REAP-50-GGUF)
  - [Akicou/INTELLECT-3-REAP-50-GGUF](https://huggingface.co/Akicou/INTELLECT-3-REAP-50-GGUF)
  - [Akicou/INTELLECT-3-REAP-50-heretic-GGUF](https://huggingface.co/Akicou/INTELLECT-3-REAP-50-heretic-GGUF)
  - [Akicou/MiniMax-M2.1-REAP-40-GGUF](https://huggingface.co/Akicou/MiniMax-M2.1-REAP-40-GGUF)
  - [Akicou/MiniMax-M2.1-REAP-50-GGUF](https://huggingface.co/Akicou/MiniMax-M2.1-REAP-50-GGUF)
  <!-- END-GGUF -->

*Check my HuggingFace profile for available models.*

---

## Adding a New Model

See [A1_how_to_add_model.md](./A1_how_to_add_model.md) for detailed instructions on adding a new HuggingFace MoE model for REAP pruning.

---

## Apple Silicon (MLX) Support

REAP now supports Apple Silicon devices through MLX backend! This enables running REAP on Mac Studio, MacBook Pro, and other Apple devices with M1/M2/M3 chips.

### Installation for MLX

```bash
# Install MLX dependencies
pip install mlx mlx-lm

# Install REAP
pip install -e .
```

### Using MLX Backend

```bash
# Use MLX backend (auto-detects Apple Silicon)
python -m reap.prune \
    --backend mlx \
    --model_name "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-MLX" \
    --compression_ratio 0.5 \
    --prune_method "reap"
```

### MLX Configuration

A default MLX configuration is provided at `src/reap/configs/mlx_config.yaml`:

```yaml
backend: mlx
model:
  path: "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-MLX"
pruning:
  strategy: "activation"
  activation_threshold: 0.01
```

### Supported MLX Models

MLX requires models in MLX format. You can:

1. **Use pre-converted models** from `mlx-community` on HuggingFace
2. **Convert HuggingFace models** using `mlx-lm`:

```bash
# Convert a HuggingFace model to MLX
python -m mlx_lm.convert --hf-path deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --mlx-path ./models/deepseek-lite-mlx
```

### MLX vs PyTorch Backend

| Feature | PyTorch | MLX |
|---------|---------|-----|
| Platform | NVIDIA GPUs (CUDA) | Apple Silicon (M1/M2/M3) |
| Model Format | HuggingFace | MLX-converted |
| Evaluation | vLLM server | mlx-lm direct |
| Memory | VRAM limited | Unified memory |
| Performance | Fast on NVIDIA | Optimized for Apple |

### Performance on Mac Studio M3 Ultra

- 26-30% faster than Ollama for LLM inference
- Token generation: 20-31 tokens/sec (varies by model size)
- Unified memory supports models up to 1T parameters
- Power: <200W under load

### ⚠️ Important: Untested Implementation

**The MLX backend implementation is currently untested.** This code was written without access to Apple Silicon hardware for runtime verification.

**Status:**
- ✅ Code structure complete
- ✅ All abstractions defined
- ✅ MLX patterns implemented based on documentation
- ⚠️ **Runtime testing required on Mac Studio M3 Ultra**

**Expected to work** based on:
- MLX documentation and examples
- MLX-LM package patterns
- Parallel structure with tested PyTorch backend

**Help needed:** If you have a Mac Studio, please test and report issues!

---

---

## Running REAP - Complete Arguments Guide

### Basic Usage

```bash
python -m reap.prune --model_name <MODEL> --compression_ratio <RATIO>
```

### Complete Command with All Arguments

```bash
# Environment variables (optional)
export HF_TOKEN=your_huggingface_token_here              # Required for auto-upload to HuggingFace
export DISCORD_WEBHOOK=your_discord_webhook_url_here      # Optional: For progress notifications

# Main pruning command
python -m reap.prune \
    # ===== REQUIRED ARGUMENTS =====
    --model_name "MiniMaxAI/MiniMax-M2.5" \
    --compression_ratio "0.1,0.2,0.3,0.4,0.5" \
    --prune_method "reap" \
    \
    # ===== MODEL ARGUMENTS =====
    # --num_experts_per_tok_override 8 \                # Override number of experts per token (optional)
    \
    # ===== DATASET ARGUMENTS =====
    --dataset_name "theblackcat102/evol-codealpaca-v1" \  # Dataset for observation
    # --dataset_config_name "all" \                    # Dataset configuration name (default: "all")
    # --split "train" \                                # Dataset split to use (default: "train")
    # --shuffle \                                      # Whether to shuffle dataset (default: True)
    \
    # ===== OBSERVER ARGUMENTS =====
    # --samples_per_category 1024 \                    # Number of samples per category (default: 1024)
    # --split_by_category \                            # Split dataset by category (default: False)
    # --select_only_categories "category1,category2" \  # Select specific categories only (optional)
    # --model_max_length 2048 \                        # Maximum sequence length (default: 2048)
    # --return_vllm_tokens_prompt \                    # Return vLLM tokens prompt (default: False)
    # --truncate \                                     # Truncate sequences to max length (default: False)
    # --overwrite_observations \                       # Overwrite existing observer data (default: False)
    # --load_observer_state "path/to/observer.pt" \    # Load saved observer state instead of running observation
    # --distance_measure "cosine" \                    # Distance function: angular, euclidean, jsd, cka, cosine (default: "angular")
    # --output_file_name "observations_1024_cosine.pt" \ # Output filename for observer data
    # --record_pruning_metrics_only \                  # Only record pruning metrics to reduce memory (default: False)
    # --renormalize_router_weights \                   # Renormalize topk router weights if norm_topk_prob is True (default: False)
    # --load_in_4bit \                                 # Load model in 4-bit quantization for observation (default: False)
    \
    # ===== CLUSTERING ARGUMENTS =====
    # --expert_sim "ttm" \                             # Expert similarity method: ttm, dynamic_ttm, characteristic_activation, routed_characteristic_activation, router_logits, online_characteristic_activation_dist (default: "ttm")
    # --num_clusters 128 \                             # Number of clusters per layer (default: auto-calculated from compression_ratio)
    # --cluster_method "agglomerative" \               # Clustering method: agglomerative, kmeans, spectral, mc_smoe (default: "agglomerative")
    # --linkage_method "average" \                     # Linkage method for agglomerative: ward, complete, average, single (default: "average")
    # --frequency_penalty \                            # Apply frequency penalty to expert similarity (default: True)
    # --softmax_temperature 1.0 \                      # Temperature for softmax scaling (default: None)
    # --multi_layer 4 \                                # Number of layers to merge at once (default: None, all layers separately)
    # --max_cluster_size 32 \                          # Maximum experts per cluster (default: None)
    # --singleton_super_experts \                      # Keep super experts in singleton clusters (default: False)
    # --singleton_outlier_experts \                    # Keep outlier experts in singleton clusters (default: False)
    \
    # ===== PRUNING ARGUMENTS =====
    # --n_experts_to_prune 128 \                       # Number of experts to prune (default: auto-calculated from compression_ratio)
    # --perserve_super_experts \                       # Preserve super experts (last 25% layers excluded) (default: False)
    # --perserve_outliers \                            # Preserve outlier experts (all layers included) (default: False)
    # --overwrite_pruned_model \                       # Overwrite existing pruned models (default: False - skips existing)
    \
    # ===== REAP ARGUMENTS =====
    # --seed 42 \                                      # Random seed for reproducibility (default: 42)
    # --debug \                                        # Enable debug mode for verbose output (default: False)
    # --profile \                                      # Enable profiling before run to avoid OOM (default: True)
    # --run_observer_only \                            # Only run observer, skip pruning (default: False)
    --upload_calibration_to_hf \                      # Auto-upload calibration .pt file to HuggingFace (default: False)
    --discord_webhook "$DISCORD_WEBHOOK" \            # Discord webhook URL for progress notifications (default: None)
    --upload_pruned_to_hf \                            # Auto-upload pruned models to HuggingFace (default: False)
    # --verify_model_config \                          # Verify model config before pruning (default: True)
    # --do_eval \                                      # Run evaluation after pruning (default: True)
    # --plot_clusters \                                # Plot clusters after clustering (default: True)
    # --smoke_test \                                   # Run smoke test on merged model (default: True)
    \
    # ===== EVALUATION ARGUMENTS =====
    # --use_server \                                   # Use vLLM server for evaluation (default: True)
    # --greedy \                                      # Use greedy decoding (default: True)
    # --temperature 0.7 \                              # Sampling temperature (default: 0.7)
    # --top_p 0.8 \                                    # Top-p for nucleus sampling (default: 0.8)
    # --top_k 20 \                                     # Top-k for sampling (default: 20)
    # --min_p 0.0 \                                    # Min probability for sampling (default: 0.0)
    # --run_lm_eval \                                  # Run lm-eval tasks (default: True)
    # --run_evalplus \                                # Run evalplus tasks (default: True)
    # --run_livecodebench \                           # Run livecodebench tasks (default: True)
    # --run_wildbench \                               # Run wildbench tasks (default: False)
    # --run_math \                                    # Run math tasks (default: False)
    # --lm_eval_tasks "winogrande,arc_challenge,boolq" \ # LM Eval tasks to run
    # --evalplus_tasks "mbpp,humaneval" \              # EvalPlus tasks to run
    # --vllm_port 8000 \                               # Port for vLLM server (default: 8000)
    # --parallel_tasks 32 \                           # Parallel tasks for evalplus (default: 32)
```

### Common Use Cases

#### Quick Start (Minimum Required)
```bash
python -m reap.prune \
    --model_name "MiniMaxAI/MiniMax-M2.5" \
    --compression_ratio 0.5 \
    --prune_method "reap"
```

#### Multi-Ratio Pruning with Auto-Upload
```bash
export HF_TOKEN=your_token
export DISCORD_WEBHOOK=your_webhook

python -m reap.prune \
    --model_name "MiniMaxAI/MiniMax-M2.5" \
    --compression_ratio "0.1,0.2,0.3,0.4,0.5" \
    --prune_method "reap" \
    --upload_pruned_to_hf \
    --upload_calibration_to_hf \
    --discord_webhook "$DISCORD_WEBHOOK"
```

#### Observation Only (Save for Later)
```bash
python -m reap.prune \
    --model_name "MiniMaxAI/MiniMax-M2.5" \
    --run_observer_only \
    --samples_per_category 2048 \
    --distance_measure "cosine"
```

#### Load Saved Observer Data for Pruning
```bash
python -m reap.prune \
    --model_name "MiniMaxAI/MiniMax-M2.5" \
    --compression_ratio "0.1,0.2,0.3,0.4,0.5" \
    --prune_method "reap" \
    --load_observer_state "artifacts/MiniMax-M2.5/combined/all/observations_1024_cosine.pt"
```

#### Force Re-pruning Existing Models
```bash
python -m reap.prune \
    --model_name "MiniMaxAI/MiniMax-M2.5" \
    --compression_ratio "0.1,0.2,0.3,0.4,0.5" \
    --prune_method "reap" \
    --overwrite_pruned_model
```

#### Low VRAM Mode (4-bit Quantization for Observation)
```bash
python -m reap.prune \
    --model_name "MiniMaxAI/MiniMax-M2.5" \
    --compression_ratio 0.5 \
    --prune_method "reap" \
    --load_in_4bit
```

#### Skip Evaluation
```bash
python -m reap.prune \
    --model_name "MiniMaxAI/MiniMax-M2.5" \
    --compression_ratio 0.5 \
    --prune_method "reap" \
    --do_eval False
```

### Pruning Methods Explained

| Method | Description |
|--------|-------------|
| `reap` | Router-weighted Expert Activation Pruning (recommended) |
| `reap_l2` | REAP with L2 distance metric |
| `frequency` | Prune based on expert activation frequency |
| `weighted_frequency_sum` | Weighted sum of frequency-based metrics |
| `ean_ca`, `ean_sum`, `ean_mean` | Expert Activation Norm-based methods |
| `weighted_ean_sum`, `weighted_ean_sum_l2` | Weighted EAN variants |
| `max_activations` | Prune based on maximum activation values |
| `under_average` | Prune experts below average activation per layer |

### Compression Ratios Explained

The `--compression_ratio` determines what percentage of experts to remove:

- `0.1` = Remove 10% of experts (keep 90%)
- `0.2` = Remove 20% of experts (keep 80%)
- `0.3` = Remove 30% of experts (keep 70%)
- `0.4` = Remove 40% of experts (keep 60%)
- `0.5` = Remove 50% of experts (keep 50%)

For a 256-expert model with ratio `0.5`: 256 × (1 - 0.5) = **128 experts remaining**

### Auto-Upload Formats

| Upload Type | Repo Name Format | Example |
|-------------|------------------|---------|
| Calibration | `{username}/{MODEL}-{SampleSize}-OBS` | `Akicou/MiniMax-M2-5-1024-OBS` |
| Pruned Model | `{username}/{MODEL}-REAP-{Compression%}` | `Akicou/MiniMax-M2-5-REAP-50` |

### Discord Notifications

When `--discord_webhook` is set, you'll receive notifications for:
- 🚀 Process started
- ✅ Model loaded
- 🔍 Observation started
- 📂 Category progress (25%, 50%, 75%, 100%)
- ✅ Observation complete
- ✂️ Pruning started
- ✅ Pruning complete (per ratio)
- 📊 Evaluation started (if enabled)
- 🎉 Process complete

---

## Citation

```bibtex
@misc{lasby-reap,
    title       = {{REAP the Experts: Why Pruning Prevails for One-Shot MoE compression}},
    author      = {Lasby, Mike and Lazarevich, Ivan and Sinnadurai, Nish and Lie, Sean and Ioannou, Yani and Thangarasa, Vithursan},
    year        = {2025},
    publisher   = {arXiv},
    note        = {arXiv:2510.13999v1 [cs]},
    url         = {https://arxiv.org/abs/2510.13999v1},
}
```
