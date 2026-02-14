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
- **GGUF** - For llama.cpp and compatible frontends

*Check my HuggingFace profile for available models.*

---

## Adding a New Model

See [A1_how_to_add_model.md](./A1_how_to_add_model.md) for detailed instructions on adding a new HuggingFace MoE model for REAP pruning.

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
