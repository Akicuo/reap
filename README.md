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
