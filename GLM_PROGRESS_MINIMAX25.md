# Adding MiniMax-M2.5 to REAP: Step-by-Step Process

This document describes the complete process of adding the MiniMax-M2.5 model to the REAP pruning pipeline.

## Overview

**Model:** MiniMaxAI/MiniMax-M2.5
**Goal:** Configure the model for REAP expert pruning

## Step 1: Inspect the Model Config

First, we loaded the model configuration to understand its structure:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained('MiniMaxAI/MiniMax-M2.5', trust_remote_code=True)

print(f'Model type: {config.model_type}')
print(f'num_local_experts: {config.num_local_experts}')
print(f'num_experts_per_tok: {config.num_experts_per_tok}')
print(f'hidden_size: {config.hidden_size}')
print(f'intermediate_size: {config.intermediate_size}')
```

**Key findings from config:**
- Model class: `MiniMaxM2ForCausalLM`
- Model type: `minimax_m2`
- `num_local_experts: 256` (not `num_experts` like other models)
- `num_experts_per_tok: 8`
- `shared_intermediate_size: 0` (no shared experts)
- `use_routing_bias: True`

## Step 2: Download and Inspect Model Source Code

Since the model uses custom code, we fetched the modeling file directly from HuggingFace:

```python
import requests

url = "https://huggingface.co/MiniMaxAI/MiniMax-M2.5/raw/main/modeling_minimax_m2.py"
response = requests.get(url)
content = response.text
```

## Step 3: Find the MoE Block Class

We searched for MoE-related classes:

```python
import re
all_classes = re.findall(r'class (\w+)\([^)]*\):', content)

# Filter for MoE/expert-related classes
moe_classes = [cls for cls in all_classes if any(k in cls.lower() for k in ['moe', 'expert'])]
```

**Found classes:**
- `MiniMaxM2Experts` - The experts module (inherits from nn.ModuleList)
- `MiniMaxM2SparseMoeBlock` - The MoE block

## Step 4: Inspect MiniMaxM2SparseMoeBlock

We extracted the MoE block class definition:

```python
class MiniMaxM2SparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = MiniMaxM2Experts(config)
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))
```

**Key findings:**
- Router attribute: `gate` (nn.Linear layer)
- Experts attribute: `experts`
- Uses `config.num_local_experts` (not standard `num_experts`)

## Step 5: Inspect MiniMaxM2Experts (Individual Expert Structure)

```python
class MiniMaxM2Experts(nn.ModuleList):
    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        for _ in range(self.num_experts):
            self.append(MiniMaxM2MLP(config))
```

Each expert is a `MiniMaxM2MLP`, so we needed to check that:

```python
class MiniMaxM2MLP(nn.Module):
    def __init__(self, config: MiniMaxM2Config):
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
```

**Critical discovery:** The projections are `w1`, `w2`, `w3` - NOT `gate_proj`, `up_proj`, `down_proj`!

- `w1` = gate projection
- `w3` = up projection
- `w2` = down projection

## Step 6: Determine if Experts are Fused

Looking at `MiniMaxM2Experts`, it inherits from `nn.ModuleList` and appends individual `MiniMaxM2MLP` instances. This means:
- **NOT fused** - each expert is a separate module

## Step 7: Compile Configuration

Based on all findings, we compiled the configuration:

### MODEL_ATTRS entry:

```python
"MiniMaxM2ForCausalLM": {
    "moe_block": "mlp",
    "gate_proj": "w1",          # NOT gate_proj
    "up_proj": "w3",            # NOT up_proj
    "down_proj": "w2",          # NOT down_proj
    "experts": "experts",
    "fused": False,
    "router": "gate",
    "num_experts": "num_local_experts",     # NOT num_experts
    "num_experts_per_tok": "num_experts_per_tok",
}
```

### Observer Config:

```python
@dataclass
class MiniMaxM2ObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: "MiniMaxM2SparseMoeBlock"
    num_experts_attr_name: "config.num_local_experts"
    top_k_attr_name: "config.num_experts_per_tok"
    fused_experts: False
```

## Step 8: Add to Codebase

1. **model_util.py** - Added MODEL_ATTRS entry
2. **observer.py** - Added observer config class and registry entry

## Key Differences from Standard MoE Models

| Attribute | Standard (e.g., Qwen3) | MiniMax-M2.5 |
|-----------|------------------------|--------------|
| gate_proj | `gate_proj` | `w1` |
| up_proj | `up_proj` | `w3` |
| down_proj | `down_proj` | `w2` |
| num_experts key | `num_experts` | `num_local_experts` |
| Router | `gate` or `router` | `gate` |

## Challenges Encountered

1. **torch/torchvision version mismatch** - Initial loading failed due to compatibility issues
2. **Custom model code** - Model uses `trust_remote_code=True` with custom implementation
3. **Non-standard projection names** - w1/w2/w3 instead of gate_proj/up_proj/down_proj
4. **Different config key** - `num_local_experts` instead of `num_experts`

## Running Pruning

```bash
python -m reap.prune \
    --model_name "MiniMaxAI/MiniMax-M2.5" \
    --compression_ratio 0.5 \
    --prune_method reap \
    --dataset_name "theblackcat102/evol-codealpaca-v1"
```
