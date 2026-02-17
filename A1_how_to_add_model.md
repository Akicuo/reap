# How to Add a New MoE Model for REAP Pruning

This guide explains how to add a HuggingFace Mixture-of-Experts (MoE) model to the REAP pruning pipeline.

## Quick Start

Many models will work automatically thanks to auto-detection and built-in verification. Try running your model first:

```bash
python -m reap.prune --model_name "your-org/your-model" --compression_ratio 0.5
```

**NEW:** REAP now includes automatic model configuration verification that runs before pruning. It will check:
- Model is in `MODEL_ATTRS`
- Observer config exists
- Model structure matches configuration
- All required attributes are present

If verification fails, you'll see detailed error messages telling you exactly what's missing.

---

## Step 1: Register Model Attributes

Edit `src/reap/model_util.py` and add an entry to the `MODEL_ATTRS` dictionary.

**Key:** Use `model.__class__.__name__` as the key (e.g., `"Qwen3MoeForCausalLM"`).

```python
MODEL_ATTRS = {
    "YourModelForCausalLM": {
        "moe_block": "mlp",                    # Attribute name of MoE submodule in decoder layer
        "gate_proj": "gate_proj",              # Expert gate projection attribute
        "up_proj": "up_proj",                  # Expert up projection attribute
        "down_proj": "down_proj",              # Expert down projection attribute
        "experts": "experts",                  # ModuleList containing the experts
        "fused": False,                        # True if using FusedMoE (like Llama4)
        "router": "gate",                      # Router/gate attribute name in MoE block
        "num_experts": "num_experts",          # Config key for total experts per layer
        "num_experts_per_tok": "num_experts_per_tok",  # Config key for experts per token
    },
}
```

### Finding the MoE Block Location

The MoE block location varies by model. Common locations:

| Location | Models |
|----------|--------|
| `mlp` | Qwen3, Mixtral, most standard MoE |
| `block_sparse_moe` | MiniMax-M2.5, some custom implementations |
| `feed_forward` | Llama4 |
| `moe` | Some DeepSeek models |

### How to Find These Values

Load your model and inspect its structure:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-org/your-model", trust_remote_code=True)

# Get model class name
print(f"Model class: {model.__class__.__name__}")

# Inspect config for expert counts
print(f"Config attributes: {[a for a in dir(model.config) if not a.startswith('_')]}")
print(f"num_experts: {getattr(model.config, 'num_experts', 'NOT FOUND')}")
print(f"num_local_experts: {getattr(model.config, 'num_local_experts', 'NOT FOUND')}")
print(f"num_experts_per_tok: {getattr(model.config, 'num_experts_per_tok', 'NOT FOUND')}")

# Inspect first layer's MoE block
layer = model.model.layers[0]
for attr in ["mlp", "block_sparse_moe", "moe", "feed_forward"]:
    if hasattr(layer, attr):
        moe = getattr(layer, attr)
        print(f"MoE block found at: {attr}")
        print(f"MoE class: {moe.__class__.__name__}")
        print(f"MoE attributes: {[a for a in dir(moe) if not a.startswith('_')]}")

        # Check for experts
        if hasattr(moe, "experts"):
            expert = moe.experts[0]
            print(f"Expert attributes: {[a for a in dir(expert) if not a.startswith('_')]}")

        # Check for router
        for router_name in ["gate", "router", "gating"]:
            if hasattr(moe, router_name):
                print(f"Router found at: {router_name}")
                break
        break
```

### How to Detect Fused Experts

Fused experts combine `gate_proj` and `up_proj` into a single tensor for efficiency. You need to know this to set the `"fused"` flag correctly.

**Quick detection script:**

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-org/your-model", trust_remote_code=True)

# Find MoE block
for layer in model.model.layers:
    for attr in ["mlp", "block_sparse_moe", "moe", "feed_forward"]:
        if hasattr(layer, attr):
            moe = getattr(layer, attr)
            if hasattr(moe, 'experts'):
                experts = moe.experts

                # Check for fused experts
                is_fused = hasattr(experts, 'gate_up_proj')

                print(f"MoE class: {moe.__class__.__name__}")
                print(f"Experts class: {experts.__class__.__name__}")
                print(f"Fused: {is_fused}")

                if is_fused:
                    print(f"  gate_up_proj shape: {experts.gate_up_proj.shape}")
                    print(f"  down_proj shape: {experts.down_proj.shape}")
                else:
                    # Check individual expert structure
                    expert = experts[0]
                    print(f"  Expert attributes: {[a for a in dir(expert) if not a.startswith('_')]}")
                break
```

**Key differences:**

| | Non-Fused | Fused |
|--|-----------|-------|
| **Structure** | `experts` is `ModuleList[Expert]` | `experts` is a single module with tensor weights |
| **Projections** | Separate `gate_proj`, `up_proj`, `down_proj` per expert | Combined `gate_up_proj` tensor, separate `down_proj` |
| **Weight Storage** | `[num_experts]` Module objects | Single tensor `[num_experts, 2*intermediate, hidden]` |
| **Examples** | Qwen3, Mixtral, DeepSeek, MiniMax-M2.5 | Llama4, GLM-4.7-Flash |

**Detection logic:**
```python
# If this exists -> FUSED
moe.experts.gate_up_proj  # Tensor [num_experts, 2*intermediate_size, hidden_size]

# If this exists -> NOT FUSED
moe.experts[0].gate_proj  # nn.Linear for each expert separately
```

### Special Cases

**Fused Experts (like Llama4, GLM-4.7-Flash):**
```python
"Glm4MoeLiteForCausalLM": {
    "moe_block": "mlp",
    "gate_proj": "gate_up_proj",  # Fused gate+up projection
    "up_proj": "gate_up_proj",    # Same tensor
    "down_proj": "down_proj",
    "experts": "experts",
    "fused": True,                # Set to True
    "router": "gate",
    "num_experts": "n_routed_experts",
    "num_experts_per_tok": "num_experts_per_tok",
},
```

**Non-Standard Projection Names (like MiniMax-M2.5):**
```python
"MiniMaxM2ForCausalLM": {
    "moe_block": "block_sparse_moe",  # NOT "mlp"!
    "gate_proj": "w1",               # NOT gate_proj
    "up_proj": "w3",                 # NOT up_proj
    "down_proj": "w2",               # NOT down_proj
    "experts": "experts",
    "fused": False,
    "router": "gate",
    "num_experts": "num_local_experts",     # NOT num_experts
    "num_experts_per_tok": "num_experts_per_tok",
},
```

**Nested Router Weight (like LongCat):**
```python
"LongcatCausalLM": {
    # ... other attributes ...
    "router": "router",
    "router_weight_attr": "classifier.weight",  # Router uses classifier.weight, not weight directly
},
```

---

## Step 2: Register Observer Config

Edit `src/reap/observer.py` and add an entry to `OBSERVER_CONFIG_REGISTRY`.

### IMPORTANT: Config Path vs Direct Attribute Access

**NEW:** Some models don't have a `config` attribute in their MoE block. For these, you must use **direct attribute access**.

**Option A: Config Path (Standard - MoE block has config attribute)**
```python
@dataclass
class YourModelObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "YourMoEBlockClassName"
    num_experts_attr_name: str = "config.num_experts"  # Access via model.config
    top_k_attr_name: str = "config.num_experts_per_tok"
    fused_experts: bool = False
```

**Option B: Direct Attribute (For models like MiniMax-M2.5 without config)**
```python
@dataclass
class MiniMaxM2ObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "MiniMaxM2SparseMoeBlock"
    num_experts_attr_name: str = "experts.num_experts"  # Direct: moe.experts.num_experts
    top_k_attr_name: str = "top_k"                      # Direct: moe.top_k
    fused_experts: bool = False
```

**How to detect which to use:**

```python
moe = model.model.layers[0].mlp  # or wherever MoE block is

# Check if MoE block has config attribute
if hasattr(moe, 'config'):
    print("MoE block has config - use config.path format")
    # Use: "config.num_experts"
else:
    print("MoE block has NO config - use direct attribute format")
    # Use: "experts.num_experts" or similar
```

### Complete Example

```python
from dataclasses import dataclass

@dataclass
class YourModelObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "YourMoEBlockClassName"
    num_experts_attr_name: str = "num_experts"  # or "config.num_experts" or "experts.num_experts"
    top_k_attr_name: str = "num_experts_per_tok"  # or "config.top_k" or "top_k"
    fused_experts: bool = False

OBSERVER_CONFIG_REGISTRY = {
    # ... existing entries ...
    "YourModelForCausalLM": YourModelObserverHookConfig,
}
```

### Finding MoE Block Class Name

```python
# Find the MoE block class name
for layer in model.model.layers:
    for attr in ["mlp", "block_sparse_moe", "moe"]:
        if hasattr(layer, attr):
            moe = getattr(layer, attr)
            if hasattr(moe, "experts"):
                print(f"MoE class name: {moe.__class__.__name__}")
                break
```

### Finding num_experts and top_k Attributes

```python
moe = model.model.layers[0].mlp  # or wherever MoE block is

# IMPORTANT: Check if MoE block has config
has_config = hasattr(moe, 'config')
print(f"MoE block has config attribute: {has_config}")

# Check direct attributes ON MoE block
for attr in ["num_experts", "num_local_experts", "n_routed_experts"]:
    if hasattr(moe, attr):
        print(f"Found (direct): {attr}")

# Check if MoE block has experts with num_experts
if hasattr(moe, 'experts'):
    experts = moe.experts
    if hasattr(experts, 'num_experts'):
        print(f"Found (experts.num_experts): experts.num_experts")

# Check config attributes (only if has_config is True)
if has_config:
    for attr in ["num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"]:
        if hasattr(moe.config, attr):
            print(f"Found (config): config.{attr}")

# For top_k - check MoE block directly first, then config
for attr in ["top_k", "num_experts_per_tok", "k", "moe_k"]:
    if hasattr(moe, attr):
        print(f"Found top_k (direct): {attr}")
        break
if has_config:
    for attr in ["num_experts_per_tok", "top_k"]:
        if hasattr(moe.config, attr):
            print(f"Found top_k (config): config.{attr}")
```

---

## Step 3: Handle Router Logits (If Needed)

Some models don't return `router_logits` from their MoE forward method. If you see warnings like:

```
Could not retrieve router_logits for YourMoEBlock at layer 0
```

You need to patch the model. Edit `src/reap/models/auto_patch.py`:

### Option A: Add to Generic Patching

If your model follows common patterns, add its class name to the appropriate list:

```python
def needs_patching(model: nn.Module) -> bool:
    # Add to no_patch_needed if model returns router_logits properly
    no_patch_needed = [
        # ... existing ...
        "YourModelForCausalLM",  # Add here if it returns router_logits
    ]

    # Add to patch_required if model needs patching
    patch_required = [
        # ... existing ...
        "YourModelForCausalLM",  # Add here if it needs patching
    ]
```

### Option B: Create Custom Patcher

For complex models, create a specific patcher:

```python
def _patch_your_model(model: nn.Module) -> int:
    """Specific patcher for YourModel."""
    patched_count = 0

    for layer in model.model.layers:
        if hasattr(layer, 'mlp') and layer.mlp.__class__.__name__ == 'YourMoEBlock':
            moe = layer.mlp

            if hasattr(moe, '_reap_patched'):
                continue

            hidden_size = moe.config.hidden_size

            def make_patched_forward(m, h_size):
                original_forward = m.forward.__func__ if hasattr(m.forward, '__func__') else m.forward

                def patched_forward(self, hidden_states):
                    # Compute router logits
                    hidden_flat = hidden_states.view(-1, h_size)
                    router_logits = F.linear(
                        hidden_flat.type(torch.float32),
                        self.gate.weight.type(torch.float32)  # Adjust as needed
                    )
                    self._last_router_logits = router_logits
                    return original_forward(self, hidden_states)

                return patched_forward

            moe.forward = types.MethodType(make_patched_forward(moe, hidden_size), moe)
            moe._reap_patched = True
            patched_count += 1

    return patched_count
```

Then add to `patch_specific_model()`:

```python
def patch_specific_model(model: nn.Module, model_class_name: str) -> int:
    specific_patchers = {
        # ... existing ...
        "YourModelForCausalLM": _patch_your_model,
    }
```

---

## Step 4: Handle Local Model Patches (If Needed)

If your model needs custom model code (not in HuggingFace transformers), create a local patch:

1. Create `artifacts/models/YourModel/` directory
2. Add your custom `modeling_your_model.py` and/or `tokenization_your_model.py`
3. Add mapping in `model_util.py`:

```python
def patched_model_map(model: str):
    if model == "your-org/your-model":
        return "artifacts/models/YourModel"
    # ... rest of function
```

---

## Step 5: Test Your Integration

Run the pruning pipeline:

```bash
# Step 1: Verify config (runs automatically)
python -m reap.prune \
    --model_name "your-org/your-model" \
    --compression_ratio 0.1 \
    --verify_model_config True

# Step 2: Test observer with small sample
python -m reap.prune \
    --model_name "your-org/your-model" \
    --compression_ratio 0.1 \
    --prune_method reap \
    --run_observer_only True \
    --samples_per_category 128

# Step 3: Full pruning test
python -m reap.prune \
    --model_name "your-org/your-model" \
    --compression_ratio 0.5 \
    --prune_method reap
```

---

## Checklist

- [ ] Added entry to `MODEL_ATTRS` in `model_util.py`
- [ ] Created observer config dataclass in `observer.py`
- [ ] Added entry to `OBSERVER_CONFIG_REGISTRY` in `observer.py`
- [ ] Verified config path vs direct attribute access for observer
- [ ] (If needed) Added patcher in `auto_patch.py`
- [ ] (If needed) Added local model mapping in `patched_model_map()`
- [ ] Tested with `--verify_model_config True`
- [ ] Tested observer with `--run_observer_only True`
- [ ] Tested full pruning pipeline

---

## Reference: Currently Supported Models

| Model | Class Name | MoE Block | Fused | Notes |
|-------|-----------|-----------|-------|-------|
| Qwen3 MoE | `Qwen3MoeForCausalLM` | `Qwen3MoeSparseMoeBlock` | No | Standard |
| Llama4 | `Llama4ForCausalLM` | `Llama4TextMoe` | Yes | Fused experts |
| Mixtral | `MixtralForCausalLM` | `MixtralSparseMoeBlock` | No | Standard |
| MiniMax-M2.5 | `MiniMaxM2ForCausalLM` | `MiniMaxM2SparseMoeBlock` | No | Non-standard projections (w1/w2/w3), block_sparse_moe location |
| DeepSeek V2/V3 | `DeepseekV2ForCausalLM` | `DeepseekV2MoE` | No | Uses `experts_per_rank` |
| ERNIE 4.5 | `Ernie4_5_MoEForCausalLM` | `Ernie4_5_MoeMLP` | No | Custom patch |
| GLM-4.5 | `Glm4MoeForCausalLM` | `Glm4MoeMoE` | No | Custom patch |
| GLM-4.7-Flash | `Glm4MoeLiteForCausalLM` | `Glm4MoeLiteMoE` | Yes | Fused grouped_mm |
| LongCat | `LongcatCausalLM` | `LongcatMoE` | No | Nested router |

---

## Troubleshooting

### Model Verification Failed
**Error:** `Model X not in MODEL_ATTRS` or `No observer configuration registered for model X`

**Solution:** Add entries to `MODEL_ATTRS` and `OBSERVER_CONFIG_REGISTRY` in the appropriate files.

### "Module X does not have expected 'num_experts' or 'top_k' attributes"
**Error:** Observer tried to access `config.num_experts` but MoE block doesn't have a config attribute.

**Solution:** Use direct attribute access like `"experts.num_experts"` or `"top_k"` instead of `"config.num_experts"`.

### "Could not retrieve router_logits"
Your model needs patching. See Step 3.

### "MoE block missing attribute 'mlp'"
**Error:** MODEL_ATTRS has wrong `moe_block` value.

**Solution:** Check where the MoE block is actually located (might be `block_sparse_moe`, `feed_forward`, etc.).

### "AttributeError: 'X' object has no attribute 'gate_proj'"
**Error:** Wrong projection names in MODEL_ATTRS.

**Solution:** Inspect the expert structure to find actual projection names (might be `w1`, `w2`, `w3` for some models).

### CUDA OOM during observation
Use `--load_in_4bit True` to load the model in 4-bit quantization during observation.

### Cache/DynamicCache errors
**Error:** `CacheLayerMixin.__init__() got an unexpected keyword argument 'max_cache_len'`

**Solution:** Add a custom patcher that sets `model.config.use_cache = False`.

### Smoke test failures
Smoke tests may fail for some models (like MiniMax-M2.5) due to cache issues, but this doesn't affect the pruned model quality. The pruned models work fine.
