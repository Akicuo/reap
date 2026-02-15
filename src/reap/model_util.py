import torch
import logging
from typing import Any

logger = logging.getLogger(__name__)


MODEL_ATTRS = {
    "Qwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Qwen3-Coder-30B-A3B-Instruct": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "NonUniformQwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Llama4ForCausalLM": {
        "moe_block": "feed_forward",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "MixtralForCausalLM": {
        "moe_block": "block_sparse_moe",
        "gate_proj": "w3",
        "up_proj": "w1",
        "down_proj": "w2",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "DeepseekV2ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoEForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "moe_k",
    },
    "gpt-oss-20b": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Glm4MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "SolarOpenForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

    "VaetkiForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    # XiaomiMiMo/MiMo-V2-Flash - 309B MoE model
    "MiMoV2FlashForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    # PrimeIntellect/INTELLECT-3 - DeepSeek V3 based architecture
    "DeepseekV3ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    # Kimi-K2-Thinking - DeepSeek V3 based architecture
    "KimiK2ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    # GLM-4.7-Flash (zai-org/GLM-4.7-Flash) - glm4_moe_lite architecture
    # Layer 0 is dense (Glm4MoeLiteMLP), layers 1-46 are MoE (Glm4MoeLiteMoE)
    # Experts are fused in Glm4MoeLiteNaiveMoe with gate_up_proj tensor
    "Glm4MoeLiteForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    # meituan-longcat/LongCat-Flash-Thinking-2601 - 560B MoE model
    # MoE uses LongcatMoE with LongcatTopkRouter (router.classifier is the gate)
    # 512 real experts + 256 identity "zero experts" (total 768 from router's perspective)
    # top_k = 12, uses MLA attention similar to DeepSeek
    "LongcatCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "router",  # LongcatTopkRouter module
        "router_weight_attr": "classifier.weight",  # router.classifier is the gate Linear
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "moe_topk",
    },
    # Alternative name for LongCat model
    "LongcatForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "router",
        "router_weight_attr": "classifier.weight",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "moe_topk",
    },
    # MiniMaxAI/MiniMax-M2.5 - Uses w1/w2/w3 projections, MoE block at block_sparse_moe
    "MiniMaxM2ForCausalLM": {
        "moe_block": "block_sparse_moe",
        "gate_proj": "w1",
        "up_proj": "w3",
        "down_proj": "w2",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },

}


def _infer_model_attrs_from_model(model) -> dict:
    """
    Infer MODEL_ATTRS configuration by inspecting a loaded model.
    
    This analyzes the model structure to determine:
    - MoE block attribute name
    - Projection names (gate, up, down)
    - Whether experts are fused
    - Router attribute name
    - Config keys for num_experts and num_experts_per_tok
    """
    attrs = {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    }
    
    # Try to find a layer with MoE
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            # Find MoE block attribute
            for moe_attr in ["mlp", "block_sparse_moe", "moe", "feed_forward", "ffn"]:
                if hasattr(layer, moe_attr):
                    moe = getattr(layer, moe_attr)
                    # Check if it has experts (MoE indicator)
                    if hasattr(moe, 'experts'):
                        attrs["moe_block"] = moe_attr
                        
                        # Check for fused experts
                        if hasattr(moe.experts, 'gate_up_proj'):
                            attrs["fused"] = True
                            attrs["gate_proj"] = "gate_up_proj"
                            attrs["up_proj"] = "gate_up_proj"
                        elif hasattr(moe, 'experts') and len(list(moe.experts)) > 0:
                            expert = list(moe.experts)[0]
                            # Find projection names
                            for proj_name in ["gate_proj", "w3", "wi_0"]:
                                if hasattr(expert, proj_name):
                                    attrs["gate_proj"] = proj_name
                                    break
                            for proj_name in ["up_proj", "w1", "wi_1", "fc1"]:
                                if hasattr(expert, proj_name):
                                    attrs["up_proj"] = proj_name
                                    break
                            for proj_name in ["down_proj", "w2", "wo", "fc2"]:
                                if hasattr(expert, proj_name):
                                    attrs["down_proj"] = proj_name
                                    break
                        
                        # Find router attribute
                        for router_name in ["gate", "router", "gating"]:
                            if hasattr(moe, router_name):
                                attrs["router"] = router_name
                                # Check if router uses classifier pattern (like LongCat)
                                router_module = getattr(moe, router_name)
                                if hasattr(router_module, 'classifier') and isinstance(router_module.classifier, torch.nn.Linear):
                                    attrs["router_weight_attr"] = "classifier.weight"
                                break
                        
                        break
            if attrs["moe_block"] != "mlp" or hasattr(getattr(layer, "mlp", None), "experts"):
                break
    
    # Infer num_experts config key from model.config
    if hasattr(model, 'config'):
        config = model.config
        for key in ["num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"]:
            if hasattr(config, key):
                attrs["num_experts"] = key
                break
        for key in ["num_experts_per_tok", "top_k", "moe_k", "num_selected_experts"]:
            if hasattr(config, key):
                attrs["num_experts_per_tok"] = key
                break
    
    return attrs


def ensure_model_registered(model) -> bool:
    """
    Ensure a model is registered in MODEL_ATTRS.
    
    If the model class is not in MODEL_ATTRS, this function will:
    1. Analyze the model structure
    2. Generate appropriate MODEL_ATTRS configuration
    3. Inject it into MODEL_ATTRS at runtime
    
    Args:
        model: The loaded model to check/register
        
    Returns:
        True if model was already registered or successfully auto-registered,
        False if registration failed.
    """
    model_class = model.__class__.__name__
    
    if model_class in MODEL_ATTRS:
        logger.debug(f"Model {model_class} already in MODEL_ATTRS")
        return True
    
    logger.info(f"Model {model_class} not in MODEL_ATTRS, attempting auto-registration...")
    
    try:
        # Infer attributes from the loaded model
        inferred_attrs = _infer_model_attrs_from_model(model)
        
        # Register the model
        MODEL_ATTRS[model_class] = inferred_attrs
        
        logger.info(f"Auto-registered MODEL_ATTRS for {model_class}: {inferred_attrs}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to auto-register MODEL_ATTRS for {model_class}: {e}")
        return False


def get_moe(model, layer):
    model_class = model.__class__.__name__
    if model_class not in MODEL_ATTRS:
        ensure_model_registered(model)
    moe_attr_name = MODEL_ATTRS.get(model_class, {}).get("moe_block", "mlp")
    return getattr(model.model.layers[layer], moe_attr_name)


def assert_merge(model, merged_moe, cluster_label):
    model_attr = MODEL_ATTRS.get(model.__class__.__name__)
    assert hasattr(merged_moe, "experts"), (
        "The merged module must have an 'experts' attribute."
    )

    gate_proj = model_attr["gate_proj"]
    down_proj = model_attr["down_proj"]

    if model_attr["fused"]:
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert torch.allclose(
                    getattr(merged_moe.experts, gate_proj)[dom_expert],
                    getattr(merged_moe.experts, gate_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
                assert torch.allclose(
                    getattr(merged_moe.experts, down_proj)[dom_expert],
                    getattr(merged_moe.experts, down_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
    else:
        up_proj = model_attr["up_proj"]
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert (
                    getattr(merged_moe.experts[dom_expert], up_proj).weight
                    == getattr(merged_moe.experts[expert], up_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], down_proj).weight
                    == getattr(merged_moe.experts[expert], down_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], gate_proj).weight
                    == getattr(merged_moe.experts[expert], gate_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."


def patched_model_map(model: str):
    patched = False
    model_name = model

    if model == "deepseek-ai/DeepSeek-V2-Lite-Chat":
        patched = True
        model_name = "artifacts/models/DeepSeek-V2-Lite-Chat"

    # until hf version lands
    if model == "baidu/ERNIE-4.5-21B-A3B-PT":
        patched = True
        model_name = "artifacts/models/ERNIE-4.5-21B-A3B-PT"

    if model == "Qwen/NonUniformQwen3-30B-A3B":
        patched = True
        model_name = "artifacts/models/NonUniformQwen3-30B-A3B"

    if model == "zai-org/GLM-4.5-Air":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air"

    if model == "zai-org/GLM-4.5-Air-FP8":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air-FP8"

    if model == "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8":
        patched = True
        model_name = "artifacts/models/Qwen3-Coder-480B-A35B-Instruct-FP8"

    # --- Model-specific patches for problematic models ---
    
    # PrimeIntellect/INTELLECT-3 - uses DeepSeek V3 architecture
    # The model may have config detection issues, handled in main loading logic
    if model == "PrimeIntellect/INTELLECT-3":
        patched = False  # No local patch needed, handled by config detection fix
        logger.info(f"PrimeIntellect/INTELLECT-3 detected - will use auto-detection and config handling")

    # moonshotai/Kimi-K2-Thinking - pre-quantized with CompressedTensorsConfig
    if model == "moonshotai/Kimi-K2-Thinking":
        patched = False  # No local patch needed, handled by pre-quantization detection
        logger.info(f"moonshotai/Kimi-K2-Thinking detected - will skip 4-bit quantization to avoid conflicts")

    if patched:
        logger.info(f"Using patched model for {model} from: {model_name}")
    return model_name


def assert_tied_weights(model, clusters_labels):
    model_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    for layer_idx in clusters_labels:
        clusters = clusters_labels[layer_idx]
        moe = get_moe(model, layer_idx)
        experts = getattr(moe, model_attrs["experts"])
        for cluster_idx in torch.unique(clusters):
            experts_in_cluster = torch.where(clusters == cluster_idx)[0].tolist()
            dom_expert = experts[experts_in_cluster[0]]
            for attr in ["up_proj", "down_proj", "gate_proj"]:
                for expert_idx in experts_in_cluster:
                    if expert_idx == dom_expert:
                        continue
                    expert = experts[expert_idx]
                    proj = getattr(expert, attr)
                    weight = proj.weight
                    dom_proj = getattr(dom_expert, attr)
                    dom_weight = dom_proj.weight
                    if not torch.allclose(weight, dom_weight):
                        print(
                            f"Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and attr {attr} are not tied!"
                        )
                        print(f"Max diff: {torch.abs(weight - dom_weight).max()}")
                    # check adapters
                    for lora_adapter in ["lora_A", "lora_B"]:
                        if hasattr(proj, lora_adapter):
                            lora_weight = getattr(proj, lora_adapter).default.weight
                            dom_lora_weight = getattr(
                                dom_proj, lora_adapter
                            ).default.weight
                            if not torch.allclose(lora_weight, dom_lora_weight):
                                print(
                                    f"LoRA Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and adapter {lora_adapter} are not tied!"
                                )
                                print(
                                    f"Max diff: {torch.abs(lora_weight - dom_lora_weight).max()}"
                                )

def get_super_expert_indices(observer_data, include_last_layers: bool = False):
    logger.info("Identifying super experts to preserve...")
    quantile = 99.5
    times = 10
    all_max_activations = [layer['max_activations'] for layer in observer_data.values()]
    num_layers = len(all_max_activations)
    all_max_activations = torch.cat(all_max_activations).flatten()
    percentile_threshold = torch.quantile(all_max_activations, quantile / 100.0).item()
    abs_threshold = all_max_activations.max().item() / times
    final_threshold = max(percentile_threshold, abs_threshold)
    # reshape back into per layer data
    all_max_activations = all_max_activations.reshape(num_layers, -1)
    super_experts_mask = all_max_activations > final_threshold
    if not include_last_layers:
        # only consider first 75% of layers for super experts
        logger.info(
            "Only considering first 75% of layers for super expert "
            "identification since perserve_outliers is False"
        )
        num_layers = int(num_layers * 0.75)
        super_experts_mask[num_layers:, :] = False
    super_expert_idx = torch.argwhere(super_experts_mask)
    logger.info(f"Identified {super_experts_mask.sum().item()} super experts with threshold: {final_threshold:.4f}")
    return super_expert_idx


def verify_model_config(model_name: str, model=None) -> dict[str, Any]:
    """
    Verify that all model configurations are correct for REAP pruning.

    Args:
        model_name: Name of the model to verify
        model: Optional pre-loaded model instance. If None, will try to load.

    Returns:
        Dictionary with verification results:
        {
            "valid": bool,
            "model_class": str,
            "model_attrs": dict | None,
            "observer_config": str | None,
            "errors": list[str],
            "warnings": list[str],
            "details": dict,
        }
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from reap.observer import OBSERVER_CONFIG_REGISTRY
    import traceback as tb

    errors = []
    warnings = []
    details = {}

    logger.info(f"Verifying model configuration for: {model_name}")

    # Step 1: Get model class name
    try:
        if model is None:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_class = config.architectures[0] if config.architectures else None
            details["config_class"] = config.__class__.__name__
        else:
            model_class = model.__class__.__name__

        if not model_class:
            errors.append("Could not determine model class from config.architectures")
            return _format_verification_result(False, None, None, errors, warnings, details)

        details["model_class"] = model_class
        logger.info(f"Model class: {model_class}")

    except Exception as e:
        errors.append(f"Failed to get model config: {e}")
        return _format_verification_result(False, None, None, errors, warnings, details)

    # Step 2: Check MODEL_ATTRS
    model_attrs = MODEL_ATTRS.get(model_class)
    if model_attrs is None:
        errors.append(
            f"Model class '{model_class}' not found in MODEL_ATTRS. "
            f"Supported classes: {list(MODEL_ATTRS.keys())}"
        )
    else:
        logger.info(f"✅ MODEL_ATTRS found for {model_class}")
        details["model_attrs"] = model_attrs

        # Verify required MODEL_ATTRS fields
        required_fields = ["moe_block", "gate_proj", "up_proj", "down_proj", "experts", "router", "num_experts"]
        missing_fields = [f for f in required_fields if f not in model_attrs]
        if missing_fields:
            errors.append(f"MODEL_ATTRS missing required fields: {missing_fields}")
        else:
            logger.info(f"✅ All required MODEL_ATTRS fields present")

    # Step 3: Check Observer Config
    observer_config = OBSERVER_CONFIG_REGISTRY.get(model_class)
    if observer_config is None:
        errors.append(
            f"Model class '{model_class}' not found in OBSERVER_CONFIG_REGISTRY. "
            f"Supported classes: {list(OBSERVER_CONFIG_REGISTRY.keys())}"
        )
    else:
        logger.info(f"✅ Observer config found for {model_class}")
        details["observer_config"] = observer_config.__class__.__name__

    # Step 4: If model provided, verify structure matches MODEL_ATTRS
    if model is not None and model_attrs:
        try:
            structure_errors = _verify_model_structure(model, model_class, model_attrs)
            if structure_errors:
                errors.extend(structure_errors)
            else:
                logger.info(f"✅ Model structure matches MODEL_ATTRS")
        except Exception as e:
            errors.append(f"Failed to verify model structure: {e}\n{tb.format_exc()}")

    # Step 5: Warnings for common issues
    if model is not None and hasattr(model, 'config'):
        if hasattr(model.config, 'quantization_config') and model.config.quantization_config:
            if "MiniMax" in model_class and "4bit" in str(model.config.quantization_config).lower():
                warnings.append("MiniMax-M2.5 models may have issues with pre-quantization. Ensure quantization_config is handled.")

    valid = len(errors) == 0
    return _format_verification_result(valid, model_class, model_attrs, errors, warnings, details)


def _verify_model_structure(model: Any, model_class: str, model_attrs: dict[str, Any]) -> list[str]:
    """Verify that the actual model structure matches MODEL_ATTRS."""
    errors = []

    # Find a decoder layer to inspect
    layers = None
    if hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model.model, 'decoder'):
            if hasattr(model.model.decoder, 'layers'):
                layers = model.model.decoder.layers
    elif hasattr(model, 'layers'):
        layers = model.layers

    if layers is None or len(layers) == 0:
        return ["Could not find any decoder layers in the model"]

    # Check first layer
    layer = layers[0]
    moe_block_path = model_attrs.get("moe_block")
    if not moe_block_path:
        return ["MODEL_ATTRS missing 'moe_block' path"]

    # Navigate to MoE block
    moe_block = None
    current = layer
    for attr in moe_block_path.split('.'):
        if hasattr(current, attr):
            moe_block = getattr(current, attr)
            current = moe_block
        else:
            errors.append(f"Layer 0 missing attribute '{moe_block_path}' (failed at '{attr}')")
            return errors

    if moe_block is None:
        errors.append(f"Could not find MoE block at path '{moe_block_path}' in layer 0")
        return errors

    logger.info(f"✅ Found MoE block: {moe_block.__class__.__name__}")

    # Check experts
    experts_path = model_attrs.get("experts")
    if experts_path:
        parts = experts_path.split('.')
        current = moe_block
        for i, part in enumerate(parts[:-1]) if len(parts) > 1 else []:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                errors.append(f"MoE block missing attribute '{part}' in experts path")
                return errors

        if hasattr(current, parts[-1]):
            experts = getattr(current, parts[-1])
            if hasattr(experts, '__len__'):
                num_experts = len(experts)
                logger.info(f"✅ Found {num_experts} experts")
            else:
                errors.append(f"Experts at '{experts_path}' is not a list/array")
        else:
            errors.append(f"MoE block missing 'experts' attribute at '{experts_path}'")

    # Check router
    router_path = model_attrs.get("router")
    if router_path and hasattr(moe_block, router_path):
        logger.info(f"✅ Found router: {getattr(moe_block, router_path).__class__.__name__}")
    elif router_path:
        errors.append(f"MoE block missing 'router' attribute at '{router_path}'")

    return errors


def _format_verification_result(
    valid: bool,
    model_class: str | None,
    model_attrs: dict[str, Any] | None,
    errors: list[str],
    warnings: list[str],
    details: dict[str, Any],
) -> dict[str, Any]:
    """Format verification results into a structured dictionary."""
    return {
        "valid": valid,
        "model_class": model_class,
        "model_attrs": model_attrs,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


def print_verification_result(result: dict[str, Any]) -> None:
    """Print verification results in a formatted way."""
    print("\n" + "=" * 70)
    print("REAP Model Configuration Verification")
    print("=" * 70)

    if result["model_class"]:
        print(f"\n📦 Model Class: {result['model_class']}")

    if result["valid"]:
        print("\n✅ Configuration is VALID for REAP pruning!")
    else:
        print("\n❌ Configuration has ERRORS - pruning will likely FAIL!")

    if result["model_attrs"]:
        print(f"\n🔧 MODEL_ATTRS:")
        for key, value in result["model_attrs"].items():
            print(f"   {key}: {value}")

    if result["details"].get("observer_config"):
        print(f"\n🔍 Observer Config: {result['details']['observer_config']}")

    if result["warnings"]:
        print(f"\n⚠️  WARNINGS ({len(result['warnings'])}):")
        for warning in result["warnings"]:
            print(f"   - {warning}")

    if result["errors"]:
        print(f"\n❌ ERRORS ({len(result['errors'])}):")
        for error in result["errors"]:
            print(f"   - {error}")

    print("\n" + "=" * 70)


def register_llama_with_vllm():
    from vllm.model_executor.models import ModelRegistry
    print("Registering Llama4ForCausalLM with vLLM")
    ModelRegistry.register_model("Llama4ForCausalLM", "vllm.model_executor.models.llama4:Llama4ForCausalLM")