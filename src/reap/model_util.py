import torch
import logging

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
    # MiniMax-M2.1-PRISM - likely similar to DeepSeek architecture
    "MiniMaxForCausalLM": {
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

    # Ex0bit/MiniMax-M2.1-PRISM - custom BitConfig tokenizer issues
    if model == "Ex0bit/MiniMax-M2.1-PRISM":
        patched = False  # No local patch needed, handled by tokenizer fallback
        logger.info(f"Ex0bit/MiniMax-M2.1-PRISM detected - will use tokenizer fallback handling")

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

def register_llama_with_vllm():
    from vllm.model_executor.models import ModelRegistry
    print("Registering Llama4ForCausalLM with vLLM")
    ModelRegistry.register_model("Llama4ForCausalLM", "vllm.model_executor.models.llama4:Llama4ForCausalLM")