from __future__ import annotations
import time
import logging
import dataclasses
import pathlib
import time
import math
from typing import Any
import gc
import yaml
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, BitsAndBytesConfig

from accelerate.utils import set_seed
from accelerate.hooks import remove_hook_from_module

from reap.transformers_compat import apply_transformers_compat_patches

from reap.main import record_activations, smoke_test, create_results_directory
from reap.args import (
    ReapArgs,
    ModelArgs,
    EvalArgs,
    PruneArgs,
    ObserverArgs,
    DatasetArgs,
    ClusterArgs,
    parse_compression_ratios,
)
from reap.data import DATASET_REGISTRY
from reap.cluster import (
    get_penalty_vector,
    hierarchical_clustering,
    dynamic_frequency_penalized_clustering,
)
from reap.model_util import get_moe, assert_merge, MODEL_ATTRS, patched_model_map, get_super_expert_indices, ensure_model_registered
from reap.observer import ensure_observer_config, generate_pruning_report
from reap.models.auto_patch import auto_patch_moe, patch_specific_model, needs_patching
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean-like environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def dump_args_to_yaml(
    pruned_model_dir: pathlib.Path,
    reap_args: ReapArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    model_args: ModelArgs,
    eval_args: EvalArgs,
    prune_args: PruneArgs,
    cluster_args: ClusterArgs,
):
    """Dump all arguments to a YAML file."""
    all_args = {
        "reap_args": dataclasses.asdict(reap_args),
        "ds_args": dataclasses.asdict(ds_args),
        "obs_args": dataclasses.asdict(obs_args),
        "model_args": dataclasses.asdict(model_args),
        "eval_args": dataclasses.asdict(eval_args),
        "prune_args": dataclasses.asdict(prune_args),
        "cluster_args": dataclasses.asdict(cluster_args),
    }

    def convert_paths_to_str(data):
        if isinstance(data, dict):
            return {k: convert_paths_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_paths_to_str(i) for i in data]
        elif isinstance(data, pathlib.Path):
            return str(data)
        else:
            return data

    serializable_args = convert_paths_to_str(all_args)

    output_path = pruned_model_dir / "reap_args.yaml"
    with open(output_path, "w") as f:
        yaml.dump(serializable_args, f, default_flow_style=False)
    logger.info(f"All arguments saved to {output_path}")


def load_model_with_vllm(model_name: str, local_only: bool = False):
    """
    Load a model using vLLM and extract the underlying HuggingFace model.
    
    This is useful for FP8 pre-quantized models that are designed for vLLM
    and cannot be loaded directly with transformers.
    
    Args:
        model_name: HuggingFace model name or path
        local_only: Whether to only use local files
        
    Returns:
        The underlying model that can be used for pruning, or None if vLLM is not available
    """
    try:
        from vllm import LLM
        from vllm.model_executor.models import ModelRegistry
    except ImportError:
        logger.warning("vLLM is not installed. Install with: pip install vllm")
        return None
    
    logger.info(f"Loading model '{model_name}' with vLLM...")
    
    # Clear any existing CUDA memory before vLLM initialization
    torch.cuda.empty_cache()
    gc.collect()
    
    # Auto-detect number of GPUs for tensor parallelism
    num_gpus = torch.cuda.device_count()
    # Use power of 2 GPUs for tensor parallelism (1, 2, 4, 8)
    if num_gpus >= 8:
        tensor_parallel_size = 8
    elif num_gpus >= 4:
        tensor_parallel_size = 4
    elif num_gpus >= 2:
        tensor_parallel_size = 2
    else:
        tensor_parallel_size = 1
    logger.info(f"Detected {num_gpus} GPUs, using tensor_parallel_size={tensor_parallel_size}")
    
    try:
        # Create vLLM LLM instance - this handles FP8 loading automatically
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="auto",
            # Use tensor parallelism across available GPUs
            tensor_parallel_size=tensor_parallel_size,
            # Disable CUDA graph to allow model access
            enforce_eager=True,
            # Lower GPU memory utilization to avoid conflicts with previously initialized CUDA
            gpu_memory_utilization=0.50,
        )
        
        # Try to extract the underlying model
        # vLLM stores the model in the model_executor
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_executor'):
            model_executor = llm.llm_engine.model_executor
            
            # For single GPU, the model is directly accessible
            if hasattr(model_executor, 'driver_worker'):
                worker = model_executor.driver_worker
                if hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'model'):
                    vllm_model = worker.model_runner.model
                    
                    # The vLLM model wraps the HuggingFace model
                    if hasattr(vllm_model, 'model'):
                        hf_model = vllm_model.model
                        logger.info(f"Extracted HuggingFace model: {hf_model.__class__.__name__}")
                        
                        # We need to wrap this in a way that's compatible with our pruning code
                        # Create a wrapper class that mimics AutoModelForCausalLM behavior
                        class VLLMModelWrapper(torch.nn.Module):
                            def __init__(self, vllm_llm, inner_model):
                                super().__init__()
                                self._vllm_llm = vllm_llm  # Keep reference to prevent GC
                                self._inner_model = inner_model
                                # Copy attributes from inner model
                                self.config = getattr(inner_model, 'config', None)
                                self.device = next(inner_model.parameters()).device
                                
                            def __getattr__(self, name):
                                if name.startswith('_') or name in ['config', 'device', 'forward', 'parameters', 'named_parameters', 'modules', 'named_modules', 'state_dict', 'load_state_dict']:
                                    return super().__getattribute__(name)
                                return getattr(self._inner_model, name)
                            
                            def forward(self, *args, **kwargs):
                                return self._inner_model(*args, **kwargs)
                            
                            def parameters(self, recurse=True):
                                return self._inner_model.parameters(recurse)
                            
                            def named_parameters(self, prefix='', recurse=True):
                                return self._inner_model.named_parameters(prefix, recurse)
                            
                            def modules(self):
                                return self._inner_model.modules()
                            
                            def named_modules(self, memo=None, prefix='', remove_duplicate=True):
                                return self._inner_model.named_modules(memo, prefix, remove_duplicate)
                            
                            def state_dict(self, *args, **kwargs):
                                return self._inner_model.state_dict(*args, **kwargs)
                            
                            def save_pretrained(self, save_directory, **kwargs):
                                """Save the model in HuggingFace format."""
                                import os
                                os.makedirs(save_directory, exist_ok=True)
                                
                                # Save config if available
                                if self.config is not None:
                                    self.config.save_pretrained(save_directory)
                                
                                # Save model weights
                                state_dict = self.state_dict()
                                save_path = os.path.join(save_directory, "pytorch_model.bin")
                                torch.save(state_dict, save_path)
                                logger.info(f"Saved model weights to {save_path}")
                        
                        wrapper = VLLMModelWrapper(llm, hf_model)
                        return wrapper
        
        logger.warning("Could not extract model from vLLM. Model structure may have changed.")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load model with vLLM: {e}")
        import traceback
        traceback.print_exc()
        return None


def prune(
    observer_data,
    model,
    tokenizer,
    reap_args,
    prune_args,
    n_experts_to_prune,
    pruned_model_dir,
    category_expert_map: dict = None,
):
    """
    Prune the model based on the observer data and clustering.
    
    Args:
        observer_data: Dictionary containing observer metrics per layer
        model: The model to prune
        tokenizer: The tokenizer
        reap_args: Reap configuration arguments
        prune_args: Pruning configuration arguments
        n_experts_to_prune: Number of experts to prune per layer
        pruned_model_dir: Directory to save the pruned model
        category_expert_map: Optional mapping of layer_idx -> expert_idx -> category name
    
    Returns:
        pruned_model_dir: Path to the pruned model directory
    """
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    
    # Track which experts are pruned per layer for the report
    experts_to_prune_per_layer: dict[int, torch.Tensor] = {}

    for layer in observer_data:
        if "expert_proba" not in observer_data[layer]:
            # Calculate expert probabilities if not already present
            observer_data[layer]["expert_proba"] = (
                observer_data[layer]["expert_frequency"]
                / observer_data[layer]["total_tokens"]
            )

    if prune_args.perserve_super_experts or prune_args.perserve_outliers:
        super_expert_idx = get_super_expert_indices(observer_data, include_last_layers=prune_args.perserve_outliers)
        metrics = [
            "expert_proba",
            "ean_sum",
            "ean_mean",
            "weighted_expert_frequency_sum",
            "weighted_ean_sum",
            "reap",
            "reap_l2",
            "weighted_ean_sum_l2",
        ]
        for layer in observer_data:
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][:, 1]
            if len(super_experts_in_layer) > 0:
                for metric in metrics:
                    if metric in observer_data[layer]:
                        observer_data[layer][metric][super_experts_in_layer] = float("inf")

    for layer in tqdm(observer_data, "Pruning layers..."):
        num_experts = observer_data[layer]["expert_frequency"].shape[0]
        if prune_args.prune_method == "under_average":
            # Under-average pruning: prune all experts with activation count below average
            # Example: experts with counts [22, 16, 14, 1] -> avg = ceil(53/4) = 14
            # Expert 4 (count=1) is below 14, so it gets pruned
            expert_frequency = observer_data[layer]["expert_frequency"]
            if isinstance(expert_frequency, torch.Tensor):
                freq_values = expert_frequency.float()
            else:
                freq_values = torch.tensor(expert_frequency, dtype=torch.float32)
            
            total_activations = freq_values.sum().item()
            # Round UP to next integer (ceiling)
            average_threshold = math.ceil(total_activations / num_experts)
            
            # Find all experts below average
            below_average_mask = freq_values < average_threshold
            experts_to_prune = torch.where(below_average_mask)[0]
            
            # Ensure we don't prune ALL experts - keep at least 1
            if len(experts_to_prune) >= num_experts:
                # Keep the expert with highest frequency
                _, best_expert = torch.topk(freq_values, 1, largest=True)
                experts_to_prune = torch.tensor(
                    [i for i in range(num_experts) if i != best_expert.item()],
                    device=freq_values.device
                )
            
            logger.info(
                f"Layer {layer}: avg threshold={average_threshold}, "
                f"pruning {len(experts_to_prune)}/{num_experts} experts below average"
            )
        elif prune_args.prune_method == "ean_ca":
            ean = torch.zeros(num_experts, device=model.device, dtype=torch.float32)
            for i in range(num_experts):
                ean[i] = torch.linalg.norm(
                    observer_data[layer]["routed_characteristic_activation"][i], dim=-1
                ).sum()
            _, experts_to_prune = torch.topk(ean, n_experts_to_prune, largest=False)
        else:
            prune_method = prune_args.prune_method
            if prune_method == "frequency":
                prune_method = "expert_frequency"
            saliency_data = observer_data[layer].get(prune_method)
            if saliency_data is None:
                raise ValueError(
                    f"Prune method {prune_args.prune_method} not found in observer data for layer {layer}. "
                    f"Available keys: {list(observer_data[layer].keys())}"
                )
            _, experts_to_prune = torch.topk(
                saliency_data, n_experts_to_prune, largest=False
            )
        
        # Store experts to prune for the report
        experts_to_prune_per_layer[layer] = experts_to_prune.clone()

        retained_expert_indicies = [
            i for i in range(num_experts) if i not in experts_to_prune
        ]
        # prune experts
        moe = get_moe(model, layer)
        if not model_attrs["fused"]:
            all_experts = getattr(moe, model_attrs["experts"])
            retained_experts = [all_experts[i] for i in retained_expert_indicies]
            retained_experts = torch.nn.ModuleList(retained_experts)
            setattr(moe, model_attrs["experts"], retained_experts)
            if model.__class__.__name__.lower() == "Ernie4_5_MoEForCausalLM".lower():
                # transformers version >=4.54
                # prune expert score correction bias too
                moe.moe_statics.e_score_correction_bias.data = (
                    moe.moe_statics.e_score_correction_bias.data[
                        :, retained_expert_indicies
                    ]
                )

            # prune router
            router = getattr(moe, model_attrs["router"])
            
            # Handle different router structures:
            # - Standard: router.weight (direct Linear layer)
            # - LongCat-style: router.classifier.weight (router is a module with classifier attr)
            router_weight_attr = model_attrs.get("router_weight_attr", None)
            if router_weight_attr and "." in router_weight_attr:
                # Handle nested attribute like "classifier.weight"
                parts = router_weight_attr.split(".")
                inner_module = router
                for part in parts[:-1]:
                    inner_module = getattr(inner_module, part)
                weight_attr = parts[-1]
                # Update weight
                setattr(inner_module, weight_attr, 
                        getattr(inner_module, weight_attr)[retained_expert_indicies, :])
                # Update bias if present
                bias_attr = weight_attr.replace("weight", "bias")
                if hasattr(inner_module, bias_attr) and getattr(inner_module, bias_attr) is not None:
                    setattr(inner_module, bias_attr,
                            getattr(inner_module, bias_attr)[retained_expert_indicies])
                # Update out_features
                if hasattr(inner_module, "out_features"):
                    inner_module.out_features = len(retained_expert_indicies)
                # Also update n_routed_experts on router if present (LongCat has this)
                if hasattr(router, "n_routed_experts"):
                    router.n_routed_experts = len(retained_expert_indicies)
            else:
                # Standard router with direct weight attribute
                router.weight.data = router.weight.data[retained_expert_indicies, :]
                if getattr(router, "bias", None):
                    router.bias.data = router.bias.data[retained_expert_indicies]
                router.out_features = len(retained_expert_indicies)
            
            # Handle e_score_correction_bias (present in some models like LongCat, Ernie)
            if hasattr(router, "e_score_correction_bias"):
                router.e_score_correction_bias.data = (
                    router.e_score_correction_bias.data[retained_expert_indicies]
                )
            setattr(moe, model_attrs["router"], router)
        else:
            # prune fused experts (Llama4, Glm4MoeLite, etc.)
            moe.experts.gate_up_proj.data = moe.experts.gate_up_proj[
                retained_expert_indicies
            ]
            moe.experts.down_proj.data = moe.experts.down_proj[retained_expert_indicies]
            if hasattr(moe, 'num_experts'):
                moe.num_experts = len(retained_expert_indicies)
            
            # Use model_attrs to get the correct router attribute name
            router_attr = model_attrs.get("router", "router")
            router = getattr(moe, router_attr)
            router.weight.data = router.weight.data[retained_expert_indicies]
            router.out_features = len(retained_expert_indicies)
            if hasattr(router, "num_experts"):  # transformers >= 4.54+
                router.num_experts = len(retained_expert_indicies)
            # Handle e_score_correction_bias if present (some models like GLM)
            if hasattr(router, "e_score_correction_bias"):
                router.e_score_correction_bias.data = router.e_score_correction_bias.data[retained_expert_indicies]

    # patch config and dump
    logger.info("Saving pruned model...")
    retained_experts = len(retained_expert_indicies)
    setattr(model.config, model_attrs["num_experts"], retained_experts)
    if model.__class__.__name__ == "Ernie4_5_MoeForCausalLM":  # remote-code verson
        model.config.moe_capacity = [
            retained_experts,
            retained_experts,
            retained_experts,
        ]

    pruned_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        model.save_pretrained(pruned_model_dir)
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.warning(f"CUDA error during save_pretrained: {e}")
            logger.info("Falling back to manual save with safetensors...")
            # Fallback: save manually without triggering deepspeed import
            from safetensors.torch import save_file
            state_dict = model.state_dict()
            # Move all tensors to CPU for saving
            state_dict = {k: v.cpu() for k, v in state_dict.items()}
            save_file(state_dict, pruned_model_dir / "model.safetensors")
            model.config.save_pretrained(pruned_model_dir)
            logger.info("Model saved using fallback method (safetensors)")
        else:
            raise
    end = time.time()
    logger.info(
        f"Pruned model saved to {pruned_model_dir} in {end - start:.2f} seconds"
    )
    
    # Generate and save pruning report
    logger.info("Generating pruning report...")
    try:
        pruning_report = generate_pruning_report(
            observer_data=observer_data,
            experts_to_prune_per_layer=experts_to_prune_per_layer,
            model_name=model.__class__.__name__,
            prune_method=prune_args.prune_method,
            category_expert_map=category_expert_map,
        )
        report_path = pruned_model_dir / "pruning_report.md"
        pruning_report.save(report_path)
        logger.info(f"Pruning report saved to {report_path}")
    except Exception as e:
        logger.warning(f"Failed to generate pruning report: {e}")
    
    return pruned_model_dir


def get_pruned_model_dir(
    results_dir,
    n_experts_to_prune: int | None,
    total_experts: int,
    prune_args,
    seed: int,
    renorm: bool,
) -> pathlib.Path:
    pruned_model_name = f"{prune_args.prune_method}"
    if prune_args.perserve_super_experts:
        pruned_model_name += "-perserve_super"
    elif prune_args.perserve_outliers:
        pruned_model_name += "-perserve_outlier"
    if renorm:
        pruned_model_name += f"-renorm_{str(renorm).lower()}"
    pruned_model_name += f"-seed_{seed}"
    
    # For under_average, we don't have a fixed compression ratio
    if prune_args.prune_method == "under_average":
        pruned_model_name += "-dynamic"
    else:
        compression_ratio_str = f"{(n_experts_to_prune / total_experts):.2f}"
        pruned_model_name += f"-{compression_ratio_str}"
    
    pruned_model_dir = results_dir / "pruned_models" / pruned_model_name
    logger.info(f"Using seed {seed}, pruned model dir: {pruned_model_dir}")
    return pruned_model_dir


def reload_model_in_full_precision(
    model_name: str,
    tokenizer: AutoTokenizer,
    local_only: bool,
    model_class_name: str,
):
    """
    Reload the model in full precision (bfloat16) for pruning.
    
    This is used after observer analysis with 4-bit quantization to ensure
    the pruned model is saved in full precision, not 4-bit.
    """
    logger.info("Reloading model in full precision (bfloat16) for pruning...")
    
    # Check if model is pre-quantized to avoid conflicts
    is_pre_quantized = False
    pre_quant_config = None
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_only)
        if hasattr(config, 'quantization_config') and config.quantization_config:
            is_pre_quantized = True
            pre_quant_config = config.quantization_config
            logger.info(f"Model appears to be pre-quantized: {pre_quant_config}")
    except Exception as config_error:
        logger.warning(f"Could not check for pre-quantization: {config_error}")
    
    # Determine the correct quantization config to use
    # For pre-quantized models (FP8, etc.), don't pass quantization_config
    # Let transformers handle it automatically from the model's config
    final_quant_config = None
    if is_pre_quantized:
        logger.info("Model is pre-quantized, will load with native quantization (no extra config needed)")
    
    # For pre-quantized models, we still load without additional quantization
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # Force bfloat16
            trust_remote_code=True,
            local_files_only=local_only,
            quantization_config=final_quant_config,  # Use the correct config
        )
    except TypeError as dtype_error:
        # Some models with custom code don't accept torch_dtype parameter
        if "unexpected keyword argument" in str(dtype_error):
            logger.warning(f"Model doesn't accept torch_dtype parameter: {dtype_error}")
            logger.info("Retrying without torch_dtype...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=local_only,
                quantization_config=final_quant_config,
            )
        else:
            raise
    except OSError as model_load_error:
        # Handle config detection issues
        if "does not appear to have a file named configuration" in str(model_load_error):
            logger.warning(f"Config detection failed during reload: {model_load_error}")
            
            # Special handling for PrimeIntellect/INTELLECT-3 (GLM4 misdetection fix)
            if model_name == "PrimeIntellect/INTELLECT-3":
                logger.info("PrimeIntellect/INTELLECT-3 detected - overriding GLM4 config misdetection during reload")
                # Use the same complex config handling as in main loading
                from transformers import AutoConfig
                try:
                    # First, check if any config files exist in the model repo manually
                    import os
                    from huggingface_hub import snapshot_download
                    # Download config.json only temporarily to inspect it
                    temp_dir = snapshot_download(model_name, allow_patterns=["config.json"], local_files_only=local_only)
                    
                    # Try to read the config.json directly to get the model type
                    config_path = os.path.join(temp_dir, "config.json")
                    if os.path.exists(config_path):
                        import json
                        with open(config_path, 'r') as f:
                            config_dict = json.load(f)
                        # Get the model type from the config
                        model_type = config_dict.get("model_type", "deepseek_v3")
                        logger.info(f"Detected model type from config.json: {model_type}")
                        
                        if model_type in ("glm4_moe", "glm4_moe_lite"):
                            # Use transformers built-in GLM4 config if available
                            try:
                                if model_type == "glm4_moe_lite":
                                    from transformers import Glm4MoeLiteConfig
                                    logger.info("Using built-in Glm4MoeLiteConfig")
                                    config = Glm4MoeLiteConfig.from_pretrained(model_name, trust_remote_code=True)
                                else:
                                    from transformers import Glm4MoeConfig
                                    logger.info("Using built-in Glm4MoeConfig")
                                    config = Glm4MoeConfig.from_pretrained(model_name, trust_remote_code=True)
                            except ImportError:
                                logger.warning("GLM4 config not found in transformers, trying fallback")
                                # Fallback to generic config but with correct model_type
                                from transformers import PretrainedConfig
                                config = PretrainedConfig(
                                    model_type=model_type,
                                    trust_remote_code=True
                                )
                        else:
                            # Try to load config with the actual model type
                            config = AutoConfig.from_pretrained(
                                model_name,
                                trust_remote_code=True,
                                local_files_only=local_only,
                            )
                    else:
                        # If config.json doesn't exist, try to create config based on known architecture
                        logger.info("Config.json not found, assuming DeepSeekV3-like architecture")
                        from transformers import PretrainedConfig
                        config = PretrainedConfig(
                            model_type="deepseek",
                            architectures=["AutoModelForCausalLM"],
                            vocab_size=102400,
                            hidden_size=4096,
                            num_hidden_layers=28,
                            num_attention_heads=32,
                            intermediate_size=11008,
                            max_position_embeddings=32768,
                            rms_norm_eps=1e-06,
                            rope_theta=10000.0,
                            trust_remote_code=True,
                        )
                except Exception:
                    # Final fallback
                    try:
                        from transformers import Glm4MoeConfig
                        config = Glm4MoeConfig(vocab_size=151552, hidden_size=4096, num_layers=40)
                    except:
                        from transformers import PretrainedConfig
                        config = PretrainedConfig(
                            model_type="glm4_moe",
                            architectures=["Glm4MoeForCausalLM"],
                            vocab_size=151552,
                            hidden_size=4096,
                            num_hidden_layers=40,
                            trust_remote_code=True,
                        )
            else:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    local_files_only=local_only,
                    quantization_config=final_quant_config,
                )
            except TypeError as inner_dtype_error:
                if "unexpected keyword argument" in str(inner_dtype_error):
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        config=config,
                        device_map="auto",
                        trust_remote_code=True,
                        local_files_only=local_only,
                        quantization_config=final_quant_config,
                    )
                else:
                    raise
        else:
            raise
    
    # Re-apply patches for the new model instance
    # 1. Ensure model is in MODEL_ATTRS (auto-register if needed)
    if not ensure_model_registered(model):
        logger.warning(f"Could not auto-register MODEL_ATTRS for {model_class_name}, pruning may fail")
    
    # 2. Ensure model has observer config (auto-register if needed)
    if not ensure_observer_config(model):
        logger.warning(f"Could not auto-register observer config for {model_class_name}, pruning may fail")
    
    # 3. Auto-patch MoE blocks for router_logits if needed
    if needs_patching(model):
        logger.info(f"Auto-patching MoE blocks for {model_class_name} to expose router_logits")
        patched_count = patch_specific_model(model, model_class_name)
        if patched_count == 0:
            # Try generic patcher
            patched_count = auto_patch_moe(model)
        if patched_count == 0:
            logger.warning(f"Could not patch any MoE blocks for {model_class_name}, pruning may fail")
    
    return model


def main():
    parser = HfArgumentParser(
        (
            ReapArgs,
            DatasetArgs,
            ObserverArgs,
            ModelArgs,
            EvalArgs,
            PruneArgs,
            ClusterArgs,
        )
    )
    reap_args, ds_args, obs_args, model_args, eval_args, prune_args, cluster_args = (
        parser.parse_args_into_dataclasses()
    )
    if prune_args.perserve_super_experts and prune_args.perserve_outliers:
        raise ValueError("Only one of perserve_super_experts or perserve_outliers can be set to True.")
    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    # get local patched model if req'd
    model_name = patched_model_map(model_args.model_name)
    # Apply compat patches BEFORE importing any trust_remote_code modules
    apply_transformers_compat_patches()
    
    # --- FIX 1: Handle tokenizer loading for models with custom configs ---
    # Some models (like MiniMax-M2.1-PRISM, LongCat) use custom config classes that aren't in tokenizer mapping
    # LongCat specifically has tokenizer_class: "PreTrainedTokenizer" which is a base class
    # and can't be auto-loaded - need to use PreTrainedTokenizerFast for tokenizer.json
    try:
        # Check if this is a model with base class tokenizer_class that needs special handling
        if "longcat" in model_name.lower():
            logger.info("LongCat model detected - using PreTrainedTokenizerFast to load tokenizer.json")
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except (KeyError, ValueError) as e:
        logger.warning(f"Standard tokenizer loading failed: {e}")
        logger.info("Attempting fallback tokenizer loading with trust_remote_code and custom handling...")
        # Try with additional trust_remote_code and fallback options
        try:
            from transformers import AutoConfig
            # Get config to understand the model type
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            logger.info(f"Model config type: {config.__class__.__name__}")
            
            # Special handling for BitConfig (MiniMax-M2.1-PRISM)
            if config.__class__.__name__ == "BitConfig":
                logger.info("BitConfig detected - using generic tokenizer fallback")
                # Try loading with slow tokenizer only (not fast)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    use_fast=False,
                )
            # Special handling for LongCat models
            elif config.__class__.__name__ == "LongcatConfig" or "longcat" in model_name.lower():
                logger.info("LongCat model detected - using PreTrainedTokenizerFast fallback")
                # LongCat has tokenizer.json but no explicit tokenizer class
                # Try loading with PreTrainedTokenizerFast directly
                try:
                    from transformers import PreTrainedTokenizerFast
                    tokenizer = PreTrainedTokenizerFast.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                    )
                except Exception as longcat_tok_err:
                    logger.warning(f"PreTrainedTokenizerFast failed: {longcat_tok_err}")
                    # Try with LlamaTokenizerFast as LongCat may use similar tokenizer
                    from transformers import LlamaTokenizerFast
                    tokenizer = LlamaTokenizerFast.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                    )
            else:
                # Try loading tokenizer with explicit trust_remote_code
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    # Add fallback for models with custom tokenizers
                    use_fast=False,
                )
        except Exception as tokenizer_error:
            logger.error(f"Fallback tokenizer loading also failed: {tokenizer_error}")
            # Final fallback: try to load with just the path
            try:
                logger.info("Attempting final fallback with PreTrainedTokenizerFast")
                from transformers import PreTrainedTokenizerFast
                tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
            except Exception:
                # Last resort: Create a basic tokenizer 
                logger.info("Creating basic fallback tokenizer")
                try:
                    # Use the slow tokenizer instead of trying to build a fast one
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        use_fast=False
                    )
                except Exception:
                    logger.warning("Creating basic fallback tokenizer with standard parameters")
                    from transformers import GPT2TokenizerFast  # Using GPT2 as a generic fallback
                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                    # Update special tokens if possible
                    if hasattr(tokenizer, 'add_special_tokens'):
                        tokenizer.add_special_tokens({
                            'pad_token': '<pad>',
                            'unk_token': '<unk>',
                            'bos_token': '<s>',
                            'eos_token': '</s>',
                        })
                    logger.warning("Using generic GPT2 tokenizer as final fallback - may not be fully compatible")
    
    # load model
    local_only = _env_flag("REAP_LOCAL_FILES_ONLY", True)
    quantization_config = None
    
    # --- FIX 2: Detect pre-quantized models and skip BitsAndBytesConfig ---
    # Check if model is already quantized (e.g., Kimi-K2-Thinking with CompressedTensorsConfig)
    is_pre_quantized = False
    pre_quant_config = None
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_only)
        if hasattr(config, 'quantization_config') and config.quantization_config:
            is_pre_quantized = True
            pre_quant_config = config.quantization_config
            logger.info(f"Model appears to be pre-quantized: {pre_quant_config}")
            logger.info("Will skip BitsAndBytesConfig to avoid conflicts")
    except Exception as config_error:
        logger.warning(f"Could not check for pre-quantization: {config_error}")
    
    if obs_args.load_in_4bit and not is_pre_quantized:
        logger.info("Loading model in 4-bit quantization to reduce VRAM during expert analysis.")
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as e:
            logger.warning(
                "4-bit requested but quantization config could not be created (likely missing bitsandbytes). "
                "Falling back to non-quantized load. Error: %s",
                e,
            )
            quantization_config = None
    elif is_pre_quantized:
        logger.info("Model is pre-quantized, will load with native quantization (no extra config needed)")
        # For pre-quantized models (FP8, etc.), don't pass quantization_config
        # Let transformers handle it automatically from the model's config
        quantization_config = None
    
    # --- FIX 3: Handle model loading with fallback for config detection issues ---
    # Suppress transformers config logging to avoid serialization bug in some models
    import logging as std_logging
    transformers_config_logger = std_logging.getLogger("transformers.configuration_utils")
    old_config_level = transformers_config_logger.level
    transformers_config_logger.setLevel(std_logging.ERROR)
    
    try:
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": local_only,
        }
        
        # For pre-quantized FP8 models, don't pass any quantization config or dtype
        # Let the model's native loading handle everything
        if not is_pre_quantized:
            load_kwargs["torch_dtype"] = "auto"
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except TypeError as dtype_error:
        # Some models with custom code don't accept torch_dtype parameter
        if "unexpected keyword argument" in str(dtype_error):
            logger.warning(f"Model doesn't accept torch_dtype parameter: {dtype_error}")
            logger.info("Retrying without torch_dtype...")
            load_kwargs.pop("torch_dtype", None)
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        else:
            raise
    except ValueError as model_type_error:
        logger.warning(f"Model type not recognized: {model_type_error}")
        logger.info("Retrying with explicit config loading...")
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=local_only,
            )
        except Exception as config_error:
            logger.warning(f"AutoConfig failed for model type: {config_error}")
            from transformers import PretrainedConfig
            config = PretrainedConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=local_only,
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_only,
            quantization_config=quantization_config,
        )
    except AttributeError as attr_error:
        # Handle bug in transformers where quantization_config.to_dict() is called on None
        if "NoneType" in str(attr_error) and "to_dict" in str(attr_error):
            logger.warning(f"Transformers config serialization bug detected: {attr_error}")
            logger.info("Retrying with minimal parameters...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=local_only,
            )
        else:
            raise
    except RuntimeError as runtime_error:
        # Handle FP8 weight loading issues for pre-quantized models
        if "size mismatch" in str(runtime_error) and "weight_scale" in str(runtime_error):
            logger.warning(f"FP8 weight loading issue detected: {runtime_error}")
            logger.info("This model appears to be pre-quantized for vLLM. Attempting vLLM-based loading...")
            
            # Try vLLM-based loading for FP8 models
            model = None
            try:
                model = load_model_with_vllm(model_name, local_only)
                if model is not None:
                    logger.info("Successfully loaded model via vLLM extraction")
            except Exception as vllm_error:
                logger.warning(f"vLLM loading failed: {vllm_error}")
            
            if model is None:
                logger.info("Trying standard loading without device_map...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        local_files_only=local_only,
                        low_cpu_mem_usage=False,
                    )
                    model = model.to("cuda")
                except Exception as fallback_error:
                    logger.error(f"All loading methods failed.")
                    logger.error(f"vLLM error: {vllm_error if 'vllm_error' in dir() else 'N/A'}")
                    logger.error(f"Standard error: {fallback_error}")
                    raise RuntimeError(
                        f"Cannot load FP8 pre-quantized model '{model_name}'. "
                        f"This model requires vLLM for inference. Install vLLM with: pip install vllm\n"
                        f"Original error: {runtime_error}"
                    )
        else:
            raise
    except OSError as model_load_error:
        # Handle cases like PrimeIntellect/INTELLECT-3 where config detection fails
        if "does not appear to have a file named configuration" in str(model_load_error):
            logger.warning(f"Config detection failed: {model_load_error}")
            logger.info("Attempting to load with explicit config class handling...")
            
            # Special handling for PrimeIntellect/INTELLECT-3 (GLM4 misdetection fix)
            if model_name == "PrimeIntellect/INTELLECT-3":
                logger.info("PrimeIntellect/INTELLECT-3 detected - overriding GLM4 config misdetection")
                # Try to get the actual model config type
                from transformers import AutoConfig
                try:
                    # First, check if any config files exist in the model repo manually
                    import os
                    from huggingface_hub import snapshot_download
                    # Download config.json only temporarily to inspect it
                    temp_dir = snapshot_download(model_name, allow_patterns=["config.json"], local_files_only=local_only)
                    
                    # Try to read the config.json directly to get the model type
                    config_path = os.path.join(temp_dir, "config.json")
                    if os.path.exists(config_path):
                        import json
                        with open(config_path, 'r') as f:
                            config_dict = json.load(f)
                        # Get the model type from the config
                        model_type = config_dict.get("model_type", "deepseek_v3")
                        logger.info(f"Detected model type from config.json: {model_type}")
                        
                        if model_type in ("glm4_moe", "glm4_moe_lite"):
                            # Use transformers built-in GLM4 config if available
                            try:
                                if model_type == "glm4_moe_lite":
                                    from transformers import Glm4MoeLiteConfig
                                    logger.info("Using built-in Glm4MoeLiteConfig")
                                    config = Glm4MoeLiteConfig.from_pretrained(model_name, trust_remote_code=True)
                                else:
                                    from transformers import Glm4MoeConfig
                                    logger.info("Using built-in Glm4MoeConfig")
                                    config = Glm4MoeConfig.from_pretrained(model_name, trust_remote_code=True)
                                
                                # Instantiating model directly to bypass AutoModel's remote code check
                                from transformers import Glm4MoeForCausalLM
                                logger.info("Instantiating Glm4MoeForCausalLM directly to bypass missing remote code")
                                model = Glm4MoeForCausalLM.from_pretrained(
                                    model_name,
                                    config=config,
                                    device_map="auto",
                                    trust_remote_code=True,
                                    local_files_only=local_only,
                                    quantization_config=quantization_config,
                                )
                                # Skip the AutoModelForCausalLM call below
                                return model 
                            except ImportError:
                                logger.warning("GLM4 config/Glm4MoeForCausalLM not found in transformers, trying fallback")
                                # Fallback to generic config but with correct model_type
                                from transformers import PretrainedConfig
                                config = PretrainedConfig(
                                    model_type=model_type,
                                    trust_remote_code=True
                                )
                        else:
                            # Try to load config with the actual model type
                            # First try loading with trust_remote_code to get the real config class
                            config = AutoConfig.from_pretrained(
                                model_name,
                                trust_remote_code=True,
                                local_files_only=local_only,
                            )
                    else:
                        # If config.json doesn't exist, try to create config based on known architecture
                        logger.info("Config.json not found, assuming DeepSeekV3-like architecture")
                        from transformers import PretrainedConfig
                        
                        # Define a minimal DeepSeekV3-like config
                        config_attrs = {
                            "model_type": "deepseek_v3",
                            "architectures": ["DeepseekV3ForCausalLM"],
                            "vocab_size": 102400,  # Common for DeepSeek models
                            "hidden_size": 4096,  # Default, will be overridden if config loads
                            "num_hidden_layers": 28,
                            "num_attention_heads": 32,
                            "num_key_value_heads": 32,
                            "intermediate_size": 11008,
                            "max_position_embeddings": 32768,
                            "rms_norm_eps": 1e-06,
                            "rope_theta": 10000.0,
                            "attn_implementation": "flash_attention_2",
                            "trust_remote_code": True,
                        }
                        from transformers import DeepseekV2Config  # Use similar base config
                        try:
                            config = DeepseekV2Config(**config_attrs)
                        except:
                            # If DeepseekV2Config doesn't exist, use generic config
                            config = PretrainedConfig(**config_attrs)
                except Exception as deep_error:
                    logger.warning(f"Deep approach failed: {deep_error}")
                    logger.info("Using minimal fallback config for PrimeIntellect/INTELLECT-3")
                    # Create a basic config that's compatible with AutoModelForCausalLM
                    try:
                        from transformers import Glm4MoeConfig
                        config = Glm4MoeConfig(vocab_size=151552, hidden_size=4096, num_layers=40)
                    except:
                        from transformers import PretrainedConfig
                        config = PretrainedConfig(
                            model_type="glm4_moe",  # Use the detected type
                            architectures=["Glm4MoeForCausalLM"],  
                            vocab_size=151552,
                            hidden_size=4096,
                            num_hidden_layers=40,
                            trust_remote_code=True,
                        )
            else:
                # Try loading with explicit config class from transformers
                from transformers import AutoConfig
                try:
                    # Force load config first
                    config = AutoConfig.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        local_files_only=local_only,
                    )
                    logger.info(f"Loaded config: {config.__class__.__name__}")
                except Exception as config_error:
                    logger.warning(f"Could not load config normally: {config_error}")
                    logger.info("Using generic config fallback")
                    from transformers import PretrainedConfig
                    config = PretrainedConfig(
                        trust_remote_code=True,
                    )
            
            # Now try loading model with the config
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=local_only,
                quantization_config=quantization_config,
            )
        else:
            raise
    finally:
        transformers_config_logger.setLevel(old_config_level)
    
    # --- AUTO-DETECTION AND PATCHING FOR UNSUPPORTED MODELS ---
    model_class_name = model.__class__.__name__
    
    # 1. Ensure model is in MODEL_ATTRS (auto-register if needed)
    if not ensure_model_registered(model):
        logger.warning(f"Could not auto-register MODEL_ATTRS for {model_class_name}, pruning may fail")
    
    # 2. Ensure model has observer config (auto-register if needed)
    if not ensure_observer_config(model):
        logger.warning(f"Could not auto-register observer config for {model_class_name}, pruning may fail")
    
    # 3. Auto-patch MoE blocks for router_logits if needed
    if needs_patching(model):
        logger.info(f"Auto-patching MoE blocks for {model_class_name} to expose router_logits")
        patched_count = patch_specific_model(model, model_class_name)
        if patched_count == 0:
            # Try generic patcher
            patched_count = auto_patch_moe(model)
        if patched_count == 0:
            logger.warning(f"Could not patch any MoE blocks for {model_class_name}, observer may fail")
    
    # record activations or load previously recorded activations
    logger.info(
        f"Running observer to collect activation data for model {model_args.model_name} on dataset {ds_args.dataset_name}."
    )
    observer_data = record_activations(
        model,
        tokenizer,
        reap_args,
        model_args,
        ds_args,
        obs_args,
        results_dir,
    )
    if reap_args.run_observer_only:
        logger.info(
            "Observer run completed. Exiting after collecting activation data since "
            "`run_observer_only` is set to True."
        )
        return

    # pruning
    logger.info("Start of pruning")
    total_experts = len(
        observer_data[next(iter(observer_data))]["expert_frequency"]
    )

    compression_ratios = parse_compression_ratios(cluster_args.compression_ratio)
    if prune_args.prune_method == "under_average":
        n_experts_to_prune_list = [None]
        logger.info(
            "Using under_average pruning: experts below average activation count will be pruned per layer"
        )
    else:
        if prune_args.n_experts_to_prune is not None:
            if compression_ratios and len(compression_ratios) > 1:
                raise ValueError(
                    "Multiple compression ratios cannot be used with n_experts_to_prune."
                )
            n_experts_to_prune_list = [prune_args.n_experts_to_prune]
        else:
            if compression_ratios is None:
                raise ValueError(
                    "Either n_experts_to_prune or compression_ratio must be set for pruning."
                )
            if len(compression_ratios) > 1 and prune_args.prune_method != "reap":
                raise ValueError(
                    "Multiple compression ratios are only supported when prune_method is 'reap'."
                )
            n_experts_to_prune_list = [
                int(total_experts * ratio) for ratio in compression_ratios
            ]
            for ratio, n_to_prune in zip(compression_ratios, n_experts_to_prune_list):
                logger.info(
                    f"Calculated n_experts to prune: {n_to_prune} from compression_ratio: {ratio}"
                )

    # Extract category-expert map from observer data if available
    category_expert_map = None
    if "__dominant_category_per_expert__" in observer_data:
        category_expert_map = observer_data.pop("__dominant_category_per_expert__")
        # Also remove the raw category frequency data as it's not needed for pruning
        observer_data.pop("__category_expert_frequency__", None)

    pruned_model_dirs = []
    multi_ratio = len(n_experts_to_prune_list) > 1

    def _load_model_for_pruning(force_reload: bool) -> Any:
        nonlocal model
        if force_reload or obs_args.load_in_4bit:
            if model is not None:
                remove_hook_from_module(model, recurse=True)
                model.to("cpu")
                del model
                torch.cuda.empty_cache()
                gc.collect()
            model = reload_model_in_full_precision(
                model_name,
                tokenizer,
                local_only,
                model_class_name,
            )
        return model

    for n_experts_to_prune in n_experts_to_prune_list:
        pruned_model_dir = get_pruned_model_dir(
            results_dir, n_experts_to_prune, total_experts, prune_args, reap_args.seed, obs_args.renormalize_router_weights
        )
        if (
            pruned_model_dir.exists()
            and list(pruned_model_dir.glob("*.safetensors"))
            and not prune_args.overwrite_pruned_model
        ):
            logger.info(
                f"Pruned model directory {pruned_model_dir} already exists and contains pruned model files. "
                "Skipping pruning step."
            )
            pruned_model_dirs.append(pruned_model_dir)
            continue

        if obs_args.load_in_4bit:
            logger.info("4-bit quantization was used for observer analysis.")
        if obs_args.load_in_4bit or multi_ratio:
            logger.info("Reloading model in full precision (bfloat16) for pruning...")

        model = _load_model_for_pruning(force_reload=multi_ratio)

        if prune_args.prune_method == "under_average":
            logger.info("Pruning model with under_average per-layer thresholds...")
        else:
            logger.info(f"Pruning model to {total_experts - n_experts_to_prune} experts...")

        # --- FIX: Ensure model registration and observer config before pruning ---
        # This handles cases where reload_model_in_full_precision wasn't called
        # or if the model state needs refreshing
        model_class_name = model.__class__.__name__
        ensure_model_registered(model)
        ensure_observer_config(model)

        prune(
            observer_data,
            model,
            tokenizer,
            reap_args,
            prune_args,
            n_experts_to_prune,
            pruned_model_dir,
            category_expert_map=category_expert_map,
        )
        logger.info("pruning completed.")

        # smoke test
        if reap_args.smoke_test:
            logger.info("Running smoke test on the merged model...")
            try:
                smoke_test(model, tokenizer)
            except Exception as e:
                logger.error(f"Smoke test failed: {e}")
                pass

        tokenizer.save_pretrained(pruned_model_dir)
        if model_name == "artifacts/models/GLM-4.5-Air":
            # move modelling file
            source_file = pathlib.Path(model_name) / "modeling_glm4_moe.py"
            target_file = pruned_model_dir / "modeling_glm4_moe.py"
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied modeling_glm4_moe.py to {pruned_model_dir}")
            else:
                raise RuntimeError(
                    f"Source file {source_file} does not exist. Cannot copy to {target_file}."
                )

        logger.info("Pruning completed.")

        dump_args_to_yaml(
            pruned_model_dir,
            reap_args,
            ds_args,
            obs_args,
            model_args,
            eval_args,
            prune_args,
            cluster_args,
        )
        pruned_model_dirs.append(pruned_model_dir)

    # eval
    if reap_args.do_eval and pruned_model_dirs:
        from reap.eval import run_evaluate
        if model is not None:
            remove_hook_from_module(model, recurse=True)
            model.to("cpu")
            del model
        del observer_data
        torch.cuda.empty_cache()
        gc.collect()
        for pruned_model_dir in pruned_model_dirs:
            model_args.model_name = pruned_model_dir
            run_evaluate(model_args, pruned_model_dir / "eval", eval_args, reap_args.seed)


if __name__ == "__main__":
    main()
