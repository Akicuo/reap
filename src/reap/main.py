from __future__ import annotations
import time
import pickle
import logging
import dataclasses
import pathlib
import re
import time
from typing import Any
import gc
import yaml
import shutil
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, BitsAndBytesConfig

from accelerate.utils import set_seed
from accelerate.hooks import remove_hook_from_module

from reap.transformers_compat import apply_transformers_compat_patches

from reap.args import (
    ReapArgs,
    ModelArgs,
    DatasetArgs,
    ObserverArgs,
    ClusterArgs,
    KdArgs,
    EvalArgs,
    MergeArgs,
)
from reap.merge import MergeMethod, MoEExpertMerger
from reap.data import DATASET_REGISTRY
from reap.observer import OBSERVER_CONFIG_REGISTRY, MoETransformerObserver, ensure_observer_config
from reap.cluster import (
    get_penalty_vector,
    hierarchical_clustering,
    dynamic_frequency_penalized_clustering,
    multi_layer_hierarchical_clustering,
    mc_smoe_clustering,
    multi_layer_kmeans_clustering,
    multi_layer_kmeans_clustering_on_ca,
    restricted_hierarchical_clustering,
    kmeans_clustering
)
from reap.model_util import get_moe, assert_merge, MODEL_ATTRS, patched_model_map, get_super_expert_indices, ensure_model_registered
from reap.models.auto_patch import auto_patch_moe, patch_specific_model, needs_patching
from reap.eval import run_evaluate
from reap.cluster_plots import plot_cluster_analysis
from reap.metrics import get_distance_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> tuple[dataclasses.Dataclass]:
    parser = HfArgumentParser(
        (
            ReapArgs,
            ModelArgs,
            DatasetArgs,
            ObserverArgs,
            ClusterArgs,
            KdArgs,
            EvalArgs,
            MergeArgs,
        )
    )
    args = parser.parse_args_into_dataclasses()
    return args


def str_to_directory_name(s: str) -> str:
    """Convert a string to a valid directory name by replacing special characters."""
    return re.sub(r"[^\w\-_.]", "_", s)


def create_results_directory(model_name: str, dataset_name: str) -> pathlib.Path:
    """Create a clean directory name from model and dataset names."""
    model_clean = model_name.split("/")[-1]
    dataset_clean = dataset_name.split("/")[-1]

    # Create clean directory name by removing special characters
    model_clean = str_to_directory_name(model_clean)
    dataset_clean = str_to_directory_name(dataset_clean)

    results_dir = pathlib.Path("./artifacts") / model_clean / dataset_clean

    if results_dir.exists():
        logger.warning(f"Directory '{results_dir}' already exists")
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created artifacts directory: {results_dir}")

    return results_dir


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean-like environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def record_activations(
    model, tokenizer, reap_args, model_args, ds_args, obs_args, results_dir
):
    if ds_args.dataset_name == "combined":
        # just return the combined data
        cat_dir = results_dir / "all"
        f_name = cat_dir / obs_args.output_file_name
        if f_name.exists():
            return torch.load(f_name, weights_only=False)
        else:
            raise RuntimeError(
                f"Combined dataset requested but no pre-recorded data found at {f_name}"
            )
    try:
        if ds_args.dataset_name == "allenai/c4":
            file_url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
            c4_single_file_dataset = load_dataset(
                "json", data_files={"train": file_url}, split="train", streaming=False
            )
            raw_ds = c4_single_file_dataset
        else:
            raw_ds = load_dataset(ds_args.dataset_name, split=ds_args.split)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{ds_args.dataset_name}': {e}")

    # load dataset processor
    proc_cls = DATASET_REGISTRY.get(ds_args.dataset_name)
    if proc_cls is None:
        raise ValueError(
            f"No DatasetProcessor registered for '{ds_args.dataset_name}'. "
            f"Supported: {list(DATASET_REGISTRY.keys())}"
        )

    # init processor & process dataset
    processor = proc_cls(
        dataset=raw_ds,
        tokenizer=tokenizer,
        max_input_len=obs_args.model_max_length,
        split=ds_args.split,
        split_by_category=obs_args.split_by_category,
        return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
        truncate=obs_args.truncate,
    )
    category_data_batches = processor.get_processed_dataset(
        samples_per_category=obs_args.samples_per_category,
    )
    logger.info(
        "Loaded and processed data for categories: %s",
        str(list(category_data_batches.keys())),
    )

    # load observer and hook model
    try:
        renormalize_router_weights = getattr(model.config, "norm_topk_prob", False) and obs_args.renormalize_router_weights
        if renormalize_router_weights:
            logger.info("Renormalizing topk router weights to sum to 1.")
        observer_config = OBSERVER_CONFIG_REGISTRY[model.__class__.__name__](
            # distance_measure=obs_args.distance_measure,
            distance_measure='cosine',
            renormalize_router_weights=renormalize_router_weights,
            record_pruning_metrics_only=obs_args.record_pruning_metrics_only,
        )
    except KeyError:
        raise ValueError(
            f"No observer configuration registered for model '{model.__class__.__name__}'. "
            f"Supported: {list(OBSERVER_CONFIG_REGISTRY.keys())}"
        )
    observer = MoETransformerObserver(
        model=model,
        hook_config=observer_config,
    )

    if reap_args.profile:
        # profile at max len
        with torch.no_grad():
            try:
                model_max_length = obs_args.model_max_length
                if model_max_length is None:
                    model_max_length = tokenizer.model_max_length
                logger.info(f"Profiling at model max length: {model_max_length}.")
                s = "hello " * model_max_length
                tokenized = tokenizer(
                    [s],
                    return_tensors="pt",
                    truncation=True,
                    max_length=model_max_length,
                )
                tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
                for _ in range(2):
                    _ = model(**tokenized)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to run model with max input length {model_max_length}: {e}"
                )
        logger.info(
            f"Model {model_args.model_name} successfully loaded and profiled at max length {model_max_length}."
        )
        observer.reset()

    # run samples over model and save observer state
    with torch.no_grad():
        for category, cat_data in category_data_batches.items():
            logger.info(f"Processing category: {category}...")
            cat_dir = results_dir / str_to_directory_name(category)
            cat_dir.mkdir(parents=True, exist_ok=True)
            f_name = cat_dir / obs_args.output_file_name
            if f_name.exists() and not obs_args.overwrite_observations:
                logger.info(
                    f"Category '{category}' previously processed. Skipping to next category..."
                )
                continue
            try:
                logger.info("No previous data found @ %s", f_name)
                for sample in tqdm(cat_data, desc=f"Processing {category} samples"):
                    model(sample.to(model.device))
            except Exception as e:
                logger.error(f"Error processing category '{category}'")
                logger.info(
                    f"Saving partial results for category '{category}' and exiting"
                )
                observer.save_state(cat_dir / "partial.pkl")
                logger.info(
                    f"{category} data processed and saved to "
                    f"{cat_dir / obs_args.output_file_name}"
                )
                raise e
            observer.save_state(cat_dir / obs_args.output_file_name)
            observer.reset()
            logger.info(
                f"{category} data processed and saved to "
                f"{cat_dir / obs_args.output_file_name}"
            )
    observer.close_hooks()
    with open(f"{cat_dir / obs_args.output_file_name}", "rb") as f:
        observer_data = torch.load(f, weights_only=False)
    return observer_data


def cluster(
    data: dict[int, dict[str, Any]],
    num_clusters: int,
    cluster_args: ClusterArgs,
    distance_measure: str,
    results_dir: pathlib.Path,
) -> dict[int, torch.Tensor]:
    """Cluster the model's experts based on the specified clustering method."""
    logger.info(f"Clustering experts using settings:\n{cluster_args.__str__()}\n")

    cluster_labels = {}
    distances = {}
    all_layer_expert_proba = {}
    if cluster_args.singleton_super_experts or cluster_args.singleton_outlier_experts:
        super_expert_idx = get_super_expert_indices(data, include_last_layers=cluster_args.singleton_outlier_experts)
    for layer in tqdm(data, "Clustering experts..."):
        expert_prob = data[layer]["expert_frequency"] / data[layer]["total_tokens"]
        ttm_sim_matrix = None
        try:
            ttm_sim_matrix = data[layer]["ttm_similarity_matrix"]
        except KeyError:
            pass
        online_characteristic_activation_dist = None
        try:
            online_characteristic_activation_dist = data[layer][
                "online_characteristic_activation_dist"
            ]
        except KeyError:
            pass
        ca = data[layer]["characteristic_activation"]
        routed_ca = None
        try:
            routed_ca = data[layer]["routed_characteristic_activation"]
        except KeyError:
            pass
        router_logits = data[layer]["router_logit_similiarity"]

        expert_similarity_scores = {
            "ttm": ttm_sim_matrix,
            "dynamic_ttm": ttm_sim_matrix,
            "characteristic_activation": ca,
            "routed_characteristic_activation": routed_ca,
            "router_logits": router_logits,
            "online_characteristic_activation_dist": online_characteristic_activation_dist,
        }
        distance = expert_similarity_scores[cluster_args.expert_sim]

        if cluster_args.expert_sim in [
            "characteristic_activation",
            "routed_characteristic_activation",
            "router_logits",
        ] and cluster_args.cluster_method != "kmeans":
            # get NxN similarity matrix for vector metrics
            distance_fn = get_distance_fn(distance_measure)
            distance = distance_fn(distance.unsqueeze(0), distance.unsqueeze(1))

        
        if cluster_args.singleton_super_experts:
            # set super expert distance to max
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][:, 1]
            if len(super_experts_in_layer) > 0:
                max_value = torch.finfo(distance.dtype).max
                distance[:, super_experts_in_layer] = max_value
                distance[super_experts_in_layer, :] = max_value

        distances[layer] = distance
        all_layer_expert_proba[layer] = expert_prob
        if cluster_args.multi_layer or cluster_args.cluster_method == "mc_smoe":
            continue
        if cluster_args.frequency_penalty and cluster_args.expert_sim != "dynamic_ttm":
            penalty = get_penalty_vector(
                expert_prob,
                cluster_args.softmax_temperature,
            )
            penalty_matrix = penalty.unsqueeze(0) + penalty.unsqueeze(1)
            penalized_distance = distance * penalty_matrix
            penalized_distance[penalized_distance.isnan()] = float("inf")
            distance = penalized_distance

        if cluster_args.expert_sim == "dynamic_ttm":
            cluster_label = dynamic_frequency_penalized_clustering(
                distance,
                expert_prob,
                num_clusters,
                cluster_args.softmax_temperature,
            )

        elif cluster_args.cluster_method == "agglomerative":
            if (
                hasattr(cluster_args, "max_cluster_size")
                and cluster_args.max_cluster_size is None
            ):
                cluster_label = hierarchical_clustering(
                    distance,
                    cluster_args.linkage_method,
                    num_clusters,
                )
            else:
                cluster_label = restricted_hierarchical_clustering(
                    distance,
                    cluster_args.linkage_method,
                    num_clusters,
                    max_cluster_size=cluster_args.max_cluster_size,
                )
            if isinstance(cluster_label, np.ndarray):
                cluster_label = torch.tensor(cluster_label)

        elif cluster_args.cluster_method == "kmeans":
            cluster_label = kmeans_clustering(distance, num_clusters)

        else:
            raise NotImplementedError(
                f"Clustering method '{cluster_args.cluster_method}' is not implemented."
            )
        cluster_labels[layer] = cluster_label

    if cluster_args.multi_layer:
        # we have parsed distances, time to cluster across layers]
        logger.info(
            f"Multi layer clustering with multi_layer={cluster_args.multi_layer}"
        )
        if cluster_args.cluster_method == "agglomerative":
            cluster_labels = multi_layer_hierarchical_clustering(
                distances,
                cluster_args.multi_layer,
                cluster_args.linkage_method,
                num_clusters,
            )
        elif cluster_args.cluster_method == "kmeans": 
            # try v2:
            if cluster_args.expert_sim != 'characteristic_activation':
                raise ValueError("multi_layer kmeans clustering on ca only implemented for characteristic_activation expert sim")
            cluster_labels = multi_layer_kmeans_clustering_on_ca(
                distances,
                num_layers=cluster_args.multi_layer,
                n_clusters=num_clusters,
            )
            
            # cluster_labels = multi_layer_kmeans_clustering(
            #     distances,
            #     num_layers=cluster_args.multi_layer,
            #     n_clusters=num_clusters,
            # )

    if cluster_args.cluster_method == "mc_smoe":
        logger.info(f"Performing MC-SMoE adpative layer-wise merging...")
        cluster_labels = mc_smoe_clustering(
            distances,
            all_layer_expert_proba,
            total_clusters=len(distances) * num_clusters,
        )
    return cluster_labels


def merge(
    model: nn.Module,
    cluster_labels: dict[int, torch.Tensor],
    observer_data: dict[int, dict[str, Any]],
    merge_args: MergeArgs,
):
    """Merge experts based on the clustering results."""
    logger.info(f"Merging experts using method '{merge_args.merge_method}'")
    model_attrs = MODEL_ATTRS[model.__class__.__name__]

    try:
        merge_method = MergeMethod(merge_args.merge_method)
    except ValueError:
        raise NotImplementedError(
            f"Merge method '{merge_args.merge_method}' is not implemented. "
            f"Supported methods: {[method.value for method in MergeMethod]}"
        )

    for layer_idx, layer in enumerate(tqdm(cluster_labels, "Merging layers...")):
        if merge_args.skip_first and layer_idx == 0:
            logger.info(
                f"Skipping merging for layer {layer_idx} as per 'skip_first' argument."
            )
            continue

        if merge_args.skip_last and layer_idx == len(cluster_labels) - 1:
            logger.info(
                f"Skipping merging for layer {layer_idx} as per 'skip_last' argument."
            )
            continue

        expert_proba = (
            observer_data[layer]["expert_frequency"]
            / observer_data[layer]["total_tokens"]
        )
        cluster_label = cluster_labels[layer]
        moe = get_moe(model, layer)
        merger = MoEExpertMerger(
            moe=moe,
            cluster_label=cluster_label,
            expert_proba=expert_proba,
            model_attrs=model_attrs,
            merge_method=merge_method,
            dom_as_base=merge_args.dom_as_base,
            select_top_k=merge_args.select_top_k,
            permute=merge_args.permute,
            tie_tensors=merge_args.save_as_tied_params,
        )
        merger.merge_experts()
        # in case of non-uniform compression, update num_experts
        # TODO deal with router too
        # setattr(getattr(moe, model_attrs["num_experts"]), model_attrs["num_experts"], len(cluster_label.unique()))
        assert_merge(model, moe, cluster_label)


def save_merged_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    merged_model_dir: pathlib.Path,
    safe_serialization,
) -> pathlib.Path:
    logger.info("Saving merged model...")
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        model.save_pretrained(merged_model_dir, safe_serialization=safe_serialization)
        tokenizer.save_pretrained(merged_model_dir)
    except Exception as e:
        import pdb; breakpoint()
    end = time.time()
    logger.info(
        f"Merged model saved to {merged_model_dir} in {end - start:.2f} seconds"
    )
    return merged_model_dir


@torch.no_grad()
def smoke_test(model: nn.Module, tokenizer: AutoTokenizer):
    """Run a smoke test to ensure the model is functioning correctly."""
    prompt = "What is your name?"
    test_input = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        test_input,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
        # enable_thinking=False,
    ).to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=50,
        do_sample=True,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    logger.info("Smoke test response: %s", response[0])


def get_model_dir(
    results_dir, num_clusters, cluster_labels, cluster_args, obs_args, merge_args
) -> pathlib.Path:
    cluster_desc = cluster_args.cluster_description
    if not cluster_desc:
        cluster_desc = (
            f"{cluster_args.expert_sim}_{obs_args.distance_measure}_{num_clusters}_"
            f"{cluster_args.linkage_method}_freq-penalty-{cluster_args.frequency_penalty}"
            f"_softmax-{cluster_args.softmax_temperature}_multi_layer-{cluster_args.multi_layer}"
        )
        if cluster_args.max_cluster_size is not None:
            cluster_desc += f"_max_size-{cluster_args.max_cluster_size}"
    merge_model_subdir_name = merge_args.merged_model_dir_name
    
    if not merge_model_subdir_name:
        merge_model_subdir_name = f"{merge_args.merge_method}-permute_{merge_args.permute}-skip_first_{merge_args.skip_first}-skip_last_{merge_args.skip_last}-multilayer_{cluster_args.multi_layer}"

    # Check for non uniform compression
    non_uniform_cluster_labels = (
        len(
            torch.unique(
                torch.tensor(
                    [
                        len(torch.unique(clusters))
                        for clusters in cluster_labels.values()
                    ]
                )
            )
        )
        > 1
    )
    if (
        non_uniform_cluster_labels
        or cluster_args.multi_layer
        or merge_args.skip_first
        or merge_args.skip_last
    ):
        logger.info("Detected non-uniform compression across layers.")
        merge_model_parent_dir_name = "non_uniform_merged_models"
    else:
        merge_model_parent_dir_name = "merged_models"

    merged_model_dir = (
        results_dir
        / merge_model_parent_dir_name
        / merge_model_subdir_name
        / cluster_desc
    )
    return merged_model_dir


def dump_args_to_yaml(
    merged_model_dir: pathlib.Path,
    reap_args: ReapArgs,
    model_args: ModelArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    cluster_args: ClusterArgs,
    kd_args: KdArgs,
    eval_args: EvalArgs,
    merge_args: MergeArgs,
):
    """Dump all arguments to a YAML file."""
    all_args = {
        "reap_args": dataclasses.asdict(reap_args),
        "model_args": dataclasses.asdict(model_args),
        "ds_args": dataclasses.asdict(ds_args),
        "obs_args": dataclasses.asdict(obs_args),
        "cluster_args": dataclasses.asdict(cluster_args),
        "kd_args": dataclasses.asdict(kd_args),
        "eval_args": dataclasses.asdict(eval_args),
        "merge_args": dataclasses.asdict(merge_args),
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

    output_path = merged_model_dir / "reap_args.yaml"
    with open(output_path, "w") as f:
        yaml.dump(serializable_args, f, default_flow_style=False)
    logger.info(f"All arguments saved to {output_path}")


def reload_model_in_full_precision(
    model_name: str,
    tokenizer: AutoTokenizer,
    local_only: bool,
    model_class_name: str,
):
    """
    Reload the model in full precision (bfloat16) for merging.
    
    This is used after observer analysis with 4-bit quantization to ensure
    the merged model is saved in full precision, not 4-bit.
    """
    logger.info("Reloading model in full precision (bfloat16) for merging...")
    
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
    final_quant_config = None
    if is_pre_quantized:
        logger.info("Model is pre-quantized, using original quantization config for reload")
        final_quant_config = pre_quant_config
        # Convert dict to proper class if needed
        if isinstance(final_quant_config, dict):
            try:
                from transformers import CompressedTensorsConfig
                if 'format' in final_quant_config and final_quant_config['format'] == 'pack-quantized':
                    final_quant_config = CompressedTensorsConfig.from_dict(final_quant_config)
            except ImportError:
                logger.warning("CompressedTensorsConfig not available, using original config dict")
            except Exception:
                logger.warning("Could not convert quantization config to CompressedTensorsConfig, using original")
    else:
        # No additional quantization needed if not pre-quantized
        final_quant_config = None
    
    # For pre-quantized models, we still load without additional quantization
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,  # Force bfloat16
            trust_remote_code=True,
            local_files_only=local_only,
            quantization_config=final_quant_config,  # Use the correct config
        )
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
                        
                        if model_type == "glm4_moe":
                            # Use transformers built-in Glm4MoeConfig if available
                            try:
                                from transformers import Glm4MoeConfig
                                logger.info("Using built-in Glm4MoeConfig")
                                config = Glm4MoeConfig.from_pretrained(model_name, trust_remote_code=True)
                            except ImportError:
                                logger.warning("Glm4MoeConfig not found in transformers, trying fallback")
                                # Fallback to generic config but with correct model_type
                                from transformers import PretrainedConfig
                                config = PretrainedConfig(
                                    model_type="glm4_moe",
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
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                device_map="auto",
                dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=local_only,
                quantization_config=final_quant_config,  # Use the correct config
            )
        else:
            raise
    
    # Re-apply patches for the new model instance
    # 1. Ensure model is in MODEL_ATTRS (auto-register if needed)
    if not ensure_model_registered(model):
        logger.warning(f"Could not auto-register MODEL_ATTRS for {model_class_name}, merging may fail")
    
    # 2. Ensure model has observer config (auto-register if needed)
    if not ensure_observer_config(model):
        logger.warning(f"Could not auto-register observer config for {model_class_name}, merging may fail")
    
    # 3. Auto-patch MoE blocks for router_logits if needed
    if needs_patching(model):
        logger.info(f"Auto-patching MoE blocks for {model_class_name} to expose router_logits")
        patched_count = patch_specific_model(model, model_class_name)
        if patched_count == 0:
            # Try generic patcher
            patched_count = auto_patch_moe(model)
        if patched_count == 0:
            logger.warning(f"Could not patch any MoE blocks for {model_class_name}, merging may fail")
    
    return model


def main():
    (
        reap_args,
        model_args,
        ds_args,
        obs_args,
        cluster_args,
        kd_args,
        eval_args,
        merge_args,
    ) = parse_args()
    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    if cluster_args.singleton_super_experts and cluster_args.singleton_outlier_experts:
        raise ValueError(
            "Both 'singleton_super_experts' in clustering and 'perserve_super_experts' in merging cannot be set to True."
        )
    # get local patched model if req'd
    model_name = patched_model_map(model_args.model_name)
    # Apply compat patches BEFORE importing any trust_remote_code modules
    apply_transformers_compat_patches()
    
    # --- FIX 1: Handle tokenizer loading for models with custom configs ---
    # Some models (like MiniMax-M2.1-PRISM) use custom config classes that aren't in tokenizer mapping
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except KeyError as e:
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
                logger.info("Attempting final fallback with direct file loading")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=False,
                    # Skip slow-fast tokenizer conflicts by forcing slow tokenizer
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
    
    # Fix TRANSFORMERS_CACHE deprecation warning
    if os.getenv("TRANSFORMERS_CACHE") is not None:
        logger.warning("TRANSFORMERS_CACHE is deprecated - use HF_HOME instead. Unsetting TRANSFORMERS_CACHE and setting HF_HOME to /workspace/.cache/huggingface")
        os.unsetenv("TRANSFORMERS_CACHE")
        os.environ["HF_HOME"] = os.getenv("HF_HOME", "/workspace/.cache/huggingface")
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    
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
    
    quantization_config = None
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
        logger.info("Model is pre-quantized, using original quantization config")
        # Use the original quantization config directly to avoid class mismatch issues
        quantization_config = pre_quant_config
        # But we need to ensure we pass the correct class type
        # If quantization_config is a dict (like in Kimi-K2), try to convert it
        if isinstance(quantization_config, dict):
            try:
                # Try to import CompressedTensorsConfig if available
                from transformers import CompressedTensorsConfig
                if 'format' in quantization_config and quantization_config['format'] == 'pack-quantized':
                    quantization_config = CompressedTensorsConfig.from_dict(quantization_config)
            except ImportError:
                logger.warning("CompressedTensorsConfig not available, using original config dict")
            except Exception:
                logger.warning("Could not convert quantization config to CompressedTensorsConfig, using original")
    
    # --- FIX 3: Handle model loading with fallback for config detection issues ---
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
            trust_remote_code=True,
            local_files_only=local_only,
            quantization_config=quantization_config,
        )
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
                        
                        if model_type == "glm4_moe":
                            # Use transformers built-in Glm4MoeConfig if available
                            try:
                                from transformers import Glm4MoeConfig
                                logger.info("Using built-in Glm4MoeConfig")
                                config = Glm4MoeConfig.from_pretrained(model_name, trust_remote_code=True)
                            except ImportError:
                                logger.warning("Glm4MoeConfig not found in transformers, trying fallback")
                                # Fallback to generic config but with correct model_type
                                from transformers import PretrainedConfig
                                config = PretrainedConfig(
                                    model_type="glm4_moe",
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
                            model_type="glm4_moe",
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
    
    # --- AUTO-DETECTION AND PATCHING FOR UNSUPPORTED MODELS ---
    model_class_name = model.__class__.__name__
    
    # 1. Ensure model is in MODEL_ATTRS (auto-register if needed)
    if not ensure_model_registered(model):
        logger.warning(f"Could not auto-register MODEL_ATTRS for {model_class_name}, merging may fail")
    
    # 2. Ensure model has observer config (auto-register if needed)
    if not ensure_observer_config(model):
        logger.warning(f"Could not auto-register observer config for {model_class_name}, merging may fail")
    
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

    # clustering
    logger.info("Start of clustering")
    num_clusters = cluster_args.num_clusters
    if num_clusters is None:
        if cluster_args.compression_ratio is None:
            raise ValueError(
                "Either num_clusters or compression_ratio must be set for clustering."
            )
        else:
            # Calculate num_clusters from compression_ratio
            if not merge_args.skip_first and not merge_args.skip_last:
                total_experts = len(
                    observer_data[next(iter(observer_data))]["expert_frequency"]
                )
                num_clusters = int(total_experts * (1 - cluster_args.compression_ratio))
            else:
                # If skipping first or last layer, adjust total_experts accordingly
                experts_per_layer = len(
                    observer_data[next(iter(observer_data))]["expert_frequency"]
                )
                layers = len(observer_data)
                total_experts = layers * experts_per_layer
                total_clusters = int(
                    total_experts * (1 - cluster_args.compression_ratio)
                )
                total_layers = len(observer_data)
                if merge_args.skip_first:
                    total_layers -= 1
                if merge_args.skip_last:
                    total_layers -= 1
                num_clusters = int(total_clusters / total_layers)
            logger.info(
                f"Calculated num_clusters: {num_clusters} from compression_ratio: {cluster_args.compression_ratio}"
            )
    cluster_labels = cluster(
        observer_data,
        num_clusters,
        cluster_args,
        obs_args.distance_measure,
        results_dir,
    )
    logger.info("Clustering completed.")

    # merging
    logging.info("Start of merging")
    merged_model_dir = get_model_dir(
        results_dir,
        num_clusters,
        cluster_labels,
        cluster_args,
        obs_args,
        merge_args,
    )
    if (
        merged_model_dir.exists()
        and list(merged_model_dir.glob("*.safetensors"))
        and not merge_args.overwrite_merged_model
    ):
        logger.info(
            f"Merged model files already exist in {merged_model_dir}. Skipping merging."
        )
    else:
        # If 4-bit was used for observer, reload model in full precision for merging
        if obs_args.load_in_4bit:
            logger.info("4-bit quantization was used for observer analysis.")
            logger.info("Reloading model in full precision (bfloat16) for merging...")
            
            # Clean up the 4-bit model
            remove_hook_from_module(model, recurse=True)
            model.to("cpu")
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            # Reload in full precision
            model = reload_model_in_full_precision(
                model_name,
                tokenizer,
                local_only,
                model_class_name,
            )
        
        merge(
            model,
            cluster_labels,
            # num_clusters,
            observer_data,
            merge_args,
        )
        logger.info("Merging completed.")
        logger.info("Saving merged model...")
        merged_model_dir = save_merged_model(
            model,
            tokenizer,
            merged_model_dir,
            safe_serialization=True if not merge_args.save_as_tied_params else False,
        )
        logger.info(f"Merged model saved to {merged_model_dir}.")

        # save clustering results
        logger.info("Saving clustering results...")
        cluster_analysis_dir = merged_model_dir / "clusters"
        cluster_analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(cluster_analysis_dir / "clusters.pkl", "wb") as f:
            pickle.dump(cluster_labels, f)

        if reap_args.plot_clusters:
            logger.info("Plotting clusters analysis...")
            plot_cluster_analysis(
                cluster_labels,
                cluster_analysis_dir,
                merge_args.skip_first,
                merge_args.skip_last,
            )
        logger.info(
            f"Clustering results saved to {merged_model_dir / cluster_analysis_dir}"
        )

        # smoke test
        if reap_args.smoke_test:
            logger.info("Running smoke test on the merged model...")
            try:
                smoke_test(model, tokenizer)
            except Exception as e:
                logger.error(f"Smoke test failed: {e}")
                pass

        dump_args_to_yaml(
            merged_model_dir,
            reap_args,
            model_args,
            ds_args,
            obs_args,
            cluster_args,
            kd_args,
            eval_args,
            merge_args,
        )

        if model_name == "artifacts/models/GLM-4.5-Air":
            # move modelling file
            source_file = pathlib.Path(model_name) / "modeling_glm4_moe.py"
            target_file = merged_model_dir / "modeling_glm4_moe.py"
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied modeling_glm4_moe.py to {merged_model_dir}")
            else:
                raise RuntimeError(
                    f"Source file {source_file} does not exist. Cannot copy to {target_file}."
                )

    # eval
    if reap_args.do_eval:
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")
        del model
        del observer_data
        del cluster_labels
        torch.cuda.empty_cache()
        gc.collect()
        model_args.model_name = merged_model_dir
        run_evaluate(model_args, merged_model_dir / "eval", eval_args, reap_args.seed)


if __name__ == "__main__":
    main()
