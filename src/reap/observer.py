from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
import gc
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from dataclasses import dataclass
import logging
import pathlib
from functools import reduce

from reap.metrics import (
    ttm_online,
    get_routed_characteristic_activation,
    ca_dist_online,
    OnlineStatsTracker,
    get_distance_fn,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Expert Pruning Report Types ---

def _clean_category_name(category: str) -> str:
    """Convert dataset names to cleaner display names."""
    # Map of known dataset names to cleaner display names
    category_display_map = {
        "theblackcat102/evol-codealpaca-v1": "Coding",
        "Salesforce/xlam-function-calling-60k": "FunctionCalling",
        "SWE-bench/SWE-smith-trajectories": "SWE-Bench",
        "m-a-p/CodeFeedback-Filtered-Instruction": "CodeFeedback",
        "ise-uiuc/Magicoder-Evol-Instruct-110K": "Magicoder",
        "allenai/c4": "C4-General",
        "euclaise/WritingPrompts_curated": "Writing",
        "allenai/tulu-3-sft-personas-math": "Math",
    }
    
    if category in category_display_map:
        return category_display_map[category]
    
    # Try to extract a cleaner name from the path
    if "/" in category:
        # Take the part after the slash and clean it up
        name = category.split("/")[-1]
        # Remove common suffixes
        for suffix in ["-Instruct", "-v1", "-110K", "-60k", "-curated"]:
            name = name.replace(suffix, "")
        return name.replace("-", "_").replace("_", " ").title().replace(" ", "")
    
    return category


@dataclass
class ExpertPruningInfo:
    """Information about a single expert's pruning decision."""
    layer_idx: int
    expert_idx: int
    activation_count: int
    pruned: bool
    classification: str = "unknown"
    saliency_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "layer": self.layer_idx,
            "expert_idx": self.expert_idx,
            "activation_count": self.activation_count,
            "pruned": "Y" if self.pruned else "N",
            "classification": _clean_category_name(self.classification),
            "saliency_score": self.saliency_score,
        }


@dataclass 
class LayerPruningReport:
    """Pruning report for a single layer."""
    layer_idx: int
    experts: list[ExpertPruningInfo]
    total_tokens: int
    n_pruned: int
    n_retained: int
    
    def to_markdown(self) -> str:
        """Generate markdown table for this layer."""
        lines = [
            f"### Layer {self.layer_idx}",
            f"",
            f"**Total Tokens Processed:** {self.total_tokens:,}",
            f"**Experts Pruned:** {self.n_pruned} | **Retained:** {self.n_retained}",
            f"",
            "| Expert Index | Counts of Activation | Pruned? (Y or N) | Classification | Saliency Score |",
            "|--------------|---------------------|------------------|----------------|----------------|",
        ]
        for expert in sorted(self.experts, key=lambda e: e.expert_idx):
            clean_classification = _clean_category_name(expert.classification)
            lines.append(
                f"| {expert.expert_idx} | {expert.activation_count:,} | "
                f"{('Y' if expert.pruned else 'N')} | {clean_classification} | "
                f"{expert.saliency_score:.4f} |"
            )
        return "\n".join(lines)


@dataclass
class PruningReport:
    """Complete pruning report across all layers."""
    model_name: str
    prune_method: str
    compression_ratio: float
    total_experts_before: int
    total_experts_after: int
    layers: list[LayerPruningReport]
    category_expert_map: Optional[dict] = None
    
    def to_markdown(self) -> str:
        """Generate full markdown report."""
        lines = [
            "# Expert Pruning Report",
            "",
            f"**Model:** {self.model_name}",
            f"**Pruning Method:** {self.prune_method}",
            f"**Compression Ratio:** {self.compression_ratio:.2%}",
            f"**Total Experts Before:** {self.total_experts_before}",
            f"**Total Experts After:** {self.total_experts_after}",
            "",
            "---",
            "",
        ]
        
        for layer_report in self.layers:
            lines.append(layer_report.to_markdown())
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Summary statistics
        lines.extend([
            "## Summary Statistics",
            "",
            "| Layer | Pruned Count | Retained Count | Pruning Rate |",
            "|-------|--------------|----------------|--------------|",
        ])
        for layer_report in self.layers:
            total = layer_report.n_pruned + layer_report.n_retained
            rate = layer_report.n_pruned / total if total > 0 else 0
            lines.append(
                f"| {layer_report.layer_idx} | {layer_report.n_pruned} | "
                f"{layer_report.n_retained} | {rate:.2%} |"
            )
        
        return "\n".join(lines)
    
    def save(self, file_path: str | pathlib.Path):
        """Save report to markdown file."""
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
        logger.info(f"Pruning report saved to {file_path}")


class BaseTransformerObserverHookConfig:
    state_attr_name: str = "hook_state"
    hook_attr_name: str = "hooks"
    module_name_to_hook_regex: Optional[str] = None
    module_class_name_to_hook_regex: Optional[nn.Module] = None


class BaseTransformerObserver(ABC):
    def __init__(
        self,
        model,
        hook_config: Optional[BaseTransformerObserverHookConfig] = None,
    ):
        self.model = model
        self.hook_config = hook_config
        self.hooks = []
        self.state: dict[Any, Any] = {}
        self._hook_model()
        logger.info(
            "%s initialized for %s.",
            self.__class__.__name__,
            self.model.__class__.__name__,
        )

    @abstractmethod
    def _hook_factory(self, module: nn.Module, layer_number: int) -> callable:
        """
        Factory method to create a hook function for the given module.
        This method should be implemented by subclasses to define how the
        hook function should behave.
        """
        raise NotImplementedError("Subclasses must implement _hook_factory method.")

    def report_state(self) -> dict[str, Any]:
        """
        Method to report the current state of the observer. Can be overridden to inject
        custom behaviours.
        """
        return self.state

    def close_hooks(self):
        """Close all hooks registered to the model."""
        self.reset()  # Reset the state before closing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.debug("All hooks closed for %s.", self.model.__class__.__name__)

    def reset(self):
        """Reset the observer state."""
        del self.state
        gc.collect()
        self.state = {}
        logger.debug("Observer state reset for %s.", self.model.__class__.__name__)

    def save_state(self, file_path: str | pathlib.Path):
        self._move_state_tensors_to_cpu()
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.report_state()
        with open(file_path, "wb") as f:
            torch.save(state_dict, f)
        logger.info("State saved to %s", file_path)

    def _move_state_tensors_to_cpu(self):
        """
        Move all tensors in the state dictionary to CPU.
        This is useful before saving the state to avoid GPU memory issues.
        """
        for layer_number, layer_state in self.state.items():
            for key, value in layer_state.items():
                if isinstance(value, torch.Tensor):
                    self.state[layer_number][key] = value.cpu()

    def _validate_hook_config(self):
        if self.hook_config is None:
            return
        if (
            self.hook_config.module_name_to_hook_regex is None
            and self.hook_config.module_class_name_to_hook_regex is None
        ):
            raise ValueError(
                "At least one of 'module_n`ame_to_hook_regex' or "
                "'module_type_to_hook_regex' must be provided in the hook config."
            )
        if (
            self.hook_config.module_name_to_hook_regex is not None
            and self.hook_config.module_class_name_to_hook_regex is not None
        ):
            logger.warning(
                "Both 'module_name_to_hook_regex' and 'module_type_to_hook_regex' are "
                "provided. Both conditions must be satisfied to hook the module."
            )

    def _hook_model(self):
        for name, module in self.model.named_modules():
            hook_module = False
            if (
                self.hook_config.module_name_to_hook_regex
                and re.search(self.hook_config.module_name_to_hook_regex, name)
            ) or (
                self.hook_config.module_class_name_to_hook_regex
                and module.__class__.__name__
                == self.hook_config.module_class_name_to_hook_regex
            ):
                hook_module = True
            if hook_module:
                layer_number = int(re.search(r"\d+", name).group(0))
                hook_fn = self._hook_factory(module, layer_number)
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
                logger.info("Hooked module: %s at layer %d", name, layer_number)
        if len(self.hooks) == 0:
            raise ValueError(
                "No modules matched the provided hook configuration. "
                "Check your hook configuration settings."
            )

    @classmethod
    def _get_registry_for_cls(cls) -> dict[str, type[BaseTransformerObserver]]:
        """Helper to get the registry from the specific class 'cls'."""
        if not hasattr(cls, "_architecture_registry") or not isinstance(
            cls._architecture_registry, dict
        ):
            raise AttributeError(
                f"Class {cls.__name__} must define its own "
                "`_architecture_registry: dict[str, type] = {{}}` "
                f"to use the common registration/creation methods."
            )
        return cls._architecture_registry

    @classmethod
    def register_implementation(cls, *arch_names: str):
        """
        Class method decorator to register a concrete observer implementation.
        'cls' is the class on which this decorator's factory is called (e.g.,
        MoEExpertObserver) 'sub_cls' is the class being decorated
        (e.g., Llama4MoEExpertObserver).
        """

        def decorator(sub_cls: type[BaseTransformerObserver]):
            registry = cls._get_registry_for_cls()

            for name in arch_names:
                if name in registry:
                    raise RuntimeError(
                        f"Architecture {name} already registered with "
                        f"{registry[name].__name__} for {cls.__name__}."
                    )
                registry[name] = sub_cls
            return sub_cls

        return decorator

    @classmethod
    def create_from_registry(
        cls,
        model: nn.Module,
        hook_config: Optional[BaseTransformerObserverHookConfig] = None,
        return_rank_0_only: bool = True,
        **kwargs: Any,
    ) -> BaseTransformerObserver:
        registry = cls._get_registry_for_cls()
        model_cls_name = model.__class__.__name__

        specific_observer_cls = registry.get(model_cls_name)

        if specific_observer_cls:
            return specific_observer_cls(
                model,
                hook_config=hook_config,
                return_rank_0_only=return_rank_0_only,
                **kwargs,
            )
        else:
            raise ValueError(
                "Unsupported architecture for "
                f"{cls.__name__}: {model_cls_name}. "
                "Registered architectures in "
                f"{cls.__name__}._architecture_registry: "
                f"{list(registry.keys())}"
            )


# --- MoE Transformer Observer ---------------------------------------------------------


@dataclass
class MoETransformerObserverConfig(BaseTransformerObserverHookConfig):
    num_experts_attr_name: str = "num_experts"
    top_k_attr_name: str = "top_k"
    fused_experts: bool = False
    distance_measure: str = "angular"
    renormalize_router_weights: bool = False
    record_pruning_metrics_only: bool = False
    track_category_expert_frequency: bool = True  # Track which categories activate which experts


class MoETransformerObserver(BaseTransformerObserver):
    """MoE Transformer Observer for all methods including both pruning and merging."""
    
    def __init__(self, model, hook_config=None):
        self._current_category: Optional[str] = None
        self._category_expert_frequency: dict[str, dict[int, torch.Tensor]] = {}
        super().__init__(model, hook_config)
    
    def set_category(self, category: str):
        """Set the current category being processed for category-aware tracking."""
        self._current_category = category
        if category not in self._category_expert_frequency:
            self._category_expert_frequency[category] = {}
    
    def get_category_expert_frequency(self) -> dict[str, dict[int, torch.Tensor]]:
        """Get the per-category expert frequency mapping."""
        return self._category_expert_frequency
    
    def get_dominant_category_per_expert(self) -> dict[int, dict[int, str]]:
        """
        For each layer and expert, return the category that activated it the most.
        
        Returns:
            Dict mapping layer_idx -> expert_idx -> dominant_category
        """
        result: dict[int, dict[int, str]] = {}
        
        if not self._category_expert_frequency:
            return result
        
        # Get all layers
        all_layers = set()
        for cat_data in self._category_expert_frequency.values():
            all_layers.update(cat_data.keys())
        
        for layer_idx in sorted(all_layers):
            result[layer_idx] = {}
            
            # Get number of experts from any category that has this layer
            num_experts = 0
            for cat_data in self._category_expert_frequency.values():
                if layer_idx in cat_data:
                    num_experts = len(cat_data[layer_idx])
                    break
            
            for expert_idx in range(num_experts):
                max_count = 0
                dominant_cat = "unknown"
                
                for category, cat_data in self._category_expert_frequency.items():
                    if layer_idx in cat_data:
                        freq = cat_data[layer_idx]
                        if expert_idx < len(freq):
                            count = freq[expert_idx].item()
                            if count > max_count:
                                max_count = count
                                dominant_cat = category
                
                result[layer_idx][expert_idx] = dominant_cat
        
        return result

    def report_state(self) -> dict[str, Any]:
        """
        Method to report the current state of the observer. Can be overridden to inject
        custom behaviours.
        """
        result = {
            layer_num: {
                k: v.mean if isinstance(v, OnlineStatsTracker) else v
                for k, v in layer_state.items()
            }
            for layer_num, layer_state in self.state.items()
        }
        # Include category-expert frequency map if available
        if self._category_expert_frequency:
            result["__category_expert_frequency__"] = self._category_expert_frequency
            result["__dominant_category_per_expert__"] = self.get_dominant_category_per_expert()
        return result

    def _initialize_state(self, output: torch.Tensor, num_experts: int):
        # get device and shape info
        output_hidden_states = output[0]
        device = "cpu"
        hidden_dim = output_hidden_states.shape[-1]
        layer_state = {}

        # unnormalized states (counts)
        layer_state["total_tokens"] = torch.tensor(0, device=device, dtype=torch.long)
        layer_state["expert_frequency"] = torch.zeros(
            num_experts, device=device, dtype=torch.long
        )
        layer_state["pairwise_expert_frequency"] = torch.zeros(
            num_experts, num_experts, dtype=torch.long, device=device
        )

        if not self.hook_config.record_pruning_metrics_only:
            # per routed token normalized states
            layer_state["ttm_similarity_matrix"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=(num_experts, num_experts),
                device=device,
                dtype=torch.float32,
            )
            layer_state["routed_characteristic_activation"] = OnlineStatsTracker(
                shape=(num_experts, hidden_dim),
                count_shape=(num_experts, hidden_dim),
                device=device,
                dtype=torch.float32,
            )
            # HC-SMoE
            layer_state["characteristic_activation"] = OnlineStatsTracker(
                shape=(num_experts, hidden_dim),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            # SubMoE
            layer_state["online_characteristic_activation_dist"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            # per total token normalized states -> MC-SMoE
            layer_state["router_logit_similiarity"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )

        # Expert Activation Norm
        layer_state["ean_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )
        layer_state["weighted_ean_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )
        layer_state["ean_mean"] = OnlineStatsTracker(
            shape=(num_experts,),
            count_shape=(num_experts,),
            device=device,
            dtype=torch.float32,
        )
        layer_state["reap"] = OnlineStatsTracker(
            shape=(num_experts,),
            count_shape=(num_experts,),
            device=device,
            dtype=torch.float32,
        )

        # Weighted frequency
        layer_state["weighted_expert_frequency_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )

        # super experts
        layer_state["max_activations"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float32, requires_grad=False
        )

        return layer_state

    def _hook_factory(self, module: nn.Module, layer_number: int) -> callable:
        distance_fn = get_distance_fn("cosine") # always use cosine for online dist. metrics
        num_experts = reduce(
            getattr, self.hook_config.num_experts_attr_name.split("."), module
        )
        top_k = reduce(getattr, self.hook_config.top_k_attr_name.split("."), module)
        if num_experts is None or top_k is None:
            raise ValueError(
                f"Module {module.__class__.__name__} at layer {layer_number} "
                "does not have expected 'num_experts' or 'top_k' attributes. Check "
                "HookConfig settings."
            )

        @torch.no_grad()
        def _hook_fn(module, args, output):
            input = args[0]  # (batch_size, seq_len, hidden_dim)
            device = input.device
            batch_size, sequence_length, hidden_dim = input.shape
            flat_input = input.view(-1, hidden_dim)  # total_seq_len, hidden
            
            # --- ROUTER LOGITS RETRIEVAL WITH FALLBACK CHAIN ---
            router_logits = None
            
            # Fallback 1: Check for _last_router_logits attribute (auto-patched models)
            if hasattr(module, '_last_router_logits') and module._last_router_logits is not None:
                router_logits = module._last_router_logits
                # Clear after retrieval to avoid stale data
                module._last_router_logits = None
            
            # Fallback 2: Check if output is a tuple with router_logits
            elif isinstance(output, tuple) and len(output) >= 2:
                # Last element is typically router_logits
                router_logits = output[-1]
            
            # Fallback 3: Check for router_logits attribute on module
            elif hasattr(module, 'router_logits'):
                router_logits = module.router_logits
            
            # Fallback 4: Compute from gate/router weights if available
            elif hasattr(module, 'gate') and hasattr(module.gate, 'weight'):
                try:
                    gate_weight = module.gate.weight
                    router_logits = F.linear(
                        flat_input.to(gate_weight.dtype),
                        gate_weight
                    )
                except Exception:
                    pass
            elif hasattr(module, 'router') and hasattr(module.router, 'weight'):
                try:
                    router_weight = module.router.weight
                    router_logits = F.linear(
                        flat_input.to(router_weight.dtype),
                        router_weight
                    )
                except Exception:
                    pass
            # Fallback 4b: Handle LongCat-style router (router.classifier.weight pattern)
            elif hasattr(module, 'router') and hasattr(module.router, 'classifier'):
                try:
                    # LongcatTopkRouter uses router.classifier as the gate Linear layer
                    classifier_weight = module.router.classifier.weight
                    router_logits = F.linear(
                        flat_input.to(classifier_weight.dtype),
                        classifier_weight
                    )
                except Exception:
                    pass
            
            # Fallback 5: Create placeholder router_logits if all else fails
            if router_logits is None:
                logger.warning(
                    f"Could not retrieve router_logits for {module.__class__.__name__} at layer {layer_number}. "
                    "Creating placeholder. Some metrics may be inaccurate."
                )
                router_logits = torch.zeros(
                    flat_input.shape[0], num_experts,
                    device=device, dtype=flat_input.dtype
                )
            
            # Ensure router_logits is on the right device and has right shape
            if router_logits.device != device:
                router_logits = router_logits.to(device)
            
            # Handle models with zero/identity experts (like LongCat)
            # If router_logits has more columns than num_experts (e.g., 768 vs 512),
            # truncate to only track the real experts we can prune
            if router_logits.shape[-1] > num_experts:
                logger.debug(
                    f"Truncating router_logits from {router_logits.shape[-1]} to {num_experts} experts "
                    "(model may have zero/identity experts that can't be pruned)"
                )
                router_logits = router_logits[:, :num_experts]
            
            # Get selected experts from router_logits
            _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
            
            if layer_number not in self.state:
                self.state[layer_number] = self._initialize_state(output, num_experts)
            
            activations = torch.zeros((num_experts, *flat_input.shape), device=device)

            if self.hook_config.fused_experts:
                # Check if this is a grouped_mm style fused expert (like Glm4MoeLiteNaiveMoe)
                # These require routing indices and can't be called directly
                experts_module = module.experts
                is_grouped_mm = (
                    hasattr(experts_module, 'gate_up_proj') and 
                    hasattr(experts_module, 'down_proj') and
                    isinstance(experts_module.gate_up_proj, torch.Tensor)
                )
                
                if is_grouped_mm:
                    # Grouped_mm fused experts - compute activations manually per expert
                    # gate_up_proj shape: [num_experts, 2*intermediate, hidden_dim]
                    # down_proj shape: [num_experts, hidden_dim, intermediate]
                    gate_up_proj = experts_module.gate_up_proj  # [E, 2*I, H]
                    down_proj = experts_module.down_proj  # [E, H, I]
                    
                    intermediate_size = down_proj.shape[2]
                    
                    for expert_idx in range(num_experts):
                        # Get this expert's weights
                        expert_gate_up = gate_up_proj[expert_idx]  # [2*I, H]
                        expert_down = down_proj[expert_idx]  # [H, I]
                        
                        # Compute forward: gate_up = input @ gate_up.T
                        gate_up_out = F.linear(flat_input.to(expert_gate_up.dtype), expert_gate_up)  # [tokens, 2*I]
                        
                        # Split into gate and up
                        gate_out, up_out = gate_up_out.chunk(2, dim=-1)  # each [tokens, I]
                        
                        # SiLU activation on gate, multiply with up
                        hidden = F.silu(gate_out) * up_out  # [tokens, I]
                        
                        # Down projection
                        output = F.linear(hidden, expert_down)  # [tokens, H]
                        
                        activations[expert_idx] = output.to(device)
                else:
                    # Standard fused experts path (like Llama4)
                    router_indices = (
                        torch.arange(batch_size * sequence_length, device=device)
                        .view(1, -1)
                        .expand(num_experts, -1)
                    )
                    router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
                    routed_in = torch.gather(
                        input=flat_input,
                        dim=0,
                        index=router_indices,
                    ).to(device)
                    # record unweighted activations for all experts
                    routed_out = module.experts(routed_in)
                    activations = routed_out.view(num_experts, *flat_input.shape)

            else:  # loop based MoE execution
                for idx, expert in enumerate(module.experts):
                    activations[idx] = expert(flat_input).to(
                        device
                    )  # (num_experts, total_seq_len, hidden_dim)

            del flat_input
            num_tokens = batch_size * sequence_length
            num_tokens = torch.tensor(num_tokens, device="cpu", dtype=torch.long)

            # --- PRUNE/MERGE SALIENCY CRITERIA --------------------------------
            # expert frequency
            expert_frequency = torch.bincount(
                selected_experts.view(-1), minlength=num_experts
            ).to(device)
            pairwise_expert_frequency = expert_frequency.unsqueeze(
                0
            ) + expert_frequency.unsqueeze(1)
            pairwise_expert_frequency = pairwise_expert_frequency.to(device)

            self.state[layer_number]["total_tokens"] += num_tokens
            self.state[layer_number]["expert_frequency"] += expert_frequency.to(
                "cpu", torch.long
            )
            self.state[layer_number]["pairwise_expert_frequency"] += (
                pairwise_expert_frequency.to("cpu", torch.long)
            )
            
            # Track per-category expert frequency if category is set
            if (self.hook_config.track_category_expert_frequency and 
                self._current_category is not None):
                cat = self._current_category
                if cat not in self._category_expert_frequency:
                    self._category_expert_frequency[cat] = {}
                if layer_number not in self._category_expert_frequency[cat]:
                    self._category_expert_frequency[cat][layer_number] = torch.zeros(
                        num_experts, device="cpu", dtype=torch.long
                    )
                self._category_expert_frequency[cat][layer_number] += expert_frequency.to(
                    "cpu", torch.long
                )

            # Merging critera
            if not self.hook_config.record_pruning_metrics_only:
                ttm_similarity_matrix = ttm_online(
                    activations,
                    selected_experts,
                    distance_callable=distance_fn,
                    num_experts=num_experts,
                    pairwise_expert_frequency=pairwise_expert_frequency,
                )

                # ttm_similarity_matrix with pairwise frequency counts
                self.state[layer_number]["ttm_similarity_matrix"].update(
                    ttm_similarity_matrix, pairwise_expert_frequency
                )
                del ttm_similarity_matrix

                routed_characteristic_activation = get_routed_characteristic_activation(
                    activations,
                    selected_experts,
                    expert_frequency,
                    device,
                    hidden_dim,
                    num_experts,
                )

                # routed_characteristic_activation with expert frequency counts
                expert_freq_expanded = expert_frequency.unsqueeze(-1).expand(
                    (-1, hidden_dim)
                )
                self.state[layer_number]["routed_characteristic_activation"].update(
                    routed_characteristic_activation, expert_freq_expanded
                )
                del expert_freq_expanded, routed_characteristic_activation

                online_characteristic_activation_dist = ca_dist_online(
                    activations,
                    distance_callable=distance_fn,
                ).to(device="cpu")

                # online_characteristic_activation_dist with expert frequency counts
                self.state[layer_number]["online_characteristic_activation_dist"].update(
                    online_characteristic_activation_dist, num_tokens
                )
                del online_characteristic_activation_dist

                # router logit similarity -> must align with distance_fn shape expectations
                # dim 0 "batch" dim, dims 1,2 expert pairwise, dim 3 token logits
                router_logit_sim = (
                    distance_fn(
                        router_logits.permute(1, 0).view(
                            1, num_experts, 1, -1
                        ),  # 1, num_experts, 1, logits
                        router_logits.permute(1, 0).view(
                            1, 1, num_experts, -1
                        ),  # 1, 1, num_experts, logits
                    )
                    .squeeze()
                    .to(device="cpu")
                )  # yields (num_experts, num_experts)

                # router_logit_similarity with total tokens count
                self.state[layer_number]["router_logit_similiarity"].update(
                    router_logit_sim, num_tokens
                )
                del router_logit_sim

                # characteristic_activation with total tokens count
                self.state[layer_number]["characteristic_activation"].update(
                    activations.mean(dim=1), num_tokens
                )

            # Pruning criteria
            ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
            ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
            weighted_ean_sum = torch.zeros(
                num_experts, device=device, dtype=torch.float64
            )
            reap = torch.zeros(
                num_experts, device=device, dtype=torch.float32
            )
            weighted_expert_frequency_sum = torch.zeros(
                num_experts, device=device, dtype=torch.float64
            )
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float).to(
                device
            )  # tok, num_experts
            prior_max_activations = self.state[layer_number]["max_activations"]
            # renormalize
            if self.hook_config.renormalize_router_weights:
                topk_weights = torch.gather(
                    routing_weights,
                    1,
                    selected_experts,
                )  # (total_tokens, top_k)
                routing_weights = routing_weights / topk_weights.sum(
                    dim=-1, keepdim=True
                )
                routing_weights = torch.clamp(
                    routing_weights, min=torch.finfo(routing_weights.dtype).eps
                )
                # routing_weights = routing_weights.to(device)

            for i in range(num_experts):
                active_mask = (selected_experts == i).any(dim=-1).to(device)
                if not active_mask.any():
                    continue
                active_router_weights = routing_weights[active_mask, i]
                ean_norm = torch.linalg.norm(activations[i, active_mask, :], dim=-1)
                ean_sum[i] = ean_norm.sum().to(device)
                ean_mean[i] = ean_norm.mean().to(device)
                weighted_expert_frequency_sum[i] = active_router_weights.sum().to(
                    device
                )
                weighted_ean_sum[i] = (
                    (ean_norm * active_router_weights).sum().to(device)
                )
                reap[i] = (
                    (ean_norm * active_router_weights).mean().to(device)
                )

                # super experts
                selected_activations = activations[i, active_mask, :]
                selected_activations_max = selected_activations.max().to(device="cpu")
                if selected_activations_max > prior_max_activations[i]:
                    self.state[layer_number]["max_activations"][i] = (
                        selected_activations_max
                    )
                    prior_max_activations[i] = selected_activations_max

            # ean
            self.state[layer_number]["ean_sum"] += ean_sum.to(device="cpu")
            self.state[layer_number]["ean_mean"].update(ean_mean, expert_frequency)
            self.state[layer_number]["weighted_ean_sum"] += weighted_ean_sum.to(
                device="cpu"
            )
            if reap.sum() == 0:
                print("debug")
            self.state[layer_number]["reap"].update(
                reap, expert_frequency
            )

            # weighted_expert_frequency_sum
            
            self.state[layer_number]["weighted_expert_frequency_sum"] += (
                weighted_expert_frequency_sum.to(device="cpu")
            )

            # --- CLEAN UP -------------------------------------------------------------
            del (
                activations,
                selected_experts,
                router_logits,
                expert_frequency,
                pairwise_expert_frequency,
                prior_max_activations,
            )
            gc.collect()

        return _hook_fn


# --- Concrete Config Implementations ----


@dataclass
class Qwen3MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Qwen3MoeSparseMoeBlock"


@dataclass
class Llama4MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Llama4TextMoe"
    fused_experts: bool = True  # Llama4 uses fused experts


@dataclass
class MixtralMoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "MixtralSparseMoeBlock"


@dataclass
class DeepSeekMoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "DeepseekV2MoE"
    num_experts_attr_name: str = "experts_per_rank"  # only for ep=1!
    top_k_attr_name: str = "num_experts_per_tok"
    fused_experts: bool = False


@dataclass
class Ernie4_5MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Ernie4_5_MoeMLP"
    num_experts_attr_name: str = "num_local_experts"
    top_k_attr_name: str = "k"

    # hf in tree implementation below:
    # module_class_name_to_hook_regex: Optional[str] = "Ernie4_5_MoESparseMoeBlock"
    # num_experts_attr_name: str = "num_experts"
    # top_k_attr_name: str = "top_k"
    fused_experts: bool = False


@dataclass
class Glm44MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Glm4MoeMoE"
    num_experts_attr_name: str = "config.n_routed_experts"
    top_k_attr_name: str = "config.num_experts_per_tok"
    fused_experts: bool = False


@dataclass
class SolarOpenForCausalLMObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "SolarOpenMoE"
    num_experts_attr_name: str = "config.n_routed_experts"
    top_k_attr_name: str = "config.num_experts_per_tok"
    fused_experts: bool = False

@dataclass
class VaetkiForCausalLMObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "VaetkiForCausalLM"
    num_experts_attr_name: str = "n_routed_experts"
    top_k_attr_name: str = "num_experts_per_tok"
    fused_experts: bool = False


@dataclass
class MiMoV2FlashObserverHookConfig(MoETransformerObserverConfig):
    """Observer config for XiaomiMiMo/MiMo-V2-Flash - 309B MoE model."""
    module_class_name_to_hook_regex: Optional[str] = "MiMoV2FlashMoE"
    num_experts_attr_name: str = "num_experts"
    top_k_attr_name: str = "num_experts_per_tok"
    fused_experts: bool = False


@dataclass
class Glm4MoeLiteObserverHookConfig(MoETransformerObserverConfig):
    """Observer config for zai-org/GLM-4.7-Flash (glm4_moe_lite architecture).
    
    Layer 0 is dense (Glm4MoeLiteMLP), layers 1-46 are MoE (Glm4MoeLiteMoE).
    Experts are fused in Glm4MoeLiteNaiveMoe with gate_up_proj tensor [64, 3072, 2048].
    """
    module_class_name_to_hook_regex: Optional[str] = "Glm4MoeLiteMoE"
    num_experts_attr_name: str = "config.n_routed_experts"
    top_k_attr_name: str = "config.num_experts_per_tok"
    fused_experts: bool = True


@dataclass
class LongcatMoEObserverHookConfig(MoETransformerObserverConfig):
    """Observer config for meituan-longcat/LongCat-Flash-Thinking-2601 (560B MoE model).
    
    MoE architecture:
    - LongcatMoE is the MoE block, located at decoder_layer.mlp
    - LongcatTopkRouter is the router (router.classifier is the Linear gate)
    - 512 real experts (LongcatMLP) + 256 identity "zero experts" = 768 total from router's view
    - top_k = 12 (moe_topk in config)
    - Uses MLA attention (Multi-head Latent Attention) similar to DeepSeek
    - zero_expert_type can be "identity" (pass-through) or "drop" (zero output)
    
    The router computes logits via router.classifier (Linear layer), not directly from router.weight.
    
    NOTE: num_experts_attr_name uses "config.n_routed_experts" (512) for real prunable experts.
    The router has 768 logits (including zero experts), but these are truncated in the observer
    to only track the 512 real experts that have actual MLP weights and can be pruned.
    Zero experts (indices 512-767) are virtual (identity/drop) with no weights.
    """
    module_class_name_to_hook_regex: Optional[str] = "LongcatMoE"
    # Use config.n_routed_experts (512) for real experts that can be pruned
    # Router logits will be truncated from 768 to 512 in the observer hook
    num_experts_attr_name: str = "config.n_routed_experts"
    top_k_attr_name: str = "router.top_k"  # LongcatTopkRouter has self.top_k = config.moe_topk
    fused_experts: bool = False


@dataclass
class MiniMaxM2ObserverHookConfig(MoETransformerObserverConfig):
    """Observer config for MiniMaxAI/MiniMax-M2.5.

    MoE architecture:
    - MiniMaxM2SparseMoeBlock is the MoE block
    - Uses w1/w2/w3 projections (not gate_proj/up_proj/down_proj)
    - 256 experts, top_k=8
    - Router is 'gate' (Linear layer)
    - Note: MoE block stores config values directly as attributes (top_k, experts.num_experts)
    """
    module_class_name_to_hook_regex: Optional[str] = "MiniMaxM2SparseMoeBlock"
    num_experts_attr_name: str = "experts.num_experts"
    top_k_attr_name: str = "top_k"
    fused_experts: bool = False


def _infer_moe_class_name(model) -> str | None:
    """Infer the MoE block class name by inspecting the model structure."""
    moe_patterns = ["MoE", "SparseMoeBlock", "MoeBlock", "MoeMLP", "ExpertLayer"]
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            for attr_name in ["mlp", "block_sparse_moe", "moe", "feed_forward", "ffn"]:
                if hasattr(layer, attr_name):
                    module = getattr(layer, attr_name)
                    class_name = module.__class__.__name__
                    # Check if it looks like an MoE block
                    if any(p.lower() in class_name.lower() for p in moe_patterns):
                        return class_name
                    # Check if it has experts attribute (MoE indicator)
                    if hasattr(module, 'experts'):
                        return class_name
    return None


def _infer_num_experts_attr(model) -> str:
    """Infer the attribute path for num_experts on MoE blocks."""
    if hasattr(model, 'config'):
        config = model.config
        # Check config-level attributes
        for key in ["num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"]:
            if hasattr(config, key):
                return f"config.{key}"
    
    # Check MoE module attributes
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            for attr_name in ["mlp", "block_sparse_moe", "moe", "feed_forward"]:
                if hasattr(layer, attr_name):
                    moe = getattr(layer, attr_name)
                    for key in ["num_experts", "num_local_experts", "n_routed_experts", "experts_per_rank"]:
                        if hasattr(moe, key):
                            return key
                    if hasattr(moe, 'config'):
                        for key in ["num_experts", "num_local_experts", "n_routed_experts"]:
                            if hasattr(moe.config, key):
                                return f"config.{key}"
    
    return "num_experts"


def _infer_top_k_attr(model) -> str:
    """Infer the attribute path for num_experts_per_tok on MoE blocks."""
    if hasattr(model, 'config'):
        config = model.config
        for key in ["num_experts_per_tok", "top_k", "moe_k", "num_selected_experts", "k"]:
            if hasattr(config, key):
                return f"config.{key}"
    
    # Check MoE module attributes
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            for attr_name in ["mlp", "block_sparse_moe", "moe", "feed_forward"]:
                if hasattr(layer, attr_name):
                    moe = getattr(layer, attr_name)
                    for key in ["top_k", "num_experts_per_tok", "k"]:
                        if hasattr(moe, key):
                            return key
                    if hasattr(moe, 'config'):
                        for key in ["num_experts_per_tok", "top_k", "k"]:
                            if hasattr(moe.config, key):
                                return f"config.{key}"
    
    return "num_experts_per_tok"


def _detect_fused_experts(model) -> bool:
    """Detect if the model uses fused expert implementation."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            for attr_name in ["mlp", "block_sparse_moe", "moe", "feed_forward"]:
                if hasattr(layer, attr_name):
                    moe = getattr(layer, attr_name)
                    if hasattr(moe, 'experts'):
                        # Fused experts have gate_up_proj as a single tensor
                        if hasattr(moe.experts, 'gate_up_proj'):
                            return True
    return False


def generate_pruning_report(
    observer_data: dict[int, dict[str, Any]],
    experts_to_prune_per_layer: dict[int, torch.Tensor],
    model_name: str,
    prune_method: str,
    category_expert_map: Optional[dict[int, dict[int, str]]] = None,
) -> PruningReport:
    """
    Generate a detailed pruning report showing which experts were pruned.
    
    Args:
        observer_data: Observer state dictionary containing expert frequencies and saliency metrics
        experts_to_prune_per_layer: Dict mapping layer_idx -> tensor of expert indices to prune
        model_name: Name of the model being pruned
        prune_method: The pruning method used (e.g., "frequency", "reap", "ean_sum")
        category_expert_map: Optional mapping of layer_idx -> expert_idx -> category name
    
    Returns:
        PruningReport object containing detailed per-expert pruning decisions
    """
    layer_reports = []
    total_experts_before = 0
    total_experts_after = 0
    
    for layer_idx in sorted(observer_data.keys()):
        layer_data = observer_data[layer_idx]
        expert_frequency = layer_data["expert_frequency"]
        total_tokens = layer_data["total_tokens"].item() if isinstance(layer_data["total_tokens"], torch.Tensor) else layer_data["total_tokens"]
        num_experts = len(expert_frequency)
        
        # Get experts to prune for this layer
        pruned_experts = set()
        if layer_idx in experts_to_prune_per_layer:
            pruned_tensor = experts_to_prune_per_layer[layer_idx]
            if isinstance(pruned_tensor, torch.Tensor):
                pruned_experts = set(pruned_tensor.tolist())
            else:
                pruned_experts = set(pruned_tensor)
        
        # Get saliency scores based on prune method
        saliency_key = prune_method
        if prune_method == "frequency":
            saliency_key = "expert_frequency"
        
        saliency_data = layer_data.get(saliency_key, expert_frequency)
        if isinstance(saliency_data, OnlineStatsTracker):
            saliency_data = saliency_data.mean
        
        # Build expert info list
        expert_infos = []
        for expert_idx in range(num_experts):
            # Get activation count
            activation_count = expert_frequency[expert_idx].item() if isinstance(
                expert_frequency[expert_idx], torch.Tensor
            ) else int(expert_frequency[expert_idx])
            
            # Get classification from category map
            classification = "unknown"
            if category_expert_map and layer_idx in category_expert_map:
                classification = category_expert_map[layer_idx].get(expert_idx, "unknown")
            
            # Get saliency score
            saliency_score = 0.0
            if saliency_data is not None and expert_idx < len(saliency_data):
                score = saliency_data[expert_idx]
                saliency_score = score.item() if isinstance(score, torch.Tensor) else float(score)
            
            expert_info = ExpertPruningInfo(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                activation_count=activation_count,
                pruned=expert_idx in pruned_experts,
                classification=classification,
                saliency_score=saliency_score,
            )
            expert_infos.append(expert_info)
        
        n_pruned = len(pruned_experts)
        n_retained = num_experts - n_pruned
        
        layer_report = LayerPruningReport(
            layer_idx=layer_idx,
            experts=expert_infos,
            total_tokens=total_tokens,
            n_pruned=n_pruned,
            n_retained=n_retained,
        )
        layer_reports.append(layer_report)
        
        total_experts_before += num_experts
        total_experts_after += n_retained
    
    # Calculate compression ratio
    compression_ratio = 1 - (total_experts_after / total_experts_before) if total_experts_before > 0 else 0
    
    return PruningReport(
        model_name=model_name,
        prune_method=prune_method,
        compression_ratio=compression_ratio,
        total_experts_before=total_experts_before,
        total_experts_after=total_experts_after,
        layers=layer_reports,
        category_expert_map=category_expert_map,
    )


def ensure_observer_config(model) -> bool:
    """
    Ensure a model has an observer config registered.
    
    If the model class is not in OBSERVER_CONFIG_REGISTRY, this function will:
    1. Analyze the model structure to infer MoE block class name
    2. Infer num_experts and top_k attribute paths
    3. Create and register a dynamic observer config class
    
    Args:
        model: The loaded model to check/register
        
    Returns:
        True if model was already registered or successfully auto-registered,
        False if registration failed.
    """
    model_class = model.__class__.__name__
    
    if model_class in OBSERVER_CONFIG_REGISTRY:
        logger.debug(f"Model {model_class} already in OBSERVER_CONFIG_REGISTRY")
        return True
    
    logger.info(f"Model {model_class} not in OBSERVER_CONFIG_REGISTRY, attempting auto-registration...")
    
    try:
        # Infer MoE block class name
        moe_class_name = _infer_moe_class_name(model)
        if moe_class_name is None:
            # Fallback: use model class name with MoE suffix
            base_name = model_class.replace("ForCausalLM", "").replace("ForConditionalGeneration", "")
            moe_class_name = f"{base_name}MoE"
            logger.warning(f"Could not detect MoE class, using fallback: {moe_class_name}")
        
        # Infer attribute paths
        num_experts_attr = _infer_num_experts_attr(model)
        top_k_attr = _infer_top_k_attr(model)
        is_fused = _detect_fused_experts(model)
        
        # Create dynamic config class
        @dataclass
        class DynamicObserverConfig(MoETransformerObserverConfig):
            module_class_name_to_hook_regex: Optional[str] = moe_class_name
            num_experts_attr_name: str = num_experts_attr
            top_k_attr_name: str = top_k_attr
            fused_experts: bool = is_fused
        
        # Register the config
        OBSERVER_CONFIG_REGISTRY[model_class] = DynamicObserverConfig
        
        logger.info(
            f"Auto-registered observer config for {model_class}: "
            f"moe_class={moe_class_name}, num_experts_attr={num_experts_attr}, "
            f"top_k_attr={top_k_attr}, fused={is_fused}"
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to auto-register observer config for {model_class}: {e}")
        return False


OBSERVER_CONFIG_REGISTRY = {
    "Qwen3MoeForCausalLM": Qwen3MoEObserverHookConfig,
    "NonUniformQwen3MoeForCausalLM": Qwen3MoEObserverHookConfig,
    "Llama4ForCausalLM": Llama4MoEObserverHookConfig,
    "MixtralForCausalLM": MixtralMoEObserverHookConfig,
    "DeepseekV2ForCausalLM": DeepSeekMoEObserverHookConfig,
    "Ernie4_5_MoEForCausalLM": Ernie4_5MoEObserverHookConfig,
    "Ernie4_5_MoeForCausalLM": Ernie4_5MoEObserverHookConfig,
    "Glm4MoeForCausalLM": Glm44MoEObserverHookConfig,
    "SolarOpenForCausalLM": SolarOpenForCausalLMObserverHookConfig,
    "VaetkiForCausalLM": VaetkiForCausalLMObserverHookConfig,
    # PrimeIntellect/INTELLECT-3 - DeepSeek V3 based
    "DeepseekV3ForCausalLM": DeepSeekMoEObserverHookConfig,
    # Kimi-K2-Thinking - DeepSeek V3 based
    "KimiK2ForCausalLM": DeepSeekMoEObserverHookConfig,
    # XiaomiMiMo/MiMo-V2-Flash - 309B MoE model
    "MiMoV2FlashForCausalLM": MiMoV2FlashObserverHookConfig,
    # GLM-4.7-Flash (zai-org/GLM-4.7-Flash) - glm4_moe_lite architecture
    # Layer 0 is dense, layers 1-46 are MoE with fused experts
    "Glm4MoeLiteForCausalLM": Glm4MoeLiteObserverHookConfig,
    # meituan-longcat/LongCat-Flash-Thinking-2601 - 560B MoE model
    # 512 real experts + 256 identity zero experts, top_k=12, MLA attention
    "LongcatCausalLM": LongcatMoEObserverHookConfig,
    "LongcatForCausalLM": LongcatMoEObserverHookConfig,
    # MiniMaxAI/MiniMax-M2.5 - Uses w1/w2/w3 projections
    "MiniMaxM2ForCausalLM": MiniMaxM2ObserverHookConfig,
}
