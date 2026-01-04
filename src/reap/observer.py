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


class MoETransformerObserver(BaseTransformerObserver):
    """MoE Transformer Observer for all methods including both pruning and merging."""

    def report_state(self) -> dict[str, Any]:
        """
        Method to report the current state of the observer. Can be overridden to inject
        custom behaviours.
        """
        return {
            layer_num: {
                k: v.mean if isinstance(v, OnlineStatsTracker) else v
                for k, v in layer_state.items()
            }
            for layer_num, layer_state in self.state.items()
        }

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
            
            # Get selected experts from router_logits
            _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
            
            if layer_number not in self.state:
                self.state[layer_number] = self._initialize_state(output, num_experts)
            
            activations = torch.zeros((num_experts, *flat_input.shape), device=device)

            if self.hook_config.fused_experts:
                # Fused experts path
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
    # MiniMax-M2.1-PRISM - similar to DeepSeek
    "MiniMaxForCausalLM": DeepSeekMoEObserverHookConfig,
    # Kimi-K2-Thinking - DeepSeek V3 based
    "KimiK2ForCausalLM": DeepSeekMoEObserverHookConfig,
}
