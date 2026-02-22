"""MLX weight operations for REAP framework."""

import logging
from typing import Any, Dict, List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn

from ...core.merging_base import MergeMethod

logger = logging.getLogger(__name__)


class MlxWeightOps:
    """
    Weight manipulation operations for MLX models.

    MLX arrays are immutable, so operations create new arrays rather than
    modifying in place.
    """

    @staticmethod
    def get_expert_weights(model: nn.Module, layer_idx: int, expert_idx: int) -> Dict[str, mx.array]:
        """
        Extract weights for a specific expert.

        Args:
            model: The MLX model
            layer_idx: Layer index
            expert_idx: Expert index

        Returns:
            Dictionary mapping weight names to MLX arrays
        """
        weights = {}

        # Navigate to layer
        moe_block = MlxWeightOps._get_moe_block(model, layer_idx)
        if moe_block is None:
            raise ValueError(f"No MoE block found at layer {layer_idx}")

        # Get experts module
        experts = MlxWeightOps._get_experts_module(moe_block)
        if experts is None:
            raise ValueError(f"No experts module in layer {layer_idx}")

        # Check if fused
        is_fused = MlxWeightOps._is_fused_experts(moe_block)

        if is_fused:
            # Fused: gate_up_proj combined
            if hasattr(experts, 'gate_up_proj'):
                weights['gate_up_proj.weight'] = experts.gate_up_proj[expert_idx]
            if hasattr(experts, 'down_proj'):
                weights['down_proj.weight'] = experts.down_proj[expert_idx]
        else:
            # Non-fused: separate projections
            expert = experts[expert_idx] if expert_idx < len(experts) else None
            if expert is None:
                raise ValueError(f"Expert {expert_idx} not found")

            # Try common projection names
            for proj in ['gate_proj', 'w1']:
                if hasattr(expert, proj):
                    weights[f'{proj}.weight'] = getattr(expert, proj).weight
                    break

            for proj in ['up_proj', 'w3']:
                if hasattr(expert, proj):
                    weights[f'{proj}.weight'] = getattr(expert, proj).weight
                    break

            for proj in ['down_proj', 'w2']:
                if hasattr(expert, proj):
                    weights[f'{proj}.weight'] = getattr(expert, proj).weight
                    break

        return weights

    @staticmethod
    def set_expert_weights(model: nn.Module, layer_idx: int, expert_idx: int,
                          weights: Dict[str, mx.array]):
        """
        Set weights for a specific expert.

        Note: MLX arrays are immutable, so we update parameters dict.

        Args:
            model: The MLX model
            layer_idx: Layer index
            expert_idx: Expert index
            weights: Dictionary of weight arrays
        """
        # Build parameter paths and update
        params = model.parameters()

        for weight_name, weight_value in weights.items():
            # Remove .weight suffix
            name = weight_name.replace('.weight', '')
            param_path = MlxWeightOps._build_param_path(layer_idx, expert_idx, name, model)
            if param_path:
                params[param_path] = weight_value

        # Update model
        model.update(params)
        mx.eval(model.parameters())

    @staticmethod
    def merge_experts(source_expert: Dict[str, mx.array],
                     target_expert: Dict[str, mx.array],
                     alpha: float = 0.5,
                     method: MergeMethod = MergeMethod.AVERAGE) -> Dict[str, mx.array]:
        """
        Merge two experts by combining their weights.

        Args:
            source_expert: Source expert weights
            target_expert: Target expert weights
            alpha: Merge coefficient (0 = keep target, 1 = full source, 0.5 = average)
            method: Merging method

        Returns:
            Merged weights dictionary
        """
        merged = {}

        if method == MergeMethod.AVERAGE:
            # Simple average
            for key in source_expert.keys():
                if key in target_expert:
                    merged[key] = (source_expert[key] + target_expert[key]) / 2
                else:
                    merged[key] = source_expert[key]

        elif method == MergeMethod.FREQUENCY_WEIGHTED_AVERAGE:
            # Weighted by frequency (passed externally, simple average here)
            for key in source_expert.keys():
                if key in target_expert:
                    merged[key] = alpha * source_expert[key] + (1 - alpha) * target_expert[key]
                else:
                    merged[key] = source_expert[key]

        else:
            # Default to weighted average
            for key in source_expert.keys():
                if key in target_expert:
                    merged[key] = alpha * source_expert[key] + (1 - alpha) * target_expert[key]
                else:
                    merged[key] = source_expert[key]

        return merged

    @staticmethod
    def remove_expert(model: nn.Module, layer_idx: int, expert_idx: int):
        """
        Remove an expert from the model.

        Args:
            model: The MLX model
            layer_idx: Layer index
            expert_idx: Expert index to remove
        """
        moe_block = MlxWeightOps._get_moe_block(model, layer_idx)
        if moe_block is None:
            raise ValueError(f"No MoE block at layer {layer_idx}")

        experts = MlxWeightOps._get_experts_module(moe_block)
        if experts is None:
            raise ValueError(f"No experts module in layer {layer_idx}")

        # Remove from ModuleList
        if hasattr(experts, '__delitem__'):
            del experts[expert_idx]

        # Update config
        if hasattr(moe_block, 'num_local_experts'):
            moe_block.num_local_experts = len(experts)
        elif hasattr(moe_block, 'num_experts'):
            moe_block.num_experts = len(experts)

        # Update router weights
        router = MlxWeightOps._get_router(moe_block)
        if router is not None and hasattr(router, 'weight'):
            gate_weight = router.weight
            # Remove row for this expert
            router.weight = mx.concatenate([
                gate_weight[:expert_idx],
                gate_weight[expert_idx+1:]
            ], axis=0)

        mx.eval(model.parameters())

    @staticmethod
    def compute_cosine_similarity(weights_a: Dict[str, mx.array],
                                  weights_b: Dict[str, mx.array]) -> float:
        """
        Compute cosine similarity between two experts.

        Args:
            weights_a: First expert's weights
            weights_b: Second expert's weights

        Returns:
            Cosine similarity score
        """
        # Flatten weights into vectors
        vec_a = MlxWeightOps._flatten_weights(weights_a)
        vec_b = MlxWeightOps._flatten_weights(weights_b)

        # Ensure same size
        min_size = min(vec_a.size, vec_b.size)
        vec_a = vec_a[:min_size]
        vec_b = vec_b[:min_size]

        # Cosine similarity
        dot_product = mx.sum(vec_a * vec_b)
        norm_a = mx.sqrt(mx.sum(vec_a * vec_a))
        norm_b = mx.sqrt(mx.sum(vec_b * vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return (dot_product / (norm_a * norm_b)).item()

    @staticmethod
    def compute_euclidean_distance(weights_a: Dict[str, mx.array],
                                   weights_b: Dict[str, mx.array]) -> float:
        """Compute Euclidean distance between two experts."""
        vec_a = MlxWeightOps._flatten_weights(weights_a)
        vec_b = MlxWeightOps._flatten_weights(weights_b)

        min_size = min(vec_a.size, vec_b.size)
        vec_a = vec_a[:min_size]
        vec_b = vec_b[:min_size]

        return mx.sqrt(mx.sum((vec_a - vec_b) ** 2)).item()

    @staticmethod
    def permute_weights(weights: Dict[str, mx.array],
                       permutation: List[int]) -> Dict[str, mx.array]:
        """
        Permute weights according to a permutation.

        Args:
            weights: Expert weights
            permutation: Permutation to apply

        Returns:
            Permuted weights
        """
        permuted = {}
        for key, value in weights.items():
            if value.ndim >= 1:
                # Permute along first dimension
                permuted[key] = value[permutation]
            else:
                permuted[key] = value
        return permuted

    @staticmethod
    def save_model(model: nn.Module, save_path: str):
        """
        Save MLX model to disk.

        Args:
            model: The model
            save_path: Directory to save to
        """
        from mlx_lm.utils import save_weights
        import os

        os.makedirs(save_path, exist_ok=True)
        save_weights(save_path, model.parameters())
        logger.info(f"Saved model to {save_path}")

    # Helper methods

    @staticmethod
    def _get_moe_block(model: nn.Module, layer_idx: int) -> Optional[Any]:
        """Get MoE block for a layer."""
        for attr in ['model', 'transformer']:
            if hasattr(model, attr):
                container = getattr(model, attr)
                if hasattr(container, 'layers') and layer_idx < len(container.layers):
                    layer = container.layers[layer_idx]
                    for moe_attr in ['mlp', 'moe', 'block_sparse_moe', 'feed_forward']:
                        if hasattr(layer, moe_attr):
                            return getattr(layer, moe_attr)
        return None

    @staticmethod
    def _get_experts_module(moe_block: Any) -> Optional[Any]:
        """Get experts module from MoE block."""
        for attr in ['experts', 'local_experts', 'expert']:
            if hasattr(moe_block, attr):
                return getattr(moe_block, attr)
        return None

    @staticmethod
    def _get_router(moe_block: Any) -> Optional[Any]:
        """Get router/gate from MoE block."""
        for attr in ['gate', 'router', 'gating_network', 'switch']:
            if hasattr(moe_block, attr):
                return getattr(moe_block, attr)
        return None

    @staticmethod
    def _is_fused_experts(moe_block: Any) -> bool:
        """Check if experts use fused weights."""
        experts = MlxWeightOps._get_experts_module(moe_block)
        if experts is not None and hasattr(experts, 'gate_up_proj'):
            return True
        return False

    @staticmethod
    def _flatten_weights(weights: Dict[str, mx.array]) -> mx.array:
        """Flatten all weights into a single vector."""
        vectors = []
        for weight in weights.values():
            vectors.append(weight.flatten())
        return mx.concatenate(vectors)

    @staticmethod
    def _build_param_path(layer_idx: int, expert_idx: int, weight_name: str,
                          model: nn.Module) -> Optional[str]:
        """Build parameter path for weight updates."""
        # Detect MoE block location
        moe_block = MlxWeightOps._get_moe_block(model, layer_idx)
        if moe_block is None:
            return None

        # Determine block type
        for moe_attr in ['mlp', 'moe', 'block_sparse_moe', 'feed_forward']:
            if hasattr(model.model.layers[layer_idx], moe_attr):
                return f"model.layers.{layer_idx}.{moe_attr}.experts.{expert_idx}.{weight_name}"

        return f"layers.{layer_idx}.mlp.experts.{expert_idx}.{weight_name}"
