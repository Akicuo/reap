"""MLX expert pruning for REAP framework."""

import logging
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn

from ...core.base_backend import BaseBackend, PruningResult
from ...core.observer_base import ObserverMetrics
from ...core.merging_base import MergeMethod

logger = logging.getLogger(__name__)


class MlxPruner:
    """
    MLX implementation of expert pruning.

    Removes low-saliency experts from MoE models.
    """

    def __init__(self, backend: BaseBackend):
        """
        Initialize pruner.

        Args:
            backend: MLX backend instance
        """
        self.backend = backend

    def prune_by_activation(self, model: Any, observer_metrics: ObserverMetrics,
                           threshold: float) -> PruningResult:
        """
        Prune experts with activation frequency below threshold.

        Args:
            model: The model to prune
            observer_metrics: Metrics from observer
            threshold: Minimum activation frequency (0-1)

        Returns:
            PruningResult with details
        """
        logger.info(f"Pruning by activation threshold: {threshold}")

        experts_to_remove = []

        # Group activations by layer and expert
        for activation in observer_metrics.expert_activations:
            if activation.activation_frequency < threshold:
                experts_to_remove.append((activation.layer_idx, activation.expert_idx))

        return self.prune_custom(model, experts_to_remove)

    def prune_by_count(self, model: Any, observer_metrics: ObserverMetrics,
                      keep_n: int) -> PruningResult:
        """
        Keep only top N experts per layer by activation count.

        Args:
            model: The model to prune
            observer_metrics: Metrics from observer
            keep_n: Number of experts to keep per layer

        Returns:
            PruningResult with details
        """
        logger.info(f"Pruning to keep top {keep_n} experts per layer")

        experts_to_remove = []

        # Group by layer
        layer_experts: Dict[int, List] = {}
        for activation in observer_metrics.expert_activations:
            if activation.layer_idx not in layer_experts:
                layer_experts[activation.layer_idx] = []
            layer_experts[activation.layer_idx].append(activation)

        # Find experts to remove (bottom ones)
        for layer_idx, activations in layer_experts.items():
            # Sort by activation count (ascending)
            sorted_activations = sorted(activations, key=lambda x: x.activation_count)

            # Remove all but top N
            if len(sorted_activations) > keep_n:
                for activation in sorted_activations[:-keep_n]:
                    experts_to_remove.append((activation.layer_idx, activation.expert_idx))

        return self.prune_custom(model, experts_to_remove)

    def prune_custom(self, model: Any, experts_to_remove: List[Tuple[int, int]]) -> PruningResult:
        """
        Remove specified experts from model.

        Args:
            model: The model to prune
            experts_to_remove: List of (layer_idx, expert_idx) pairs

        Returns:
            PruningResult with details
        """
        logger.info(f"Pruning {len(experts_to_remove)} experts")

        # Get metrics before
        metrics_before = self._compute_model_metrics(model)

        # Sort by (layer, expert) in descending order to avoid index shifting
        experts_to_remove = sorted(experts_to_remove, key=lambda x: (x[0], x[1]), reverse=True)

        # Remove experts
        for layer_idx, expert_idx in experts_to_remove:
            try:
                self.backend.remove_expert(model, layer_idx, expert_idx)
                logger.debug(f"Removed expert {expert_idx} from layer {layer_idx}")
            except Exception as e:
                logger.warning(f"Failed to remove expert {expert_idx} from layer {layer_idx}: {e}")

        # Get metrics after
        metrics_after = self._compute_model_metrics(model)

        # Calculate size reduction
        experts_before = metrics_before.get('total_experts', 0)
        experts_after = metrics_after.get('total_experts', 0)
        size_reduction = (experts_before - experts_after) / max(experts_before, 1)

        logger.info(f"Pruning complete: {experts_before} -> {experts_after} experts ({size_reduction:.1%} reduction)")

        return PruningResult(
            experts_removed=experts_to_remove,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            model_size_reduction=size_reduction,
        )

    def _compute_model_metrics(self, model: Any) -> Dict[str, float]:
        """Compute model metrics."""
        total_experts = 0
        total_params = 0

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                # Check each MoE location
                for moe_attr in ['mlp', 'moe', 'block_sparse_moe', 'feed_forward']:
                    if hasattr(layer, moe_attr):
                        moe_block = getattr(layer, moe_attr)
                        num_experts = self.backend.get_num_experts(model, list(model.model.layers).index(layer))
                        total_experts += num_experts

                        # Count parameters
                        if hasattr(moe_block, 'experts'):
                            for expert in moe_block.experts:
                                for param in expert.parameters():
                                    total_params += param.size

        return {
            "total_experts": total_experts,
            "total_parameters": total_params,
        }


class MlxMerger:
    """
    MLX implementation of expert merging.

    Combines similar experts to reduce model size.
    """

    def __init__(self, backend: BaseBackend):
        """
        Initialize merger.

        Args:
            backend: MLX backend instance
        """
        self.backend = backend
        from . import weight_ops
        self.weight_ops = weight_ops.MlxWeightOps

    def merge_similar_experts(self, model: Any, layer_idx: int,
                             similarity_threshold: float = 0.9,
                             method: MergeMethod = MergeMethod.FREQUENCY_WEIGHTED_AVERAGE,
                             frequency_data: Dict[int, int] = None) -> List[Tuple[int, int]]:
        """
        Find and merge similar experts within a layer.

        Args:
            model: The model
            layer_idx: Layer index
            similarity_threshold: Cosine similarity threshold
            method: Merging method
            frequency_data: Expert activation frequencies

        Returns:
            List of (expert_a, expert_b) pairs merged
        """
        logger.info(f"Merging similar experts in layer {layer_idx} with threshold {similarity_threshold}")

        # Get all experts in layer
        num_experts = self.backend.get_num_experts(model, layer_idx)
        if num_experts == 0:
            return []

        # Compute pairwise similarities
        similarities = self._compute_pairwise_similarities(model, layer_idx)

        # Find pairs above threshold
        pairs_to_merge = []
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                if similarities[i][j] >= similarity_threshold:
                    pairs_to_merge.append((i, j))

        # Merge pairs (process in reverse order to avoid index shifting)
        merged_pairs = []
        for expert_a, expert_b in sorted(pairs_to_merge, reverse=True):
            if self._merge_expert_pair(model, layer_idx, expert_a, expert_b,
                                       method, frequency_data):
                merged_pairs.append((expert_a, expert_b))

        logger.info(f"Merged {len(merged_pairs)} expert pairs in layer {layer_idx}")
        return merged_pairs

    def merge_by_clusters(self, model: Any, clusters: List[Any],
                         method: MergeMethod = MergeMethod.FREQUENCY_WEIGHTED_AVERAGE,
                         frequency_data: Dict[Tuple[int, int], int] = None):
        """
        Merge experts according to pre-computed clusters.

        Args:
            model: The model
            clusters: List of cluster assignments
            method: Merging method
            frequency_data: Expert activation frequencies

        Returns:
            MergeResult
        """
        logger.info("Merging experts by clusters")

        metrics_before = self._compute_model_metrics(model)

        merged_pairs = []

        for cluster in clusters:
            layer_idx = cluster.layer_idx
            expert_clusters = cluster.expert_clusters

            # Group experts by cluster
            cluster_groups: Dict[int, List[int]] = {}
            for expert_idx, cluster_id in enumerate(expert_clusters):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(expert_idx)

            # Merge each cluster (keep first, merge others into it)
            for cluster_id, expert_indices in cluster_groups.items():
                if len(expert_indices) <= 1:
                    continue

                # Sort and merge in reverse
                expert_indices = sorted(expert_indices, reverse=True)
                keep_expert = expert_indices[0]

                for merge_expert in expert_indices[1:]:
                    if self._merge_expert_pair(model, layer_idx, keep_expert, merge_expert,
                                             method, frequency_data):
                        merged_pairs.append((layer_idx, keep_expert, merge_expert))

        metrics_after = self._compute_model_metrics(model)

        return MergeResult(
            merged_pairs=merged_pairs,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            model_size_reduction=(metrics_before['total_experts'] - metrics_after['total_experts']) /
                                max(metrics_before['total_experts'], 1),
        )

    def _merge_expert_pair(self, model: Any, layer_idx: int,
                          expert_a: int, expert_b: int,
                          method: MergeMethod,
                          frequency_data: Dict[Tuple[int, int], int] = None) -> bool:
        """
        Merge expert_b into expert_a, then remove expert_b.

        Args:
            model: The model
            layer_idx: Layer index
            expert_a: Expert to keep
            expert_b: Expert to merge and remove
            method: Merging method
            frequency_data: Expert frequencies

        Returns:
            True if merge succeeded
        """
        try:
            weights_a = self.backend.get_expert_weights(model, layer_idx, expert_a)
            weights_b = self.backend.get_expert_weights(model, layer_idx, expert_b)

            # Calculate merge coefficient
            alpha = 0.5  # Default equal weight

            if frequency_data is not None:
                freq_a = frequency_data.get((layer_idx, expert_a), 1)
                freq_b = frequency_data.get((layer_idx, expert_b), 1)
                total = freq_a + freq_b
                if total > 0:
                    alpha = freq_b / total  # Weight by frequency

            # Merge weights
            merged_weights = self.weight_ops.merge_experts(weights_b, weights_a, alpha, method)

            # Update expert_a
            self.backend.set_expert_weights(model, layer_idx, expert_a, merged_weights)

            # Remove expert_b
            self.backend.remove_expert(model, layer_idx, expert_b)

            return True
        except Exception as e:
            logger.warning(f"Failed to merge experts {expert_a} and {expert_b}: {e}")
            return False

    def _compute_pairwise_similarities(self, model: Any, layer_idx: int) -> List[List[float]]:
        """Compute cosine similarity between all expert pairs in a layer."""
        num_experts = self.backend.get_num_experts(model, layer_idx)
        if num_experts == 0:
            return []

        similarities = [[0.0] * num_experts for _ in range(num_experts)]

        # Get all expert weights
        expert_weights = []
        for i in range(num_experts):
            try:
                weights = self.backend.get_expert_weights(model, layer_idx, i)
                expert_weights.append(weights)
            except Exception:
                expert_weights.append(None)

        # Compute similarities
        for i in range(num_experts):
            for j in range(i, num_experts):
                if expert_weights[i] is None or expert_weights[j] is None:
                    similarities[i][j] = 0.0
                    similarities[j][i] = 0.0
                    continue

                sim = self.weight_ops.compute_cosine_similarity(expert_weights[i], expert_weights[j])
                similarities[i][j] = float(sim)
                similarities[j][i] = float(sim)

        return similarities

    def _compute_model_metrics(self, model: Any) -> Dict[str, float]:
        """Compute model metrics."""
        total_experts = 0
        total_params = 0

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                for moe_attr in ['mlp', 'moe', 'block_sparse_moe', 'feed_forward']:
                    if hasattr(layer, moe_attr):
                        moe_block = getattr(layer, moe_attr)
                        num_experts = self.backend.get_num_experts(model, list(model.model.layers).index(layer))
                        total_experts += num_experts

        return {
            "total_experts": total_experts,
            "total_parameters": total_params,
        }
