"""Abstract base class for expert merging operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MergeMethod(Enum):
    """Supported expert merging methods."""
    FREQUENCY_WEIGHTED_AVERAGE = "frequency_weighted_average"
    AVERAGE = "average"
    TIES = "ties"
    MULTISLERP = "multislerp"
    SCE = "sce"  # Stein Complement Estimator
    KARCHER = "karcher"  # Karcher mean
    SUBMOE = "submoe"


@dataclass
class MergeResult:
    """Result of a merging operation."""
    merged_pairs: List[Tuple[int, int, int]]  # (layer_idx, expert_a, expert_b)
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    model_size_reduction: float


@dataclass
class ClusterAssignment:
    """Assignment of experts to clusters."""
    layer_idx: int
    expert_clusters: List[int]  # cluster_id for each expert
    num_clusters: int


class BaseMerger(ABC):
    """
    Abstract interface for expert merging.

    Merging combines similar experts within a cluster by averaging their weights.
    This is an alternative to pruning - instead of removing experts, similar ones
    are merged to reduce model size.
    """

    def __init__(self, backend: 'BaseBackend'):
        """
        Initialize merger.

        Args:
            backend: Backend instance (PyTorch or MLX)
        """
        self.backend = backend

    @abstractmethod
    def merge_similar_experts(self, model: Any, layer_idx: int,
                             similarity_threshold: float = 0.9,
                             method: MergeMethod = MergeMethod.FREQUENCY_WEIGHTED_AVERAGE,
                             frequency_data: Optional[Dict[int, int]] = None) -> List[Tuple[int, int]]:
        """
        Find and merge similar experts within a layer.

        Args:
            model: The model
            layer_idx: Layer index
            similarity_threshold: Cosine similarity threshold for merging
            method: Merging method to use
            frequency_data: Expert activation frequencies (for weighted methods)

        Returns:
            List of (expert_a, expert_b) pairs that were merged
        """
        pass

    @abstractmethod
    def merge_by_clusters(self, model: Any, clusters: List[ClusterAssignment],
                         method: MergeMethod = MergeMethod.FREQUENCY_WEIGHTED_AVERAGE,
                         frequency_data: Optional[Dict[Tuple[int, int], int]] = None) -> MergeResult:
        """
        Merge experts according to pre-computed clusters.

        Args:
            model: The model
            clusters: List of cluster assignments per layer
            method: Merging method to use
            frequency_data: Expert activation frequencies

        Returns:
            MergeResult with details
        """
        pass

    @abstractmethod
    def merge_expert_pair(self, model: Any, layer_idx: int,
                         expert_a: int, expert_b: int,
                         alpha: float = 0.5,
                         method: MergeMethod = MergeMethod.AVERAGE) -> bool:
        """
        Merge expert_b into expert_a, then remove expert_b.

        Args:
            model: The model
            layer_idx: Layer index
            expert_a: First expert index (will be kept)
            expert_b: Second expert index (will be removed)
            alpha: Merge coefficient (0 = keep A, 1 = keep B, 0.5 = average)
            method: Merging method

        Returns:
            True if merge succeeded, False otherwise
        """
        pass

    @abstractmethod
    def compute_pairwise_similarities(self, model: Any, layer_idx: int,
                                     metric: str = "cosine") -> Any:
        """
        Compute pairwise similarity between all experts in a layer.

        Args:
            model: The model
            layer_idx: Layer index
            metric: Similarity metric ("cosine", "euclidean", "angular")

        Returns:
            Similarity matrix (num_experts x num_experts)
        """
        pass


class BaseClusterer(ABC):
    """
    Abstract interface for expert clustering.

    Clustering groups similar experts together before merging.
    """

    def __init__(self, backend: 'BaseBackend'):
        """
        Initialize clusterer.

        Args:
            backend: Backend instance
        """
        self.backend = backend

    @abstractmethod
    def cluster_layer(self, model: Any, layer_idx: int,
                     num_clusters: int,
                     similarity_metric: str = "cosine",
                     linkage_method: str = "average") -> ClusterAssignment:
        """
        Cluster experts within a single layer.

        Args:
            model: The model
            layer_idx: Layer index
            num_clusters: Number of clusters to create
            similarity_metric: Similarity metric for clustering
            linkage_method: Linkage method (for hierarchical clustering)

        Returns:
            ClusterAssignment with cluster IDs for each expert
        """
        pass

    @abstractmethod
    def cluster_all_layers(self, model: Any,
                          num_clusters: int,
                          similarity_metric: str = "cosine",
                          linkage_method: str = "average") -> List[ClusterAssignment]:
        """
        Cluster experts across all layers.

        Args:
            model: The model
            num_clusters: Number of clusters per layer
            similarity_metric: Similarity metric
            linkage_method: Linkage method

        Returns:
            List of ClusterAssignment, one per MoE layer
        """
        pass

    @abstractmethod
    def compute_expert_similarity(self, weights_a: Dict[str, Any],
                                  weights_b: Dict[str, Any],
                                  metric: str = "cosine") -> float:
        """
        Compute similarity between two experts.

        Args:
            weights_a: First expert's weights
            weights_b: Second expert's weights
            metric: Similarity metric

        Returns:
            Similarity score
        """
        pass
