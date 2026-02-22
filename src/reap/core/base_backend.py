"""Abstract base class for REAP backends (PyTorch/MLX)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Metadata about a loaded model."""
    name: str
    num_experts: int
    num_layers: int
    hidden_size: int
    expert_config: Dict[str, Any]

    # MoE-specific fields
    num_experts_per_tok: int = 0
    moe_layer_interval: int = 1
    expert_intermediate_size: int = 0

    # Architecture-specific
    fused_experts: bool = False
    moe_block_location: str = "mlp"  # "mlp", "block_sparse_moe", "feed_forward"


class BaseBackend(ABC):
    """
    Abstract base class for REAP backends.

    Defines the interface that both PyTorch and MLX backends must implement.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._get_device()

    @abstractmethod
    def _get_device(self) -> Any:
        """
        Get device handle for the backend.

        Returns:
            Device object (e.g., torch.device, mx.Device)
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> Tuple[Any, ModelInfo]:
        """
        Load model from path.

        Args:
            model_path: Path to model (HuggingFace ID, local path, or mlx-community path)

        Returns:
            Tuple of (model, ModelInfo)
        """
        pass

    @abstractmethod
    def load_tokenizer(self, model_path: str) -> Any:
        """
        Load tokenizer for the model.

        Args:
            model_path: Path to model

        Returns:
            Tokenizer object
        """
        pass

    @abstractmethod
    def get_expert_weights(self, model: Any, layer_idx: int, expert_idx: int) -> Dict[str, Any]:
        """
        Extract weights for a specific expert.

        Args:
            model: The model
            layer_idx: Layer index
            expert_idx: Expert index within layer

        Returns:
            Dictionary mapping weight names to weight tensors
        """
        pass

    @abstractmethod
    def set_expert_weights(self, model: Any, layer_idx: int, expert_idx: int, weights: Dict[str, Any]):
        """
        Set weights for a specific expert.

        Args:
            model: The model
            layer_idx: Layer index
            expert_idx: Expert index within layer
            weights: Dictionary mapping weight names to weight tensors
        """
        pass

    @abstractmethod
    def remove_expert(self, model: Any, layer_idx: int, expert_idx: int):
        """
        Remove an expert from the model.

        Args:
            model: The model
            layer_idx: Layer index
            expert_idx: Expert index within layer
        """
        pass

    @abstractmethod
    def get_num_experts(self, model: Any, layer_idx: int) -> int:
        """
        Get number of experts in a layer.

        Args:
            model: The model
            layer_idx: Layer index

        Returns:
            Number of experts
        """
        pass

    @abstractmethod
    def save_model(self, model: Any, save_path: str):
        """
        Save model to disk.

        Args:
            model: The model
            save_path: Path to save
        """
        pass

    @abstractmethod
    def create_optimizer(self, params: Any, lr: float) -> Any:
        """
        Create optimizer for training/merging.

        Args:
            params: Model parameters
            lr: Learning rate

        Returns:
            Optimizer object
        """
        pass

    @abstractmethod
    def empty_cache(self):
        """Clear device cache to free memory."""
        pass

    @abstractmethod
    def get_device_count(self) -> int:
        """Get number of available devices."""
        pass


@dataclass
class ExpertWeights:
    """Container for expert weights."""
    gate_proj: Any  # Weight tensor
    up_proj: Any  # Weight tensor
    down_proj: Any  # Weight tensor


@dataclass
class PruningResult:
    """Result of a pruning operation."""
    experts_removed: List[Tuple[int, int]]  # (layer_idx, expert_idx) pairs
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    model_size_reduction: float


class BasePruner(ABC):
    """Abstract interface for expert pruning."""

    def __init__(self, backend: BaseBackend):
        self.backend = backend

    @abstractmethod
    def prune_by_activation(self, model: Any, observer_metrics: 'ObserverMetrics',
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
        pass

    @abstractmethod
    def prune_by_count(self, model: Any, observer_metrics: 'ObserverMetrics',
                      keep_n: int) -> PruningResult:
        """
        Keep top N experts per layer.

        Args:
            model: The model to prune
            observer_metrics: Metrics from observer
            keep_n: Number of experts to keep per layer

        Returns:
            PruningResult with details
        """
        pass

    @abstractmethod
    def prune_custom(self, model: Any, experts_to_remove: List[Tuple[int, int]]) -> PruningResult:
        """
        Remove specific experts from model.

        Args:
            model: The model to prune
            experts_to_remove: List of (layer_idx, expert_idx) pairs

        Returns:
            PruningResult with details
        """
        pass
