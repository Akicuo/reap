"""Abstract base class for observer system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class ExpertActivation:
    """Information about a single expert's activation."""
    layer_idx: int
    expert_idx: int
    activation_count: int
    tokens_processed: int
    activation_frequency: float  # activations / total_tokens
    classification: str = "unknown"
    saliency_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer": self.layer_idx,
            "expert_idx": self.expert_idx,
            "activation_count": self.activation_count,
            "tokens_processed": self.tokens_processed,
            "activation_frequency": self.activation_frequency,
            "classification": self.classification,
            "saliency_score": self.saliency_score,
        }


@dataclass
class LayerStats:
    """Statistics for a single layer."""
    layer_idx: int
    total_activations: int
    active_experts: int
    avg_activation_per_expert: float
    expert_counts: Dict[int, int] = field(default_factory=dict)


@dataclass
class ObserverMetrics:
    """Metrics collected by the observer."""
    total_tokens: int
    expert_activations: List[ExpertActivation]
    layer_wise_stats: Dict[int, LayerStats]
    temporal_activations: List[Dict[int, int]] = field(default_factory=list)

    def get_expert_count(self, layer_idx: int, expert_idx: int) -> int:
        """Get activation count for a specific expert."""
        for activation in self.expert_activations:
            if activation.layer_idx == layer_idx and activation.expert_idx == expert_idx:
                return activation.activation_count
        return 0

    def get_layer_experts(self, layer_idx: int) -> List[int]:
        """Get list of expert indices activated in a layer."""
        return [a.expert_idx for a in self.expert_activations if a.layer_idx == layer_idx]


class BaseObserver(ABC):
    """
    Abstract observer for monitoring expert activations.

    This defines the interface that both PyTorch and MLX observers must implement.
    """

    def __init__(self, model: Any, layers_to_monitor: Optional[List[int]] = None):
        """
        Initialize observer.

        Args:
            model: The model to observe
            layers_to_monitor: List of layer indices to monitor (None = all MoE layers)
        """
        self.model = model
        self.layers_to_monitor = layers_to_monitor or []
        self.metrics: Optional[ObserverMetrics] = None
        self._hooks: List[Any] = []

    @abstractmethod
    def start_observing(self):
        """
        Attach observation hooks/wrappers to the model.

        This method should:
        1. Find MoE layers in the model
        2. Attach hooks or wrap forward methods
        3. Initialize tracking data structures
        """
        pass

    @abstractmethod
    def stop_observing(self):
        """
        Remove observation hooks/wrappers from the model.

        This should restore the model to its original state.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> ObserverMetrics:
        """
        Return collected metrics.

        Returns:
            ObserverMetrics with all collected data
        """
        pass

    @abstractmethod
    def reset(self):
        """Clear all collected metrics."""
        pass

    @abstractmethod
    def process_batch(self, inputs: Any, outputs: Any, layer_idx: int):
        """
        Process a single batch through the observer.

        This is called by hooks/wrappers during forward pass.

        Args:
            inputs: Input tensors
            outputs: Output tensors
            layer_idx: Layer index
        """
        pass

    @abstractmethod
    def record_activation(self, layer_idx: int, expert_idx: int, count: int = 1):
        """
        Record an expert activation.

        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            count: Number of activations to record
        """
        pass

    @abstractmethod
    def get_layer_expert_indices(self, layer_output: Any, layer_idx: int) -> List[int]:
        """
        Extract which experts were activated from layer output.

        This is architecture-specific and must be implemented per model type.

        Args:
            layer_output: Output from the MoE layer
            layer_idx: Layer index

        Returns:
            List of expert indices that were activated
        """
        pass

    def is_observing(self) -> bool:
        """Check if observer is currently active."""
        return len(self._hooks) > 0


@dataclass
class MoEBlockInfo:
    """Information about an MoE block in the model."""
    layer_idx: int
    module_path: str  # e.g., "model.layers.0.mlp"
    num_experts: int
    experts_per_tok: int
    fused: bool


class BaseModelAdapter(ABC):
    """
    Abstract base class for model-specific adapters.

    Different MoE architectures (DeepSeek, Mixtral, Qwen3, etc.) have different
    internal structures. Adapters provide the interface to work with these.
    """

    @abstractmethod
    def get_num_experts(self, model: Any, layer_idx: int) -> int:
        """Get number of experts in a layer."""
        pass

    @abstractmethod
    def get_experts_per_tok(self, model: Any, layer_idx: int) -> int:
        """Get number of experts selected per token."""
        pass

    @abstractmethod
    def is_fused(self, model: Any, layer_idx: int) -> bool:
        """Check if experts use fused weights."""
        pass

    @abstractmethod
    def find_moe_blocks(self, model: Any) -> List[MoEBlockInfo]:
        """Find all MoE blocks in the model."""
        pass

    @abstractmethod
    def get_expert_indices_from_output(self, output: Any) -> List[int]:
        """Extract expert indices from layer output."""
        pass

    @abstractmethod
    def get_moe_block(self, model: Any, layer_idx: int) -> Any:
        """Get the MoE block module for a layer."""
        pass
