"""MLX observer implementation for REAP framework.

⚠️ UNTESTED IMPLEMENTATION ⚠️
This code was developed without access to Apple Silicon hardware for runtime testing.
The implementation follows MLX documentation patterns but has not been verified on actual hardware.
Please test on Mac Studio M3 Ultra and report any issues.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn

from ...core.observer_base import (
    BaseObserver,
    ExpertActivation,
    ObserverMetrics,
    LayerStats,
    MoEBlockInfo,
)

logger = logging.getLogger(__name__)


class MlxObserver(BaseObserver):
    """
    MLX Observer implementation using module wrapping.

    Since MLX lacks forward hooks, we wrap MoE layer forward methods
    with tracking logic.
    """

    def __init__(self, model: nn.Module, layers_to_monitor: Optional[List[int]] = None):
        """
        Initialize MLX observer.

        Args:
            model: The MLX model to observe
            layers_to_monitor: List of layer indices to monitor (None = all MoE layers)
        """
        super().__init__(model, layers_to_monitor)
        self.original_forward: Dict[int, Callable] = {}  # Store original forward methods
        self.activation_data: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.token_count = 0
        self.moe_blocks: Dict[int, MoEBlockInfo] = {}

    def start_observing(self):
        """
        Attach observation wrappers to MoE layers.

        This method:
        1. Finds all MoE layers in the model
        2. Stores original forward methods
        3. Wraps forward methods with observation logic
        """
        logger.info("Starting MLX observer...")

        # Reset state
        self.activation_data = defaultdict(lambda: defaultdict(int))
        self.token_count = 0
        self.original_forward = {}

        # Find MoE layers
        self.moe_blocks = self._find_all_moe_blocks()
        logger.info(f"Found {len(self.moe_blocks)} MoE blocks")

        # Wrap each MoE layer's forward method
        for layer_idx, block_info in self.moe_blocks.items():
            if self.layers_to_monitor and layer_idx not in self.layers_to_monitor:
                continue

            moe_module = self._get_moe_module_by_info(block_info)
            if moe_module is None:
                logger.warning(f"Could not get MoE module for layer {layer_idx}")
                continue

            # Store original forward
            self.original_forward[layer_idx] = moe_module.forward

            # Create and set wrapped forward
            wrapped_forward = self._create_observed_forward(layer_idx, block_info)
            moe_module.forward = wrapped_forward

        logger.info(f"Started observing {len(self.original_forward)} MoE layers")

    def _find_all_moe_blocks(self) -> Dict[int, MoEBlockInfo]:
        """
        Find all MoE blocks in the model.

        Returns:
            Dictionary mapping layer_idx to MoEBlockInfo
        """
        moe_blocks = {}

        # Try common model structures
        model_root = self.model
        for attr in ['model', 'transformer']:
            if hasattr(self.model, attr):
                model_root = getattr(self.model, attr)
                break

        if not hasattr(model_root, 'layers'):
            # Try alternative structures
            if hasattr(model_root, 'layers'):
                model_root = model_root
            elif hasattr(self.model, 'layers'):
                model_root = self.model
            else:
                logger.warning("Could not find layers container in model")
                return moe_blocks

        layers = model_root.layers
        num_layers = len(layers) if hasattr(layers, '__len__') else 0

        for layer_idx in range(num_layers):
            layer = layers[layer_idx]

            # Try common MoE block locations
            for moe_attr in ['mlp', 'moe', 'block_sparse_moe', 'feed_forward']:
                if hasattr(layer, moe_attr):
                    moe_block = getattr(layer, moe_attr)
                    block_info = self._extract_moe_block_info(
                        layer_idx, moe_attr, moe_block, layer
                    )
                    if block_info is not None:
                        moe_blocks[layer_idx] = block_info
                        break

        return moe_blocks

    def _extract_moe_block_info(
        self, layer_idx: int, moe_attr: str, moe_block: Any, layer: Any
    ) -> Optional[MoEBlockInfo]:
        """Extract information about an MoE block."""
        # Get number of experts
        num_experts = 0
        experts_per_tok = 0
        fused = False

        # Try config attributes
        for attr in ['num_local_experts', 'num_experts', 'n_routed_experts']:
            if hasattr(moe_block, attr):
                num_experts = getattr(moe_block, attr)
                break

        # Try experts module
        if num_experts == 0:
            experts = self._get_experts_from_block(moe_block)
            if experts is not None:
                if hasattr(experts, '__len__'):
                    num_experts = len(experts)
                elif hasattr(experts, 'num_experts'):
                    num_experts = experts.num_experts

        # Get experts per token
        for attr in ['num_experts_per_tok', 'top_k', 'k', 'moe_k']:
            if hasattr(moe_block, attr):
                experts_per_tok = getattr(moe_block, attr)
                break

        # Check for fused experts
        experts = self._get_experts_from_block(moe_block)
        if experts is not None and hasattr(experts, 'gate_up_proj'):
            fused = True

        if num_experts == 0:
            return None

        return MoEBlockInfo(
            layer_idx=layer_idx,
            module_path=f"layers.{layer_idx}.{moe_attr}",
            num_experts=num_experts,
            experts_per_tok=experts_per_tok,
            fused=fused,
        )

    def _get_experts_from_block(self, moe_block: Any) -> Optional[Any]:
        """Get the experts module from an MoE block."""
        for attr in ['experts', 'local_experts', 'expert']:
            if hasattr(moe_block, attr):
                return getattr(moe_block, attr)
        return None

    def _get_moe_module_by_info(self, block_info: MoEBlockInfo) -> Optional[Any]:
        """Get the MoE module using MoEBlockInfo."""
        # Navigate to the module using the stored path
        parts = block_info.module_path.split('.')
        current = self.model

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

        return current

    def _create_observed_forward(self, layer_idx: int, block_info: MoEBlockInfo) -> Callable:
        """
        Create a wrapped forward function that tracks expert activations.

        Args:
            layer_idx: Layer index
            block_info: Information about the MoE block

        Returns:
            Wrapped forward function
        """

        def observed_forward(*args, **kwargs):
            # Call original forward
            original_fwd = self.original_forward[layer_idx]
            outputs = original_fwd(*args, **kwargs)

            # Extract expert routing information
            expert_indices = self._extract_expert_indices(outputs, block_info)

            # Record activations
            if layer_idx not in self.activation_data:
                self.activation_data[layer_idx] = defaultdict(int)

            for expert_idx in expert_indices:
                self.activation_data[layer_idx][expert_idx] += 1

            # Track token count
            self._track_token_count(args)

            return outputs

        return observed_forward

    def _extract_expert_indices(self, outputs: Any, block_info: MoEBlockInfo) -> List[int]:
        """
        Extract which experts were activated from forward output.

        This is architecture-specific and may need customization per model.

        Args:
            outputs: Output from the MoE layer forward
            block_info: Information about the MoE block

        Returns:
            List of expert indices that were activated
        """
        expert_indices = []

        # Method 1: Check for router output in tuple
        if isinstance(outputs, tuple) and len(outputs) > 1:
            for item in outputs:
                if hasattr(item, 'expert_indices'):
                    indices = item.expert_indices
                    if hasattr(indices, 'tolist'):
                        expert_indices.extend(indices.tolist())
                    else:
                        expert_indices.extend(list(indices))

        # Method 2: Try to find routing logits in outputs
        if isinstance(outputs, tuple):
            for item in outputs:
                if isinstance(item, mx.array) and item.ndim >= 2:
                    # Might be routing weights - take top-k
                    k = block_info.experts_per_tok if block_info.experts_per_tok > 0 else 4
                    top_k_indices = mx.argpartition(item, k=-k, axis=-1)[..., -k:]
                    expert_indices.extend(top_k_indices.flatten().tolist())

        # Method 3: Check layer attributes (some models store routing info)
        moe_module = self._get_moe_module_by_info(block_info)
        if moe_module is not None:
            for attr in ['last_expert_indices', 'expert_mask', 'selected_experts']:
                if hasattr(moe_module, attr):
                    indices = getattr(moe_module, attr)
                    if isinstance(indices, mx.array):
                        expert_indices.extend(indices.flatten().tolist())
                    elif isinstance(indices, list):
                        expert_indices.extend(indices)

        return list(set(int(idx) for idx in expert_indices if idx >= 0))

    def _track_token_count(self, args: tuple):
        """Track number of tokens processed."""
        # First arg is usually hidden_states
        if args and hasattr(args[0], 'shape'):
            shape = args[0].shape
            if len(shape) >= 2:
                # (batch_size, seq_len, ...)
                self.token_count += shape[0] * shape[1]

    def stop_observing(self):
        """
        Remove observation wrappers and restore original forward methods.

        This restores the model to its original state.
        """
        logger.info("Stopping MLX observer...")

        for layer_idx, block_info in self.moe_blocks.items():
            moe_module = self._get_moe_module_by_info(block_info)
            if moe_module is not None and layer_idx in self.original_forward:
                moe_module.forward = self.original_forward[layer_idx]

        self.original_forward.clear()
        logger.info("Observer stopped, original forward methods restored")

    def get_metrics(self) -> ObserverMetrics:
        """
        Calculate and return observer metrics.

        Returns:
            ObserverMetrics with all collected data
        """
        expert_activations = []

        for layer_idx, expert_counts in self.activation_data.items():
            block_info = self.moe_blocks.get(layer_idx)
            if block_info is None:
                continue

            total_layer_activations = sum(expert_counts.values())

            for expert_idx, count in expert_counts.items():
                frequency = count / max(self.token_count, 1)

                expert_activations.append(ExpertActivation(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    activation_count=count,
                    tokens_processed=self.token_count,
                    activation_frequency=frequency,
                ))

        layer_stats = {}
        for layer_idx, expert_counts in self.activation_data.items():
            block_info = self.moe_blocks.get(layer_idx)
            num_experts = block_info.num_experts if block_info else 0

            total = sum(expert_counts.values())
            active = len(expert_counts)

            layer_stats[layer_idx] = LayerStats(
                layer_idx=layer_idx,
                total_activations=total,
                active_experts=active,
                avg_activation_per_expert=total / max(active, 1),
                expert_counts=dict(expert_counts),
            )

        return ObserverMetrics(
            total_tokens=self.token_count,
            expert_activations=expert_activations,
            layer_wise_stats=layer_stats,
            temporal_activations=[],
        )

    def reset(self):
        """Clear all collected metrics."""
        self.activation_data = defaultdict(lambda: defaultdict(int))
        self.token_count = 0

    def process_batch(self, inputs: mx.array, outputs: mx.array, layer_idx: int):
        """
        Process a batch through the observer.

        This is called manually when not using hooks.
        For MLX, we use the wrapped forward methods instead.

        Args:
            inputs: Input tensors
            outputs: Output tensors
            layer_idx: Layer index
        """
        # Token counting
        if hasattr(inputs, 'shape'):
            shape = inputs.shape
            if len(shape) >= 2:
                self.token_count += shape[0] * shape[1]

    def record_activation(self, layer_idx: int, expert_idx: int, count: int = 1):
        """
        Record an expert activation.

        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            count: Number of activations to record
        """
        self.activation_data[layer_idx][expert_idx] += count

    def get_layer_expert_indices(self, layer_output: Any, layer_idx: int) -> List[int]:
        """
        Extract expert indices from layer output.

        Args:
            layer_output: Output from MoE layer
            layer_idx: Layer index

        Returns:
            List of expert indices
        """
        block_info = self.moe_blocks.get(layer_idx)
        if block_info is None:
            return []

        return self._extract_expert_indices(layer_output, block_info)
