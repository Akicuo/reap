"""MLX-specific adapters for Mixtral MoE models."""

from ...core.observer_base import BaseModelAdapter, MoEBlockInfo
from typing import Any, Dict, List, Optional
import mlx.nn as nn


class MixtralMlxAdapter(BaseModelAdapter):
    """MLX adapter for Mixtral MoE models."""

    def get_num_experts(self, model: Any, layer_idx: int) -> int:
        """Get number of experts in Mixtral layer."""
        moe_block = self._get_moe_block(model, layer_idx)
        if moe_block is None:
            return 0

        # Mixtral uses num_local_experts
        for attr in ['num_local_experts', 'num_experts']:
            if hasattr(moe_block, attr):
                return getattr(moe_block, attr)

        experts = self._get_experts_module(moe_block)
        if experts is not None and hasattr(experts, '__len__'):
            return len(experts)

        return 0

    def get_experts_per_tok(self, model: Any, layer_idx: int) -> int:
        """Get number of experts selected per token."""
        moe_block = self._get_moe_block(model, layer_idx)
        if moe_block is None:
            return 0

        for attr in ['num_experts_per_tok', 'top_k', 'local_experts_per_tok']:
            if hasattr(moe_block, attr):
                return getattr(moe_block, attr)

        return 0

    def is_fused(self, model: Any, layer_idx: int) -> bool:
        """Check if experts use fused weights."""
        moe_block = self._get_moe_block(model, layer_idx)
        if moe_block is None:
            return False

        experts = self._get_experts_module(moe_block)
        if experts is not None and hasattr(experts, 'gate_up_proj'):
            return True

        return False

    def find_moe_blocks(self, model: Any) -> List[MoEBlockInfo]:
        """Find all MoE blocks in the model."""
        blocks = []

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for i, layer in enumerate(model.model.layers):
                # Mixtral uses block_sparse_moe
                if hasattr(layer, 'block_sparse_moe'):
                    moe_block = layer.block_sparse_moe
                    blocks.append(MoEBlockInfo(
                        layer_idx=i,
                        module_path=f"model.layers.{i}.block_sparse_moe",
                        num_experts=self.get_num_experts(model, i),
                        experts_per_tok=self.get_experts_per_tok(model, i),
                        fused=self.is_fused(model, i),
                    ))

        return blocks

    def get_expert_indices_from_output(self, output: Any) -> List[int]:
        """Extract expert indices from layer output."""
        indices = []

        if isinstance(output, tuple):
            for item in output:
                if hasattr(item, 'expert_indices'):
                    indices.extend(item.expert_indices.tolist())

        return indices

    def get_moe_block(self, model: Any, layer_idx: int) -> Any:
        """Get the MoE block module for a layer."""
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            if layer_idx < len(model.model.layers):
                layer = model.model.layers[layer_idx]
                if hasattr(layer, 'block_sparse_moe'):
                    return layer.block_sparse_moe
        return None

    def _get_moe_block(self, model: Any, layer_idx: int) -> Optional[Any]:
        """Get MoE block for a layer."""
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            if layer_idx < len(model.model.layers):
                layer = model.model.layers[layer_idx]
                for attr in ['block_sparse_moe', 'moe', 'mlp']:
                    if hasattr(layer, attr):
                        return getattr(layer, attr)
        return None

    def _get_experts_module(self, moe_block: Any) -> Optional[Any]:
        """Get experts module from MoE block."""
        for attr in ['experts', 'local_experts']:
            if hasattr(moe_block, attr):
                return getattr(moe_block, attr)
        return None
