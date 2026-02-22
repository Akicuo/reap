"""MLX model loader for REAP framework.

⚠️ UNTESTED IMPLEMENTATION ⚠️
This code was developed without access to Apple Silicon hardware for runtime testing.
The implementation follows MLX documentation patterns but has not been verified on actual hardware.
Please test on Mac Studio M3 Ultra and report any issues.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import os

from ...core.base_backend import BaseBackend, ModelInfo

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available. Install with: pip install mlx")

try:
    from mlx_lm import load, generate
    from mlx_lm.utils import load_config
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    logger.warning("mlx-lm not available. Install with: pip install mlx-lm")


class MlxBackend(BaseBackend):
    """
    MLX backend implementation for REAP.

    Uses mlx-lm for loading models and MLX arrays for computation.
    Optimized for Apple Silicon (M1/M2/M3 chips).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MLX backend.

        Args:
            config: Configuration dictionary
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for MlxBackend. Install with: pip install mlx")
        if not MLX_LM_AVAILABLE:
            raise ImportError("mlx-lm is required for MlxBackend. Install with: pip install mlx-lm")

        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._model_info: Optional[ModelInfo] = None

    def _get_device(self) -> Any:
        """
        Get MLX device.

        MLX uses Metal automatically on Apple Silicon.
        Can specify GPU with mx.set_default_device(mx.gpu).
        """
        if mx.metal.is_available():
            return mx.gpu
        return mx.cpu

    def load_model(self, model_path: str) -> Tuple[Any, ModelInfo]:
        """
        Load MLX model from mlx-community or local path.

        Args:
            model_path: Path to MLX model (e.g., "mlx-community/DeepSeek-Coder-V2-Lite-Instruct")

        Returns:
            Tuple of (model, ModelInfo)
        """
        logger.info(f"Loading MLX model from: {model_path}")

        # Load model and tokenizer using mlx-lm
        self._model, self._tokenizer = load(model_path)

        # Extract model configuration
        config = load_config(model_path)

        # Build ModelInfo
        self._model_info = ModelInfo(
            name=config.get("model_type", config.get("architectures", ["unknown"])[0] if isinstance(config.get("architectures"), list) else "unknown"),
            num_experts=self._extract_num_experts(config),
            num_layers=config.get("num_hidden_layers", config.get("n_layer", 0)),
            hidden_size=config.get("hidden_size", config.get("n_embd", 0)),
            expert_config=self._extract_expert_config(config),
            num_experts_per_tok=config.get("num_experts_per_tok", config.get("moe_expert_k", 0)),
            moe_layer_interval=config.get("moe_layer_interval", 1),
            expert_intermediate_size=config.get("expert_intermediate_size", 0),
            fused_experts=self._check_fused_experts(config),
            moe_block_location=self._detect_moe_block_location(config),
        )

        logger.info(f"Loaded {self._model_info.name} with {self._model_info.num_experts} experts per layer")
        return self._model, self._model_info

    def load_tokenizer(self, model_path: str) -> Any:
        """
        Load tokenizer for the model.

        Args:
            model_path: Path to model

        Returns:
            Tokenizer object
        """
        if self._tokenizer is None:
            from mlx_lm import load
            _, self._tokenizer = load(model_path)
        return self._tokenizer

    def _extract_num_experts(self, config: Dict[str, Any]) -> int:
        """
        Extract number of experts from config.

        MoE architectures vary in config structure.
        """
        # Try common config keys
        for key in ["num_local_experts", "moe_num_experts", "n_routed_experts",
                    "expert_layer_num", "num_experts"]:
            if key in config:
                return int(config[key])

        # Try nested config
        for key in ["moe", "ffn", "mlp"]:
            if key in config and isinstance(config[key], dict):
                for subkey in ["num_experts", "n_experts", "num_local_experts"]:
                    if subkey in config[key]:
                        return int(config[key][subkey])

        return 0

    def _extract_expert_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract MoE-specific configuration."""
        return {
            "num_experts_per_tok": config.get("num_experts_per_tok", config.get("moe_expert_k", 0)),
            "expert_intermediate_size": config.get("expert_intermediate_size", 0),
            "moe_layer_interval": config.get("moe_layer_interval", 1),
            "capacity_factor": config.get("capacity_factor", 1.0),
            "drop_tokens": config.get("drop_tokens", False),
        }

    def _check_fused_experts(self, config: Dict[str, Any]) -> bool:
        """
        Check if model uses fused experts.

        Fused experts combine gate_proj and up_proj into a single tensor.
        """
        # Check for fused config flags
        for key in ["fused_experts", "use_fused_moe", "fused_mlp"]:
            if config.get(key, False):
                return True

        # Check architecture name
        model_type = config.get("model_type", "").lower()
        arch = config.get("architectures", [])
        if isinstance(arch, list) and arch:
            model_type = arch[0].lower()

        fused_patterns = ["llama4", "glm4.*lite", "flash"]
        import re
        for pattern in fused_patterns:
            if re.search(pattern, model_type):
                return True

        return False

    def _detect_moe_block_location(self, config: Dict[str, Any]) -> str:
        """
        Detect where MoE blocks are located in the model.

        Returns: "mlp", "block_sparse_moe", "feed_forward", or "moe"
        """
        model_type = config.get("model_type", "").lower()
        arch = config.get("architectures", [])
        if isinstance(arch, list) and arch:
            model_type = arch[0].lower()

        # Architecture-based detection
        if "llama" in model_type:
            return "feed_forward"
        elif "mixtral" in model_type:
            return "block_sparse_moe"
        elif "qwen" in model_type or "deepseek" in model_type:
            return "mlp"
        elif "minimax" in model_type:
            return "block_sparse_moe"

        # Default
        return "mlp"

    def get_expert_weights(self, model: Any, layer_idx: int, expert_idx: int) -> Dict[str, Any]:
        """
        Extract weights for a specific expert.

        Args:
            model: The model
            layer_idx: Layer index
            expert_idx: Expert index within layer

        Returns:
            Dictionary mapping weight names to MLX arrays
        """
        moe_block = self._get_moe_block(model, layer_idx)
        if moe_block is None:
            raise ValueError(f"No MoE block found at layer {layer_idx}")

        weights = {}

        # Get the expert module
        experts = self._get_experts_module(moe_block)
        if experts is None or expert_idx >= len(experts):
            raise ValueError(f"Expert {expert_idx} not found in layer {layer_idx}")

        if self._model_info and self._model_info.fused_experts:
            # Fused experts: gate_up_proj is combined
            if hasattr(experts, 'gate_up_proj'):
                gate_up_weight = experts.gate_up_proj[expert_idx]
                weights['gate_up_proj.weight'] = gate_up_weight
            if hasattr(experts, 'down_proj'):
                down_weight = experts.down_proj[expert_idx]
                weights['down_proj.weight'] = down_weight
        else:
            # Non-fused: separate gate, up, down
            expert = experts[expert_idx]

            # Try common projection names
            for proj_name in ['gate_proj', 'w1', 'gate']:
                if hasattr(expert, proj_name):
                    weights[f'{proj_name}.weight'] = getattr(expert, proj_name).weight
                    break

            for proj_name in ['up_proj', 'w3', 'up']:
                if hasattr(expert, proj_name):
                    weights[f'{proj_name}.weight'] = getattr(expert, proj_name).weight
                    break

            for proj_name in ['down_proj', 'w2', 'down']:
                if hasattr(expert, proj_name):
                    weights[f'{proj_name}.weight'] = getattr(expert, proj_name).weight
                    break

        return weights

    def set_expert_weights(self, model: Any, layer_idx: int, expert_idx: int, weights: Dict[str, Any]):
        """
        Set weights for a specific expert.

        Note: MLX arrays are immutable, so we need to update parameters dict
        and call model.update().

        Args:
            model: The model
            layer_idx: Layer index
            expert_idx: Expert index within layer
            weights: Dictionary mapping weight names to MLX arrays
        """
        moe_block = self._get_moe_block(model, layer_idx)
        if moe_block is None:
            raise ValueError(f"No MoE block found at layer {layer_idx}")

        experts = self._get_experts_module(moe_block)
        if experts is None or expert_idx >= len(experts):
            raise ValueError(f"Expert {expert_idx} not found in layer {layer_idx}")

        # Update weights
        # MLX uses tree-based parameter updates
        params = model.parameters()

        for weight_name, weight_value in weights.items():
            # Build parameter path
            # Example: layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight
            param_path = self._build_param_path(layer_idx, expert_idx, weight_name)
            params[param_path] = weight_value

        # Update model parameters
        model.update(params)
        mx.eval(model.parameters())

    def remove_expert(self, model: Any, layer_idx: int, expert_idx: int):
        """
        Remove an expert from the model.

        This is complex and may require:
        1. Removing expert from the experts list/array
        2. Updating routing logic
        3. Adjusting normalization

        Args:
            model: The model
            layer_idx: Layer index
            expert_idx: Expert index within layer
        """
        moe_block = self._get_moe_block(model, layer_idx)
        if moe_block is None:
            raise ValueError(f"No MoE block found at layer {layer_idx}")

        experts = self._get_experts_module(moe_block)
        if experts is None:
            raise ValueError(f"No experts module found in layer {layer_idx}")

        # For ModuleList, we can remove by index
        if hasattr(experts, '__delitem__'):
            del experts[expert_idx]

        # Update num_experts in config
        if hasattr(moe_block, 'num_local_experts'):
            moe_block.num_local_experts = len(experts)
        elif hasattr(moe_block, 'num_experts'):
            moe_block.num_experts = len(experts)

        # Update gate/router output dimension if needed
        router = self._get_router(moe_block)
        if router is not None and hasattr(router, 'weight'):
            gate_weight = router.weight
            # Remove the row corresponding to this expert
            router.weight = mx.concatenate([
                gate_weight[:expert_idx],
                gate_weight[expert_idx+1:]
            ], axis=0)

        # Materialize changes
        mx.eval(model.parameters())

    def get_num_experts(self, model: Any, layer_idx: int) -> int:
        """
        Get number of experts in a layer.

        Args:
            model: The model
            layer_idx: Layer index

        Returns:
            Number of experts
        """
        if self._model_info and self._model_info.num_experts > 0:
            return self._model_info.num_experts

        moe_block = self._get_moe_block(model, layer_idx)
        if moe_block is None:
            return 0

        # Try config attributes
        for attr in ['num_local_experts', 'num_experts', 'n_routed_experts', 'moe_num_experts']:
            if hasattr(moe_block, attr):
                return getattr(moe_block, attr)

        # Count from experts module
        experts = self._get_experts_module(moe_block)
        if experts is not None:
            if hasattr(experts, '__len__'):
                return len(experts)
            elif hasattr(experts, 'num_experts'):
                return experts.num_experts

        return 0

    def save_model(self, model: Any, save_path: str):
        """
        Save model to disk in MLX format.

        Args:
            model: The model
            save_path: Path to save directory
        """
        from mlx_lm.utils import save_weights

        # Create directory if needed
        os.makedirs(save_path, exist_ok=True)

        # Save weights
        save_weights(save_path, model.parameters())

        logger.info(f"Saved model to {save_path}")

    def create_optimizer(self, params: Any, lr: float) -> Any:
        """
        Create MLX optimizer.

        Args:
            params: Model parameters
            lr: Learning rate

        Returns:
            MLX optimizer
        """
        return mx.optim.AdamW(learning_rate=lr)

    def empty_cache(self):
        """Clear MLX cache to free memory."""
        # MLX manages memory automatically
        # Explicit evaluation can help free memory
        mx.eval([])

    def get_device_count(self) -> int:
        """Get number of available MLX devices."""
        # MLX on Apple Silicon has unified memory
        # Returns 1 if Metal is available, 0 otherwise
        return 1 if mx.metal.is_available() else 0

    # Helper methods

    def _get_moe_block(self, model: Any, layer_idx: int) -> Optional[Any]:
        """Get the MoE block for a given layer."""
        # Common model structures
        for attr in ['model', 'layers', 'transformer']:
            if hasattr(model, attr):
                container = getattr(model, attr)
                if hasattr(container, 'layers') and layer_idx < len(container.layers):
                    layer = container.layers[layer_idx]

                    # Try common MoE block locations
                    for moe_attr in ['mlp', 'moe', 'block_sparse_moe', 'feed_forward']:
                        if hasattr(layer, moe_attr):
                            return getattr(layer, moe_attr)

        return None

    def _get_experts_module(self, moe_block: Any) -> Optional[Any]:
        """Get the experts module from an MoE block."""
        for attr in ['experts', 'local_experts', 'expert']:
            if hasattr(moe_block, attr):
                return getattr(moe_block, attr)
        return None

    def _get_router(self, moe_block: Any) -> Optional[Any]:
        """Get the router/gate module from an MoE block."""
        for attr in ['gate', 'router', 'gating_network', 'switch']:
            if hasattr(moe_block, attr):
                return getattr(moe_block, attr)
        return None

    def _build_param_path(self, layer_idx: int, expert_idx: int, weight_name: str) -> str:
        """Build parameter path for weight updates."""
        # Remove .weight suffix if present
        name = weight_name.replace('.weight', '')

        # Common path patterns
        if self._model_info:
            moe_loc = self._model_info.moe_block_location
            return f"layers.{layer_idx}.{moe_loc}.experts.{expert_idx}.{name}"

        return f"layers.{layer_idx}.mlp.experts.{expert_idx}.{name}"
