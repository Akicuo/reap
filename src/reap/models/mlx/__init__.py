"""MLX-specific model adapters for REAP framework."""

from .deepseek_mlx import DeepSeekMlxAdapter
from .mixtral_mlx import MixtralMlxAdapter
from .qwen_mlx import QwenMlxAdapter

__all__ = [
    "DeepSeekMlxAdapter",
    "MixtralMlxAdapter",
    "QwenMlxAdapter",
]

# Registry for model adapters
MLX_ADAPTERS = {
    "DeepseekV2ForCausalLM": DeepSeekMlxAdapter,
    "DeepseekV3ForCausalLM": DeepSeekMlxAdapter,
    "MixtralForCausalLM": MixtralMlxAdapter,
    "Qwen3MoeForCausalLM": QwenMlxAdapter,
}


def get_mlx_adapter(model_class_name: str):
    """Get MLX adapter for a model class."""
    return MLX_ADAPTERS.get(model_class_name)
