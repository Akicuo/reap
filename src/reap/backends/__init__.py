"""Backend registry for REAP framework."""

from typing import Dict, Type, List, Optional
import logging

from ..core.base_backend import BaseBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """
    Registry for available backends (PyTorch, MLX, etc.).

    Backends are registered by name and can be retrieved for instantiation.
    """

    _backends: Dict[str, Type[BaseBackend]] = {}
    _backend_info: Dict[str, Dict[str, str]] = {}

    @classmethod
    def register(cls, name: str, backend_class: Type[BaseBackend],
                description: str = "", requires_device: str = "any"):
        """
        Register a backend.

        Args:
            name: Backend name (e.g., "torch", "mlx")
            backend_class: Backend class
            description: Human-readable description
            requires_device: Device requirement ("cuda", "metal", "any")
        """
        cls._backends[name] = backend_class
        cls._backend_info[name] = {
            "description": description,
            "requires_device": requires_device,
        }
        logger.info(f"Registered backend: {name}")

    @classmethod
    def get_backend(cls, name: str) -> Type[BaseBackend]:
        """
        Get a backend class by name.

        Args:
            name: Backend name

        Returns:
            Backend class

        Raises:
            ValueError: If backend not found
        """
        if name not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(
                f"Unknown backend: {name}. Available backends: {available}"
            )
        return cls._backends[name]

    @classmethod
    def list_backends(cls) -> List[str]:
        """Get list of registered backend names."""
        return list(cls._backends.keys())

    @classmethod
    def get_backend_info(cls, name: str) -> Optional[Dict[str, str]]:
        """
        Get information about a backend.

        Args:
            name: Backend name

        Returns:
            Dictionary with backend info or None
        """
        return cls._backend_info.get(name)

    @classmethod
    def is_available(cls, name: str) -> bool:
        """
        Check if a backend is available on the current system.

        Args:
            name: Backend name

        Returns:
            True if backend is available
        """
        if name not in cls._backends:
            return False

        backend_info = cls._backend_info.get(name, {})
        device_requirement = backend_info.get("requires_device", "any")

        if device_requirement == "cuda":
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        elif device_requirement == "metal":
            try:
                import mlx.core as mx
                return mx.metal.is_available()
            except ImportError:
                return False

        return True


def get_available_backends() -> List[str]:
    """Get list of backends available on the current system."""
    return [name for name in BackendRegistry.list_backends()
            if BackendRegistry.is_available(name)]


def auto_detect_backend() -> Optional[str]:
    """
    Auto-detect the best available backend.

    Prefers MLX on Apple Silicon, PyTorch/CUDA on NVIDIA GPUs.

    Returns:
        Backend name or None
    """
    # Check for Apple Silicon
    try:
        import platform
        if platform.processor() == "arm" and BackendRegistry.is_available("mlx"):
            return "mlx"
    except Exception:
        pass

    # Check for CUDA
    if BackendRegistry.is_available("torch"):
        try:
            import torch
            if torch.cuda.is_available():
                return "torch"
        except Exception:
            pass

    # Fall back to first available
    available = get_available_backends()
    return available[0] if available else None
