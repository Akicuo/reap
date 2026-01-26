"""
Compatibility shims for HuggingFace `transformers` + `trust_remote_code`.

Some model repos expect different signatures for internal utilities across
Transformers versions. Example: Solar-Open uses `@check_model_inputs()` as a
decorator factory, but some Transformers versions expose `check_model_inputs`
as a plain decorator requiring `func`.

We patch these in-process *before* importing remote code via from_pretrained().

IMPORTANT: Some patches (like rope_config_validation) need to be applied BEFORE
any model-specific modules are imported, because they do top-level imports.
We apply critical patches immediately when this module is imported.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)


def _early_patch_rope_config_validation() -> None:
    """
    EARLY patch for rope_config_validation - must run before any model code imports.
    
    This function was deprecated and removed in newer transformers versions.
    Models like LongCat-Flash-Thinking-2601 import it at module level.
    """
    try:
        from transformers import modeling_rope_utils as mru
    except ImportError:
        return
    
    if hasattr(mru, "rope_config_validation"):
        return
    
    def rope_config_validation(config, rope_scaling=None):
        """Compatibility shim - validation moved to RotaryEmbeddingConfigMixin.validate_rope"""
        pass
    
    mru.rope_config_validation = rope_config_validation
    logger.debug("[transformers_compat] Early-patched rope_config_validation")


# Apply critical patches IMMEDIATELY when this module is imported
_early_patch_rope_config_validation()

F = TypeVar("F", bound=Callable[..., Any])


def _patch_check_model_inputs() -> None:
    try:
        # location used by many releases
        from transformers import modeling_utils as mu  # type: ignore
    except Exception:
        return

    if not hasattr(mu, "check_model_inputs"):
        return

    orig = mu.check_model_inputs
    try:
        sig = inspect.signature(orig)
    except Exception:
        return

    # If it already supports being called with no args, we do nothing.
    # We specifically patch the case where signature is (func, ...) and the
    # remote code uses @check_model_inputs().
    params = list(sig.parameters.values())
    if not params:
        return

    first = params[0]
    if first.default is not inspect._empty:
        # Already optional (factory-like)
        return

    def check_model_inputs_compat(func: F | None = None, *args: Any, **kwargs: Any):
        # If used as @check_model_inputs() -> func is None
        if func is None:
            def decorator(f: F) -> F:
                return orig(f, *args, **kwargs)

            return decorator
        # If used as @check_model_inputs -> func is callable
        return orig(func, *args, **kwargs)

    mu.check_model_inputs = check_model_inputs_compat  # type: ignore[assignment]
    logger.info("[transformers_compat] Patched transformers.modeling_utils.check_model_inputs()")


def apply_transformers_compat_patches() -> None:
    """Apply all known compat shims (safe to call multiple times).
    
    Note: rope_config_validation is patched immediately at module import time,
    not here, because it needs to be available before any model code imports.
    """
    _patch_check_model_inputs()
    # rope_config_validation is already patched at import time via _early_patch_rope_config_validation()


