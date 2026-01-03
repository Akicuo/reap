"""
Compatibility shims for HuggingFace `transformers` + `trust_remote_code`.

Some model repos expect different signatures for internal utilities across
Transformers versions. Example: Solar-Open uses `@check_model_inputs()` as a
decorator factory, but some Transformers versions expose `check_model_inputs`
as a plain decorator requiring `func`.

We patch these in-process *before* importing remote code via from_pretrained().
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

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
    """Apply all known compat shims (safe to call multiple times)."""
    _patch_check_model_inputs()


