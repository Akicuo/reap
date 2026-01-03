"""
Model-specific patches and utilities for REAP.

This module contains:
- auto_patch: Automatic MoE patching for router_logits exposure
- Model-specific modeling files for patched models
"""

from reap.models.auto_patch import (
    auto_patch_moe,
    patch_specific_model,
    needs_patching,
    patch_moe_block,
)

__all__ = [
    "auto_patch_moe",
    "patch_specific_model", 
    "needs_patching",
    "patch_moe_block",
]

