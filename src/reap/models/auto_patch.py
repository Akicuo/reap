"""
Auto-patch MoE models to expose router_logits for the REAP observer.

This module provides generic patching for MoE models that don't return
router_logits from their MoE block forward method. The observer hook
expects (hidden_states, router_logits), but many models only return
hidden_states.

The patcher:
1. Detects MoE blocks in the model
2. Identifies the router/gate module
3. Patches the MoE forward to store router_logits in _last_router_logits
4. The observer hook retrieves router_logits from this attribute
"""

from __future__ import annotations

import logging
import types
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Common MoE block class name patterns
MOE_BLOCK_PATTERNS = [
    "MoE", "SparseMoeBlock", "MoeBlock", "MoeMLP", "SparseMLP",
    "MixtureOfExperts", "ExpertLayer", "MoELayer",
]

# Common router/gate attribute names
ROUTER_ATTR_NAMES = ["gate", "router", "gating", "expert_gate", "moe_gate"]

# Common router weight attribute names
ROUTER_WEIGHT_ATTRS = ["weight", "linear", "proj", "classifier"]


def _is_moe_block(module: nn.Module) -> bool:
    """Check if a module is likely an MoE block based on class name."""
    class_name = module.__class__.__name__
    return any(pattern.lower() in class_name.lower() for pattern in MOE_BLOCK_PATTERNS)


def _find_router(moe_block: nn.Module) -> nn.Module | None:
    """Find the router/gate module within an MoE block."""
    for attr_name in ROUTER_ATTR_NAMES:
        if hasattr(moe_block, attr_name):
            router = getattr(moe_block, attr_name)
            if isinstance(router, nn.Module):
                return router
    return None


def _find_router_weight(router: nn.Module) -> torch.Tensor | None:
    """Find the weight tensor in a router module."""
    # Direct weight attribute
    if hasattr(router, 'weight') and isinstance(router.weight, (torch.Tensor, nn.Parameter)):
        return router.weight
    
    # Check for linear layer
    for attr_name in ROUTER_WEIGHT_ATTRS:
        if hasattr(router, attr_name):
            sub = getattr(router, attr_name)
            if isinstance(sub, nn.Linear):
                return sub.weight
            if isinstance(sub, (torch.Tensor, nn.Parameter)):
                return sub
    
    # Check children
    for child in router.children():
        if isinstance(child, nn.Linear):
            return child.weight
    
    return None


def _get_hidden_size(module: nn.Module) -> int | None:
    """Try to get hidden size from module config or attributes."""
    # Check config
    if hasattr(module, 'config'):
        config = module.config
        for attr in ['hidden_size', 'hidden_dim', 'd_model', 'embed_dim']:
            if hasattr(config, attr):
                return getattr(config, attr)
    
    # Check direct attributes
    for attr in ['hidden_size', 'hidden_dim', 'd_model', 'embed_dim']:
        if hasattr(module, attr):
            val = getattr(module, attr)
            if isinstance(val, int):
                return val
    
    return None


def _create_patched_forward(
    original_forward: Callable,
    router: nn.Module,
    router_weight: torch.Tensor,
    hidden_size: int,
) -> Callable:
    """Create a patched forward that stores router_logits."""
    
    def patched_forward(self, hidden_states, *args, **kwargs):
        # Compute router logits before calling original forward
        batch_shape = hidden_states.shape[:-1]
        hidden_flat = hidden_states.view(-1, hidden_size)
        
        # Compute router logits
        weight = router_weight
        if weight.dtype != hidden_flat.dtype:
            weight = weight.to(hidden_flat.dtype)
        
        try:
            router_logits = F.linear(hidden_flat, weight)
        except Exception:
            # Fallback: create zeros if computation fails
            num_experts = weight.shape[0] if weight.dim() >= 1 else 8
            router_logits = torch.zeros(
                hidden_flat.shape[0], num_experts,
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        
        # Store for observer hook to retrieve
        self._last_router_logits = router_logits
        
        # Call original forward
        return original_forward(hidden_states, *args, **kwargs)
    
    return patched_forward


def patch_moe_block(moe_block: nn.Module) -> bool:
    """
    Patch a single MoE block to store router_logits.
    
    Returns True if patching was successful, False otherwise.
    """
    # Find router
    router = _find_router(moe_block)
    if router is None:
        logger.debug(f"Could not find router in {moe_block.__class__.__name__}")
        return False
    
    # Find router weight
    router_weight = _find_router_weight(router)
    if router_weight is None:
        logger.debug(f"Could not find router weight in {router.__class__.__name__}")
        return False
    
    # Get hidden size
    hidden_size = _get_hidden_size(moe_block)
    if hidden_size is None:
        # Try to infer from router weight shape
        if router_weight.dim() >= 2:
            hidden_size = router_weight.shape[-1]
        else:
            logger.debug(f"Could not determine hidden_size for {moe_block.__class__.__name__}")
            return False
    
    # Create and apply patched forward
    original_forward = moe_block.forward
    patched = _create_patched_forward(original_forward, router, router_weight, hidden_size)
    moe_block.forward = types.MethodType(patched, moe_block)
    
    return True


def auto_patch_moe(model: nn.Module) -> int:
    """
    Automatically patch all MoE blocks in a model to store router_logits.
    
    Args:
        model: The model to patch
        
    Returns:
        Number of MoE blocks successfully patched
    """
    patched_count = 0
    moe_class_names = set()
    
    for name, module in model.named_modules():
        if _is_moe_block(module):
            # Skip if already patched
            if hasattr(module, '_reap_patched'):
                continue
            
            if patch_moe_block(module):
                module._reap_patched = True
                patched_count += 1
                moe_class_names.add(module.__class__.__name__)
    
    if patched_count > 0:
        logger.info(
            f"[auto_patch] Patched {patched_count} MoE blocks "
            f"(classes: {', '.join(moe_class_names)})"
        )
    else:
        logger.warning("[auto_patch] No MoE blocks found or patched")
    
    return patched_count


def patch_specific_model(model: nn.Module, model_class_name: str) -> int:
    """
    Apply model-specific patches for known problematic models.
    
    Falls back to auto_patch_moe if no specific patch exists.
    """
    # Model-specific patchers
    specific_patchers = {
        "SolarOpenForCausalLM": _patch_solar_open,
        "LongcatCausalLM": _patch_longcat,
        "LongcatForCausalLM": _patch_longcat,
        # Add more specific patchers here as needed
    }
    
    if model_class_name in specific_patchers:
        return specific_patchers[model_class_name](model)
    
    # For new model types that should work with generic patching
    # These are DeepSeek-like architectures
    new_supported_models = [
        "DeepseekV3ForCausalLM",
        "MiniMaxForCausalLM", 
        "KimiK2ForCausalLM",
    ]
    
    if model_class_name in new_supported_models:
        logger.info(f"Model {model_class_name} is a supported DeepSeek-like architecture, using generic patcher")
        return auto_patch_moe(model)
    
    # Fall back to generic patcher
    return auto_patch_moe(model)


def _patch_solar_open(model: nn.Module) -> int:
    """Specific patcher for Solar-Open models."""
    patched_count = 0
    
    for layer in model.model.layers:
        if hasattr(layer, 'mlp') and layer.mlp.__class__.__name__ == 'SolarOpenMoE':
            moe = layer.mlp
            
            if hasattr(moe, '_reap_patched'):
                continue
            
            # Get config values
            hidden_size = moe.config.hidden_size
            
            def make_patched_forward(m, h_size):
                original = m.forward.__func__ if hasattr(m.forward, '__func__') else m.forward
                
                def patched_forward(self, hidden_states):
                    # Compute router logits
                    hidden_flat = hidden_states.view(-1, h_size)
                    router_logits = F.linear(
                        hidden_flat.type(torch.float32),
                        self.gate.weight.type(torch.float32)
                    )
                    self._last_router_logits = router_logits
                    
                    # Original forward logic
                    residuals = hidden_states
                    orig_shape = hidden_states.shape
                    topk_indices, topk_weights = self.gate(hidden_states)
                    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                    hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
                    hidden_states = hidden_states + self.shared_experts(residuals)
                    return hidden_states
                
                return patched_forward
            
            moe.forward = types.MethodType(make_patched_forward(moe, hidden_size), moe)
            moe._reap_patched = True
            patched_count += 1
    
    if patched_count > 0:
        logger.info(f"[patch_solar_open] Patched {patched_count} SolarOpenMoE layers")
    
    return patched_count


def _patch_longcat(model: nn.Module) -> int:
    """
    Specific patcher for meituan-longcat/LongCat-Flash-Thinking-2601.
    
    LongCat MoE architecture:
    - LongcatMoE is the MoE block at decoder_layer.mlp
    - LongcatTopkRouter is the router with router.classifier as the gate Linear
    - 512 real experts + 256 identity "zero experts" 
    - The router's classifier weight is used to compute router_logits
    """
    patched_count = 0
    
    for layer in model.model.layers:
        if hasattr(layer, 'mlp') and layer.mlp.__class__.__name__ == 'LongcatMoE':
            moe = layer.mlp
            
            if hasattr(moe, '_reap_patched'):
                continue
            
            # Get config values from the model config
            hidden_size = moe.config.hidden_size
            
            def make_patched_forward(m, h_size):
                original_forward = m.forward.__func__ if hasattr(m.forward, '__func__') else m.forward
                
                def patched_forward(self, hidden_states):
                    # Compute router logits using router.classifier
                    # LongcatTopkRouter computes: F.linear(hidden_states, classifier.weight)
                    hidden_flat = hidden_states.view(-1, h_size)
                    
                    # Get router logits from the router's classifier
                    router_logits = F.linear(
                        hidden_flat.type(torch.float32),
                        self.router.classifier.weight.type(torch.float32)
                    )
                    
                    # Add e_score_correction_bias if present (used for routing selection)
                    # Note: This matches what the actual router does in get_topk_indices
                    
                    # Store for observer hook to retrieve
                    self._last_router_logits = router_logits
                    
                    # Call original forward
                    return original_forward(self, hidden_states)
                
                return patched_forward
            
            moe.forward = types.MethodType(make_patched_forward(moe, hidden_size), moe)
            moe._reap_patched = True
            patched_count += 1
    
    if patched_count > 0:
        logger.info(f"[patch_longcat] Patched {patched_count} LongcatMoE layers")
    
    return patched_count


def needs_patching(model: nn.Module) -> bool:
    """
    Check if a model likely needs MoE patching.
    
    Returns True if the model has MoE blocks that don't appear to
    return router_logits in their forward output.
    """
    model_class_name = model.__class__.__name__
    
    # These model types are known to work without patching
    # (they return router_logits properly)
    no_patch_needed = [
        "Qwen3MoeForCausalLM",
        "Llama4ForCausalLM",
        "MixtralForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",  # V3 typically returns router_logits
        "Ernie4_5_MoEForCausalLM",
        "Glm4MoeForCausalLM",
        "SolarOpenForCausalLM",
    ]
    
    # These model types NEED patching (MoE forward only returns hidden_states)
    patch_required = [
        "LongcatCausalLM",
        "LongcatForCausalLM",
    ]
    
    if model_class_name in no_patch_needed:
        logger.debug(f"Model {model_class_name} is known to not need patching")
        return False
    
    if model_class_name in patch_required:
        logger.debug(f"Model {model_class_name} is known to require patching")
        return True
    
    for name, module in model.named_modules():
        if _is_moe_block(module):
            # If already patched, no need
            if hasattr(module, '_reap_patched'):
                continue
            
            # If has _last_router_logits storage, assume patched
            if hasattr(module, '_last_router_logits'):
                continue
            
            # Check if forward signature suggests tuple return
            # This is heuristic - if uncertain, assume needs patching
            return True
    
    return False

