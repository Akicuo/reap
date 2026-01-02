"""
Model Analyzer for HuggingFace Models - NO MODEL LOADING

IMPORTANT: This module NEVER loads model weights into memory.
It uses safetensors metadata inspection and config.json parsing
to deeply analyze model architecture without requiring a GPU.

This is designed to help AI agents understand model structure and
dynamically add support for new MoE models by:
1. Analyzing tensor names to discover expert/router patterns
2. Extracting model configuration from config.json
3. Generating MODEL_ATTRS-compatible dictionaries for new models
4. Identifying MoE-specific components (experts, gates, routers)

Usage:
    python -m reap.model_analyzer /path/to/model/directory
    
    # Or programmatically:
    from reap.model_analyzer import ModelAnalyzer
    analyzer = ModelAnalyzer("/path/to/model")
    analysis = analyzer.analyze()
    print(analysis.suggest_model_attrs())
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional
import textwrap

# NOTE: We only use safetensors for metadata inspection - weights are NEVER loaded
from safetensors import safe_open
from safetensors.torch import safe_open as torch_safe_open

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class TensorInfo:
    """Information about a single tensor WITHOUT loading its data."""
    name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    
    @property
    def num_params(self) -> int:
        """Calculate number of parameters from shape."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result


@dataclass
class ExpertInfo:
    """Information about a discovered expert module."""
    layer_idx: int
    expert_idx: int
    tensors: list[TensorInfo] = field(default_factory=list)
    
    @property
    def param_count(self) -> int:
        return sum(t.num_params for t in self.tensors)


@dataclass 
class MoELayerInfo:
    """Information about a single MoE layer."""
    layer_idx: int
    num_experts: int
    experts: list[ExpertInfo] = field(default_factory=list)
    router_tensors: list[TensorInfo] = field(default_factory=list)
    has_shared_expert: bool = False
    shared_expert_tensors: list[TensorInfo] = field(default_factory=list)


@dataclass
class ModelAnalysis:
    """Complete analysis of a model's architecture."""
    model_path: pathlib.Path
    config: dict[str, Any]
    tensor_infos: list[TensorInfo]
    moe_layers: list[MoELayerInfo]
    
    # Discovered patterns
    expert_pattern: str | None = None
    router_pattern: str | None = None
    gate_proj_name: str | None = None
    up_proj_name: str | None = None
    down_proj_name: str | None = None
    moe_block_name: str | None = None
    
    # Model metadata
    model_type: str | None = None
    num_layers: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    is_fused_experts: bool = False
    
    def suggest_model_attrs(self) -> dict[str, Any]:
        """
        Generate a MODEL_ATTRS-compatible dictionary for this model.
        
        This can be used by AI agents to add support for new models
        by inserting into reap.model_util.MODEL_ATTRS.
        """
        return {
            "moe_block": self.moe_block_name or "mlp",
            "gate_proj": self.gate_proj_name or "gate_proj",
            "up_proj": self.up_proj_name or "up_proj",
            "down_proj": self.down_proj_name or "down_proj",
            "experts": "experts",
            "fused": self.is_fused_experts,
            "router": "gate",
            "num_experts": self._suggest_num_experts_attr(),
            "num_experts_per_tok": self._suggest_num_experts_per_tok_attr(),
        }
    
    def _suggest_num_experts_attr(self) -> str:
        """Suggest the config attribute name for num_experts."""
        config = self.config
        candidates = [
            "num_experts", "num_local_experts", "n_routed_experts",
            "moe_num_experts", "num_moe_experts", "n_experts"
        ]
        for attr in candidates:
            if attr in config:
                return attr
        return "num_experts"
    
    def _suggest_num_experts_per_tok_attr(self) -> str:
        """Suggest the config attribute name for num_experts_per_tok."""
        config = self.config
        candidates = [
            "num_experts_per_tok", "top_k", "moe_k", "num_selected_experts",
            "experts_per_tok", "k"
        ]
        for attr in candidates:
            if attr in config:
                return attr
        return "num_experts_per_tok"
    
    def suggest_observer_config(self) -> dict[str, Any]:
        """
        Generate an observer hook config for this model.
        
        This can be used by AI agents to add a new entry to 
        OBSERVER_CONFIG_REGISTRY in reap.observer.
        """
        # Try to determine the MoE block class name
        moe_class_name = self._infer_moe_class_name()
        
        return {
            "module_class_name_to_hook_regex": moe_class_name,
            "num_experts_attr_name": self._suggest_num_experts_attr(),
            "top_k_attr_name": self._suggest_num_experts_per_tok_attr(),
            "fused_experts": self.is_fused_experts,
        }
    
    def _infer_moe_class_name(self) -> str | None:
        """Attempt to infer the MoE block class name from model type."""
        model_type = self.config.get("model_type", "").lower()
        architectures = self.config.get("architectures", [])
        
        # Common patterns
        type_to_class = {
            "qwen3_moe": "Qwen3MoeSparseMoeBlock",
            "qwen2_moe": "Qwen2MoeSparseMoeBlock",
            "mixtral": "MixtralSparseMoeBlock",
            "deepseek_v2": "DeepseekV2MoE",
            "llama4": "Llama4TextMoe",
        }
        
        for key, cls in type_to_class.items():
            if key in model_type:
                return cls
        
        # Fallback: generate from architecture name
        if architectures:
            arch = architectures[0]
            if "moe" in arch.lower() and "ForCausalLM" in arch:
                return f"{arch.replace('ForCausalLM', '')}SparseMoeBlock"
            # If we can't infer a block, at least return the model class so users can refine
            return arch

        # Last resort: build a class-like string from model_type
        if model_type:
            cleaned = "".join(part.capitalize() for part in re.split(r"[_\\-]", model_type) if part)
            return cleaned or None
        
        return None

    def get_architecture_key(self) -> str:
        """
        Choose a registry key for MODEL_ATTRS / OBSERVER_CONFIG_REGISTRY.
        Preference order:
          1) First entry in config['architectures']
          2) model_type
          3) Folder name
        """
        architectures = self.config.get("architectures", [])
        if architectures:
            return architectures[0]
        if self.model_type:
            return self.model_type
        return self.model_path.name
    
    def get_summary(self) -> str:
        """Generate a human-readable summary of the model analysis."""
        lines = [
            "=" * 60,
            "MODEL ANALYSIS SUMMARY (NO WEIGHTS LOADED)",
            "=" * 60,
            f"Model Path: {self.model_path}",
            f"Model Type: {self.model_type or 'Unknown'}",
            f"Architectures: {self.config.get('architectures', ['Unknown'])}",
            "",
            "--- Model Configuration ---",
            f"Number of Layers: {self.num_layers}",
            f"Hidden Size: {self.hidden_size}",
            f"Intermediate Size: {self.intermediate_size}",
            f"Number of Experts: {self.num_experts}",
            f"Experts per Token (top-k): {self.num_experts_per_tok}",
            f"Is Fused Experts: {self.is_fused_experts}",
            "",
            "--- Discovered Patterns ---",
            f"MoE Block Name: {self.moe_block_name or 'Not found'}",
            f"Expert Pattern: {self.expert_pattern or 'Not found'}",
            f"Router Pattern: {self.router_pattern or 'Not found'}",
            f"Gate Proj: {self.gate_proj_name or 'Not found'}",
            f"Up Proj: {self.up_proj_name or 'Not found'}",
            f"Down Proj: {self.down_proj_name or 'Not found'}",
            "",
            f"--- MoE Layers Found: {len(self.moe_layers)} ---",
        ]
        
        for moe_layer in self.moe_layers[:3]:  # Show first 3
            lines.append(
                f"  Layer {moe_layer.layer_idx}: {moe_layer.num_experts} experts, "
                f"has_shared={moe_layer.has_shared_expert}"
            )
        if len(self.moe_layers) > 3:
            lines.append(f"  ... and {len(self.moe_layers) - 3} more layers")
        
        lines.extend([
            "",
            "--- Suggested MODEL_ATTRS Entry ---",
            json.dumps(self.suggest_model_attrs(), indent=2),
            "",
            "--- Suggested Observer Config ---",
            json.dumps(self.suggest_observer_config(), indent=2),
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Export analysis as a dictionary for programmatic use."""
        return {
            "model_path": str(self.model_path),
            "model_type": self.model_type,
            "config": self.config,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "is_fused_experts": self.is_fused_experts,
            "discovered_patterns": {
                "moe_block_name": self.moe_block_name,
                "expert_pattern": self.expert_pattern,
                "router_pattern": self.router_pattern,
                "gate_proj_name": self.gate_proj_name,
                "up_proj_name": self.up_proj_name,
                "down_proj_name": self.down_proj_name,
            },
            "suggested_model_attrs": self.suggest_model_attrs(),
            "suggested_observer_config": self.suggest_observer_config(),
            "moe_layer_count": len(self.moe_layers),
            "total_tensors": len(self.tensor_infos),
        }


class ModelAnalyzer:
    """
    Analyze HuggingFace model architecture WITHOUT loading weights.
    
    This class uses safetensors metadata inspection and config.json
    parsing to understand model structure. At NO POINT are model
    weights loaded into memory, making this safe to run on any machine
    regardless of GPU availability.
    
    The analysis can be used by AI agents to:
    1. Understand new model architectures
    2. Generate MODEL_ATTRS entries for reap.model_util
    3. Generate observer configs for reap.observer
    4. Identify MoE-specific patterns and components
    """
    
    # Common tensor name patterns for MoE components
    EXPERT_PATTERNS = [
        r"layers\.(\d+)\..*experts\.(\d+)\.",  # Most common
        r"layers\.(\d+)\..*expert_(\d+)\.",
        r"decoder\.layers\.(\d+)\..*experts\.(\d+)\.",
        r"model\.layers\.(\d+)\..*experts\.(\d+)\.",
        r"transformer\.h\.(\d+)\..*experts\.(\d+)\.",
    ]
    
    ROUTER_PATTERNS = [
        r"layers\.(\d+)\..*gate\.weight",
        r"layers\.(\d+)\..*router\.weight",
        r"layers\.(\d+)\..*gating\.weight",
        r"model\.layers\.(\d+)\..*gate\.weight",
    ]
    
    PROJ_PATTERNS = {
        "gate": [r"gate_proj", r"w3", r"gate_up_proj", r"wi_0"],
        "up": [r"up_proj", r"w1", r"wi_1", r"fc1"],
        "down": [r"down_proj", r"w2", r"wo", r"fc2"],
    }
    
    def __init__(self, model_path: str | pathlib.Path):
        """
        Initialize the analyzer with a path to model files.
        
        Args:
            model_path: Path to directory containing model files
                        (config.json, *.safetensors)
        
        NOTE: No model weights are loaded during initialization.
        """
        self.model_path = pathlib.Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        
        self.config: dict[str, Any] = {}
        self.tensor_infos: list[TensorInfo] = []
        self._tensor_names_by_pattern: dict[str, list[str]] = defaultdict(list)
    
    def analyze(self) -> ModelAnalysis:
        """
        Perform deep analysis of the model architecture.
        
        Returns:
            ModelAnalysis object containing all discovered information.
        
        NOTE: This method DOES NOT load model weights into memory.
              It only reads metadata from safetensors files and config.json.
        """
        logger.info(f"Analyzing model at: {self.model_path}")
        logger.info("NOTE: No model weights will be loaded into memory.")
        
        # Step 1: Load and parse config.json
        self._load_config()
        
        # Step 2: Inspect safetensors files (metadata only, NO weight loading)
        self._inspect_safetensors()
        
        # Step 3: Analyze tensor patterns to discover MoE structure
        moe_layers = self._analyze_moe_structure()
        
        # Step 4: Discover projection naming patterns
        patterns = self._discover_patterns()
        
        # Step 5: Build and return analysis
        analysis = ModelAnalysis(
            model_path=self.model_path,
            config=self.config,
            tensor_infos=self.tensor_infos,
            moe_layers=moe_layers,
            **patterns,
            model_type=self.config.get("model_type"),
            num_layers=self._get_num_layers(),
            num_experts=self._get_num_experts(),
            num_experts_per_tok=self._get_num_experts_per_tok(),
            hidden_size=self.config.get("hidden_size", 0),
            intermediate_size=self.config.get("intermediate_size", 0),
            is_fused_experts=self._detect_fused_experts(),
        )
        
        logger.info("Analysis complete. No weights were loaded.")
        return analysis
    
    def _load_config(self) -> None:
        """Load and parse config.json."""
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            logger.warning(f"config.json not found at {config_path}")
            return
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        logger.info(f"Loaded config.json: model_type={self.config.get('model_type')}")
    
    def _inspect_safetensors(self) -> None:
        """
        Inspect safetensors files to extract tensor metadata.
        
        IMPORTANT: This only reads metadata (names, shapes, dtypes).
        Actual tensor data/weights are NEVER loaded into memory.
        """
        safetensor_files = list(self.model_path.glob("*.safetensors"))
        
        if not safetensor_files:
            logger.warning("No .safetensors files found. Trying model.safetensors.index.json...")
            index_path = self.model_path / "model.safetensors.index.json"
            if index_path.exists():
                with open(index_path, "r") as f:
                    index = json.load(f)
                    # Get unique safetensor files from index
                    weight_map = index.get("weight_map", {})
                    safetensor_files = list(set(
                        self.model_path / fname for fname in weight_map.values()
                    ))
        
        if not safetensor_files:
            logger.error("No safetensors files found!")
            return
        
        logger.info(f"Found {len(safetensor_files)} safetensors file(s)")
        
        for st_file in safetensor_files:
            if not st_file.exists():
                logger.warning(f"Safetensors file not found: {st_file}")
                continue
                
            logger.info(f"Inspecting metadata from: {st_file.name}")
            
            # NOTE: safe_open with framework="pt" does NOT load tensors
            # It only provides access to metadata until you explicitly
            # call get_tensor(), which we NEVER do.
            try:
                with safe_open(st_file, framework="pt") as f:
                    for tensor_name in f.keys():
                        # Get tensor metadata WITHOUT loading the tensor
                        slice_obj = f.get_slice(tensor_name)
                        tensor_shape = slice_obj.get_shape()
                        # Use get_dtype() method instead of dtype attribute
                        tensor_dtype = str(slice_obj.get_dtype())
                        
                        # Calculate size (approximate, for info only)
                        dtype_sizes = {
                            "F32": 4, "F16": 2, "BF16": 2, 
                            "I32": 4, "I16": 2, "I8": 1,
                            "torch.float32": 4, "torch.float16": 2,
                            "torch.bfloat16": 2, "torch.int32": 4,
                        }
                        dtype_size = dtype_sizes.get(tensor_dtype, 2)
                        num_elements = 1
                        for dim in tensor_shape:
                            num_elements *= dim
                        size_bytes = num_elements * dtype_size
                        
                        tensor_info = TensorInfo(
                            name=tensor_name,
                            shape=tuple(tensor_shape),
                            dtype=tensor_dtype,
                            size_bytes=size_bytes,
                        )
                        self.tensor_infos.append(tensor_info)
            except Exception as e:
                logger.error(f"Error inspecting {st_file}: {e}")
        
        logger.info(f"Discovered {len(self.tensor_infos)} tensors (metadata only)")
    
    def _analyze_moe_structure(self) -> list[MoELayerInfo]:
        """Analyze tensor names to discover MoE layer structure."""
        moe_layers: dict[int, MoELayerInfo] = {}
        expert_tensors: dict[tuple[int, int], list[TensorInfo]] = defaultdict(list)
        router_tensors: dict[int, list[TensorInfo]] = defaultdict(list)
        
        for tensor in self.tensor_infos:
            name = tensor.name
            
            # Check for expert patterns
            for pattern in self.EXPERT_PATTERNS:
                match = re.search(pattern, name)
                if match:
                    layer_idx = int(match.group(1))
                    expert_idx = int(match.group(2))
                    expert_tensors[(layer_idx, expert_idx)].append(tensor)
                    
                    if layer_idx not in moe_layers:
                        moe_layers[layer_idx] = MoELayerInfo(
                            layer_idx=layer_idx,
                            num_experts=0,
                        )
                    break
            
            # Check for router patterns
            for pattern in self.ROUTER_PATTERNS:
                if re.search(pattern, name):
                    match = re.search(r"layers\.(\d+)\.", name)
                    if match:
                        layer_idx = int(match.group(1))
                        router_tensors[layer_idx].append(tensor)
                    break
            
            # Check for shared expert
            if "shared_expert" in name.lower():
                match = re.search(r"layers\.(\d+)\.", name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx in moe_layers:
                        moe_layers[layer_idx].has_shared_expert = True
                        moe_layers[layer_idx].shared_expert_tensors.append(tensor)
        
        # Build expert info objects
        for (layer_idx, expert_idx), tensors in expert_tensors.items():
            if layer_idx in moe_layers:
                expert = ExpertInfo(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    tensors=tensors,
                )
                moe_layers[layer_idx].experts.append(expert)
        
        # Update expert counts and router tensors
        for layer_idx, layer_info in moe_layers.items():
            layer_info.num_experts = len(layer_info.experts)
            layer_info.router_tensors = router_tensors.get(layer_idx, [])
        
        return sorted(moe_layers.values(), key=lambda x: x.layer_idx)
    
    def _discover_patterns(self) -> dict[str, str | None]:
        """Discover naming patterns for MoE components."""
        patterns = {
            "expert_pattern": None,
            "router_pattern": None,
            "gate_proj_name": None,
            "up_proj_name": None,
            "down_proj_name": None,
            "moe_block_name": None,
        }
        
        tensor_names = [t.name for t in self.tensor_infos]
        
        # Find projection names
        for proj_type, proj_patterns in self.PROJ_PATTERNS.items():
            for pattern in proj_patterns:
                for name in tensor_names:
                    if pattern in name:
                        if proj_type == "gate":
                            patterns["gate_proj_name"] = pattern
                        elif proj_type == "up":
                            patterns["up_proj_name"] = pattern
                        elif proj_type == "down":
                            patterns["down_proj_name"] = pattern
                        break
                if patterns.get(f"{proj_type}_proj_name"):
                    break
        
        # Find MoE block name from tensor paths
        for name in tensor_names:
            if "experts" in name:
                # Extract the parent module name
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == "experts" and i > 0:
                        patterns["moe_block_name"] = parts[i - 1]
                        break
                if patterns["moe_block_name"]:
                    break
        
        # Find expert pattern
        for name in tensor_names:
            for pattern in self.EXPERT_PATTERNS:
                if re.search(pattern, name):
                    patterns["expert_pattern"] = pattern
                    break
            if patterns["expert_pattern"]:
                break
        
        # Find router pattern  
        for name in tensor_names:
            for pattern in self.ROUTER_PATTERNS:
                if re.search(pattern, name):
                    patterns["router_pattern"] = pattern
                    break
            if patterns["router_pattern"]:
                break
        
        return patterns
    
    def _get_num_layers(self) -> int:
        """Get number of transformer layers from config."""
        for key in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
            if key in self.config:
                return self.config[key]
        return 0
    
    def _get_num_experts(self) -> int:
        """Get number of experts from config."""
        for key in ["num_experts", "num_local_experts", "n_routed_experts", 
                    "moe_num_experts", "num_moe_experts"]:
            if key in self.config:
                return self.config[key]
        return 0
    
    def _get_num_experts_per_tok(self) -> int:
        """Get number of experts per token (top-k) from config."""
        for key in ["num_experts_per_tok", "top_k", "moe_k", 
                    "num_selected_experts", "experts_per_tok"]:
            if key in self.config:
                return self.config[key]
        return 0
    
    def _detect_fused_experts(self) -> bool:
        """
        Detect if the model uses fused expert implementation.
        
        Fused experts have a single tensor for all experts' weights
        instead of separate tensors per expert.
        """
        for tensor in self.tensor_infos:
            # Fused experts typically have shape [num_experts, ...]
            # as the first dimension
            if "experts" in tensor.name and "gate_up_proj" in tensor.name:
                if len(tensor.shape) >= 2:
                    # If first dim matches num_experts, likely fused
                    if tensor.shape[0] == self._get_num_experts():
                        return True
        return False


def _sanitize_class_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    if not cleaned:
        cleaned = "CustomModel"
    if cleaned[0].isdigit():
        cleaned = f"Model_{cleaned}"
    return cleaned


def _format_model_attrs_entry(key: str, attrs: dict[str, Any]) -> str:
    lines = [f'    "{key}": {{']
    for k, v in attrs.items():
        if isinstance(v, bool):
            val = "True" if v else "False"
        elif isinstance(v, str):
            val = f'"{v}"'
        else:
            val = repr(v)
        lines.append(f'        "{k}": {val},')
    lines.append("    },")
    return "\n".join(lines)


def _format_observer_class(class_name: str, cfg: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        @dataclass
        class {class_name}(MoETransformerObserverConfig):
            module_class_name_to_hook_regex: Optional[str] = "{cfg['module_class_name_to_hook_regex']}"
            num_experts_attr_name: str = "{cfg['num_experts_attr_name']}"
            top_k_attr_name: str = "{cfg['top_k_attr_name']}"
            fused_experts: bool = {cfg['fused_experts']}
        """
    )


def _add_model_attrs_entry(model_key: str, attrs: dict[str, Any]) -> bool:
    """Insert MODEL_ATTRS entry into model_util.py if missing."""
    target_file = pathlib.Path(__file__).parent / "model_util.py"
    text = target_file.read_text()
    if f'"{model_key}"' in text:
        logger.info("MODEL_ATTRS already contains key %s; skipping add.", model_key)
        return False

    marker = "\n}\n\n\ndef get_moe"
    if marker not in text:
        logger.error("Could not locate insertion point in model_util.py")
        return False

    entry = _format_model_attrs_entry(model_key, attrs)
    new_text = text.replace(marker, f"\n{entry}\n{marker}", 1)
    target_file.write_text(new_text)
    logger.info("Added MODEL_ATTRS entry for %s", model_key)
    return True


def _add_observer_entry(model_key: str, cfg: dict[str, Any]) -> bool:
    """Insert observer class + registry entry into observer.py if missing."""
    target_file = pathlib.Path(__file__).parent / "observer.py"
    text = target_file.read_text()
    if f'"{model_key}"' in text:
        logger.info("OBSERVER_CONFIG_REGISTRY already contains key %s; skipping add.", model_key)
        return False

    class_name = f"{_sanitize_class_name(model_key)}ObserverHookConfig"
    class_def = _format_observer_class(class_name, cfg).strip() + "\n\n"

    registry_marker = "OBSERVER_CONFIG_REGISTRY = {"
    if registry_marker not in text:
        logger.error("Could not locate OBSERVER_CONFIG_REGISTRY in observer.py")
        return False

    if class_name not in text:
        text = text.replace(registry_marker, class_def + registry_marker, 1)

    # Insert registry entry before closing brace of registry dict (last occurrence of "\n}")
    end_idx = text.rfind("\n}")
    if end_idx == -1:
        logger.error("Could not locate registry closing brace in observer.py")
        return False

    entry_line = f'    "{model_key}": {class_name},\n'
    text = text[:end_idx] + entry_line + text[end_idx:]

    target_file.write_text(text)
    logger.info("Added OBSERVER_CONFIG_REGISTRY entry for %s", model_key)
    return True


def _add_config_entries(analysis: ModelAnalysis) -> None:
    """Add suggested MODEL_ATTRS and observer config to source files."""
    model_key = analysis.get_architecture_key()
    attrs = analysis.suggest_model_attrs()
    observer_cfg = analysis.suggest_observer_config()

    if not observer_cfg.get("module_class_name_to_hook_regex"):
        logger.warning(
            "Observer hook class name is missing; skipping observer registry add for %s",
            model_key,
        )
    else:
        _add_observer_entry(model_key, observer_cfg)

    _add_model_attrs_entry(model_key, attrs)


def analyze_model(model_path: str | pathlib.Path) -> ModelAnalysis:
    """
    Convenience function to analyze a model.
    
    Args:
        model_path: Path to HuggingFace model directory
        
    Returns:
        ModelAnalysis with all discovered information
        
    NOTE: No model weights are loaded into memory.
    """
    analyzer = ModelAnalyzer(model_path)
    return analyzer.analyze()


def print_analysis(model_path: str | pathlib.Path) -> None:
    """Print a detailed analysis of the model to stdout."""
    analysis = analyze_model(model_path)
    print(analysis.get_summary())


def export_analysis_json(model_path: str | pathlib.Path, output_path: str | pathlib.Path | None = None) -> str:
    """
    Export analysis as JSON for programmatic use.
    
    Args:
        model_path: Path to HuggingFace model directory
        output_path: Optional path to write JSON file
        
    Returns:
        JSON string of the analysis
    """
    analysis = analyze_model(model_path)
    analysis_dict = analysis.to_dict()
    json_str = json.dumps(analysis_dict, indent=2)
    
    if output_path:
        output_path = pathlib.Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        logger.info(f"Analysis exported to: {output_path}")
    
    return json_str


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI entry point for model analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze HuggingFace model architecture WITHOUT loading weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python -m reap.model_analyzer /path/to/model
    
    # Export to JSON
    python -m reap.model_analyzer /path/to/model --output analysis.json
    
    # Just show suggested MODEL_ATTRS
    python -m reap.model_analyzer /path/to/model --attrs-only

NOTE: This tool NEVER loads model weights into memory.
      Safe to run on any machine regardless of GPU availability.
        """
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to HuggingFace model directory containing config.json and *.safetensors"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path for analysis results"
    )
    parser.add_argument(
        "--attrs-only",
        action="store_true",
        help="Only print suggested MODEL_ATTRS dictionary"
    )
    parser.add_argument(
        "--observer-config",
        action="store_true",
        help="Only print suggested observer config"
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Add suggested MODEL_ATTRS and observer registry entries to source files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        analysis = analyze_model(args.model_path)
        
        if args.attrs_only:
            print("Suggested MODEL_ATTRS entry:")
            print(json.dumps(analysis.suggest_model_attrs(), indent=2))
        elif args.observer_config:
            print("Suggested Observer Config:")
            print(json.dumps(analysis.suggest_observer_config(), indent=2))
        elif args.output:
            export_analysis_json(args.model_path, args.output)
            print(f"Analysis exported to: {args.output}")
        elif args.add:
            _add_config_entries(analysis)
            print("Suggested configurations added to source files.")
        else:
            print(analysis.get_summary())
            
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

