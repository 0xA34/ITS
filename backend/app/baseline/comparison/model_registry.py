"""
Model Registry Module
======================
Registry for managing different detection models for baseline evaluation.

Supports:
    - Registering models with metadata
    - Loading and caching models
    - Model information (parameters, FLOPs, size)
    - Model comparison utilities

Usage:
    from app.baseline.comparison.model_registry import ModelRegistry

    registry = ModelRegistry()

    # Register a model
    registry.register(
        name="yolov8n",
        model_path="yolov8n.pt",
        description="YOLOv8 Nano - Fastest variant",
    )

    # Get model info
    info = registry.get_info("yolov8n")
    print(f"Parameters: {info.params_millions}M")

    # Load model for inference
    model = registry.load("yolov8n")
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a registered model."""

    name: str
    model_path: Optional[str] = None
    model_type: str = "yolo"  # 'yolo', 'torchvision', 'custom'
    description: str = ""

    # Model complexity
    params_millions: float = 0.0
    flops_billions: float = 0.0
    model_size_mb: float = 0.0

    # Configuration
    input_size: tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    # Source information
    source: str = ""  # e.g., "Ultralytics Official", "Custom trained"
    version: str = ""
    trained_on: str = ""  # e.g., "COCO", "Custom dataset"

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "description": self.description,
            "params_millions": self.params_millions,
            "flops_billions": self.flops_billions,
            "model_size_mb": self.model_size_mb,
            "input_size": self.input_size,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "source": self.source,
            "version": self.version,
            "trained_on": self.trained_on,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return (
            f"ModelInfo({self.name})\n"
            f"  Type: {self.model_type}\n"
            f"  Path: {self.model_path}\n"
            f"  Parameters: {self.params_millions}M\n"
            f"  FLOPs: {self.flops_billions}B\n"
            f"  Size: {self.model_size_mb}MB\n"
            f"  Input Size: {self.input_size}\n"
            f"  Description: {self.description}"
        )


class ModelRegistry:
    """
    Registry for managing detection models.

    Features:
        - Register models with metadata
        - Lazy loading and caching
        - Model information extraction
        - Support for multiple model types

    Usage:
        registry = ModelRegistry()

        # Register YOLO model
        registry.register(
            name="yolov8n",
            model_path="yolov8n.pt",
            model_type="yolo",
        )

        # Register custom model with loader function
        registry.register(
            name="custom_detector",
            model_type="custom",
            loader_fn=my_load_function,
        )

        # Get model
        model = registry.load("yolov8n")
    """

    def __init__(self):
        """Initialize model registry."""
        self._models: Dict[str, ModelInfo] = {}
        self._loaded_models: Dict[str, Any] = {}
        self._custom_loaders: Dict[str, Callable] = {}

    def register(
        self,
        name: str,
        model_path: Optional[str] = None,
        model_type: str = "yolo",
        description: str = "",
        params_millions: float = 0.0,
        flops_billions: float = 0.0,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        source: str = "",
        version: str = "",
        trained_on: str = "",
        loader_fn: Optional[Callable] = None,
        **metadata,
    ) -> ModelInfo:
        """
        Register a model in the registry.

        Args:
            name: Unique name for the model
            model_path: Path to model file
            model_type: Type of model ('yolo', 'torchvision', 'custom')
            description: Model description
            params_millions: Number of parameters in millions
            flops_billions: FLOPs in billions
            input_size: Expected input size (width, height)
            confidence_threshold: Default confidence threshold
            iou_threshold: Default IoU threshold for NMS
            source: Source of the model
            version: Model version
            trained_on: Dataset the model was trained on
            loader_fn: Custom loader function (for custom models)
            **metadata: Additional metadata

        Returns:
            ModelInfo object
        """
        # Calculate model size if path exists
        model_size_mb = 0.0
        if model_path and os.path.exists(model_path):
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        info = ModelInfo(
            name=name,
            model_path=model_path,
            model_type=model_type,
            description=description,
            params_millions=params_millions,
            flops_billions=flops_billions,
            model_size_mb=model_size_mb,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            source=source,
            version=version,
            trained_on=trained_on,
            metadata=metadata,
        )

        self._models[name] = info

        if loader_fn is not None:
            self._custom_loaders[name] = loader_fn

        logger.info(f"Registered model: {name} ({model_type})")

        return info

    def register_yolo_model(
        self,
        name: str,
        model_path: str,
        description: str = "",
        auto_extract_info: bool = True,
        **kwargs,
    ) -> ModelInfo:
        """
        Register a YOLO model with automatic info extraction.

        Args:
            name: Model name
            model_path: Path to YOLO model file
            description: Model description
            auto_extract_info: Automatically extract model info
            **kwargs: Additional arguments for register()

        Returns:
            ModelInfo object
        """
        params = 0.0
        flops = 0.0

        if auto_extract_info:
            try:
                params, flops = self._extract_yolo_info(model_path)
            except Exception as e:
                logger.warning(f"Could not extract model info: {e}")

        return self.register(
            name=name,
            model_path=model_path,
            model_type="yolo",
            description=description,
            params_millions=params,
            flops_billions=flops,
            source="YOLO",
            **kwargs,
        )

    def _extract_yolo_info(self, model_path: str) -> tuple[float, float]:
        """Extract parameters and FLOPs from YOLO model."""
        try:
            from ultralytics import YOLO

            model = YOLO(model_path)

            # Get model info
            params = sum(p.numel() for p in model.model.parameters()) / 1e6
            flops = 0.0  # Would need to profile to get accurate FLOPs

            return params, flops
        except ImportError:
            logger.warning("ultralytics not installed, cannot extract model info")
            return 0.0, 0.0
        except Exception as e:
            logger.warning(f"Error extracting model info: {e}")
            return 0.0, 0.0

    def unregister(self, name: str) -> bool:
        """
        Remove a model from the registry.

        Args:
            name: Model name

        Returns:
            True if model was removed, False if not found
        """
        if name in self._models:
            del self._models[name]

            # Also remove from cache and custom loaders
            if name in self._loaded_models:
                del self._loaded_models[name]
            if name in self._custom_loaders:
                del self._custom_loaders[name]

            logger.info(f"Unregistered model: {name}")
            return True

        return False

    def get_info(self, name: str) -> Optional[ModelInfo]:
        """
        Get information about a registered model.

        Args:
            name: Model name

        Returns:
            ModelInfo or None if not found
        """
        return self._models.get(name)

    def list_models(self) -> list[str]:
        """Get list of registered model names."""
        return list(self._models.keys())

    def list_models_info(self) -> Dict[str, ModelInfo]:
        """Get all registered models with their info."""
        return self._models.copy()

    def load(
        self,
        name: str,
        device: str = "auto",
        cache: bool = True,
    ) -> Any:
        """
        Load a model for inference.

        Args:
            name: Model name
            device: Device to load model on ('cuda', 'cpu', 'auto')
            cache: Whether to cache the loaded model

        Returns:
            Loaded model object

        Raises:
            ValueError: If model not found in registry
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found in registry")

        # Return cached model if available
        if cache and name in self._loaded_models:
            return self._loaded_models[name]

        info = self._models[name]
        model = None

        # Load based on model type
        if name in self._custom_loaders:
            # Use custom loader
            model = self._custom_loaders[name](info)
        elif info.model_type == "yolo":
            model = self._load_yolo(info, device)
        elif info.model_type == "torchvision":
            model = self._load_torchvision(info, device)
        else:
            raise ValueError(f"Unknown model type: {info.model_type}")

        # Cache if requested
        if cache and model is not None:
            self._loaded_models[name] = model

        return model

    def _load_yolo(self, info: ModelInfo, device: str) -> Any:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO

            model = YOLO(info.model_path)

            # Move to device
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"

            if device == "cuda":
                model.to("cuda")

            return model
        except ImportError:
            raise ImportError(
                "ultralytics package is required for YOLO models. "
                "Install with: pip install ultralytics"
            )

    def _load_torchvision(self, info: ModelInfo, device: str) -> Any:
        """Load torchvision model."""
        try:
            import torch
            import torchvision

            # Determine device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model based on path or name
            if info.model_path and os.path.exists(info.model_path):
                model = torch.load(info.model_path, map_location=device)
            else:
                # Try to load from torchvision.models
                model_name = info.metadata.get("torchvision_name", info.name)
                model_fn = getattr(torchvision.models.detection, model_name, None)
                if model_fn is None:
                    raise ValueError(f"Unknown torchvision model: {model_name}")
                model = model_fn(pretrained=True)

            model = model.to(device)
            model.eval()

            return model
        except ImportError:
            raise ImportError(
                "torchvision package is required for torchvision models. "
                "Install with: pip install torchvision"
            )

    def unload(self, name: str) -> bool:
        """
        Unload a cached model to free memory.

        Args:
            name: Model name

        Returns:
            True if model was unloaded, False if not in cache
        """
        if name in self._loaded_models:
            del self._loaded_models[name]

            # Force garbage collection
            import gc

            gc.collect()

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info(f"Unloaded model: {name}")
            return True

        return False

    def unload_all(self):
        """Unload all cached models."""
        for name in list(self._loaded_models.keys()):
            self.unload(name)

    def compare_models(self) -> str:
        """
        Generate comparison table of registered models.

        Returns:
            Formatted comparison string
        """
        if not self._models:
            return "No models registered."

        lines = [
            "\n" + "=" * 80,
            "REGISTERED MODELS",
            "=" * 80,
            f"{'Name':<20} {'Type':<12} {'Params (M)':<12} {'Size (MB)':<12} {'Source':<20}",
            "-" * 80,
        ]

        for name, info in sorted(self._models.items()):
            params = f"{info.params_millions:.1f}" if info.params_millions > 0 else "-"
            size = f"{info.model_size_mb:.1f}" if info.model_size_mb > 0 else "-"
            source = info.source[:18] if info.source else "-"

            lines.append(
                f"{name:<20} {info.model_type:<12} {params:<12} {size:<12} {source:<20}"
            )

        lines.append("=" * 80)
        lines.append(f"Total: {len(self._models)} models")

        return "\n".join(lines)


# Pre-defined YOLO model configurations
YOLO_MODELS = {
    "yolov8n": {
        "description": "YOLOv8 Nano - Fastest, smallest",
        "params_millions": 3.2,
        "flops_billions": 8.7,
        "source": "Ultralytics Official",
    },
    "yolov8s": {
        "description": "YOLOv8 Small - Good balance",
        "params_millions": 11.2,
        "flops_billions": 28.6,
        "source": "Ultralytics Official",
    },
    "yolov8m": {
        "description": "YOLOv8 Medium - Higher accuracy",
        "params_millions": 25.9,
        "flops_billions": 78.9,
        "source": "Ultralytics Official",
    },
    "yolov8l": {
        "description": "YOLOv8 Large - High accuracy",
        "params_millions": 43.7,
        "flops_billions": 165.2,
        "source": "Ultralytics Official",
    },
    "yolov8x": {
        "description": "YOLOv8 XLarge - Highest accuracy",
        "params_millions": 68.2,
        "flops_billions": 257.8,
        "source": "Ultralytics Official",
    },
    "yolov5n": {
        "description": "YOLOv5 Nano",
        "params_millions": 1.9,
        "flops_billions": 4.5,
        "source": "Ultralytics Official",
    },
    "yolov5s": {
        "description": "YOLOv5 Small",
        "params_millions": 7.2,
        "flops_billions": 16.5,
        "source": "Ultralytics Official",
    },
    "yolov5m": {
        "description": "YOLOv5 Medium",
        "params_millions": 21.2,
        "flops_billions": 49.0,
        "source": "Ultralytics Official",
    },
}


def create_default_registry() -> ModelRegistry:
    """
    Create a registry with pre-configured YOLO models.

    Returns:
        ModelRegistry with YOLO models registered
    """
    registry = ModelRegistry()

    for model_name, config in YOLO_MODELS.items():
        registry.register(
            name=model_name,
            model_path=f"{model_name}.pt",
            model_type="yolo",
            **config,
        )

    return registry
