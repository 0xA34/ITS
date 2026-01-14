"""
Baseline Configuration
======================
Configuration settings for baseline evaluation of vehicle detection system.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# Base paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
RESULTS_DIR = BASE_DIR / "results"


class DatasetConfig(BaseModel):
    """Configuration for datasets used in baseline evaluation."""

    # Dataset paths
    coco_path: Optional[str] = Field(
        default=str(DATA_DIR / "coco"), description="Path to COCO dataset"
    )
    ua_detrac_path: Optional[str] = Field(
        default=str(DATA_DIR / "ua_detrac"), description="Path to UA-DETRAC dataset"
    )
    kitti_path: Optional[str] = Field(
        default=str(DATA_DIR / "kitti"), description="Path to KITTI dataset"
    )
    custom_path: Optional[str] = Field(
        default=str(DATA_DIR / "custom"), description="Path to custom dataset"
    )

    # Ground truth annotations
    annotations_path: str = Field(
        default=str(GROUND_TRUTH_DIR / "annotations"),
        description="Path to ground truth annotations",
    )

    # Vehicle classes mapping (COCO format)
    vehicle_classes: dict[int, str] = Field(
        default={
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        },
        description="Mapping of class IDs to class names",
    )

    # Class name to ID reverse mapping
    @property
    def class_name_to_id(self) -> dict[str, int]:
        return {v: k for k, v in self.vehicle_classes.items()}


class ModelConfig(BaseModel):
    """Configuration for models to evaluate."""

    # Current model (YOLOv8)
    current_model: str = Field(
        default="yolov8n.pt", description="Current YOLO model being used"
    )

    # Models for comparison
    comparison_models: list[str] = Field(
        default=[
            "yolov8n.pt",  # YOLOv8 Nano
            "yolov8s.pt",  # YOLOv8 Small
            "yolov8m.pt",  # YOLOv8 Medium
            "yolov5n.pt",  # YOLOv5 Nano
            "yolov5s.pt",  # YOLOv5 Small
        ],
        description="List of models to compare",
    )

    # Model parameters
    confidence_threshold: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Confidence threshold for detections"
    )
    iou_threshold: float = Field(
        default=0.45, ge=0.0, le=1.0, description="IOU threshold for NMS"
    )

    # Inference settings
    image_size: int = Field(default=640, description="Input image size for inference")
    half_precision: bool = Field(default=True, description="Use FP16 half precision")
    device: str = Field(default="cuda", description="Device for inference (cuda/cpu)")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics."""

    # mAP settings
    iou_thresholds: list[float] = Field(
        default=[0.5, 0.75], description="IOU thresholds for mAP calculation"
    )
    map_iou_range: tuple[float, float, float] = Field(
        default=(0.5, 0.95, 0.05),
        description="IOU range for mAP@[.5:.95] (start, end, step)",
    )

    # FPS benchmark settings
    warmup_iterations: int = Field(
        default=10, description="Number of warmup iterations before timing"
    )
    benchmark_iterations: int = Field(
        default=100, description="Number of iterations for FPS benchmark"
    )

    # Evaluation modes
    evaluate_by_class: bool = Field(
        default=True, description="Calculate metrics per class"
    )
    evaluate_by_size: bool = Field(
        default=True,
        description="Calculate metrics by object size (small/medium/large)",
    )

    # Size thresholds (area in pixels^2)
    small_object_threshold: int = Field(
        default=32 * 32, description="Max area for small objects"
    )
    medium_object_threshold: int = Field(
        default=96 * 96, description="Max area for medium objects"
    )


class ComparisonConfig(BaseModel):
    """Configuration for method comparison."""

    # Reference papers/methods for comparison
    reference_methods: dict[str, dict] = Field(
        default={
            "YOLOv8-N (Official)": {
                "mAP50": 0.370,
                "mAP50-95": 0.183,
                "fps": 195,
                "params_m": 3.2,
                "source": "Ultralytics Official",
            },
            "YOLOv8-S (Official)": {
                "mAP50": 0.449,
                "mAP50-95": 0.228,
                "fps": 142,
                "params_m": 11.2,
                "source": "Ultralytics Official",
            },
            "YOLOv5-N (Official)": {
                "mAP50": 0.280,
                "mAP50-95": 0.140,
                "fps": 160,
                "params_m": 1.9,
                "source": "Ultralytics Official",
            },
            "YOLOv5-S (Official)": {
                "mAP50": 0.374,
                "mAP50-95": 0.188,
                "fps": 118,
                "params_m": 7.2,
                "source": "Ultralytics Official",
            },
            "Faster R-CNN (ResNet50)": {
                "mAP50": 0.420,
                "mAP50-95": 0.210,
                "fps": 15,
                "params_m": 41.0,
                "source": "Torchvision",
            },
            "SSD300 (VGG16)": {
                "mAP50": 0.341,
                "mAP50-95": 0.170,
                "fps": 46,
                "params_m": 24.0,
                "source": "Original Paper",
            },
        },
        description="Reference metrics from official sources",
    )

    # Comparison aspects
    compare_accuracy: bool = Field(default=True)
    compare_speed: bool = Field(default=True)
    compare_model_size: bool = Field(default=True)
    compare_by_class: bool = Field(default=True)


class BaselineConfig(BaseModel):
    """Main baseline configuration."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)

    # Output settings
    results_dir: str = Field(
        default=str(RESULTS_DIR), description="Directory to save evaluation results"
    )
    save_visualizations: bool = Field(
        default=True, description="Save visualization images"
    )
    export_format: list[str] = Field(
        default=["json", "csv", "markdown"], description="Export formats for results"
    )

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [
            self.results_dir,
            self.dataset.annotations_path,
            self.dataset.coco_path,
            self.dataset.ua_detrac_path,
            self.dataset.custom_path,
        ]
        for dir_path in dirs:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)


# Default configuration instance
default_config = BaselineConfig()


def get_config() -> BaselineConfig:
    """Get the default baseline configuration."""
    return default_config


def load_config_from_file(config_path: str) -> BaselineConfig:
    """Load configuration from a JSON file."""
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BaselineConfig(**data)


def save_config_to_file(config: BaselineConfig, config_path: str):
    """Save configuration to a JSON file."""
    import json

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)
