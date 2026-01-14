"""
Datasets Module for Baseline Evaluation
=======================================

This module provides dataset loaders for ground truth data used in
baseline evaluation of the vehicle detection system.

Supported Datasets:
    - COCO: Common Objects in Context (vehicle classes)
    - UA-DETRAC: Multi-object tracking benchmark for vehicles
    - KITTI: Autonomous driving dataset
    - Custom: Custom annotated dataset in COCO format

Usage:
    from app.baseline.datasets import COCODataset, CustomDataset

    # Load COCO dataset
    coco = COCODataset(data_path="data/coco")
    annotations = coco.load_annotations()

    # Load custom dataset
    custom = CustomDataset(data_path="data/custom")
    annotations = custom.load_annotations()
"""

from app.baseline.datasets.base_dataset import (
    BaseDataset,
    BoundingBox,
    DatasetStatistics,
    GroundTruthAnnotation,
    ImageAnnotation,
)
from app.baseline.datasets.coco_loader import COCODataset
from app.baseline.datasets.custom_loader import CustomDataset

# Aliases for backward compatibility
Annotation = GroundTruthAnnotation
ImageData = ImageAnnotation

__all__ = [
    "BaseDataset",
    "BoundingBox",
    "DatasetStatistics",
    "GroundTruthAnnotation",
    "ImageAnnotation",
    "Annotation",
    "ImageData",
    "COCODataset",
    "CustomDataset",
]
