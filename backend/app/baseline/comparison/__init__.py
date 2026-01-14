"""
Comparison Module for Baseline Evaluation
==========================================

This module provides tools for comparing different object detection
methods and models against baseline metrics.

Features:
    - Model registry for managing different detection models
    - Reference method database (official benchmarks)
    - Comparison report generation
    - Visualization of comparison results

Usage:
    from app.baseline.comparison import (
        ModelRegistry,
        ComparisonReport,
        ReferenceDatabase,
    )

    # Register models
    registry = ModelRegistry()
    registry.register("yolov8n", model_path="yolov8n.pt")
    registry.register("yolov8s", model_path="yolov8s.pt")

    # Get reference benchmarks
    references = ReferenceDatabase.get_all()

    # Generate comparison report
    report = ComparisonReport(
        title="Vehicle Detection Comparison",
    )
    report.add_model_results("YOLOv8n", map_result, benchmark_result)
    report.generate_report("output/comparison")
"""

from app.baseline.comparison.comparison_report import (
    ComparisonReport,
    ComparisonResult,
    ModelMetrics,
)
from app.baseline.comparison.model_registry import (
    YOLO_MODELS,
    ModelInfo,
    ModelRegistry,
    create_default_registry,
)
from app.baseline.comparison.reference_database import (
    DatasetType,
    ModelCategory,
    ReferenceDatabase,
    ReferenceMethod,
)

__all__ = [
    # Model Registry
    "ModelRegistry",
    "ModelInfo",
    "create_default_registry",
    "YOLO_MODELS",
    # Reference Database
    "ReferenceDatabase",
    "ReferenceMethod",
    "ModelCategory",
    "DatasetType",
    # Comparison Report
    "ComparisonReport",
    "ComparisonResult",
    "ModelMetrics",
]
