"""
Comparison Module for Baseline Evaluation
==========================================

This module provides tools for comparing different object detection
methods and models against baseline metrics.

Features:
    - Model registry for managing different detection models
    - Benchmark history for persistent storage of your results
    - Comparison report generation
    - Visualization of comparison results

Usage:
    from app.baseline.comparison import (
        ModelRegistry,
        ComparisonReport,
        BenchmarkHistory,
    )

    # Register models
    registry = ModelRegistry()
    registry.register("yolov8n", model_path="yolov8n.pt")
    registry.register("yolov8s", model_path="yolov8s.pt")

    # Save benchmark results
    history = BenchmarkHistory()
    history.save_result("yolov8n", map_50=0.72, fps=150)

    # Generate comparison report
    report = ComparisonReport(
        title="Vehicle Detection Comparison",
    )
    report.add_model_results("YOLOv8n", map_result, benchmark_result)
    report.generate_report("output/comparison")
"""

from app.baseline.comparison.benchmark_history import (
    BenchmarkHistory,
    BenchmarkResult,
)
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

__all__ = [
    # Model Registry
    "ModelRegistry",
    "ModelInfo",
    "create_default_registry",
    "YOLO_MODELS",
    # Benchmark History
    "BenchmarkHistory",
    "BenchmarkResult",
    # Comparison Report
    "ComparisonReport",
    "ComparisonResult",
    "ModelMetrics",
]
