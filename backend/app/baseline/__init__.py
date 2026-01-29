"""
Baseline Module for Vehicle Detection & Monitoring Survey Project

This module provides:
1. Ground Truth Dataset Management
2. Benchmark Metrics Calculation (mAP, Precision, Recall, F1, FPS)
3. Model Comparison Tools

Directory Structure:
    baseline/
    ├── __init__.py
    ├── config.py              # Configuration settings
    ├── datasets/              # Ground truth dataset loaders
    │   ├── __init__.py
    │   ├── base_dataset.py    # Base dataset class
    │   ├── coco_loader.py     # COCO format loader
    │   └── custom_loader.py   # Custom dataset loader
    ├── metrics/               # Evaluation metrics
    │   ├── __init__.py
    │   ├── iou.py             # IoU calculation
    │   ├── map_calculator.py  # mAP calculation
    │   ├── precision_recall.py # Precision, Recall, F1
    │   └── fps_benchmark.py   # FPS/Latency measurement
    ├── comparison/            # Model comparison tools
    │   ├── __init__.py
    │   ├── model_registry.py  # Register different models
    │   └── comparison_report.py # Generate comparison reports
    └── results/               # Output results directory

Usage:
    from app.baseline import BaselineEvaluator

    evaluator = BaselineEvaluator()
    results = evaluator.run_evaluation(
        dataset="coco",
        models=["yolov8n", "yolov8s", "yolov8m"]
    )
    evaluator.generate_report(results)
"""

from app.baseline.config import BaselineConfig

__version__ = "1.0.0"
__all__ = [
    "BaselineConfig",
]
