"""
Metrics Module for Baseline Evaluation
======================================

This module provides evaluation metrics for vehicle detection baseline.

Available Metrics:
    - IoU (Intersection over Union)
    - Precision, Recall, F1-Score
    - mAP (mean Average Precision)
    - FPS (Frames Per Second) / Latency

Usage:
    from app.baseline.metrics import (
        calculate_iou,
        calculate_iou_batch,
        calculate_precision_recall,
        calculate_map,
        FPSBenchmark,
    )

    # Calculate IoU between two boxes
    iou = calculate_iou(pred_box, gt_box)

    # Calculate precision/recall
    result = calculate_precision_recall(
        predictions, ground_truths, iou_threshold=0.5
    )

    # Calculate mAP
    map_score = calculate_map(predictions, ground_truths, class_names)

    # Benchmark FPS
    benchmark = FPSBenchmark(model_path="yolov8n.pt")
    result = benchmark.run()
"""

from app.baseline.metrics.fps_benchmark import (
    BenchmarkResult,
    FPSBenchmark,
    benchmark_model,
    print_comparison_table,
)
from app.baseline.metrics.fps_benchmark import (
    compare_models as compare_models_fps,
)
from app.baseline.metrics.iou import (
    calculate_ciou,
    calculate_diou,
    calculate_giou,
    calculate_iou,
    calculate_iou_batch,
    convert_boxes,
    match_boxes_by_iou,
)
from app.baseline.metrics.map_calculator import (
    APResult,
    MAPCalculator,
    MAPResult,
    calculate_ap,
    calculate_coco_metrics,
    calculate_map,
    print_map_summary,
)
from app.baseline.metrics.map_calculator import (
    compare_models as compare_models_map,
)
from app.baseline.metrics.precision_recall import (
    Detection,
    EvaluationResult,
    GroundTruth,
    PrecisionRecallCurve,
    calculate_confusion_matrix,
    calculate_f1_score,
    calculate_precision,
    calculate_precision_recall,
    calculate_precision_recall_by_class,
    calculate_precision_recall_curve,
    calculate_recall,
    print_evaluation_summary,
)

__all__ = [
    # IoU
    "calculate_iou",
    "calculate_iou_batch",
    "calculate_giou",
    "calculate_diou",
    "calculate_ciou",
    "match_boxes_by_iou",
    "convert_boxes",
    # Precision/Recall
    "Detection",
    "GroundTruth",
    "EvaluationResult",
    "PrecisionRecallCurve",
    "calculate_precision",
    "calculate_recall",
    "calculate_f1_score",
    "calculate_precision_recall",
    "calculate_precision_recall_curve",
    "calculate_precision_recall_by_class",
    "calculate_confusion_matrix",
    "print_evaluation_summary",
    # mAP
    "APResult",
    "MAPResult",
    "MAPCalculator",
    "calculate_ap",
    "calculate_map",
    "calculate_coco_metrics",
    "print_map_summary",
    "compare_models_map",
    # FPS
    "BenchmarkResult",
    "FPSBenchmark",
    "benchmark_model",
    "compare_models_fps",
    "print_comparison_table",
]
