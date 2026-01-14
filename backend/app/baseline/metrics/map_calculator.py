"""
mAP (mean Average Precision) Calculator
========================================
The standard metric for evaluating object detection models.

mAP@0.5: Average Precision at IoU threshold 0.5
mAP@0.75: Average Precision at IoU threshold 0.75
mAP@[.5:.95]: Average of AP at IoU thresholds from 0.5 to 0.95 (step 0.05)

Average Precision (AP):
    Area under the Precision-Recall curve for a single class.

Mean Average Precision (mAP):
    Mean of AP across all classes.

Calculation Steps:
    1. Sort predictions by confidence (descending)
    2. For each prediction, determine if it's TP or FP based on IoU with GT
    3. Calculate cumulative precision and recall
    4. Compute AP as area under interpolated PR curve
    5. Average AP across all classes for mAP
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
from app.baseline.metrics.iou import calculate_iou_batch
from app.baseline.metrics.precision_recall import (
    Detection,
    GroundTruth,
    PrecisionRecallCurve,
    calculate_precision_recall_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class APResult:
    """Average Precision result for a single class."""

    class_id: int
    class_name: str
    ap: float
    precision_recall_curve: Optional[PrecisionRecallCurve] = None
    num_predictions: int = 0
    num_ground_truths: int = 0
    iou_threshold: float = 0.5

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "ap": self.ap,
            "num_predictions": self.num_predictions,
            "num_ground_truths": self.num_ground_truths,
            "iou_threshold": self.iou_threshold,
        }


@dataclass
class MAPResult:
    """Mean Average Precision result."""

    map_score: float
    ap_per_class: dict[str, APResult] = field(default_factory=dict)
    iou_threshold: float = 0.5
    num_classes: int = 0
    total_predictions: int = 0
    total_ground_truths: int = 0

    # Additional metrics at different IoU thresholds
    map_50: Optional[float] = None  # mAP@0.5
    map_75: Optional[float] = None  # mAP@0.75
    map_50_95: Optional[float] = None  # mAP@[.5:.95]

    def to_dict(self) -> dict:
        return {
            "map": self.map_score,
            "map_50": self.map_50,
            "map_75": self.map_75,
            "map_50_95": self.map_50_95,
            "iou_threshold": self.iou_threshold,
            "num_classes": self.num_classes,
            "total_predictions": self.total_predictions,
            "total_ground_truths": self.total_ground_truths,
            "ap_per_class": {
                name: result.to_dict() for name, result in self.ap_per_class.items()
            },
        }

    def __str__(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"mAP Evaluation Results",
            f"{'=' * 60}",
            f"mAP@{self.iou_threshold}: {self.map_score:.4f}",
        ]

        if self.map_50 is not None:
            lines.append(f"mAP@0.5: {self.map_50:.4f}")
        if self.map_75 is not None:
            lines.append(f"mAP@0.75: {self.map_75:.4f}")
        if self.map_50_95 is not None:
            lines.append(f"mAP@[.5:.95]: {self.map_50_95:.4f}")

        lines.extend(
            [
                f"\nTotal Predictions: {self.total_predictions}",
                f"Total Ground Truths: {self.total_ground_truths}",
                f"Number of Classes: {self.num_classes}",
                f"\nPer-Class AP:",
                "-" * 40,
            ]
        )

        for class_name, ap_result in sorted(self.ap_per_class.items()):
            lines.append(
                f"  {class_name:<15}: AP={ap_result.ap:.4f} "
                f"(GT={ap_result.num_ground_truths}, Pred={ap_result.num_predictions})"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class MAPCalculator:
    """
    Calculator for mAP (mean Average Precision).

    Supports:
        - Single IoU threshold evaluation (e.g., mAP@0.5)
        - Multiple IoU thresholds (mAP@[.5:.95])
        - Per-class AP breakdown
        - COCO-style evaluation

    Usage:
        calculator = MAPCalculator(class_names={0: "car", 1: "truck", ...})

        # Add predictions and ground truths
        calculator.add_predictions(predictions)
        calculator.add_ground_truths(ground_truths)

        # Calculate mAP
        result = calculator.calculate()
        print(result)
    """

    def __init__(
        self,
        class_names: dict[int, str],
        iou_thresholds: Optional[list[float]] = None,
        ignore_difficult: bool = True,
    ):
        """
        Initialize mAP calculator.

        Args:
            class_names: Mapping of class_id to class_name
            iou_thresholds: List of IoU thresholds to evaluate
            ignore_difficult: Whether to ignore difficult ground truths
        """
        self.class_names = class_names
        self.class_ids = set(class_names.keys())
        self.ignore_difficult = ignore_difficult

        # Default IoU thresholds (COCO style)
        if iou_thresholds is None:
            self.iou_thresholds = [
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
            ]
        else:
            self.iou_thresholds = sorted(iou_thresholds)

        # Storage for predictions and ground truths
        self.predictions: list[Detection] = []
        self.ground_truths: list[GroundTruth] = []

        # Cache for results
        self._results_cache: dict[float, MAPResult] = {}

    def reset(self):
        """Clear all predictions and ground truths."""
        self.predictions.clear()
        self.ground_truths.clear()
        self._results_cache.clear()

    def add_prediction(
        self,
        bbox: tuple[float, float, float, float],
        class_id: int,
        confidence: float,
        image_id: str = "",
    ):
        """Add a single prediction."""
        self.predictions.append(
            Detection(
                bbox=bbox,
                class_id=class_id,
                confidence=confidence,
                image_id=image_id,
            )
        )
        self._results_cache.clear()

    def add_predictions(self, predictions: list[Detection]):
        """Add multiple predictions."""
        self.predictions.extend(predictions)
        self._results_cache.clear()

    def add_ground_truth(
        self,
        bbox: tuple[float, float, float, float],
        class_id: int,
        image_id: str = "",
        is_difficult: bool = False,
    ):
        """Add a single ground truth."""
        self.ground_truths.append(
            GroundTruth(
                bbox=bbox,
                class_id=class_id,
                image_id=image_id,
                is_difficult=is_difficult,
            )
        )
        self._results_cache.clear()

    def add_ground_truths(self, ground_truths: list[GroundTruth]):
        """Add multiple ground truths."""
        self.ground_truths.extend(ground_truths)
        self._results_cache.clear()

    def calculate_ap(
        self,
        class_id: int,
        iou_threshold: float = 0.5,
    ) -> APResult:
        """
        Calculate Average Precision for a single class.

        Args:
            class_id: Class ID to evaluate
            iou_threshold: IoU threshold for matching

        Returns:
            APResult for the class
        """
        class_name = self.class_names.get(class_id, f"class_{class_id}")

        # Filter predictions and ground truths for this class
        class_preds = [p for p in self.predictions if p.class_id == class_id]
        class_gts = [gt for gt in self.ground_truths if gt.class_id == class_id]

        # Filter difficult ground truths
        if self.ignore_difficult:
            class_gts_eval = [gt for gt in class_gts if not gt.is_difficult]
        else:
            class_gts_eval = class_gts

        num_predictions = len(class_preds)
        num_ground_truths = len(class_gts_eval)

        # Handle edge cases
        if num_predictions == 0 or num_ground_truths == 0:
            return APResult(
                class_id=class_id,
                class_name=class_name,
                ap=0.0,
                num_predictions=num_predictions,
                num_ground_truths=num_ground_truths,
                iou_threshold=iou_threshold,
            )

        # Calculate precision-recall curve
        pr_curve = calculate_precision_recall_curve(
            predictions=class_preds,
            ground_truths=class_gts_eval,
            iou_threshold=iou_threshold,
            ignore_difficult=False,  # Already filtered
        )

        return APResult(
            class_id=class_id,
            class_name=class_name,
            ap=pr_curve.ap,
            precision_recall_curve=pr_curve,
            num_predictions=num_predictions,
            num_ground_truths=num_ground_truths,
            iou_threshold=iou_threshold,
        )

    def calculate(self, iou_threshold: float = 0.5) -> MAPResult:
        """
        Calculate mAP at a specific IoU threshold.

        Args:
            iou_threshold: IoU threshold for matching

        Returns:
            MAPResult with overall mAP and per-class AP
        """
        # Check cache
        if iou_threshold in self._results_cache:
            return self._results_cache[iou_threshold]

        # Calculate AP for each class
        ap_results = {}
        ap_values = []

        for class_id, class_name in self.class_names.items():
            ap_result = self.calculate_ap(class_id, iou_threshold)
            ap_results[class_name] = ap_result

            # Only include classes with ground truths in mAP calculation
            if ap_result.num_ground_truths > 0:
                ap_values.append(ap_result.ap)

        # Calculate mean AP
        if len(ap_values) > 0:
            map_score = float(np.mean(ap_values))
        else:
            map_score = 0.0

        result = MAPResult(
            map_score=map_score,
            ap_per_class=ap_results,
            iou_threshold=iou_threshold,
            num_classes=len(ap_values),
            total_predictions=len(self.predictions),
            total_ground_truths=len(
                [
                    gt
                    for gt in self.ground_truths
                    if not (self.ignore_difficult and gt.is_difficult)
                ]
            ),
        )

        # Cache result
        self._results_cache[iou_threshold] = result

        return result

    def calculate_coco_map(self) -> MAPResult:
        """
        Calculate COCO-style mAP metrics.

        Returns:
            MAPResult with mAP@0.5, mAP@0.75, and mAP@[.5:.95]
        """
        # Calculate mAP at each IoU threshold
        map_at_thresholds = []

        for iou in self.iou_thresholds:
            result = self.calculate(iou)
            map_at_thresholds.append(result.map_score)

        # Get specific thresholds
        map_50 = self.calculate(0.5).map_score if 0.5 in self.iou_thresholds else None
        map_75 = self.calculate(0.75).map_score if 0.75 in self.iou_thresholds else None

        # Calculate mAP@[.5:.95]
        map_50_95 = float(np.mean(map_at_thresholds)) if map_at_thresholds else 0.0

        # Get detailed result at IoU=0.5 as base
        base_result = self.calculate(0.5)

        # Update with COCO metrics
        base_result.map_50 = map_50
        base_result.map_75 = map_75
        base_result.map_50_95 = map_50_95

        return base_result

    def calculate_per_size(
        self,
        iou_threshold: float = 0.5,
        small_max: int = 32 * 32,
        medium_max: int = 96 * 96,
    ) -> dict[str, float]:
        """
        Calculate mAP for different object sizes (COCO-style).

        Args:
            iou_threshold: IoU threshold for matching
            small_max: Maximum area for small objects
            medium_max: Maximum area for medium objects

        Returns:
            Dictionary with mAP for small, medium, and large objects
        """

        def get_area(bbox):
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        def categorize_by_size(items, is_gt=False):
            small, medium, large = [], [], []
            for item in items:
                area = get_area(item.bbox)
                if area < small_max:
                    small.append(item)
                elif area < medium_max:
                    medium.append(item)
                else:
                    large.append(item)
            return small, medium, large

        # Categorize predictions and ground truths
        pred_small, pred_medium, pred_large = categorize_by_size(self.predictions)
        gt_small, gt_medium, gt_large = categorize_by_size(self.ground_truths)

        results = {}

        for size_name, preds, gts in [
            ("small", pred_small, gt_small),
            ("medium", pred_medium, gt_medium),
            ("large", pred_large, gt_large),
        ]:
            if len(gts) == 0:
                results[f"mAP_{size_name}"] = 0.0
                continue

            # Create temporary calculator
            temp_calc = MAPCalculator(
                class_names=self.class_names,
                iou_thresholds=[iou_threshold],
                ignore_difficult=self.ignore_difficult,
            )
            temp_calc.predictions = preds
            temp_calc.ground_truths = gts

            result = temp_calc.calculate(iou_threshold)
            results[f"mAP_{size_name}"] = result.map_score

        return results


def calculate_ap(
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
    class_id: Optional[int] = None,
) -> float:
    """
    Calculate Average Precision.

    Convenience function for quick AP calculation.

    Args:
        predictions: List of Detection objects
        ground_truths: List of GroundTruth objects
        iou_threshold: IoU threshold for matching
        class_id: If specified, calculate AP for this class only

    Returns:
        AP value (0 to 1)
    """
    # Filter by class if specified
    if class_id is not None:
        predictions = [p for p in predictions if p.class_id == class_id]
        ground_truths = [gt for gt in ground_truths if gt.class_id == class_id]

    if len(predictions) == 0 or len(ground_truths) == 0:
        return 0.0

    pr_curve = calculate_precision_recall_curve(
        predictions=predictions,
        ground_truths=ground_truths,
        iou_threshold=iou_threshold,
    )

    return pr_curve.ap


def calculate_map(
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate mean Average Precision.

    Convenience function for quick mAP calculation.

    Args:
        predictions: List of Detection objects
        ground_truths: List of GroundTruth objects
        class_names: Mapping of class_id to class_name
        iou_threshold: IoU threshold for matching

    Returns:
        mAP value (0 to 1)
    """
    calculator = MAPCalculator(class_names=class_names)
    calculator.predictions = predictions
    calculator.ground_truths = ground_truths

    result = calculator.calculate(iou_threshold)
    return result.map_score


def calculate_coco_metrics(
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    class_names: dict[int, str],
) -> dict[str, float]:
    """
    Calculate COCO-style metrics.

    Returns metrics similar to COCO evaluation:
        - mAP@0.5
        - mAP@0.75
        - mAP@[.5:.95]
        - mAP_small, mAP_medium, mAP_large

    Args:
        predictions: List of Detection objects
        ground_truths: List of GroundTruth objects
        class_names: Mapping of class_id to class_name

    Returns:
        Dictionary with all COCO metrics
    """
    calculator = MAPCalculator(class_names=class_names)
    calculator.predictions = predictions
    calculator.ground_truths = ground_truths

    # Calculate main metrics
    coco_result = calculator.calculate_coco_map()

    # Calculate per-size metrics
    size_metrics = calculator.calculate_per_size()

    return {
        "mAP": coco_result.map_50_95,
        "mAP@0.5": coco_result.map_50,
        "mAP@0.75": coco_result.map_75,
        "mAP@[.5:.95]": coco_result.map_50_95,
        **size_metrics,
    }


def print_map_summary(result: MAPResult):
    """Print formatted mAP summary."""
    print(str(result))


def compare_models(
    model_results: dict[str, MAPResult],
) -> str:
    """
    Generate comparison table for multiple models.

    Args:
        model_results: Dictionary mapping model name to MAPResult

    Returns:
        Formatted comparison string
    """
    lines = [
        "\n" + "=" * 80,
        "MODEL COMPARISON",
        "=" * 80,
        "",
    ]

    # Header
    header = f"{'Model':<20} {'mAP@0.5':>10} {'mAP@0.75':>10} {'mAP@[.5:.95]':>12}"
    lines.append(header)
    lines.append("-" * 60)

    # Model rows
    for model_name, result in sorted(model_results.items()):
        map_50 = f"{result.map_50:.4f}" if result.map_50 else "N/A"
        map_75 = f"{result.map_75:.4f}" if result.map_75 else "N/A"
        map_50_95 = f"{result.map_50_95:.4f}" if result.map_50_95 else "N/A"

        lines.append(f"{model_name:<20} {map_50:>10} {map_75:>10} {map_50_95:>12}")

    lines.append("=" * 80)

    return "\n".join(lines)
