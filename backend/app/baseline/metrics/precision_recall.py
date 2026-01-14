"""
Precision/Recall Calculation Module
====================================
Metrics for evaluating object detection performance.

Precision: Of all detections made, how many are correct?
    Precision = TP / (TP + FP)

Recall: Of all ground truth objects, how many did we detect?
    Recall = TP / (TP + FN)

F1-Score: Harmonic mean of Precision and Recall
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

Where:
    TP (True Positive): Correct detection (IoU >= threshold)
    FP (False Positive): Incorrect detection (IoU < threshold or duplicate)
    FN (False Negative): Missed ground truth object
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
from app.baseline.metrics.iou import calculate_iou_batch, match_boxes_by_iou

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int
    confidence: float
    image_id: str = ""

    def __post_init__(self):
        self.bbox = tuple(float(x) for x in self.bbox)


@dataclass
class GroundTruth:
    """Single ground truth annotation."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int
    is_difficult: bool = False
    image_id: str = ""

    def __post_init__(self):
        self.bbox = tuple(float(x) for x in self.bbox)


@dataclass
class EvaluationResult:
    """Result of precision/recall evaluation."""

    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_predictions: int
    total_ground_truths: int
    iou_threshold: float
    class_id: Optional[int] = None
    class_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_predictions": self.total_predictions,
            "total_ground_truths": self.total_ground_truths,
            "iou_threshold": self.iou_threshold,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    def __str__(self) -> str:
        class_info = f" ({self.class_name})" if self.class_name else ""
        return (
            f"EvaluationResult{class_info}:\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall: {self.recall:.4f}\n"
            f"  F1-Score: {self.f1_score:.4f}\n"
            f"  TP: {self.true_positives}, FP: {self.false_positives}, FN: {self.false_negatives}\n"
            f"  IoU Threshold: {self.iou_threshold}"
        )


@dataclass
class PrecisionRecallPoint:
    """Single point on precision-recall curve."""

    precision: float
    recall: float
    confidence_threshold: float
    true_positives: int
    false_positives: int


@dataclass
class PrecisionRecallCurve:
    """Precision-Recall curve data."""

    points: list[PrecisionRecallPoint] = field(default_factory=list)
    ap: float = 0.0  # Average Precision (area under curve)
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    iou_threshold: float = 0.5

    def get_precision_at_recall(self, target_recall: float) -> float:
        """Get precision at a specific recall level."""
        if not self.points:
            return 0.0

        # Find the point with recall closest to target
        closest_point = min(self.points, key=lambda p: abs(p.recall - target_recall))
        return closest_point.precision

    def get_max_f1(self) -> tuple[float, float, float]:
        """Get maximum F1 score and corresponding precision/recall."""
        if not self.points:
            return 0.0, 0.0, 0.0

        max_f1 = 0.0
        best_p = 0.0
        best_r = 0.0

        for point in self.points:
            if point.precision + point.recall > 0:
                f1 = (
                    2
                    * point.precision
                    * point.recall
                    / (point.precision + point.recall)
                )
                if f1 > max_f1:
                    max_f1 = f1
                    best_p = point.precision
                    best_r = point.recall

        return max_f1, best_p, best_r

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays (recalls, precisions)."""
        if not self.points:
            return np.array([]), np.array([])

        recalls = np.array([p.recall for p in self.points])
        precisions = np.array([p.precision for p in self.points])
        return recalls, precisions


def calculate_precision(tp: int, fp: int) -> float:
    """
    Calculate precision.

    Args:
        tp: True positives
        fp: False positives

    Returns:
        Precision value (0 to 1)
    """
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calculate_recall(tp: int, fn: int) -> float:
    """
    Calculate recall.

    Args:
        tp: True positives
        fn: False negatives

    Returns:
        Recall value (0 to 1)
    """
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1-score (harmonic mean of precision and recall).

    Args:
        precision: Precision value
        recall: Recall value

    Returns:
        F1-score (0 to 1)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_precision_recall(
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
    class_id: Optional[int] = None,
    ignore_difficult: bool = True,
) -> EvaluationResult:
    """
    Calculate precision and recall for object detection.

    Args:
        predictions: List of Detection objects
        ground_truths: List of GroundTruth objects
        iou_threshold: Minimum IoU for a valid match
        class_id: If specified, evaluate only this class
        ignore_difficult: Whether to ignore difficult ground truths

    Returns:
        EvaluationResult with precision, recall, and other metrics
    """
    # Filter by class if specified
    if class_id is not None:
        predictions = [p for p in predictions if p.class_id == class_id]
        ground_truths = [gt for gt in ground_truths if gt.class_id == class_id]

    # Filter difficult ground truths if needed
    if ignore_difficult:
        non_difficult_gts = [gt for gt in ground_truths if not gt.is_difficult]
    else:
        non_difficult_gts = ground_truths

    total_predictions = len(predictions)
    total_ground_truths = len(non_difficult_gts)

    if total_predictions == 0 and total_ground_truths == 0:
        return EvaluationResult(
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            total_predictions=0,
            total_ground_truths=0,
            iou_threshold=iou_threshold,
            class_id=class_id,
        )

    if total_predictions == 0:
        return EvaluationResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            true_positives=0,
            false_positives=0,
            false_negatives=total_ground_truths,
            total_predictions=0,
            total_ground_truths=total_ground_truths,
            iou_threshold=iou_threshold,
            class_id=class_id,
        )

    if total_ground_truths == 0:
        return EvaluationResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            true_positives=0,
            false_positives=total_predictions,
            false_negatives=0,
            total_predictions=total_predictions,
            total_ground_truths=0,
            iou_threshold=iou_threshold,
            class_id=class_id,
        )

    # Group by image
    pred_by_image: dict[str, list[Detection]] = {}
    gt_by_image: dict[str, list[GroundTruth]] = {}

    for pred in predictions:
        if pred.image_id not in pred_by_image:
            pred_by_image[pred.image_id] = []
        pred_by_image[pred.image_id].append(pred)

    for gt in non_difficult_gts:
        if gt.image_id not in gt_by_image:
            gt_by_image[gt.image_id] = []
        gt_by_image[gt.image_id].append(gt)

    # Calculate TP, FP, FN
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    all_image_ids = set(pred_by_image.keys()) | set(gt_by_image.keys())

    for image_id in all_image_ids:
        img_preds = pred_by_image.get(image_id, [])
        img_gts = gt_by_image.get(image_id, [])

        if len(img_preds) == 0:
            false_negatives += len(img_gts)
            continue

        if len(img_gts) == 0:
            false_positives += len(img_preds)
            continue

        # Sort predictions by confidence (descending)
        img_preds_sorted = sorted(img_preds, key=lambda x: x.confidence, reverse=True)

        # Convert to numpy arrays
        pred_boxes = np.array([p.bbox for p in img_preds_sorted])
        gt_boxes = np.array([gt.bbox for gt in img_gts])

        # Match predictions to ground truths
        matches, unmatched_preds, unmatched_gts = match_boxes_by_iou(
            pred_boxes, gt_boxes, iou_threshold=iou_threshold
        )

        true_positives += len(matches)
        false_positives += len(unmatched_preds)
        false_negatives += len(unmatched_gts)

    # Calculate metrics
    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)
    f1 = calculate_f1_score(precision, recall)

    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1_score=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        total_predictions=total_predictions,
        total_ground_truths=total_ground_truths,
        iou_threshold=iou_threshold,
        class_id=class_id,
    )


def calculate_precision_recall_curve(
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
    class_id: Optional[int] = None,
    num_points: int = 101,
    ignore_difficult: bool = True,
) -> PrecisionRecallCurve:
    """
    Calculate precision-recall curve.

    Args:
        predictions: List of Detection objects
        ground_truths: List of GroundTruth objects
        iou_threshold: Minimum IoU for a valid match
        class_id: If specified, evaluate only this class
        num_points: Number of points on the curve
        ignore_difficult: Whether to ignore difficult ground truths

    Returns:
        PrecisionRecallCurve with points and average precision
    """
    # Filter by class if specified
    if class_id is not None:
        predictions = [p for p in predictions if p.class_id == class_id]
        ground_truths = [gt for gt in ground_truths if gt.class_id == class_id]

    # Filter difficult ground truths
    if ignore_difficult:
        ground_truths = [gt for gt in ground_truths if not gt.is_difficult]

    if len(predictions) == 0 or len(ground_truths) == 0:
        return PrecisionRecallCurve(
            points=[],
            ap=0.0,
            class_id=class_id,
            iou_threshold=iou_threshold,
        )

    # Sort predictions by confidence (descending)
    sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)

    # Group ground truths by image
    gt_by_image: dict[str, list[tuple[int, GroundTruth]]] = {}
    for i, gt in enumerate(ground_truths):
        if gt.image_id not in gt_by_image:
            gt_by_image[gt.image_id] = []
        gt_by_image[gt.image_id].append((i, gt))

    # Track which ground truths have been matched
    gt_matched = [False] * len(ground_truths)

    # Calculate TP/FP for each prediction
    tp_list = []
    fp_list = []
    confidence_list = []

    for pred in sorted_preds:
        confidence_list.append(pred.confidence)

        img_gts = gt_by_image.get(pred.image_id, [])

        if len(img_gts) == 0:
            # No ground truths in this image
            tp_list.append(0)
            fp_list.append(1)
            continue

        # Find best matching ground truth
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in img_gts:
            if gt_matched[gt_idx]:
                continue

            # Calculate IoU
            pred_box = np.array(pred.bbox)
            gt_box = np.array(gt.bbox)
            iou_matrix = calculate_iou_batch(
                pred_box.reshape(1, -1), gt_box.reshape(1, -1)
            )
            iou = iou_matrix[0, 0]

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            # True positive
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[best_gt_idx] = True
        else:
            # False positive
            tp_list.append(0)
            fp_list.append(1)

    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    # Calculate precision and recall at each point
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / len(ground_truths)

    # Build PR curve points
    points = []
    for i in range(len(sorted_preds)):
        points.append(
            PrecisionRecallPoint(
                precision=float(precisions[i]),
                recall=float(recalls[i]),
                confidence_threshold=float(confidence_list[i]),
                true_positives=int(tp_cumsum[i]),
                false_positives=int(fp_cumsum[i]),
            )
        )

    # Calculate Average Precision (AP) using 11-point interpolation or all-point
    ap = _calculate_ap_from_pr(recalls, precisions)

    return PrecisionRecallCurve(
        points=points,
        ap=ap,
        class_id=class_id,
        iou_threshold=iou_threshold,
    )


def _calculate_ap_from_pr(
    recalls: np.ndarray,
    precisions: np.ndarray,
    use_11_point: bool = False,
) -> float:
    """
    Calculate Average Precision from precision-recall arrays.

    Args:
        recalls: Array of recall values
        precisions: Array of precision values
        use_11_point: Use 11-point interpolation (VOC2007 style) or all-point (VOC2010+)

    Returns:
        Average Precision value
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    if use_11_point:
        # 11-point interpolation (VOC2007)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            # Get maximum precision at recall >= t
            mask = recalls >= t
            if mask.any():
                p = precisions[mask].max()
            else:
                p = 0.0
            ap += p / 11
        return ap
    else:
        # All-point interpolation (VOC2010+, COCO)
        # Prepend sentinel values
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Make precision monotonically decreasing
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # Find where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # Sum (Δrecall * precision)
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return float(ap)


def calculate_precision_recall_by_class(
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
    ignore_difficult: bool = True,
) -> dict[str, EvaluationResult]:
    """
    Calculate precision and recall for each class separately.

    Args:
        predictions: List of Detection objects
        ground_truths: List of GroundTruth objects
        class_names: Mapping of class_id to class_name
        iou_threshold: Minimum IoU for a valid match
        ignore_difficult: Whether to ignore difficult ground truths

    Returns:
        Dictionary mapping class names to EvaluationResult
    """
    results = {}

    for class_id, class_name in class_names.items():
        result = calculate_precision_recall(
            predictions=predictions,
            ground_truths=ground_truths,
            iou_threshold=iou_threshold,
            class_id=class_id,
            ignore_difficult=ignore_difficult,
        )
        result.class_name = class_name
        results[class_name] = result

    return results


def calculate_confusion_matrix(
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
) -> np.ndarray:
    """
    Calculate confusion matrix for multi-class object detection.

    The matrix is (num_classes + 1) x (num_classes + 1) where the last
    row/column represents background (no detection / false positive).

    Args:
        predictions: List of Detection objects
        ground_truths: List of GroundTruth objects
        class_names: Mapping of class_id to class_name
        iou_threshold: Minimum IoU for a valid match
        confidence_threshold: Minimum confidence to consider a detection

    Returns:
        Confusion matrix as numpy array
    """
    # Filter predictions by confidence
    predictions = [p for p in predictions if p.confidence >= confidence_threshold]

    num_classes = len(class_names)
    class_ids = sorted(class_names.keys())
    class_id_to_idx = {cid: idx for idx, cid in enumerate(class_ids)}

    # Matrix: rows = ground truth, cols = predictions
    # Last row/col is background
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    # Group by image
    pred_by_image: dict[str, list[Detection]] = {}
    gt_by_image: dict[str, list[GroundTruth]] = {}

    for pred in predictions:
        if pred.image_id not in pred_by_image:
            pred_by_image[pred.image_id] = []
        pred_by_image[pred.image_id].append(pred)

    for gt in ground_truths:
        if gt.image_id not in gt_by_image:
            gt_by_image[gt.image_id] = []
        gt_by_image[gt.image_id].append(gt)

    all_image_ids = set(pred_by_image.keys()) | set(gt_by_image.keys())

    for image_id in all_image_ids:
        img_preds = pred_by_image.get(image_id, [])
        img_gts = gt_by_image.get(image_id, [])

        # Track matched ground truths
        gt_matched = [False] * len(img_gts)

        # Sort predictions by confidence
        img_preds_sorted = sorted(img_preds, key=lambda x: x.confidence, reverse=True)

        for pred in img_preds_sorted:
            pred_class_idx = class_id_to_idx.get(pred.class_id)
            if pred_class_idx is None:
                continue

            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1
            best_gt_class_idx = -1

            for gt_idx, gt in enumerate(img_gts):
                if gt_matched[gt_idx]:
                    continue

                gt_class_idx = class_id_to_idx.get(gt.class_id)
                if gt_class_idx is None:
                    continue

                # Calculate IoU
                pred_box = np.array(pred.bbox)
                gt_box = np.array(gt.bbox)
                iou_matrix = calculate_iou_batch(
                    pred_box.reshape(1, -1), gt_box.reshape(1, -1)
                )
                iou = iou_matrix[0, 0]

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt_class_idx = gt_class_idx

            if best_iou >= iou_threshold:
                # Matched: add to confusion matrix
                matrix[best_gt_class_idx, pred_class_idx] += 1
                gt_matched[best_gt_idx] = True
            else:
                # False positive (background in GT, predicted class)
                matrix[num_classes, pred_class_idx] += 1

        # Count unmatched ground truths (false negatives)
        for gt_idx, gt in enumerate(img_gts):
            if not gt_matched[gt_idx]:
                gt_class_idx = class_id_to_idx.get(gt.class_id)
                if gt_class_idx is not None:
                    # Missed detection (GT class, background prediction)
                    matrix[gt_class_idx, num_classes] += 1

    return matrix


def print_evaluation_summary(
    results: dict[str, EvaluationResult],
    overall: Optional[EvaluationResult] = None,
):
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    if overall:
        print(f"\nOverall Metrics (IoU={overall.iou_threshold}):")
        print(f"  Precision: {overall.precision:.4f}")
        print(f"  Recall:    {overall.recall:.4f}")
        print(f"  F1-Score:  {overall.f1_score:.4f}")
        print(
            f"  TP: {overall.true_positives}, FP: {overall.false_positives}, FN: {overall.false_negatives}"
        )

    print(f"\nPer-Class Metrics:")
    print("-" * 70)
    print(
        f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}"
    )
    print("-" * 70)

    for class_name, result in sorted(results.items()):
        print(
            f"{class_name:<15} "
            f"{result.precision:>10.4f} "
            f"{result.recall:>10.4f} "
            f"{result.f1_score:>10.4f} "
            f"{result.true_positives:>8} "
            f"{result.false_positives:>8} "
            f"{result.false_negatives:>8}"
        )

    print("=" * 70)
