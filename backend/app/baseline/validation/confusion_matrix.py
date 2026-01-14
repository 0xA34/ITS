"""
Confusion Matrix Generator Module
=================================
Generate and visualize confusion matrices for object detection evaluation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConfusionMatrixData:
    """Data structure for confusion matrix."""

    class_names: list[str]
    matrix: np.ndarray  # Shape: (num_classes + 1, num_classes + 1) for background
    true_positives: dict[str, int] = field(default_factory=dict)
    false_positives: dict[str, int] = field(default_factory=dict)
    false_negatives: dict[str, int] = field(default_factory=dict)
    total_predictions: int = 0
    total_ground_truths: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "class_names": self.class_names,
            "matrix": self.matrix.tolist(),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_predictions": self.total_predictions,
            "total_ground_truths": self.total_ground_truths,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConfusionMatrixData":
        """Create from dictionary."""
        return cls(
            class_names=data["class_names"],
            matrix=np.array(data["matrix"]),
            true_positives=data.get("true_positives", {}),
            false_positives=data.get("false_positives", {}),
            false_negatives=data.get("false_negatives", {}),
            total_predictions=data.get("total_predictions", 0),
            total_ground_truths=data.get("total_ground_truths", 0),
        )


@dataclass
class DetectionMatch:
    """Represents a match between prediction and ground truth."""

    pred_class: int
    gt_class: int
    iou: float
    confidence: float
    is_correct: bool


class ConfusionMatrixGenerator:
    """
    Generate confusion matrices for object detection evaluation.

    For object detection, the confusion matrix tracks:
    - Correct detections (diagonal)
    - Class confusions (off-diagonal)
    - False positives (background predicted as object)
    - False negatives (object predicted as background)
    """

    def __init__(
        self,
        class_names: Optional[dict[int, str]] = None,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.25,
    ):
        """
        Initialize ConfusionMatrixGenerator.

        Args:
            class_names: Mapping of class IDs to class names
            iou_threshold: IoU threshold for matching predictions to ground truths
            confidence_threshold: Minimum confidence for predictions
        """
        self.class_names = class_names or {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        # Create ordered list of class names
        self.class_ids = sorted(self.class_names.keys())
        self.class_list = [self.class_names[cid] for cid in self.class_ids]
        self.num_classes = len(self.class_list)

        # Initialize matrix (num_classes + 1 for background)
        # Last row/column represents background (FP/FN)
        self._matrix = np.zeros(
            (self.num_classes + 1, self.num_classes + 1), dtype=np.int64
        )

        # Track statistics
        self._true_positives: dict[str, int] = {name: 0 for name in self.class_list}
        self._false_positives: dict[str, int] = {name: 0 for name in self.class_list}
        self._false_negatives: dict[str, int] = {name: 0 for name in self.class_list}
        self._total_predictions = 0
        self._total_ground_truths = 0

        # ID to index mapping
        self._id_to_idx = {cid: idx for idx, cid in enumerate(self.class_ids)}

    def reset(self):
        """Reset the confusion matrix."""
        self._matrix = np.zeros(
            (self.num_classes + 1, self.num_classes + 1), dtype=np.int64
        )
        self._true_positives = {name: 0 for name in self.class_list}
        self._false_positives = {name: 0 for name in self.class_list}
        self._false_negatives = {name: 0 for name in self.class_list}
        self._total_predictions = 0
        self._total_ground_truths = 0

    def _compute_iou(
        self,
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float],
    ) -> float:
        """
        Compute Intersection over Union (IoU) between two boxes.

        Args:
            box1: First box (x1, y1, x2, y2)
            box2: Second box (x1, y1, x2, y2)

        Returns:
            IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def add_batch(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
    ):
        """
        Add a batch of predictions and ground truths.

        Args:
            predictions: List of predictions, each with keys:
                - 'bbox': (x1, y1, x2, y2)
                - 'class_id': int
                - 'confidence': float
            ground_truths: List of ground truths, each with keys:
                - 'bbox': (x1, y1, x2, y2)
                - 'class_id': int
        """
        # Filter predictions by confidence
        filtered_preds = [
            p
            for p in predictions
            if p.get("confidence", 0) >= self.confidence_threshold
        ]

        # Filter by valid class IDs
        filtered_preds = [
            p for p in filtered_preds if p.get("class_id") in self._id_to_idx
        ]
        filtered_gts = [
            gt for gt in ground_truths if gt.get("class_id") in self._id_to_idx
        ]

        self._total_predictions += len(filtered_preds)
        self._total_ground_truths += len(filtered_gts)

        # Track which ground truths have been matched
        gt_matched = [False] * len(filtered_gts)

        # Sort predictions by confidence (descending)
        sorted_preds = sorted(
            filtered_preds, key=lambda x: x.get("confidence", 0), reverse=True
        )

        # Match predictions to ground truths
        for pred in sorted_preds:
            pred_class = pred["class_id"]
            pred_bbox = pred["bbox"]
            pred_idx = self._id_to_idx.get(pred_class)

            if pred_idx is None:
                continue

            best_iou = 0.0
            best_gt_idx = -1

            # Find best matching ground truth
            for gt_idx, gt in enumerate(filtered_gts):
                if gt_matched[gt_idx]:
                    continue

                iou = self._compute_iou(pred_bbox, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                # Matched with a ground truth
                gt_matched[best_gt_idx] = True
                gt_class = filtered_gts[best_gt_idx]["class_id"]
                gt_idx = self._id_to_idx[gt_class]

                self._matrix[gt_idx, pred_idx] += 1

                if pred_class == gt_class:
                    # True positive
                    class_name = self.class_names[pred_class]
                    self._true_positives[class_name] += 1
                else:
                    # Class confusion
                    pass
            else:
                # False positive (no matching ground truth)
                self._matrix[self.num_classes, pred_idx] += 1
                class_name = self.class_names[pred_class]
                self._false_positives[class_name] += 1

        # Count unmatched ground truths as false negatives
        for gt_idx, matched in enumerate(gt_matched):
            if not matched:
                gt_class = filtered_gts[gt_idx]["class_id"]
                gt_class_idx = self._id_to_idx[gt_class]
                self._matrix[gt_class_idx, self.num_classes] += 1
                class_name = self.class_names[gt_class]
                self._false_negatives[class_name] += 1

    def add_image_results(
        self,
        pred_boxes: np.ndarray,
        pred_classes: np.ndarray,
        pred_confidences: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ):
        """
        Add results for a single image using numpy arrays.

        Args:
            pred_boxes: Predicted boxes, shape (N, 4) in xyxy format
            pred_classes: Predicted classes, shape (N,)
            pred_confidences: Prediction confidences, shape (N,)
            gt_boxes: Ground truth boxes, shape (M, 4) in xyxy format
            gt_classes: Ground truth classes, shape (M,)
        """
        predictions = []
        for i in range(len(pred_boxes)):
            predictions.append(
                {
                    "bbox": tuple(pred_boxes[i]),
                    "class_id": int(pred_classes[i]),
                    "confidence": float(pred_confidences[i]),
                }
            )

        ground_truths = []
        for i in range(len(gt_boxes)):
            ground_truths.append(
                {
                    "bbox": tuple(gt_boxes[i]),
                    "class_id": int(gt_classes[i]),
                }
            )

        self.add_batch(predictions, ground_truths)

    def get_matrix_data(self) -> ConfusionMatrixData:
        """Get the confusion matrix data."""
        return ConfusionMatrixData(
            class_names=self.class_list + ["background"],
            matrix=self._matrix.copy(),
            true_positives=self._true_positives.copy(),
            false_positives=self._false_positives.copy(),
            false_negatives=self._false_negatives.copy(),
            total_predictions=self._total_predictions,
            total_ground_truths=self._total_ground_truths,
        )

    def get_normalized_matrix(self, mode: str = "true") -> np.ndarray:
        """
        Get normalized confusion matrix.

        Args:
            mode: Normalization mode
                - 'true': Normalize by true labels (rows sum to 1)
                - 'pred': Normalize by predictions (columns sum to 1)
                - 'all': Normalize by total count

        Returns:
            Normalized matrix as numpy array
        """
        matrix = self._matrix.astype(np.float64)

        if mode == "true":
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return matrix / row_sums
        elif mode == "pred":
            col_sums = matrix.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            return matrix / col_sums
        elif mode == "all":
            total = matrix.sum()
            return matrix / total if total > 0 else matrix
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

    def get_per_class_metrics(self) -> dict[str, dict[str, float]]:
        """
        Calculate per-class precision, recall, and F1.

        Returns:
            Dictionary with metrics for each class
        """
        metrics = {}

        for class_name in self.class_list:
            tp = self._true_positives.get(class_name, 0)
            fp = self._false_positives.get(class_name, 0)
            fn = self._false_negatives.get(class_name, 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics[class_name] = {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

        return metrics

    def generate_text_report(self) -> str:
        """Generate a text report of the confusion matrix."""
        lines = []
        lines.append("=" * 70)
        lines.append("Confusion Matrix Report")
        lines.append("=" * 70)
        lines.append(f"Total Predictions: {self._total_predictions}")
        lines.append(f"Total Ground Truths: {self._total_ground_truths}")
        lines.append(f"IoU Threshold: {self.iou_threshold}")
        lines.append(f"Confidence Threshold: {self.confidence_threshold}")
        lines.append("")

        # Print matrix
        lines.append("Confusion Matrix (rows=GT, cols=Pred):")
        lines.append("-" * 70)

        # Header
        header = "GT\\Pred".ljust(12)
        for name in self.class_list:
            header += name[:8].center(10)
        header += "BG".center(10)
        lines.append(header)
        lines.append("-" * 70)

        # Matrix rows
        for i, row_name in enumerate(self.class_list + ["BG"]):
            row_str = row_name[:10].ljust(12)
            for j in range(self.num_classes + 1):
                row_str += str(int(self._matrix[i, j])).center(10)
            lines.append(row_str)

        lines.append("")

        # Per-class metrics
        lines.append("Per-Class Metrics:")
        lines.append("-" * 70)
        lines.append(
            f"{'Class':<12} {'TP':>8} {'FP':>8} {'FN':>8} "
            f"{'Precision':>10} {'Recall':>10} {'F1':>10}"
        )
        lines.append("-" * 70)

        metrics = self.get_per_class_metrics()
        for class_name, m in metrics.items():
            lines.append(
                f"{class_name:<12} {m['true_positives']:>8} {m['false_positives']:>8} "
                f"{m['false_negatives']:>8} {m['precision']:>10.4f} "
                f"{m['recall']:>10.4f} {m['f1_score']:>10.4f}"
            )

        lines.append("=" * 70)

        return "\n".join(lines)

    def generate_html_report(self) -> str:
        """Generate an HTML report with the confusion matrix visualization."""
        metrics = self.get_per_class_metrics()
        normalized = self.get_normalized_matrix("true")

        html = """<!DOCTYPE html>
<html>
<head>
    <title>Confusion Matrix Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 40px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 30px; }
        .summary {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #2196F3;
        }
        .stat-card .label {
            color: #666;
            margin-top: 5px;
        }
        table {
            border-collapse: collapse;
            margin: 20px 0;
            width: 100%;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }
        th {
            background: #f0f0f0;
            font-weight: 600;
        }
        .matrix-cell {
            min-width: 60px;
        }
        .diagonal {
            background: #e3f2fd;
            font-weight: bold;
        }
        .high-value {
            background: #ffebee;
        }
        .metrics-table th {
            background: #e3f2fd;
        }
        .good { color: #4caf50; }
        .medium { color: #ff9800; }
        .poor { color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Confusion Matrix Report</h1>

        <div class="summary">
            <div class="stat-card">
                <div class="value">"""
        html += str(self._total_predictions)
        html += """</div>
                <div class="label">Total Predictions</div>
            </div>
            <div class="stat-card">
                <div class="value">"""
        html += str(self._total_ground_truths)
        html += """</div>
                <div class="label">Total Ground Truths</div>
            </div>
            <div class="stat-card">
                <div class="value">"""
        html += f"{self.iou_threshold}"
        html += """</div>
                <div class="label">IoU Threshold</div>
            </div>
            <div class="stat-card">
                <div class="value">"""
        html += f"{self.confidence_threshold}"
        html += """</div>
                <div class="label">Confidence Threshold</div>
            </div>
        </div>

        <h2>📊 Confusion Matrix</h2>
        <p>Rows represent ground truth classes, columns represent predictions.</p>
        <table>
            <tr>
                <th>GT \\ Pred</th>"""

        for name in self.class_list:
            html += f"<th>{name}</th>"
        html += "<th>Background</th></tr>"

        for i, row_name in enumerate(self.class_list + ["Background"]):
            html += f"<tr><th>{row_name}</th>"
            for j in range(self.num_classes + 1):
                value = int(self._matrix[i, j])
                css_class = "matrix-cell"
                if i == j:
                    css_class += " diagonal"
                elif value > 0 and i != j:
                    css_class += " high-value"
                html += f'<td class="{css_class}">{value}</td>'
            html += "</tr>"

        html += """</table>

        <h2>📈 Per-Class Metrics</h2>
        <table class="metrics-table">
            <tr>
                <th>Class</th>
                <th>TP</th>
                <th>FP</th>
                <th>FN</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
            </tr>"""

        for class_name, m in metrics.items():
            p_class = (
                "good"
                if m["precision"] >= 0.7
                else "medium"
                if m["precision"] >= 0.5
                else "poor"
            )
            r_class = (
                "good"
                if m["recall"] >= 0.7
                else "medium"
                if m["recall"] >= 0.5
                else "poor"
            )
            f_class = (
                "good"
                if m["f1_score"] >= 0.7
                else "medium"
                if m["f1_score"] >= 0.5
                else "poor"
            )

            html += f"""<tr>
                <td><strong>{class_name}</strong></td>
                <td>{m["true_positives"]}</td>
                <td>{m["false_positives"]}</td>
                <td>{m["false_negatives"]}</td>
                <td class="{p_class}">{m["precision"]:.4f}</td>
                <td class="{r_class}">{m["recall"]:.4f}</td>
                <td class="{f_class}">{m["f1_score"]:.4f}</td>
            </tr>"""

        html += """</table>

        <h2>🔍 Analysis</h2>
        <ul>"""

        # Add analysis points
        total_tp = sum(self._true_positives.values())
        total_fp = sum(self._false_positives.values())
        total_fn = sum(self._false_negatives.values())

        overall_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        )
        overall_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        )

        html += f"<li>Overall Precision: <strong>{overall_precision:.4f}</strong></li>"
        html += f"<li>Overall Recall: <strong>{overall_recall:.4f}</strong></li>"

        # Find most confused classes
        confusion_pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and self._matrix[i, j] > 0:
                    confusion_pairs.append(
                        (
                            self.class_list[i],
                            self.class_list[j],
                            int(self._matrix[i, j]),
                        )
                    )

        confusion_pairs.sort(key=lambda x: x[2], reverse=True)

        if confusion_pairs:
            html += "<li>Most common confusions:"
            html += "<ul>"
            for gt, pred, count in confusion_pairs[:5]:
                html += f"<li>{gt} → {pred}: {count} instances</li>"
            html += "</ul></li>"

        html += """</ul>
    </div>
</body>
</html>"""

        return html

    def save(self, output_path: str, formats: list[str] = None):
        """
        Save confusion matrix to files.

        Args:
            output_path: Base output path (without extension)
            formats: List of formats to save ('json', 'txt', 'html')
        """
        if formats is None:
            formats = ["json", "txt", "html"]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if "json" in formats:
            json_path = output_path.with_suffix(".json")
            data = self.get_matrix_data().to_dict()
            data["per_class_metrics"] = self.get_per_class_metrics()
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved confusion matrix JSON to {json_path}")

        if "txt" in formats:
            txt_path = output_path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(self.generate_text_report())
            logger.info(f"Saved confusion matrix text report to {txt_path}")

        if "html" in formats:
            html_path = output_path.with_suffix(".html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.generate_html_report())
            logger.info(f"Saved confusion matrix HTML report to {html_path}")

    def print_matrix(self):
        """Print the confusion matrix to console."""
        print(self.generate_text_report())
