"""
IoU (Intersection over Union) Calculation Module
=================================================
Core metric for object detection evaluation.

IoU measures the overlap between predicted and ground truth bounding boxes.
It is used for:
    1. Determining if a detection is a True Positive or False Positive
    2. Non-Maximum Suppression (NMS)
    3. Matching predictions to ground truth for mAP calculation

Formula:
    IoU = Area of Intersection / Area of Union
    IoU ∈ [0, 1], where 1 means perfect overlap
"""

from typing import Optional, Union

import numpy as np


def calculate_iou(
    box1: Union[tuple, list, np.ndarray],
    box2: Union[tuple, list, np.ndarray],
    box_format: str = "xyxy",
) -> float:
    """
    Calculate IoU between two bounding boxes.

    Args:
        box1: First bounding box
        box2: Second bounding box
        box_format: Format of boxes
            - 'xyxy': (x1, y1, x2, y2) - top-left and bottom-right corners
            - 'xywh': (x_center, y_center, width, height)
            - 'coco': (x_min, y_min, width, height)

    Returns:
        IoU value between 0 and 1
    """
    # Convert to numpy arrays
    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)

    # Convert to xyxy format if needed
    if box_format == "xywh":
        box1 = _xywh_to_xyxy(box1)
        box2 = _xywh_to_xyxy(box2)
    elif box_format == "coco":
        box1 = _coco_to_xyxy(box1)
        box2 = _coco_to_xyxy(box2)

    # Calculate intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Check for no intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    # Intersection area
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def calculate_iou_batch(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    box_format: str = "xyxy",
) -> np.ndarray:
    """
    Calculate IoU between all pairs of boxes in two arrays (vectorized).

    Args:
        boxes1: Array of shape (N, 4) - first set of boxes
        boxes2: Array of shape (M, 4) - second set of boxes
        box_format: Format of boxes ('xyxy', 'xywh', 'coco')

    Returns:
        IoU matrix of shape (N, M) where result[i, j] is IoU of boxes1[i] and boxes2[j]
    """
    boxes1 = np.array(boxes1, dtype=np.float32)
    boxes2 = np.array(boxes2, dtype=np.float32)

    # Handle empty arrays
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    # Ensure 2D arrays
    if boxes1.ndim == 1:
        boxes1 = boxes1.reshape(1, -1)
    if boxes2.ndim == 1:
        boxes2 = boxes2.reshape(1, -1)

    # Convert to xyxy format if needed
    if box_format == "xywh":
        boxes1 = np.apply_along_axis(_xywh_to_xyxy, 1, boxes1)
        boxes2 = np.apply_along_axis(_xywh_to_xyxy, 1, boxes2)
    elif box_format == "coco":
        boxes1 = np.apply_along_axis(_coco_to_xyxy, 1, boxes1)
        boxes2 = np.apply_along_axis(_coco_to_xyxy, 1, boxes2)

    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Calculate intersections using broadcasting
    # Shape: (N, M)
    x1_inter = np.maximum(boxes1[:, 0:1], boxes2[:, 0].T)  # (N, M)
    y1_inter = np.maximum(boxes1[:, 1:2], boxes2[:, 1].T)  # (N, M)
    x2_inter = np.minimum(boxes1[:, 2:3], boxes2[:, 2].T)  # (N, M)
    y2_inter = np.minimum(boxes1[:, 3:4], boxes2[:, 3].T)  # (N, M)

    # Intersection area (clip to 0 for non-overlapping)
    inter_width = np.clip(x2_inter - x1_inter, 0, None)
    inter_height = np.clip(y2_inter - y1_inter, 0, None)
    intersection = inter_width * inter_height

    # Union area
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection

    # IoU (avoid division by zero)
    iou = np.where(union > 0, intersection / union, 0)

    return iou.astype(np.float32)


def calculate_giou(
    box1: Union[tuple, list, np.ndarray],
    box2: Union[tuple, list, np.ndarray],
    box_format: str = "xyxy",
) -> float:
    """
    Calculate Generalized IoU (GIoU) between two bounding boxes.

    GIoU addresses the issue where IoU is always 0 for non-overlapping boxes,
    even if they are very close. GIoU provides a gradient for non-overlapping cases.

    Formula:
        GIoU = IoU - (Area of enclosing box - Union) / Area of enclosing box
        GIoU ∈ [-1, 1], where 1 means perfect overlap

    Args:
        box1: First bounding box
        box2: Second bounding box
        box_format: Format of boxes ('xyxy', 'xywh', 'coco')

    Returns:
        GIoU value between -1 and 1
    """
    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)

    if box_format == "xywh":
        box1 = _xywh_to_xyxy(box1)
        box2 = _xywh_to_xyxy(box2)
    elif box_format == "coco":
        box1 = _coco_to_xyxy(box1)
        box2 = _coco_to_xyxy(box2)

    # Calculate IoU
    iou = calculate_iou(box1, box2, box_format="xyxy")

    # Enclosing box
    x1_enc = min(box1[0], box2[0])
    y1_enc = min(box1[1], box2[1])
    x2_enc = max(box1[2], box2[2])
    y2_enc = max(box1[3], box2[3])

    # Areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_enc = (x2_enc - x1_enc) * (y2_enc - y1_enc)

    # Intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter > x1_inter and y2_inter > y1_inter:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    else:
        intersection = 0

    union = area1 + area2 - intersection

    if area_enc <= 0:
        return iou

    giou = iou - (area_enc - union) / area_enc

    return float(giou)


def calculate_diou(
    box1: Union[tuple, list, np.ndarray],
    box2: Union[tuple, list, np.ndarray],
    box_format: str = "xyxy",
) -> float:
    """
    Calculate Distance IoU (DIoU) between two bounding boxes.

    DIoU considers the distance between box centers in addition to IoU.
    It converges faster than GIoU during training.

    Formula:
        DIoU = IoU - (center distance)^2 / (diagonal of enclosing box)^2
        DIoU ∈ [-1, 1]

    Args:
        box1: First bounding box
        box2: Second bounding box
        box_format: Format of boxes ('xyxy', 'xywh', 'coco')

    Returns:
        DIoU value between -1 and 1
    """
    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)

    if box_format == "xywh":
        box1 = _xywh_to_xyxy(box1)
        box2 = _xywh_to_xyxy(box2)
    elif box_format == "coco":
        box1 = _coco_to_xyxy(box1)
        box2 = _coco_to_xyxy(box2)

    # Calculate IoU
    iou = calculate_iou(box1, box2, box_format="xyxy")

    # Box centers
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2

    # Center distance squared
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Enclosing box diagonal squared
    x1_enc = min(box1[0], box2[0])
    y1_enc = min(box1[1], box2[1])
    x2_enc = max(box1[2], box2[2])
    y2_enc = max(box1[3], box2[3])

    diag_sq = (x2_enc - x1_enc) ** 2 + (y2_enc - y1_enc) ** 2

    if diag_sq <= 0:
        return iou

    diou = iou - center_dist_sq / diag_sq

    return float(diou)


def calculate_ciou(
    box1: Union[tuple, list, np.ndarray],
    box2: Union[tuple, list, np.ndarray],
    box_format: str = "xyxy",
) -> float:
    """
    Calculate Complete IoU (CIoU) between two bounding boxes.

    CIoU considers overlap, center distance, and aspect ratio consistency.
    It is commonly used as a loss function for object detection training.

    Formula:
        CIoU = IoU - (center distance)^2 / (diagonal)^2 - α * v
        where v measures aspect ratio consistency and α is a trade-off parameter

    Args:
        box1: First bounding box
        box2: Second bounding box
        box_format: Format of boxes ('xyxy', 'xywh', 'coco')

    Returns:
        CIoU value
    """
    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)

    if box_format == "xywh":
        box1 = _xywh_to_xyxy(box1)
        box2 = _xywh_to_xyxy(box2)
    elif box_format == "coco":
        box1 = _coco_to_xyxy(box1)
        box2 = _coco_to_xyxy(box2)

    # Calculate IoU
    iou = calculate_iou(box1, box2, box_format="xyxy")

    # Box dimensions
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    # Box centers
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2

    # Center distance squared
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Enclosing box diagonal squared
    x1_enc = min(box1[0], box2[0])
    y1_enc = min(box1[1], box2[1])
    x2_enc = max(box1[2], box2[2])
    y2_enc = max(box1[3], box2[3])

    diag_sq = (x2_enc - x1_enc) ** 2 + (y2_enc - y1_enc) ** 2

    # Aspect ratio term
    eps = 1e-7
    arctan1 = np.arctan(w1 / (h1 + eps))
    arctan2 = np.arctan(w2 / (h2 + eps))
    v = (4 / (np.pi**2)) * ((arctan1 - arctan2) ** 2)

    # Trade-off parameter
    alpha = v / (1 - iou + v + eps) if (1 - iou + v) > 0 else 0

    if diag_sq <= 0:
        return iou - alpha * v

    ciou = iou - center_dist_sq / diag_sq - alpha * v

    return float(ciou)


def match_boxes_by_iou(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
    box_format: str = "xyxy",
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    Match predicted boxes to ground truth boxes using IoU.

    Uses greedy matching: iteratively match the pair with highest IoU
    above the threshold until no more matches are possible.

    Args:
        pred_boxes: Array of predicted boxes (N, 4)
        gt_boxes: Array of ground truth boxes (M, 4)
        iou_threshold: Minimum IoU for a valid match
        box_format: Format of boxes ('xyxy', 'xywh', 'coco')

    Returns:
        Tuple of:
            - matches: List of (pred_idx, gt_idx, iou) tuples
            - unmatched_preds: List of unmatched prediction indices
            - unmatched_gts: List of unmatched ground truth indices
    """
    pred_boxes = np.array(pred_boxes, dtype=np.float32)
    gt_boxes = np.array(gt_boxes, dtype=np.float32)

    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))

    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []

    # Calculate IoU matrix
    iou_matrix = calculate_iou_batch(pred_boxes, gt_boxes, box_format)

    matches = []
    matched_preds = set()
    matched_gts = set()

    # Greedy matching
    while True:
        # Find maximum IoU
        max_iou = -1
        max_pred_idx = -1
        max_gt_idx = -1

        for i in range(len(pred_boxes)):
            if i in matched_preds:
                continue
            for j in range(len(gt_boxes)):
                if j in matched_gts:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_pred_idx = i
                    max_gt_idx = j

        # Check if match is valid
        if max_iou < iou_threshold:
            break

        matches.append((max_pred_idx, max_gt_idx, float(max_iou)))
        matched_preds.add(max_pred_idx)
        matched_gts.add(max_gt_idx)

    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [j for j in range(len(gt_boxes)) if j not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


# ============================================================================
# Helper functions for box format conversion
# ============================================================================


def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
    x, y, w, h = box
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def _xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to (x_center, y_center, width, height)."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def _coco_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert COCO format (x_min, y_min, width, height) to (x1, y1, x2, y2)."""
    x, y, w, h = box
    return np.array([x, y, x + w, y + h])


def _xyxy_to_coco(box: np.ndarray) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to COCO format (x_min, y_min, width, height)."""
    x1, y1, x2, y2 = box
    return np.array([x1, y1, x2 - x1, y2 - y1])


def convert_boxes(
    boxes: np.ndarray,
    from_format: str,
    to_format: str,
) -> np.ndarray:
    """
    Convert boxes between different formats.

    Args:
        boxes: Array of boxes (N, 4)
        from_format: Source format ('xyxy', 'xywh', 'coco')
        to_format: Target format ('xyxy', 'xywh', 'coco')

    Returns:
        Converted boxes array
    """
    if from_format == to_format:
        return boxes.copy()

    boxes = np.array(boxes, dtype=np.float32)

    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    # Convert to xyxy first
    if from_format == "xywh":
        boxes = np.apply_along_axis(_xywh_to_xyxy, 1, boxes)
    elif from_format == "coco":
        boxes = np.apply_along_axis(_coco_to_xyxy, 1, boxes)

    # Convert from xyxy to target format
    if to_format == "xywh":
        boxes = np.apply_along_axis(_xyxy_to_xywh, 1, boxes)
    elif to_format == "coco":
        boxes = np.apply_along_axis(_xyxy_to_coco, 1, boxes)

    return boxes


# ============================================================================
# Utility functions
# ============================================================================


def box_area(box: Union[tuple, list, np.ndarray], box_format: str = "xyxy") -> float:
    """Calculate area of a bounding box."""
    box = np.array(box, dtype=np.float32)

    if box_format == "xywh":
        return float(box[2] * box[3])
    elif box_format == "coco":
        return float(box[2] * box[3])
    else:  # xyxy
        return float((box[2] - box[0]) * (box[3] - box[1]))


def box_center(
    box: Union[tuple, list, np.ndarray], box_format: str = "xyxy"
) -> tuple[float, float]:
    """Get center point of a bounding box."""
    box = np.array(box, dtype=np.float32)

    if box_format == "xywh":
        return (float(box[0]), float(box[1]))
    elif box_format == "coco":
        return (float(box[0] + box[2] / 2), float(box[1] + box[3] / 2))
    else:  # xyxy
        return (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))


def is_valid_box(box: Union[tuple, list, np.ndarray], box_format: str = "xyxy") -> bool:
    """Check if a bounding box is valid (positive area)."""
    box = np.array(box, dtype=np.float32)

    if len(box) != 4:
        return False

    if box_format == "xywh" or box_format == "coco":
        return box[2] > 0 and box[3] > 0
    else:  # xyxy
        return box[2] > box[0] and box[3] > box[1]
