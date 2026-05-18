"""
TACTILE — Task-Score-Fused NMS (NMS-IP).

Replace standard confidence-only NMS with a three-way fused score:

    fused_score = detection_confidence * TaskScore * ClassTaskPrior[class_id][task_id]

Key design decisions:
  - Division-free IoU: uses 2*intersection > union instead of intersection/union
    (consistent with FPGA hardware NMS implementation)
  - Greedy suppression with IoU threshold = 0.5
  - Returns top-K results (default K=5)
"""

import numpy as np
from typing import List, Dict


def compute_iou_division_free(box_a: List[float], box_b: List[float]) -> bool:
    """
    Division-free IoU check: returns True if IoU > 0.5.
    Uses the identity: IoU > 0.5  ⟺  2 * intersection > union

    This avoids division, mapping directly to FPGA comparator logic.

    Args:
        box_a, box_b: [x1, y1, x2, y2]

    Returns:
        True if IoU > 0.5
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return False  # No overlap

    intersection = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    # Division-free check: 2 * intersection > union  ⟺  IoU > 0.5.
    return (2 * intersection) > union


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Standard IoU computation (for evaluation purposes)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / (union + 1e-8)


def FusedNms(
    detections: List[Dict],
    task_scores: np.ndarray,
    ClassTaskPrior: np.ndarray,
    task_id: int,
    iou_threshold: float = 0.5,
    top_k: int = 5,
    use_division_free_iou: bool = True,
) -> List[Dict]:
    """
    Task-Score-Fused NMS.

    For each detection i:
        fused_i = conf_effective_i * task_score_i * ClassTaskPrior[class_i][task_id]

    Then greedy NMS with IoU suppression sorted by fused score.

    Args:
        detections: list of detection dicts with "confidence_effective" and "class_id"
        task_scores: (N,) array of task relevance scores (0..1)
        ClassTaskPrior: (80, 14) prior table
        task_id: current task ID (0..13)
        iou_threshold: IoU threshold for suppression (0.5)
        top_k: number of results to keep
        use_division_free_iou: if True, uses 2*inter > union check

    Returns:
        List of top-K detection dicts, sorted by fused score descending.
        Each dict has added "fused_score" and "TaskScore" fields.
    """
    if len(detections) == 0:
        return []

    # Compute fused scores.
    for i, det in enumerate(detections):
        conf_eff = det.get("confidence_effective", det["confidence"])
        TaskScore = float(task_scores[i]) if i < len(task_scores) else 0.5
        cls_id = det["class_id"]

        # Look up class-task prior.
        if cls_id < ClassTaskPrior.shape[0]:
            prior = float(ClassTaskPrior[cls_id, task_id])
        else:
            prior = 0.01  # Unknown class default

        # Three-way fused score.
        fused = conf_eff * TaskScore * prior

        det["fused_score"] = fused
        det["TaskScore"] = TaskScore
        det["class_prior"] = prior

    # Sort by fused score descending.
    sorted_dets = sorted(detections, key=lambda d: d["fused_score"], reverse=True)

    # Greedy NMS.
    keep = []
    for det in sorted_dets:
        suppressed = False
        for kept_det in keep:
            if use_division_free_iou:
                if compute_iou_division_free(det["bbox"], kept_det["bbox"]):
                    suppressed = True
                    break
            else:
                if compute_iou(det["bbox"], kept_det["bbox"]) > iou_threshold:
                    suppressed = True
                    break

        if not suppressed:
            keep.append(det)

        if len(keep) >= top_k:
            break

    return keep
