"""
TACTILE — SAM Soft Gating Module.

Applies the SAM-IP spatial attention mask to detection proposals.
For each proposal, if its center falls in a mask=0 cell, multiply
its confidence by SAM_ALPHA (0.25). Do NOT discard the proposal.

This is soft gating — all proposals survive, but task-irrelevant ones
get their confidence suppressed.
"""

import numpy as np
from typing import List, Dict

from ..Config.SamThresholds import SAM_ALPHA, SAM_MASK_SIZE


def apply_soft_gating(
    detections: List[Dict],
    sam_mask: np.ndarray,
    image_width: int,
    image_height: int,
    alpha: float = SAM_ALPHA
) -> List[Dict]:
    """
    Apply SAM soft gating to detection proposals.

    For each detection:
      - Compute center (cx, cy) from bbox [x1, y1, x2, y2]
      - Map center to SAM mask coordinates (40x40)
      - If mask at that cell is 0: effective_conf = conf * alpha
      - If mask at that cell is 1: effective_conf = conf (unchanged)

    Args:
        detections: list of detection dicts with "bbox" and "confidence"
        sam_mask: (40, 40) binary numpy array
        image_width: original detection image width (for coordinate mapping)
        image_height: original detection image height
        alpha: suppression factor for mask=0 regions (default 0.25)

    Returns:
        List of detection dicts with added "confidence_effective" field
    """
    mask_h, mask_w = sam_mask.shape[:2]

    gated_detections = []
    for det in detections:
        det = det.copy()  # Don't mutate original

        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Map center to mask coordinates.
        # Detection coords are in the YOLO input space (e.g., 160x160).
        mx = int(cx / image_width * mask_w)
        my = int(cy / image_height * mask_h)

        # Clamp to valid range.
        mx = max(0, min(mx, mask_w - 1))
        my = max(0, min(my, mask_h - 1))

        # Apply soft gating.
        if sam_mask[my, mx] == 0:
            det["confidence_effective"] = det["confidence"] * alpha
            det["sam_gated"] = True
        else:
            det["confidence_effective"] = det["confidence"]
            det["sam_gated"] = False

        gated_detections.append(det)

    return gated_detections
