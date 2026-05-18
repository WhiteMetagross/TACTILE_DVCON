"""
TACTILE — Class-Task Prior Table (80 x 14).

prior[c][t] = fraction of class-c instances preferred for task t
              in COCO-Tasks training data.

This module provides:
  - compute_prior_from_annotations(): offline computation from COCO-Tasks JSONs
  - load_prior_table(): load precomputed table from disk
  - get_default_prior(): heuristic fallback if annotations not yet processed
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Optional

from .Tasks import (
    NUM_TASKS, NUM_COCO_CLASSES, COCO_CATID_TO_IDX,
    TASK_RELEVANT_CLASSES, COCO_CLASS_NAMES
)

# Path to save/load the precomputed prior table.
PRIOR_TABLE_PATH = Path(__file__).parent.parent / "Weights" / "ClassTaskPrior.npy"


def compute_prior_from_annotations(annotations_dir: str) -> np.ndarray:
    """
    Compute the 80x14 class-task prior table from COCO-Tasks annotation files.

    For each task t and each COCO class c:
        prior[c][t] = count(class c is preferred in task t images)
                    / count(class c appears in task t images)

    If class c never appears in task-t images, prior[c][t] = 0.

    Args:
        annotations_dir: path to directory containing task_*_train.json files

    Returns:
        np.ndarray of shape (80, 14), dtype float32
    """
    prior = np.zeros((NUM_COCO_CLASSES, NUM_TASKS), dtype=np.float32)
    class_count = np.zeros((NUM_COCO_CLASSES, NUM_TASKS), dtype=np.float32)
    class_preferred = np.zeros((NUM_COCO_CLASSES, NUM_TASKS), dtype=np.float32)

    for task_id in range(NUM_TASKS):
        ann_file = os.path.join(annotations_dir, f"task_{task_id + 1}_train.json")
        if not os.path.exists(ann_file):
            print(f"[WARNING] Missing annotation file: {ann_file}")
            continue

        with open(ann_file, "r") as f:
            data = json.load(f)

        for ann in data["annotations"]:
            coco_cat_id = ann.get("COCO_category_id", ann.get("category_id"))

            # In COCO-Tasks, category_id is 0 (not preferred) or 1 (preferred).
            is_preferred = ann.get("category_id", 0)

            # Map COCO category ID to contiguous index.
            if coco_cat_id in COCO_CATID_TO_IDX:
                cls_idx = COCO_CATID_TO_IDX[coco_cat_id]
                class_count[cls_idx, task_id] += 1
                if is_preferred == 1:
                    class_preferred[cls_idx, task_id] += 1

    # Compute prior as fraction, avoiding division by zero.
    mask = class_count > 0
    prior[mask] = class_preferred[mask] / class_count[mask]

    # Blend with heuristic prior to prevent absolute zeros on sparse data
    heuristic = get_default_prior()
    prior = np.maximum(prior, heuristic * 0.5)

    return prior


def save_prior_table(prior: np.ndarray, path: Optional[str] = None):
    """Save the prior table to disk."""
    save_path = Path(path) if path else PRIOR_TABLE_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), prior)
    print(f"[INFO] Saved class-task prior table to {save_path}")


def load_prior_table(path: Optional[str] = None) -> np.ndarray:
    """Load precomputed prior table from disk."""
    load_path = Path(path) if path else PRIOR_TABLE_PATH
    if not load_path.exists():
        print(f"[WARNING] Prior table not found at {load_path}, using default heuristic.")
        return get_default_prior()
    prior = np.load(str(load_path))
    print(f"[INFO] Loaded class-task prior table from {load_path}")
    return prior


def get_default_prior() -> np.ndarray:
    """
    Return a heuristic prior table based on domain knowledge.
    Used as fallback when COCO-Tasks annotations are not available.
    """
    prior = np.full((NUM_COCO_CLASSES, NUM_TASKS), 0.01, dtype=np.float32)

    for task_id, class_names in TASK_RELEVANT_CLASSES.items():
        for cls_name in class_names:
            if cls_name in COCO_CLASS_NAMES:
                cls_idx = COCO_CLASS_NAMES.index(cls_name)
                prior[cls_idx, task_id] = 0.8  # High prior for known relevant classes

    return prior
