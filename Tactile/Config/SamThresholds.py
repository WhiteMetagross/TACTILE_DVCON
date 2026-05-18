"""
TACTILE — Per-task calibrated thresholds for SAM-IP binary mask.

After training SAM-IP, each task's sigmoid output is thresholded to produce a
binary 40x40 spatial mask. Thresholds are calibrated to achieve >= 97% recall
of task-relevant objects per task.

This module provides default thresholds and a calibration routine.
"""

import numpy as np
from .Tasks import NUM_TASKS

# Default thresholds: conservative (low) to achieve high recall.
# These will be calibrated per-task after SAM-IP training.
DEFAULT_SAM_THRESHOLDS = np.array([
    0.3,  # Task 0:  Serve wine
    0.3,  # Task 1:  Spread butter/jam
    0.3,  # Task 2:  Drink coffee
    0.3,  # Task 3:  Set the table
    0.3,  # Task 4:  Cut vegetables
    0.3,  # Task 5:  Serve food on a plate
    0.3,  # Task 6:  Tighten a screw
    0.3,  # Task 7:  Dig a hole
    0.3,  # Task 8:  Hang a picture
    0.3,  # Task 9:  Check the time
    0.3,  # Task 10: Make a phone call
    0.3,  # Task 11: Take a photo
    0.3,  # Task 12: Play music
    0.3,  # Task 13: Read a book
], dtype=np.float32)

# SAM soft-gating alpha: if a proposal center falls in mask=0, multiply.
# its confidence by SAM_ALPHA (do NOT discard the proposal).
SAM_ALPHA = 0.25

# SAM-IP mask resolution.
SAM_MASK_SIZE = 40

# SAM-IP input resolution (downsampled thumbnail).
SAM_INPUT_SIZE = 40


def get_thresholds() -> np.ndarray:
    """Return per-task thresholds (shape: [14])."""
    return DEFAULT_SAM_THRESHOLDS.copy()


def calibrate_thresholds(
    sam_model,
    val_loader,
    target_recall: float = 0.97
) -> np.ndarray:
    """
    Calibrate per-task thresholds to achieve target_recall on validation set.

    This is called after SAM-IP training.
    For each task, sweep threshold from 0.01 to 0.99 and pick the highest
    threshold that still achieves >= target_recall.

    Args:
        sam_model: trained SAM-IP model
        val_loader: validation data loader
        target_recall: minimum recall target (default 0.97)

    Returns:
        np.ndarray of shape (14,)
    """
    import torch

    thresholds = np.zeros(NUM_TASKS, dtype=np.float32)
    sam_model.eval()

    for task_id in range(NUM_TASKS):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"]
                labels = batch["mask"]
                task_ids = batch["task_id"]

                # Filter for this task.
                mask = task_ids == task_id
                if mask.sum() == 0:
                    continue

                preds = torch.sigmoid(sam_model(images[mask], task_ids[mask]))
                all_preds.append(preds.cpu().numpy().flatten())
                all_labels.append(labels[mask].cpu().numpy().flatten())

        if len(all_preds) == 0:
            thresholds[task_id] = 0.3
            continue

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Sweep thresholds.
        best_thresh = 0.01
        for thresh in np.arange(0.01, 0.99, 0.01):
            predicted_positive = all_preds >= thresh
            true_positive = (predicted_positive & (all_labels == 1)).sum()
            actual_positive = (all_labels == 1).sum()

            if actual_positive == 0:
                continue

            recall = true_positive / actual_positive
            if recall >= target_recall:
                best_thresh = thresh  # Keep the highest passing threshold

        thresholds[task_id] = best_thresh

    return thresholds
