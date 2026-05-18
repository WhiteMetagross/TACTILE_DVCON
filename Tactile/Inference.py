"""
TACTILE — Main Inference Pipeline.

End-to-end Pipeline: (image_path, task_id) -> top-1 task-relevant detection.

Pipeline stages:
  [1] Preprocessing — resize to 160x160 (YOLO) and 40x40 (SAM)
  [2] SAM-IP — generate 40x40 spatial attention mask
  [3] YOLOv5n — detect objects → raw proposals
  [4] SAM Soft Gating — suppress confidence of proposals in irrelevant regions
  [5] RoI Feature Extraction — 128-D features per proposal
  [6] Task Scoring — dot product with task embedding
  [7] Task-Fused NMS — three-way fused score → top-K results
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from .Config.Tasks import TASK_NAMES, NUM_TASKS, COCO_CLASS_NAMES
from .Config.ClassTaskPrior import load_prior_table, get_default_prior
from .Config.SamThresholds import get_thresholds, SAM_ALPHA
from .Models.SamIp import SAMIP
from .Models.TaskEmb import TaskEmbeddingTable
from .Models.YoloV5nWrapper import YOLOv5nDetector, INPUT_SIZE
from .Pipeline.Preprocess import load_image, preprocess_for_sam
from .Pipeline.SoftGate import apply_soft_gating
from .Pipeline.RoiExtract import extract_roi_features_simple
from .Pipeline.TaskScore import compute_task_scores
from .Pipeline.FusedNms import FusedNms


class TACTILEPipeline:
    """
    Full TACTILE Inference Pipeline.

    Usage:
        Pipeline = TACTILEPipeline(device="cpu")
        result = Pipeline.infer("image.jpg", task_id=0)
        print(result)
    """

    def __init__(
        self,
        device: str = "cpu",
        yolo_model_path: Optional[str] = None,
        sam_weights_path: Optional[str] = None,
        task_emb_path: Optional[str] = None,
        prior_table_path: Optional[str] = None,
        conf_threshold: float = 0.15,
    ):
        """
        Initialize all Pipeline components.

        Args:
            device: "cpu" or "cuda"
            yolo_model_path: path to YOLOv5n weights (downloads if None)
            sam_weights_path: path to SAM-IP weights (uses random init if None)
            task_emb_path: path to task embeddings (uses random init if None)
            prior_table_path: path to class-task prior table
            conf_threshold: minimum YOLO detection confidence
        """
        self.device = device
        print("=" * 60)
        print("TACTILE Pipeline Initialization")
        print("=" * 60)

        # [1] YOLOv5n Detector.
        print("[1/5] Loading YOLOv5n detector...")
        self.detector = YOLOv5nDetector(
            model_path=yolo_model_path,
            device=device,
            conf_threshold=conf_threshold,
        )

        # [2] SAM-IP Model.
        print("[2/5] Loading SAM-IP model...")
        self.sam = SAMIP(num_tasks=NUM_TASKS).to(device)
        if sam_weights_path and Path(sam_weights_path).exists():
            state = torch.load(sam_weights_path, map_location=device, weights_only=True)
            self.sam.load_state_dict(state)
            print(f"  Loaded SAM-IP weights from {sam_weights_path}")
        else:
            print("  Using randomly initialized SAM-IP (no trained weights)")
        self.sam.eval()
        print(f"  SAM-IP parameters: {self.sam.count_parameters()}")

        # [3] Task Embeddings.
        print("[3/5] Loading task embeddings...")
        self.TaskEmb = TaskEmbeddingTable(num_tasks=NUM_TASKS, emb_dim=128).to(device)
        if task_emb_path and Path(task_emb_path).exists():
            self.TaskEmb.load(task_emb_path)
        else:
            print("  Using randomly initialized task embeddings")

        # [4] Class-Task Prior Table.
        print("[4/5] Loading class-task prior table...")
        self.ClassTaskPrior = load_prior_table(prior_table_path)
        print(f"  Prior table shape: {self.ClassTaskPrior.shape}")

        # [5] SAM Thresholds.
        print("[5/5] Loading SAM thresholds...")
        self.SamThresholds = get_thresholds()

        print("=" * 60)
        print("Pipeline ready!")
        print("=" * 60)

    def infer(self, image_path: str, task_id: int, verbose: bool = False) -> Dict:
        """
        Run full Inference Pipeline.

        Args:
            image_path: path to input image
            task_id: integer 0-13 corresponding to one of 14 Tasks

        Returns:
            {
              "bbox": [x1, y1, x2, y2],
              "class": "wine glass",
              "confidence": 0.636,
              "task": "serve wine",
              "all_detections": [...],   # top-5 fused detections
              "timing_ms": {...}          # per-stage timing
            }
        """
        assert 0 <= task_id < NUM_TASKS, f"task_id must be in [0, 13], got {task_id}"
        timing = {}

        # ─── Stage 1: Load and Preprocess ────────────────────────────────.
        t0 = time.time()
        image_rgb = load_image(image_path)
        orig_h, orig_w = image_rgb.shape[:2]
        sam_thumbnail = preprocess_for_sam(image_rgb).to(self.device)
        timing["preprocess_ms"] = (time.time() - t0) * 1000

        # ─── Stage 2: SAM-IP spatial mask ─────────────────────────────────.
        t0 = time.time()
        task_tensor = torch.tensor([task_id], device=self.device)
        with torch.no_grad():
            sam_mask = self.sam.predict_mask(
                sam_thumbnail, task_tensor,
                threshold=float(self.SamThresholds[task_id])
            )
        sam_mask_np = sam_mask.squeeze().cpu().numpy()  # (40, 40)
        timing["sam_ms"] = (time.time() - t0) * 1000

        if verbose:
            mask_coverage = sam_mask_np.mean() * 100
            print(f"  SAM mask coverage: {mask_coverage:.1f}%")

        # ─── Stage 3: YOLOv5n detection ───────────────────────────────────.
        t0 = time.time()
        detections = self.detector.detect(image_path, input_size=INPUT_SIZE)
        timing["yolo_ms"] = (time.time() - t0) * 1000

        if verbose:
            print(f"  YOLO detections: {len(detections)}")

        if len(detections) == 0:
            return {
                "bbox": [0, 0, 0, 0],
                "class": "none",
                "confidence": 0.0,
                "task": TASK_NAMES[task_id],
                "all_detections": [],
                "timing_ms": timing,
            }

        # ─── Stage 4: SAM Soft Gating ────────────────────────────────────.
        t0 = time.time()
        detections = apply_soft_gating(
            detections, sam_mask_np,
            image_width=INPUT_SIZE, image_height=INPUT_SIZE,
            alpha=SAM_ALPHA
        )
        timing["soft_gate_ms"] = (time.time() - t0) * 1000

        # ─── Stage 5: RoI Feature Extraction ─────────────────────────────.
        t0 = time.time()
        # Use simplified feature extraction (crop-based).
        from .Pipeline.Preprocess import preprocess_for_yolo
        yolo_tensor = preprocess_for_yolo(image_rgb).to(self.device)
        proposal_features = extract_roi_features_simple(
            yolo_tensor, detections,
            feature_dim=128, device=self.device
        )
        timing["roi_ms"] = (time.time() - t0) * 1000

        # ─── Stage 6: Task Scoring ────────────────────────────────────────.
        t0 = time.time()
        with torch.no_grad():
            task_embedding = self.TaskEmb(task_tensor).squeeze(0)  # (128,)
        task_scores = compute_task_scores(proposal_features, task_embedding)
        task_scores_np = task_scores.detach().cpu().numpy()
        timing["task_score_ms"] = (time.time() - t0) * 1000

        # ─── Stage 7: Task-Fused NMS ─────────────────────────────────────.
        t0 = time.time()
        results = FusedNms(
            detections, task_scores_np, self.ClassTaskPrior,
            task_id=task_id, iou_threshold=0.5, top_k=5,
            use_division_free_iou=True,
        )
        timing["fused_nms_ms"] = (time.time() - t0) * 1000

        # ─── Assemble output ─────────────────────────────────────────────.
        total_ms = sum(timing.values())
        timing["total_ms"] = total_ms

        if len(results) > 0:
            top1 = results[0]
            bbox_orig = top1["bbox"]

            return {
                "bbox": bbox_orig,
                "class": top1["class_name"],
                "confidence": top1["fused_score"],
                "task": TASK_NAMES[task_id],
                "all_detections": results,
                "timing_ms": timing,
            }
        else:
            return {
                "bbox": [0, 0, 0, 0],
                "class": "none",
                "confidence": 0.0,
                "task": TASK_NAMES[task_id],
                "all_detections": [],
                "timing_ms": timing,
            }


def infer(image_path: str, task_id: int, device: str = "cpu") -> Dict:
    """
    Convenience function matching the required interface.

    Args:
        image_path: path to input image
        task_id: integer 0-13 corresponding to one of 14 Tasks

    Returns:
        {
          "bbox": [x1, y1, x2, y2],
          "class": "wine glass",
          "confidence": 0.636,
          "task": "serve wine"
        }
    """
    Pipeline = TACTILEPipeline(device=device)
    result = Pipeline.infer(image_path, task_id)

    # Return clean result matching spec.
    return {
        "bbox": result["bbox"],
        "class": result["class"],
        "confidence": result["confidence"],
        "task": result["task"],
    }
