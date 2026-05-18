#!/usr/bin/env python3
"""
TACTILE — Evaluation Script.

Evaluates the top-1 success rate of the TACTILE Pipeline on the test subset.
For each task:
  1. Load test images with preferred object annotations
  2. Run Inference to get top-1 detection
  3. A prediction is a "Success" if its IoU with any preferred ground-truth box > 0.5

Usage:
    python Evaluate.py --data-dir ./data
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from Tactile.Config.Tasks import TASK_NAMES, NUM_TASKS
from Tactile.Inference import TACTILEPipeline
from Tactile.Pipeline.FusedNms import compute_iou


def evaluate_task(Pipeline, data_dir: str, task_id: int, max_images: int = 100):
    """Evaluate Pipeline success rate on a single task."""
    annotations_dir = os.path.join(data_dir, "CocoTasks", "annotations")
    
    # We use the test split because it aligns with val2017 images we downloaded.
    ann_file = os.path.join(annotations_dir, f"task_{task_id + 1}_test.json")
    images_dir = os.path.join(data_dir, "Coco", "val2017")

    if not os.path.exists(ann_file):
        return {"total": 0, "success": 0, "rate": 0.0, "latency_ms": 0.0}

    with open(ann_file, "r") as f:
        data = json.load(f)

    # Build image_id -> file_name.
    img_paths = {}
    img_dims = {}
    for img_info in data["images"]:
        file_name = img_info["file_name"]
        if "COCO_" in file_name:
            file_name = file_name.split("_")[-1]
        img_paths[img_info["id"]] = file_name
        img_dims[img_info["id"]] = (img_info["width"], img_info["height"])

    # Group preferred annotations by image.
    preferred_by_img = {}
    for ann in data["annotations"]:
        if ann.get("category_id", 0) == 1:  # preferred class for this task
            img_id = ann["image_id"]
            if img_id not in preferred_by_img:
                preferred_by_img[img_id] = []
            preferred_by_img[img_id].append(ann["bbox"])  # [x, y, w, h]

    successes = 0
    total_latency = 0.0
    valid_images = 0

    img_ids = list(preferred_by_img.keys())
    if len(img_ids) > max_images:
        # Fixed seed for reproducibility.
        np.random.seed(42 + task_id)
        img_ids = np.random.choice(img_ids, max_images, replace=False)

    for img_id in tqdm(img_ids, desc=f"Eval T{task_id}", leave=False):
        if img_id not in img_paths:
            continue
        
        img_path = os.path.join(images_dir, img_paths[img_id])
        if not os.path.exists(img_path):
            continue

        valid_images += 1
        gt_bboxes = preferred_by_img[img_id]  # list of [x, y, w, h]

        # Convert GT from [x, y, w, h] to [x1, y1, x2, y2].
        gt_boxes_xyxy = []
        for x, y, w, h in gt_bboxes:
            gt_boxes_xyxy.append([x, y, x + w, y + h])

        # Run Inference.
        result = Pipeline.infer(img_path, task_id, verbose=False)
        pred_bbox = result["bbox"]

        if "timing_ms" in result:
            total_latency += result["timing_ms"].get("total_ms", 0.0)

        if result["class"] == "none":
            continue

        # Check IoU.
        is_success = False
        for gt_box in gt_boxes_xyxy:
            iou = compute_iou(pred_bbox, gt_box)
            if iou > 0.5:
                is_success = True
                break
        
        if is_success:
            successes += 1

    if valid_images == 0:
        return {"total": 0, "success": 0, "rate": 0.0, "latency_ms": 0.0}

    return {
        "total": valid_images,
        "success": successes,
        "rate": successes / valid_images,
        "latency_ms": total_latency / valid_images
    }


def main():
    parser = argparse.ArgumentParser(description="TACTILE Pipeline Evaluation")
    parser.add_argument("--data-dir", type=str, default="./CocoTaskDataset")
    parser.add_argument("--max-images", type=int, default=50, 
                        help="Max images per task to evaluate")
    parser.add_argument("--device", type=str, 
                        default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print("TACTILE — Task-Aware Evaluation")
    print(f"Evaluating Top-1 Success Rate (IoU > 0.5) | Max {args.max_images} imgs/task")
    print("=" * 70)

    # Initialize Pipeline.
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tactile", "Weights")
    sam_path = os.path.join(weights_dir, "SamIp.pth")
    emb_path = os.path.join(weights_dir, "TaskEmbeddings.npy")
    prior_path = os.path.join(weights_dir, "ClassTaskPrior.npy")

    Pipeline = TACTILEPipeline(
        device=args.device,
        sam_weights_path=sam_path if os.path.exists(sam_path) else None,
        task_emb_path=emb_path if os.path.exists(emb_path) else None,
        prior_table_path=prior_path if os.path.exists(prior_path) else None,
    )

    results = {}
    total_imgs = 0
    total_succ = 0
    total_lat = 0.0

    print("\nStarting evaluation...")
    print(f"{'Task':<25} | {'Imgs':<5} | {'Succ':<5} | {'Rate':<6} | {'Avg Latency (ms)':<15}")
    print("-" * 70)

    for task_id in range(NUM_TASKS):
        res = evaluate_task(Pipeline, args.data_dir, task_id, max_images=args.max_images)
        results[task_id] = res
        
        total_imgs += res["total"]
        total_succ += res["success"]
        total_lat += res["latency_ms"] * res["total"]

        rate_str = f"{res['rate']*100:.1f}%" if res["total"] > 0 else "N/A"
        lat_str = f"{res['latency_ms']:.1f}" if res["total"] > 0 else "N/A"
        
        print(f"{TASK_NAMES[task_id]:<25} | {res['total']:<5} | {res['success']:<5} | {rate_str:<6} | {lat_str:<15}")

    print("=" * 70)
    if total_imgs > 0:
        overall_rate = total_succ / total_imgs
        overall_lat = total_lat / total_imgs
        print(f"OVERALL RESULTS ({total_imgs} images)")
        print(f"Top-1 Success Rate: {overall_rate*100:.2f}%")
        print(f"Average Latency:    {overall_lat:.2f} ms")
    else:
        print("No valid images evaluated.")
    print("=" * 70)

if __name__ == "__main__":
    main()
