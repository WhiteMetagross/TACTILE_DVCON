#!/usr/bin/env python3
"""
TACTILE — Demo and Test Script.

Runs the full Inference Pipeline on sample images and visualizes results.
Tests all 14 Tasks to verify end-to-end correctness.

Usage:
    python Demo.py --data-dir ./data --task-id 0
    python Demo.py --data-dir ./data --test-all
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from Tactile.Config.Tasks import TASK_NAMES, NUM_TASKS
from Tactile.Inference import TACTILEPipeline


def draw_result(image_path: str, result: dict, output_path: str):
    """Draw bounding box and labels on image, save to output."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [ERROR] Cannot read: {image_path}")
        return

    h, w = img.shape[:2]

    if result["class"] != "none":
        x1, y1, x2, y2 = [int(c) for c in result["bbox"]]
        # Clamp to image bounds.
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Draw bbox.
        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label.
        label = f"{result['class']} ({result['confidence']:.3f})"
        task_label = f"Task: {result['task']}"

        # Background for text.
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Task label at top.
        cv2.putText(img, task_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imwrite(output_path, img)


def find_test_images(data_dir: str, task_id: int, max_images: int = 5):
    """Find test images for a given task from COCO-Tasks annotations."""
    annotations_dir = os.path.join(data_dir, "CocoTasks", "annotations")
    ann_file = os.path.join(annotations_dir, f"task_{task_id + 1}_test.json")

    images_dir = os.path.join(data_dir, "Coco", "val2017")
    train_images_dir = os.path.join(data_dir, "Coco", "train2017")

    if not os.path.exists(ann_file):
        # Fallback: use any images from val2017.
        if os.path.exists(images_dir):
            files = sorted(os.listdir(images_dir))[:max_images]
            return [os.path.join(images_dir, f) for f in files if f.endswith(".jpg")]
        return []

    with open(ann_file, "r") as f:
        data = json.load(f)

    # Find images that have preferred objects.
    preferred_images = set()
    for ann in data["annotations"]:
        if ann.get("category_id", 0) == 1:
            preferred_images.add(ann["image_id"])

    img_paths = {}
    for img_info in data["images"]:
        file_name = img_info["file_name"]
        if "COCO_" in file_name:
            file_name = file_name.split("_")[-1]
        img_paths[img_info["id"]] = file_name

    result_paths = []
    for img_id in list(preferred_images)[:max_images]:
        if img_id in img_paths:
            file_name = img_paths[img_id]
            # Try val2017 first, then train2017.
            for dir_path in [images_dir, train_images_dir]:
                full_path = os.path.join(dir_path, file_name)
                if os.path.exists(full_path):
                    result_paths.append(full_path)
                    break

    return result_paths


def test_single_image(Pipeline, image_path: str, task_id: int, verbose: bool = True):
    """Test Pipeline on a single image."""
    if verbose:
        print(f"\n  Image: {os.path.basename(image_path)}")
        print(f"  Task: [{task_id}] {TASK_NAMES[task_id]}")

    result = Pipeline.infer(image_path, task_id, verbose=verbose)

    if verbose:
        print(f"  Result: {result['class']} (score={result['confidence']:.4f})")
        if "timing_ms" in result:
            timing = result["timing_ms"]
            print(f"  Timing: {timing.get('total_ms', 0):.1f} ms total")
            for stage, ms in timing.items():
                if stage != "total_ms":
                    print(f"    {stage}: {ms:.1f} ms")

    return result


def test_all_tasks(Pipeline, data_dir: str, output_dir: str, max_per_task: int = 3):
    """Run Inference on all 14 Tasks and save results."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    total_time = 0
    total_images = 0

    for task_id in range(NUM_TASKS):
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {TASK_NAMES[task_id]}")
        print(f"{'='*60}")

        images = find_test_images(data_dir, task_id, max_images=max_per_task)
        if not images:
            print(f"  No test images found for task {task_id}")
            continue

        task_results = []
        for img_path in images:
            result = test_single_image(Pipeline, img_path, task_id)
            task_results.append({
                "image": os.path.basename(img_path),
                "class": result["class"],
                "confidence": result["confidence"],
                "bbox": result["bbox"],
            })

            # Save visualization.
            out_name = f"task{task_id:02d}_{os.path.basename(img_path)}"
            out_path = os.path.join(output_dir, out_name)
            draw_result(img_path, result, out_path)

            if "timing_ms" in result:
                total_time += result["timing_ms"].get("total_ms", 0)
            total_images += 1

        all_results[task_id] = task_results

    # Print summary.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {total_images}")
    if total_images > 0:
        avg_ms = total_time / total_images
        print(f"Average Inference time: {avg_ms:.1f} ms/image")
        print(f"Estimated FPS (CPU): {1000/avg_ms:.1f}" if avg_ms > 0 else "N/A")

    for task_id, results in all_results.items():
        detected = [r for r in results if r["class"] != "none"]
        print(f"  Task {task_id:2d} ({TASK_NAMES[task_id]:25s}): "
              f"{len(detected)}/{len(results)} detected, "
              f"classes: {[r['class'] for r in detected]}")

    # Save JSON results.
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="TACTILE Demo")
    parser.add_argument("--data-dir", type=str, default="./CocoTaskDataset")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a specific image")
    parser.add_argument("--task-id", type=int, default=0,
                        help="Task ID (0-13)")
    parser.add_argument("--test-all", action="store_true",
                        help="Test all 14 Tasks")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory for visualizations")
    parser.add_argument("--device", type=str,
                        default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--max-per-task", type=int, default=3,
                        help="Max images per task when testing all")
    args = parser.parse_args()

    print("=" * 60)
    print("TACTILE — Task-Aware Cascaded Inference Pipeline")
    print("DVCon India 2026 — Stage 2A Demo")
    print("=" * 60)

    # Initialize Pipeline.
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "Tactile", "Weights")
    sam_path = os.path.join(weights_dir, "SamIp.pth")
    emb_path = os.path.join(weights_dir, "TaskEmbeddings.npy")
    prior_path = os.path.join(weights_dir, "ClassTaskPrior.npy")

    Pipeline = TACTILEPipeline(
        device=args.device,
        sam_weights_path=sam_path if os.path.exists(sam_path) else None,
        task_emb_path=emb_path if os.path.exists(emb_path) else None,
        prior_table_path=prior_path if os.path.exists(prior_path) else None,
    )

    if args.test_all:
        test_all_tasks(Pipeline, args.data_dir, args.output_dir, args.max_per_task)
    elif args.image:
        result = test_single_image(Pipeline, args.image, args.task_id)
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"result_{os.path.basename(args.image)}")
        draw_result(args.image, result, out_path)
        print(f"\nVisualization saved to {out_path}")
    else:
        # Default: find images and test.
        images = find_test_images(args.data_dir, args.task_id)
        if images:
            os.makedirs(args.output_dir, exist_ok=True)
            for img_path in images[:3]:
                result = test_single_image(Pipeline, img_path, args.task_id)
                out_path = os.path.join(args.output_dir,
                                        f"result_{os.path.basename(img_path)}")
                draw_result(img_path, result, out_path)
        else:
            print(f"No test images found. Use --image to specify an image.")
            print(f"Or run: python SetupDataset.py --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()
