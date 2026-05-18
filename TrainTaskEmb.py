#!/usr/bin/env python3
"""
TACTILE — Task Embedding Training Script.

Computes task embeddings using mean-pooled RoI features of task-preferred
objects (no-train baseline, acceptable for Stage 2A).

For each task:
  1. Find all preferred objects from COCO-Tasks annotations
  2. Run YOLOv5n to extract detections
  3. Crop and compute simplified 128-D features for each preferred detection
  4. Mean-pool across all preferred instances -> 128-D task embedding
  5. L2 normalize the embedding

Usage:
    python TrainTaskEmb.py --data-dir ./data --max-images 200
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from Tactile.Config.Tasks import NUM_TASKS, TASK_NAMES
from Tactile.Models.TaskEmb import TaskEmbeddingTable


def compute_image_feature(img_path: str, bbox: list, feature_dim: int = 128) -> np.ndarray:
    """
    Compute a simplified 128-D feature from a bounding box crop.

    Args:
        img_path: path to image
        bbox: [x, y, w, h] bounding box
        feature_dim: output feature dimension

    Returns:
        (128,) normalized feature vector
    """
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(feature_dim, dtype=np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    x, y, bw, bh = bbox
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w, int(x + bw))
    y2 = min(h, int(y + bh))

    if x2 <= x1 or y2 <= y1:
        return np.zeros(feature_dim, dtype=np.float32)

    # Crop and resize to 7x7.
    crop = img[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (8, 8))  # 8x8x3 = 192 values

    # Flatten and take first feature_dim elements.
    flat = crop_resized.astype(np.float32).flatten() / 255.0

    if len(flat) >= feature_dim:
        feat = flat[:feature_dim]
    else:
        feat = np.pad(flat, (0, feature_dim - len(flat)))

    # L2 normalize.
    norm = np.linalg.norm(feat) + 1e-8
    feat = feat / norm

    return feat


def main():
    import random
    torch.manual_seed(29)
    np.random.seed(29)
    random.seed(29)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(29)
        
    parser = argparse.ArgumentParser(description="Compute Task Embeddings")
    parser.add_argument("--data-dir", type=str, default="./CocoTaskDataset")
    parser.add_argument("--max-images", type=int, default=500,
                        help="Max images per task for feature computation")
    parser.add_argument("--feature-dim", type=int, default=128)
    args = parser.parse_args()

    annotations_dir = os.path.join(args.data_dir, "CocoTasks", "annotations")
    images_dir = os.path.join(args.data_dir, "Coco", "val2017")
    train_images_dir = os.path.join(args.data_dir, "Coco", "train2017")

    if not os.path.exists(train_images_dir):
        train_images_dir = images_dir

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Tactile", "Weights")
    os.makedirs(save_dir, exist_ok=True)

    task_features = {}

    for task_id in range(NUM_TASKS):
        print(f"\n[Task {task_id}] {TASK_NAMES[task_id]}")

        split = "train" if os.path.exists(os.path.join(args.data_dir, "Coco", "train2017")) else "test"
        ann_file = os.path.join(annotations_dir, f"task_{task_id + 1}_{split}.json")
        if not os.path.exists(ann_file):
            print(f"  [WARNING] Missing: {ann_file}")
            task_features[task_id] = np.random.randn(1, args.feature_dim).astype(np.float32)
            continue

        with open(ann_file, "r") as f:
            data = json.load(f)

        # Build image_id -> file_name mapping.
        img_paths = {}
        for img_info in data["images"]:
            file_name = img_info["file_name"]
            if "COCO_" in file_name:
                file_name = file_name.split("_")[-1]
            img_paths[img_info["id"]] = file_name

        # Collect preferred bboxes.
        preferred = []  # list of (img_path, bbox)
        for ann in data["annotations"]:
            if ann.get("category_id", 0) == 1:  # preferred
                img_id = ann["image_id"]
                if img_id in img_paths:
                    file_name = img_paths[img_id]
                    img_path = os.path.join(train_images_dir, file_name)
                    if os.path.exists(img_path):
                        preferred.append((img_path, ann["bbox"]))

        # Limit to max_images.
        if len(preferred) > args.max_images:
            np.random.seed(42 + task_id)
            indices = np.random.choice(len(preferred), args.max_images, replace=False)
            preferred = [preferred[i] for i in indices]

        print(f"  Preferred instances: {len(preferred)}")

        if len(preferred) == 0:
            task_features[task_id] = np.random.randn(1, args.feature_dim).astype(np.float32)
            continue

        # Compute features.
        features = []
        for img_path, bbox in tqdm(preferred, desc="  Computing features", leave=False):
            feat = compute_image_feature(img_path, bbox, args.feature_dim)
            features.append(feat)

        task_features[task_id] = np.stack(features, axis=0)
        mean_feat = task_features[task_id].mean(axis=0)
        print(f"  Mean feature norm: {np.linalg.norm(mean_feat):.4f}")

    # Initialize embedding table.
    emb_table = TaskEmbeddingTable(num_tasks=NUM_TASKS, emb_dim=args.feature_dim)

    # Convert to torch tensors.
    torch_features = {}
    for task_id, feats in task_features.items():
        torch_features[task_id] = torch.from_numpy(feats)

    emb_table.init_from_features(torch_features)

    # Save.
    save_path = os.path.join(save_dir, "TaskEmbeddings.npy")
    emb_table.save(save_path)

    # Print summary.
    print("\n" + "=" * 60)
    print("Task Embedding Summary")
    print("=" * 60)
    all_embs = emb_table.get_all_embeddings().detach().numpy()
    for task_id in range(NUM_TASKS):
        emb = all_embs[task_id]
        print(f"  Task {task_id:2d} ({TASK_NAMES[task_id]:25s}): "
              f"norm={np.linalg.norm(emb):.4f}, "
              f"mean={emb.mean():.4f}, std={emb.std():.4f}")

    # Print pairwise similarity.
    print("\nPairwise cosine similarity (should show task clustering):")
    sim = all_embs @ all_embs.T
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    cos_sim = sim / (norms @ norms.T + 1e-8)
    print("     ", "  ".join([f"T{i:02d}" for i in range(NUM_TASKS)]))
    for i in range(NUM_TASKS):
        row = " ".join([f"{cos_sim[i, j]:5.2f}" for j in range(NUM_TASKS)])
        print(f"T{i:02d}: {row}")


if __name__ == "__main__":
    main()
