#!/usr/bin/env python3
"""
TACTILE — SAM-IP Training Script.

Trains the tiny SAM-IP spatial attention CNN to predict which 40x40 spatial
cells contain task-relevant objects for each of the 14 Tasks.

Training approach:
  - Input: 40x40 RGB thumbnail + task_id (one-hot encoded)
  - Output: 40x40 binary mask (1 = task-relevant object present)
  - Loss: BCEWithLogitsLoss with pos_weight=25 (class imbalance)
  - Target: >= 97% recall per task

Usage:
    python TrainSam.py --data-dir ./data --epochs 20
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from Tactile.Models.SamIp import SAMIP
from Tactile.Config.Tasks import NUM_TASKS, COCO_CATID_TO_IDX


class SAMTrainingDataset(Dataset):
    """
    Dataset for SAM-IP training.

    For each (image, task_id) pair:
      - Creates a 40x40 thumbnail from the image
      - Creates a 40x40 binary mask where cells containing task-preferred
        object bboxes are marked as 1
    """

    def __init__(self, annotations_dir: str, images_dir: str,
                 split: str = "train", sam_size: int = 40, img_size: int = 160):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.sam_size = sam_size
        self.img_size = img_size
        self.samples = []  # list of (image_path, task_id, preferred_bboxes)

        # Load all task annotations.
        for task_id in range(NUM_TASKS):
            ann_file = os.path.join(annotations_dir, f"task_{task_id + 1}_{split}.json")
            if not os.path.exists(ann_file):
                print(f"[WARNING] Missing: {ann_file}")
                continue

            with open(ann_file, "r") as f:
                data = json.load(f)

            # Build image_id -> annotations mapping.
            img_anns = {}
            for ann in data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in img_anns:
                    img_anns[img_id] = []
                img_anns[img_id].append(ann)

            # Build image_id -> file path mapping.
            img_paths = {}
            for img_info in data["images"]:
                # The annotations have names like 'COCO_train2014_000000262148.jpg'.
                # But our val2017 images are just '000000262148.jpg'.
                file_name = img_info["file_name"]
                if "COCO_" in file_name:
                    file_name = file_name.split("_")[-1]
                img_paths[img_info["id"]] = file_name

            # Create samples.
            for img_id, anns in img_anns.items():
                if img_id not in img_paths:
                    continue

                file_name = img_paths[img_id]
                img_path = os.path.join(images_dir, file_name)

                if not os.path.exists(img_path):
                    continue

                # Get preferred bboxes (category_id == 1 means preferred).
                preferred_bboxes = []
                for ann in anns:
                    if ann.get("category_id", 0) == 1:
                        preferred_bboxes.append(ann["bbox"])  # [x, y, w, h]

                self.samples.append((img_path, task_id, preferred_bboxes))

        print(f"[INFO] SAM dataset ({split}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, task_id, preferred_bboxes = self.samples[idx]

        # Load and resize image to thumbnail.
        img = cv2.imread(img_path)
        if img is None:
            # Return dummy data on read failure.
            return {
                "image": torch.zeros(3, self.sam_size, self.sam_size),
                "mask": torch.zeros(self.sam_size, self.sam_size),
                "task_id": torch.tensor(task_id),
            }

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Create thumbnail.
        thumbnail = cv2.resize(img, (self.sam_size, self.sam_size))
        thumbnail = torch.from_numpy(thumbnail).float() / 255.0
        thumbnail = thumbnail.permute(2, 0, 1)  # (3, 40, 40)

        # Create 40x40 binary mask.
        mask = np.zeros((self.sam_size, self.sam_size), dtype=np.float32)
        for bbox in preferred_bboxes:
            x, y, w, h = bbox
            # Convert to mask coordinates.
            x1 = int(x / orig_w * self.sam_size)
            y1 = int(y / orig_h * self.sam_size)
            x2 = int((x + w) / orig_w * self.sam_size)
            y2 = int((y + h) / orig_h * self.sam_size)
            x1 = max(0, min(x1, self.sam_size - 1))
            y1 = max(0, min(y1, self.sam_size - 1))
            x2 = max(0, min(x2, self.sam_size))
            y2 = max(0, min(y2, self.sam_size))
            mask[y1:y2, x1:x2] = 1.0

        mask = torch.from_numpy(mask)

        return {
            "image": thumbnail,
            "mask": mask,
            "task_id": torch.tensor(task_id),
        }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        task_ids = batch["task_id"].to(device)

        optimizer.zero_grad()
        logits = model(images, task_ids)  # (B, 1, 40, 40)
        logits = logits.squeeze(1)  # (B, 40, 40)

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, device, threshold=0.3):
    """Evaluate recall and precision on validation set."""
    model.eval()
    total_tp = 0
    total_fn = 0
    total_fp = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            task_ids = batch["task_id"].to(device)

            logits = model(images, task_ids).squeeze(1)
            preds = (torch.sigmoid(logits) >= threshold).float()

            tp = ((preds == 1) & (masks == 1)).sum().item()
            fn = ((preds == 0) & (masks == 1)).sum().item()
            fp = ((preds == 1) & (masks == 0)).sum().item()

            total_tp += tp
            total_fn += fn
            total_fp += fp

    recall = total_tp / max(total_tp + total_fn, 1)
    precision = total_tp / max(total_tp + total_fp, 1)

    return recall, precision


def main():
    import random
    torch.manual_seed(29)
    np.random.seed(29)
    random.seed(29)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(29)
        
    parser = argparse.ArgumentParser(description="Train SAM-IP")
    parser.add_argument("--data-dir", type=str, default="./CocoTaskDataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pos-weight", type=float, default=25.0,
                        help="Positive weight for BCE loss (class imbalance)")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    # Paths.
    annotations_dir = os.path.join(args.data_dir, "CocoTasks", "annotations")
    images_dir = os.path.join(args.data_dir, "Coco", "val2017")

    # For training, we use the COCO-Tasks train split.
    # Images come from COCO train2017, but val2017 can be used for quick testing.
    # Check if train2017 exists, else use val2017.
    train_images_dir = os.path.join(args.data_dir, "Coco", "train2017")
    if not os.path.exists(train_images_dir):
        print(f"[INFO] train2017 not found, using val2017 for both train and test")
        train_images_dir = images_dir

    # Datasets.
    print("Loading training dataset...")
    train_split = "train" if os.path.exists(os.path.join(args.data_dir, "Coco", "train2017")) else "test"
    train_dataset = SAMTrainingDataset(annotations_dir, train_images_dir, split=train_split)
    print("Loading test dataset...")
    test_dataset = SAMTrainingDataset(annotations_dir, images_dir, split="test")

    if len(train_dataset) == 0:
        print("[ERROR] No training samples found! Check data paths.")
        print(f"  Annotations: {annotations_dir}")
        print(f"  Images: {train_images_dir}")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # Model.
    model = SAMIP(num_tasks=NUM_TASKS).to(device)
    print(f"SAM-IP parameters: {model.count_parameters()}")

    # Loss with pos_weight for class imbalance.
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop.
    best_recall = 0
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Tactile", "Weights")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        # Evaluate.
        recall, precision = evaluate(model, test_loader, device)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f} | "
              f"Recall: {recall:.4f} | Precision: {precision:.4f} | LR: {lr:.6f}")

        # Save best model (by recall, targeting >= 0.97).
        if recall > best_recall:
            best_recall = recall
            save_path = os.path.join(save_dir, "SamIp.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (recall={recall:.4f}) to {save_path}")

    print(f"\nTraining complete. Best recall: {best_recall:.4f}")
    print(f"Target: >= 0.97 | {'ACHIEVED' if best_recall >= 0.97 else 'NOT YET ACHIEVED'}")


if __name__ == "__main__":
    main()
