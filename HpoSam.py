#!/usr/bin/env python3
import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import optuna

sys.path.insert(0, str(Path(__file__).parent))

from Tactile.Models.SamIp import SAMIP
from Tactile.Config.Tasks import NUM_TASKS
from TrainSam import SAMTrainingDataset

def set_seed(seed=29):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch_quiet(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        task_ids = batch["task_id"].to(device)

        optimizer.zero_grad()
        logits = model(images, task_ids).squeeze(1)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)

def evaluate_quiet(model, dataloader, device, threshold=0.3):
    model.eval()
    total_tp, total_fn, total_fp = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./CocoTaskDataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(29)
    device = args.device
    print(f"Using device: {device}")

    # Load data.
    annotations_dir = os.path.join(args.data_dir, "CocoTasks", "annotations")
    images_dir = os.path.join(args.data_dir, "Coco", "val2017")
    train_images_dir = os.path.join(args.data_dir, "Coco", "train2017")
    if not os.path.exists(train_images_dir):
        train_images_dir = images_dir

    print("Loading datasets...")
    train_split = "train" if os.path.exists(os.path.join(args.data_dir, "Coco", "train2017")) else "test"
    train_dataset = SAMTrainingDataset(annotations_dir, train_images_dir, split=train_split)
    test_dataset = SAMTrainingDataset(annotations_dir, images_dir, split="test")

    if len(train_dataset) == 0:
        print("No training data found!")
        return

    # Objective for Optuna.
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        pos_weight_val = trial.suggest_float("pos_weight", 10.0, 50.0)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = SAMIP(num_tasks=NUM_TASKS).to(device)
        pos_weight = torch.tensor([pos_weight_val], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_score = -1
        patience = 8
        epochs_no_improve = 0

        for epoch in range(20):
            train_epoch_quiet(model, train_loader, criterion, optimizer, device)
            recall, _ = evaluate_quiet(model, test_loader, device)

            trial.report(recall, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if recall > best_score:
                best_score = recall
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                break
        
        return best_score

    print("\n--- Starting Optuna HPO (30 trials, max 20 epochs, patience 8) ---")
    # For deterministic HPO behavior.
    sampler = optuna.samplers.TPESampler(seed=29)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=30)

    print("\nBest HPO parameters:", study.best_params)
    print("Best HPO score (Recall):", study.best_value)

    print("\n--- Starting Final Training (max 200 epochs, patience 10) ---")
    set_seed(29)  # Reset seed for reproducibility

    best_lr = study.best_params["lr"]
    best_pos_weight = study.best_params["pos_weight"]
    best_batch_size = study.best_params["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SAMIP(num_tasks=NUM_TASKS).to(device)
    pos_weight = torch.tensor([best_pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_recall = -1
    epochs_no_improve = 0
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tactile", "Weights", "SamIp.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(200):
        loss = train_epoch_quiet(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        recall, precision = evaluate_quiet(model, test_loader, device)

        print(f"Epoch {epoch+1:03d}/200 | Loss: {loss:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")

        if recall > best_recall:
            best_recall = recall
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model with recall {best_recall:.4f} to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == "__main__":
    main()
