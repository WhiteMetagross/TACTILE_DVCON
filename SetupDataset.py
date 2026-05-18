#!/usr/bin/env python3
"""
TACTILE — Dataset Setup Script.

Downloads:
  1. COCO-Tasks annotations from GitHub (git-lfs)
  2. COCO 2017 validation images (for testing — smaller than full train set)
  3. Precomputes class-task prior table from annotations

Usage:
    python SetupDataset.py --data-dir ./data
"""

import os
import sys
import json
import argparse
import subprocess
import urllib.request
import zipfile
from pathlib import Path

# Add parent to path.
sys.path.insert(0, str(Path(__file__).parent))


def download_coco_tasks(data_dir: str):
    """Clone COCO-Tasks annotations from GitHub."""
    coco_tasks_dir = os.path.join(data_dir, "CocoTasks")

    if os.path.exists(coco_tasks_dir) and os.path.exists(
        os.path.join(coco_tasks_dir, "annotations")
    ):
        print("[INFO] COCO-Tasks dataset already exists, skipping clone.")
        return coco_tasks_dir

    print("[INFO] Cloning COCO-Tasks dataset...")
    os.makedirs(data_dir, exist_ok=True)

    # Try HTTPS clone first (no SSH key needed).
    cmd = [
        "git", "clone",
        "-b", "cvpr2019",
        "--depth", "1",
        "https://github.com/CocoTasks/dataset.git",
        coco_tasks_dir
    ]

    try:
        subprocess.run(cmd, check=True, cwd=data_dir)
        print("[INFO] COCO-Tasks clone successful.")
    except subprocess.CalledProcessError:
        print("[WARNING] git-lfs clone failed. Trying manual download...")
        download_coco_tasks_manual(data_dir, coco_tasks_dir)

    return coco_tasks_dir


def download_coco_tasks_manual(data_dir: str, coco_tasks_dir: str):
    """Manually download COCO-Tasks annotation files from GitHub raw URLs."""
    os.makedirs(os.path.join(coco_tasks_dir, "annotations"), exist_ok=True)

    base_url = "https://raw.githubusercontent.com/CocoTasks/dataset/master/annotations/"

    for task_id in range(1, 15):
        for split in ["train", "test"]:
            filename = f"task_{task_id}_{split}.json"
            url = base_url + filename
            dest = os.path.join(coco_tasks_dir, "annotations", filename)

            if os.path.exists(dest):
                print(f"  [SKIP] {filename} already exists")
                continue

            print(f"  [DOWNLOAD] {filename}...")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                print(f"  [ERROR] Failed to download {filename}: {e}")


def download_coco_val_images(data_dir: str):
    """Download COCO 2017 validation images (5000 images, ~1GB)."""
    images_dir = os.path.join(data_dir, "Coco", "val2017")

    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 100:
        print(f"[INFO] COCO val images already exist ({len(os.listdir(images_dir))} files).")
        return images_dir

    print("[INFO] Downloading COCO 2017 validation images...")
    os.makedirs(os.path.join(data_dir, "Coco"), exist_ok=True)

    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(data_dir, "Coco", "val2017.zip")

    if not os.path.exists(zip_path):
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"  Downloaded to {zip_path}")

    print("  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(os.path.join(data_dir, "Coco"))
    print(f"  Extracted to {images_dir}")

    return images_dir


def compute_priors(annotations_dir: str, output_path: str):
    """Compute class-task prior table from COCO-Tasks annotations."""
    from Tactile.Config.ClassTaskPrior import (
        compute_prior_from_annotations, save_prior_table
    )

    print("[INFO] Computing class-task prior table...")
    prior = compute_prior_from_annotations(annotations_dir)

    # Print summary.
    import numpy as np
    nonzero = (prior > 0).sum()
    print(f"  Non-zero entries: {nonzero} / {prior.size}")
    print(f"  Max prior: {prior.max():.4f}")
    print(f"  Mean prior (non-zero): {prior[prior > 0].mean():.4f}")

    save_prior_table(prior, output_path)
    return prior


def main():
    parser = argparse.ArgumentParser(description="TACTILE Dataset Setup")
    parser.add_argument("--data-dir", type=str, default="./CocoTaskDataset",
                        help="Root data directory")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip downloading COCO images (annotation-only setup)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    # Step 1: Download COCO-Tasks annotations.
    coco_tasks_dir = download_coco_tasks(data_dir)
    annotations_dir = os.path.join(coco_tasks_dir, "annotations")

    # Step 2: Download COCO validation images.
    if not args.skip_images:
        images_dir = download_coco_val_images(data_dir)
    else:
        print("[INFO] Skipping COCO image download (--skip-images)")

    # Step 3: Compute class-task prior table.
    weights_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Tactile", "Weights"
    )
    os.makedirs(weights_dir, exist_ok=True)
    prior_path = os.path.join(weights_dir, "ClassTaskPrior.npy")

    if os.path.exists(annotations_dir):
        compute_priors(annotations_dir, prior_path)
    else:
        print("[WARNING] Annotations not found, skipping prior computation.")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"  Annotations: {annotations_dir}")
    if not args.skip_images:
        print(f"  Images: {os.path.join(data_dir, 'Coco', 'val2017')}")
    print(f"  Prior table: {prior_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
