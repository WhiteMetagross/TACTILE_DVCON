#!/usr/bin/env python3
"""
TACTILE — Comprehensive Test Suite.

Tests each Pipeline module independently, then runs end-to-end Inference.
This is the primary validation script for Stage 2A correctness.

Usage:
    python RunTests.py
    python RunTests.py --test-e2e --data-dir ./data
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_config_tasks():
    """Test task configuration module."""
    print("\n[TEST] Config.Tasks")
    from Tactile.Config.Tasks import (
        TASK_NAMES, NUM_TASKS, COCO_CLASS_NAMES, NUM_COCO_CLASSES,
        COCO_CAT_IDS, COCO_CATID_TO_IDX, COCO_IDX_TO_CATID,
        TASK_RELEVANT_CLASSES, get_annotation_filename
    )

    assert NUM_TASKS == 14, f"Expected 14 Tasks, got {NUM_TASKS}"
    assert len(TASK_NAMES) == 14
    assert NUM_COCO_CLASSES == 80
    assert len(COCO_CLASS_NAMES) == 80
    assert len(COCO_CAT_IDS) == 80
    assert len(COCO_CATID_TO_IDX) == 80
    assert len(COCO_IDX_TO_CATID) == 80

    # Test annotation filename.
    assert get_annotation_filename(0, "train") == "task_1_train.json"
    assert get_annotation_filename(13, "test") == "task_14_test.json"

    # Test category mapping roundtrip.
    for idx in range(80):
        cat_id = COCO_IDX_TO_CATID[idx]
        assert COCO_CATID_TO_IDX[cat_id] == idx, f"Roundtrip failed for idx={idx}"

    # Verify wine glass is class index 40.
    assert COCO_CLASS_NAMES[40] == "wine glass", f"Expected 'wine glass' at idx 40"
    assert COCO_CLASS_NAMES[41] == "cup"

    # Task-relevant classes should map to valid indices.
    for task_id, classes in TASK_RELEVANT_CLASSES.items():
        for cls in classes:
            assert cls in COCO_CLASS_NAMES, f"'{cls}' not in COCO class names"

    print("  PASSED - All assertions OK")
    return True


def test_class_task_prior():
    """Test class-task prior computation."""
    print("\n[TEST] Config.ClassTaskPrior")
    from Tactile.Config.ClassTaskPrior import get_default_prior, NUM_COCO_CLASSES

    prior = get_default_prior()
    assert prior.shape == (80, 14), f"Expected (80, 14), got {prior.shape}"
    assert prior.dtype == np.float32

    # Default prior should have high values for known relevant classes.
    # wine glass (idx 40) for task 0 (serve wine).
    assert prior[40, 0] == 0.8, f"Expected 0.8 for wine glass/serve wine, got {prior[40, 0]}"

    # Non-relevant classes should have low prior.
    assert prior[0, 0] == 0.01, f"Expected 0.01 for person/serve wine"

    # All values should be in [0, 1].
    assert prior.min() >= 0
    assert prior.max() <= 1

    print("  PASSED - Default prior table correct")
    return True


def test_sam_thresholds():
    """Test SAM threshold configuration."""
    print("\n[TEST] Config.SamThresholds")
    from Tactile.Config.SamThresholds import get_thresholds, SAM_ALPHA, SAM_MASK_SIZE

    thresholds = get_thresholds()
    assert thresholds.shape == (14,), f"Expected (14,), got {thresholds.shape}"
    assert SAM_ALPHA == 0.25
    assert SAM_MASK_SIZE == 40

    # All thresholds should be in (0, 1).
    assert np.all(thresholds > 0) and np.all(thresholds < 1)

    print("  PASSED - SAM thresholds OK")
    return True


def test_sam_ip_model():
    """Test SAM-IP model architecture and forward pass."""
    print("\n[TEST] Models.SamIp")
    from Tactile.Models.SamIp import SAMIP

    model = SAMIP(num_tasks=14)

    # Check parameter count.
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params}")

    # Test forward pass.
    batch_size = 4
    thumbnail = torch.randn(batch_size, 3, 40, 40)
    task_id = torch.tensor([0, 3, 7, 13])

    logits = model(thumbnail, task_id)
    assert logits.shape == (batch_size, 1, 40, 40), f"Expected (4, 1, 40, 40), got {logits.shape}"

    # Test predict_mask.
    mask = model.predict_mask(thumbnail, task_id, threshold=0.3)
    assert mask.shape == (batch_size, 40, 40), f"Expected (4, 40, 40), got {mask.shape}"
    assert mask.min() >= 0 and mask.max() <= 1

    # Test gradient flow.
    loss = logits.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"

    print("  PASSED - SAM-IP model OK")
    return True


def test_task_embeddings():
    """Test task embedding module."""
    print("\n[TEST] Models.TaskEmb")
    from Tactile.Models.TaskEmb import TaskEmbeddingTable, TaskProjectionHead

    # Test embedding table.
    emb = TaskEmbeddingTable(num_tasks=14, emb_dim=128)
    task_ids = torch.tensor([0, 5, 13])
    embeddings = emb(task_ids)
    assert embeddings.shape == (3, 128), f"Expected (3, 128), got {embeddings.shape}"

    # Test get all embeddings.
    all_embs = emb.get_all_embeddings()
    assert all_embs.shape == (14, 128)

    # Test init from features.
    task_features = {
        0: torch.randn(10, 128),
        5: torch.randn(20, 128),
    }
    emb.init_from_features(task_features)

    # Test projection head.
    proj = TaskProjectionHead(in_dim=256, out_dim=128)
    features_256 = torch.randn(5, 256)
    features_128 = proj(features_256)
    assert features_128.shape == (5, 128)

    print("  PASSED - Task embeddings OK")
    return True


def test_preprocessing():
    """Test image preprocessing module."""
    print("\n[TEST] Pipeline.Preprocess")
    from Tactile.Pipeline.Preprocess import (
        preprocess_for_yolo, preprocess_for_sam, preprocess_int8
    )

    # Create a dummy image.
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test YOLO preprocessing.
    yolo_tensor = preprocess_for_yolo(dummy_img)
    assert yolo_tensor.shape == (1, 3, 160, 160), f"Expected (1,3,160,160), got {yolo_tensor.shape}"
    assert yolo_tensor.min() >= 0 and yolo_tensor.max() <= 1

    # Test SAM preprocessing.
    sam_tensor = preprocess_for_sam(dummy_img)
    assert sam_tensor.shape == (1, 3, 40, 40), f"Expected (1,3,40,40), got {sam_tensor.shape}"

    # Test INT8 preprocessing.
    int8_result = preprocess_int8(dummy_img)
    assert int8_result.shape == (160, 160, 3)
    assert int8_result.dtype == np.int8

    print("  PASSED - Preprocessing OK")
    return True


def test_soft_gating():
    """Test SAM soft gating module."""
    print("\n[TEST] Pipeline.SoftGate")
    from Tactile.Pipeline.SoftGate import apply_soft_gating

    # Create dummy detections.
    detections = [
        {"bbox": [10, 10, 50, 50], "confidence": 0.9, "class_id": 40, "class_name": "wine glass"},
        {"bbox": [100, 100, 150, 150], "confidence": 0.8, "class_id": 41, "class_name": "cup"},
    ]

    # Create mask: left half is 1 (active), right half is 0.
    sam_mask = np.zeros((40, 40), dtype=np.float32)
    sam_mask[:, :20] = 1.0  # Left half active

    gated = apply_soft_gating(detections, sam_mask, 160, 160, alpha=0.25)

    # First detection center ~(30, 30) -> mask coords ~(7, 7) -> in active zone.
    assert gated[0]["confidence_effective"] == 0.9  # Unchanged
    assert gated[0]["sam_gated"] == False

    # Second detection center ~(125, 125) -> mask coords ~(31, 31) -> in inactive zone.
    assert abs(gated[1]["confidence_effective"] - 0.2) < 1e-6  # 0.8 * 0.25
    assert gated[1]["sam_gated"] == True

    print("  PASSED - Soft gating correctly suppresses and preserves")
    return True


def test_task_scoring():
    """Test task scoring module."""
    print("\n[TEST] Pipeline.TaskScore")
    from Tactile.Pipeline.TaskScore import compute_task_scores, compute_task_scores_int8

    # FP32 test.
    features = torch.randn(5, 128)
    TaskEmb = torch.randn(128)
    scores = compute_task_scores(features, TaskEmb)
    assert scores.shape == (5,), f"Expected (5,), got {scores.shape}"
    assert scores.min() >= 0 and scores.max() <= 1

    # INT8 test.
    features_int8 = np.random.randint(-128, 127, (5, 128), dtype=np.int8)
    emb_int8 = np.random.randint(-128, 127, (128,), dtype=np.int8)
    scores_int8 = compute_task_scores_int8(features_int8, emb_int8)
    assert scores_int8.dtype == np.uint8

    # Empty input test.
    empty_scores = compute_task_scores(torch.zeros(0, 128), TaskEmb)
    assert len(empty_scores) == 0

    print("  PASSED - Task scoring OK")
    return True


def test_fused_nms():
    """Test fused NMS module."""
    print("\n[TEST] Pipeline.FusedNms")
    from Tactile.Pipeline.FusedNms import (
        FusedNms, compute_iou, compute_iou_division_free
    )

    # Test IoU computation.
    box_a = [0, 0, 100, 100]
    box_b = [50, 50, 150, 150]
    iou = compute_iou(box_a, box_b)
    expected_iou = 2500.0 / (10000 + 10000 - 2500)
    assert abs(iou - expected_iou) < 1e-4, f"IoU mismatch: {iou} vs {expected_iou}"

    # Test division-free IoU (at 0.5 threshold: 2*2500 > 17500 => False).
    assert compute_iou_division_free(box_a, box_b) == False  # IoU ~0.143

    # Test with overlapping boxes (IoU > 0.5).
    box_c = [0, 0, 100, 100]
    box_d = [20, 20, 120, 120]
    # intersection = 80*80=6400, union = 10000+10000-6400=13600, IoU=6400/13600=0.47.
    assert compute_iou_division_free(box_c, box_d) == False  # IoU ~0.47

    box_e = [0, 0, 100, 100]
    box_f = [10, 10, 100, 100]
    # intersection = 90*90=8100, union = 10000+8100-8100=10000, IoU=0.81.
    assert compute_iou_division_free(box_e, box_f) == True  # IoU ~0.81

    # Test fused NMS with worked example (wine glass vs beer glass).
    detections = [
        {"bbox": [10, 10, 60, 60], "confidence": 0.81,
         "confidence_effective": 0.81, "class_id": 41, "class_name": "cup"},  # beer glass proxy
        {"bbox": [70, 70, 150, 150], "confidence": 0.72,
         "confidence_effective": 0.72, "class_id": 40, "class_name": "wine glass"},
    ]

    task_scores = np.array([0.18, 0.94])

    # Create a prior table where wine glass has high prior for serve wine.
    prior_table = np.full((80, 14), 0.01, dtype=np.float32)
    prior_table[40, 0] = 0.94  # wine glass, serve wine
    prior_table[41, 0] = 0.12  # cup (beer glass proxy), serve wine

    results = FusedNms(detections, task_scores, prior_table, task_id=0, top_k=5)

    # Wine glass should win due to higher fused score.
    assert len(results) == 2
    assert results[0]["class_name"] == "wine glass", \
        f"Expected wine glass to win, got {results[0]['class_name']}"
    print(f"  Wine glass fused score: {results[0]['fused_score']:.4f}")
    print(f"  Cup fused score: {results[1]['fused_score']:.4f}")
    assert results[0]["fused_score"] > results[1]["fused_score"]

    print("  PASSED - Fused NMS correctly ranks wine glass over cup")
    return True


def test_roi_extraction():
    """Test RoI feature extraction."""
    print("\n[TEST] Pipeline.RoiExtract")
    from Tactile.Pipeline.RoiExtract import extract_roi_features_simple

    image_tensor = torch.randn(1, 3, 160, 160)
    detections = [
        {"bbox": [10, 10, 50, 50]},
        {"bbox": [80, 80, 130, 130]},
    ]

    features = extract_roi_features_simple(image_tensor, detections, feature_dim=128)
    assert features.shape == (2, 128), f"Expected (2, 128), got {features.shape}"

    # Features should be L2 normalized.
    norms = torch.norm(features, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=0.1)

    # Empty detection test.
    empty = extract_roi_features_simple(image_tensor, [], feature_dim=128)
    assert empty.shape == (0, 128)

    print("  PASSED - RoI feature extraction OK")
    return True


def test_e2e_pipeline(data_dir: str):
    """Test end-to-end Pipeline."""
    print("\n[TEST] End-to-End Pipeline")

    # Find a test image.
    images_dir = os.path.join(data_dir, "Coco", "val2017")
    if not os.path.exists(images_dir):
        print("  SKIPPED - No COCO images found")
        return False

    images = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    if not images:
        print("  SKIPPED - No images in val2017")
        return False

    test_image = os.path.join(images_dir, images[0])
    print(f"  Test image: {test_image}")

    # Initialize Pipeline.
    from Tactile.Inference import TACTILEPipeline

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "Tactile", "Weights")
    sam_path = os.path.join(weights_dir, "SamIp.pth")
    emb_path = os.path.join(weights_dir, "TaskEmbeddings.npy")
    prior_path = os.path.join(weights_dir, "ClassTaskPrior.npy")

    Pipeline = TACTILEPipeline(
        device="cpu",
        sam_weights_path=sam_path if os.path.exists(sam_path) else None,
        task_emb_path=emb_path if os.path.exists(emb_path) else None,
        prior_table_path=prior_path if os.path.exists(prior_path) else None,
    )

    # Test all 14 Tasks on the same image.
    from Tactile.Config.Tasks import TASK_NAMES, NUM_TASKS
    for task_id in range(NUM_TASKS):
        result = Pipeline.infer(test_image, task_id)

        assert "bbox" in result
        assert "class" in result
        assert "confidence" in result
        assert "task" in result
        assert result["task"] == TASK_NAMES[task_id]

        print(f"  Task {task_id:2d} ({TASK_NAMES[task_id]:25s}): "
              f"{result['class']:15s} (conf={result['confidence']:.4f})")

    print("  PASSED - End-to-end Pipeline OK for all 14 Tasks")
    return True


def main():
    parser = argparse.ArgumentParser(description="TACTILE Test Suite")
    parser.add_argument("--test-e2e", action="store_true",
                        help="Include end-to-end tests (requires data)")
    parser.add_argument("--data-dir", type=str, default="./CocoTaskDataset")
    args = parser.parse_args()

    print("=" * 60)
    print("TACTILE Test Suite")
    print("=" * 60)

    results = {}
    t_start = time.time()

    # Unit tests (no data required).
    tests = [
        ("Config.Tasks", test_config_tasks),
        ("Config.ClassTaskPrior", test_class_task_prior),
        ("Config.SamThresholds", test_sam_thresholds),
        ("Models.SamIp", test_sam_ip_model),
        ("Models.TaskEmb", test_task_embeddings),
        ("Pipeline.Preprocess", test_preprocessing),
        ("Pipeline.SoftGate", test_soft_gating),
        ("Pipeline.TaskScore", test_task_scoring),
        ("Pipeline.FusedNms", test_fused_nms),
        ("Pipeline.RoiExtract", test_roi_extraction),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = "PASSED" if result else "SKIPPED"
        except Exception as e:
            results[name] = f"FAILED: {e}"
            import traceback
            traceback.print_exc()

    # End-to-end tests (requires data + Models).
    if args.test_e2e:
        try:
            result = test_e2e_pipeline(args.data_dir)
            results["e2e_pipeline"] = "PASSED" if result else "SKIPPED"
        except Exception as e:
            results["e2e_pipeline"] = f"FAILED: {e}"
            import traceback
            traceback.print_exc()

    # Summary.
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for name, status in results.items():
        icon = "✓" if "PASSED" in status else ("⚠" if "SKIP" in status else "✗")
        print(f"  {icon} {name}: {status}")
    print(f"\nTotal time: {elapsed:.1f}s")

    passed = sum(1 for s in results.values() if "PASSED" in s)
    failed = sum(1 for s in results.values() if "FAILED" in s)
    skipped = sum(1 for s in results.values() if "SKIP" in s)
    print(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
