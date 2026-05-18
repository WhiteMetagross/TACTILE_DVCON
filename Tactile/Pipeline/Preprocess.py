"""
TACTILE — Image Preprocessing Module.

Handles:
  - Resize to 160x160 for YOLOv5n input
  - Resize to 40x40 thumbnail for SAM-IP input
  - Normalization (COCO mean/std or INT8 clamping)
"""

import cv2
import numpy as np
import torch
from typing import Tuple


# COCO dataset normalization constants.
COCO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
COCO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# TACTILE resolution targets.
YOLO_INPUT_SIZE = 160
SAM_INPUT_SIZE = 40


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk as RGB numpy array.

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8, RGB order
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_for_yolo(image: np.ndarray, size: int = YOLO_INPUT_SIZE) -> torch.Tensor:
    """
    Preprocess image for YOLOv5n Inference.

    Args:
        image: (H, W, 3) RGB uint8 array
        size: target size (default 160)

    Returns:
        (1, 3, size, size) float32 tensor normalized to [0, 1]
    """
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def preprocess_for_sam(image: np.ndarray, size: int = SAM_INPUT_SIZE) -> torch.Tensor:
    """
    Preprocess image as 40x40 thumbnail for SAM-IP.

    Args:
        image: (H, W, 3) RGB uint8 array
        size: target thumbnail size (default 40)

    Returns:
        (1, 3, size, size) float32 tensor normalized to [0, 1]
    """
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 40, 40)
    return tensor


def preprocess_int8(image: np.ndarray, size: int = YOLO_INPUT_SIZE) -> np.ndarray:
    """
    INT8 preprocessing as specified in Stage 1 architecture.

    INT8 = clamp((uint8 - mean_c) * scale_c, -128, 127)

    This is for FPGA-friendly Inference. For Stage 2A we use FP32,
    but this function is provided for consistency validation.

    Args:
        image: (H, W, 3) RGB uint8 array
        size: target size

    Returns:
        (size, size, 3) int8 array
    """
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

    # INT8 quantization parameters (typical COCO calibration).
    mean_c = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # ~= COCO_MEAN * 255
    scale_c = np.array([1.0/58.395, 1.0/57.12, 1.0/57.375], dtype=np.float32)  # ~= 1/(COCO_STD*255)

    result = (resized.astype(np.float32) - mean_c) * scale_c * 128.0
    result = np.clip(result, -128, 127).astype(np.int8)

    return result


def get_original_size(image_path: str) -> Tuple[int, int]:
    """Return (height, width) of the original image."""
    img = cv2.imread(image_path)
    return img.shape[:2]
