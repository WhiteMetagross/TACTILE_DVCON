"""
TACTILE — RoI Feature Extraction Module.

For each detection proposal:
  1. RoI Align: extract 7x7 feature grid from backbone feature map
  2. Global Average Pool: 7x7xC -> C-dimensional vector
  3. Linear projection: C -> 128-D (via TaskProjectionHead)

The 128-D feature is used for task scoring via dot product with task embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import roi_align
from typing import List, Dict, Optional

from ..Models.TaskEmb import TaskProjectionHead


class RoIFeatureExtractor(nn.Module):
    """
    Extract 128-D features from backbone feature maps using RoI Align.
    """

    def __init__(self, feature_channels: int = 256, output_dim: int = 128,
                 roi_size: int = 7):
        """
        Args:
            feature_channels: number of channels in backbone feature map
            output_dim: dimensionality of output features (128)
            roi_size: RoI Align output spatial size (7x7)
        """
        super().__init__()
        self.roi_size = roi_size
        self.feature_channels = feature_channels
        self.output_dim = output_dim

        # Adaptive pooling to handle variable feature map channels.
        self.channel_adapter = nn.Conv2d(feature_channels, 256, 1, bias=False)

        # Global Average Pooling: 7x7x256 -> 256.
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Projection: 256 -> 128.
        self.projection = TaskProjectionHead(in_dim=256, out_dim=output_dim)

    def forward(self, feature_map: torch.Tensor,
                boxes: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract features for a list of bounding boxes.

        Args:
            feature_map: (1, C, H, W) backbone feature map
            boxes: list of (N, 4) tensors containing [x1, y1, x2, y2] boxes
                   in the feature map coordinate space

        Returns:
            (N, 128) feature vectors
        """
        if len(boxes) == 0 or (len(boxes) == 1 and boxes[0].shape[0] == 0):
            return torch.zeros(0, self.output_dim, device=feature_map.device)

        # Adapt channels if needed.
        if feature_map.shape[1] != 256:
            feature_map = self.channel_adapter(feature_map)

        # RoI Align.
        roi_features = roi_align(
            feature_map,
            boxes,
            output_size=self.roi_size,
            spatial_scale=feature_map.shape[-1] / 160.0,  # scale relative to input
            aligned=True,
        )  # (N, 256, 7, 7)

        # Global Average Pool.
        pooled = self.gap(roi_features).squeeze(-1).squeeze(-1)  # (N, 256)

        # Project to 128-D.
        features = self.projection(pooled)  # (N, 128)

        return features


def extract_roi_features_simple(
    image_tensor: torch.Tensor,
    detections: List[Dict],
    feature_dim: int = 128,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Simple RoI feature extraction without backbone feature maps.
    Uses cropped and resized regions as a simpler alternative.

    This is used when backbone feature map extraction is not available
    (e.g., when using the Ultralytics Inference API directly).

    Args:
        image_tensor: (1, 3, 160, 160) input image tensor
        detections: list of detection dicts with "bbox" keys
        feature_dim: output feature dimension
        device: computation device

    Returns:
        (N, feature_dim) feature tensor
    """
    import torch.nn.functional as F

    if len(detections) == 0:
        return torch.zeros(0, feature_dim, device=device)

    img = image_tensor.squeeze(0)  # (3, 160, 160)
    C, H, W = img.shape

    features = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Clamp to image bounds.
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(W, int(x2))
        y2 = min(H, int(y2))

        if x2 <= x1 or y2 <= y1:
            features.append(torch.zeros(feature_dim, device=device))
            continue

        # Crop and resize to 7x7.
        crop = img[:, y1:y2, x1:x2].unsqueeze(0)
        crop_resized = F.interpolate(crop, size=(7, 7), mode='bilinear',
                                     align_corners=False)  # (1, 3, 7, 7)

        # Simple feature: flatten and project.
        flat = crop_resized.flatten()  # 3*7*7 = 147

        # Pad or truncate to feature_dim.
        if flat.shape[0] >= feature_dim:
            feat = flat[:feature_dim]
        else:
            feat = F.pad(flat, (0, feature_dim - flat.shape[0]))

        # L2 normalize.
        feat = feat / (feat.norm() + 1e-8)
        features.append(feat)

    return torch.stack(features, dim=0)  # (N, 128)
