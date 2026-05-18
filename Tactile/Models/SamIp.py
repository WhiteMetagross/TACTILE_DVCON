"""
TACTILE — SAM-IP (Spatial Task Attention Map) Module.

A tiny 3-layer CNN that generates a 40x40 binary spatial mask identifying
image regions likely to contain task-relevant objects.

Architecture (from Stage 1 proposal):
  Layer 1: Conv2d(3+14, 4, kernel_size=3, stride=2, padding=1)  -> 20x20x4
  Layer 2: Conv2d(4, 4, kernel_size=3, stride=1, padding=1, groups=4)  -> 20x20x4 (depthwise)
  Layer 3: Conv2d(4, 1, kernel_size=1, stride=1)  -> 20x20x1
  Upsample: bilinear to 40x40

The task_id is encoded as a 14-channel one-hot map concatenated with the 3-channel
RGB thumbnail, giving 17 input channels. However, for hardware-friendliness we
actually encode task as a 14-D vector broadcast spatially to match the image size.

Total parameters: ~199 INT8 parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SAMIP(nn.Module):
    """
    Spatial Task Attention Map (SAM-IP).

    Takes a 40x40x3 RGB thumbnail and a task_id, outputs a 40x40 spatial mask.
    """

    def __init__(self, num_tasks: int = 14):
        super().__init__()
        self.num_tasks = num_tasks

        # Input: 3 (RGB) + num_tasks (one-hot task encoding) = 17 channels.
        in_channels = 3 + num_tasks

        # Layer 1: Standard Conv 3x3, stride 2 → 20x20.
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(4)

        # Layer 2: Depthwise Conv 3x3, stride 1 → 20x20.
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, groups=4, bias=True)
        self.bn2 = nn.BatchNorm2d(4)

        # Layer 3: Pointwise Conv 1x1 → 20x20x1.
        self.conv3 = nn.Conv2d(4, 1, kernel_size=1, stride=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for small network."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, thumbnail: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            thumbnail: (B, 3, 40, 40) RGB thumbnail, normalized to [0, 1]
            task_id: (B,) integer tensor of task IDs (0..13)

        Returns:
            logits: (B, 1, 40, 40) raw logits (apply sigmoid + threshold externally)
        """
        B = thumbnail.shape[0]
        device = thumbnail.device

        # Create task one-hot encoding and broadcast to spatial dimensions.
        task_onehot = F.one_hot(task_id.long(), num_classes=self.num_tasks)  # (B, 14)
        task_onehot = task_onehot.float()
        task_map = task_onehot.unsqueeze(-1).unsqueeze(-1)  # (B, 14, 1, 1)
        task_map = task_map.expand(B, self.num_tasks, 40, 40)  # (B, 14, 40, 40)

        # Concatenate RGB + task encoding.
        x = torch.cat([thumbnail, task_map], dim=1)  # (B, 17, 40, 40)

        # Forward through tiny CNN.
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 4, 20, 20)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 4, 20, 20)
        x = self.conv3(x)                    # (B, 1, 20, 20)

        # Bilinear upsample to 40x40.
        x = F.interpolate(x, size=(40, 40), mode='bilinear', align_corners=False)

        return x  # (B, 1, 40, 40) — raw logits

    def predict_mask(self, thumbnail: torch.Tensor, task_id: torch.Tensor,
                     threshold: float = 0.3) -> torch.Tensor:
        """
        Get binary mask from the SAM-IP model.

        Returns:
            mask: (B, 40, 40) binary mask (0 or 1)
        """
        logits = self.forward(thumbnail, task_id)
        probs = torch.sigmoid(logits)
        mask = (probs.squeeze(1) >= threshold).float()
        return mask

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
