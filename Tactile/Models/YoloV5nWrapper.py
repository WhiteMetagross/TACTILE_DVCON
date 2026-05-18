"""
TACTILE — YOLOv5n Detection Wrapper.

Wraps the Ultralytics YOLOv5n model for object detection at 160x160 resolution.
Provides:
  - Raw proposal extraction (bboxes, confidences, class IDs)
  - Access to intermediate feature maps for RoI extraction
  - INT8 quantization support (deferred to Stage 2B)

For Stage 2A, we use FP32 Inference with the pretrained YOLOv5n weights.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# We use ultralytics YOLO for convenience.
from ultralytics import YOLO


# Input resolution as specified in Stage 1 architecture.
INPUT_SIZE = 160


class YOLOv5nDetector:
    """
    YOLOv5n-based detector wrapper for TACTILE Pipeline.
    Uses Ultralytics YOLOv5n (or YOLOv8n as drop-in replacement with
    equivalent architecture class) pretrained on COCO.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu",
                 conf_threshold: float = 0.15, iou_threshold: float = 0.5):
        """
        Args:
            model_path: path to YOLO weights. If None, downloads yolov5nu.pt
            device: "cpu" or "cuda"
            conf_threshold: minimum confidence to keep a detection
            iou_threshold: IoU threshold for built-in NMS (we'll re-NMS later)
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load model.
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Download pre-trained YOLOv5n (ultralytics format).
            # yolov5nu = YOLOv5 nano updated to ultralytics format.
            print("[INFO] Loading YOLOv5nu pretrained on COCO...")
            self.model = YOLO("yolov5nu.pt")

        self.model.to(device)

    def detect(self, image_path: str, input_size: int = INPUT_SIZE,
               max_proposals: int = 75) -> List[Dict]:
        """
        Run detection on a single image.

        Args:
            image_path: path to input image
            input_size: resize to this resolution (default 160)
            max_proposals: maximum number of proposals to return

        Returns:
            List of detection dicts, each containing:
              - "bbox": [x1, y1, x2, y2] in original image coordinates
              - "confidence": float
              - "class_id": int (COCO contiguous index 0..79)
              - "class_name": str
        """
        # Run Inference.
        results = self.model.predict(
            source=image_path,
            imgsz=input_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=max_proposals,
            device=self.device,
            verbose=False,
        )

        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    detections.append({
                        "bbox": bbox,           # [x1, y1, x2, y2]
                        "confidence": conf,
                        "class_id": cls_id,     # contiguous index 0..79
                        "class_name": result.names[cls_id],
                    })

        # Sort by confidence descending and limit.
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[:max_proposals]

    def detect_from_tensor(self, image_tensor: torch.Tensor,
                           max_proposals: int = 75) -> List[Dict]:
        """
        Run detection on a preprocessed tensor.

        Args:
            image_tensor: (1, 3, H, W) normalized tensor
            max_proposals: maximum proposals

        Returns:
            Same format as detect()
        """
        results = self.model.predict(
            source=image_tensor,
            imgsz=INPUT_SIZE,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=max_proposals,
            device=self.device,
            verbose=False,
        )

        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    detections.append({
                        "bbox": bbox,
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": result.names[cls_id],
                    })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[:max_proposals]

    def get_feature_map(self, image_path: str, input_size: int = INPUT_SIZE) -> torch.Tensor:
        """
        Extract backbone feature map for RoI pooling.

        Returns the feature map from an intermediate layer of the backbone.
        For YOLOv5n, we use the output after the backbone (before the head).

        Returns:
            Feature map tensor of shape (1, C, H, W)
        """
        import cv2
        from torchvision import transforms

        # Read and Preprocess.
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size))

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        tensor = transform(img).unsqueeze(0).to(self.device)

        # Get the internal model and run backbone.
        internal_model = self.model.model
        # Run through the model layers to get feature maps.
        feature_maps = []

        x = tensor
        for i, layer in enumerate(internal_model.model):
            x = layer(x)
            feature_maps.append(x)

        # Return the backbone output (typically layer index ~9 for YOLOv5n).
        # The exact layer depends on the model architecture.
        # For YOLOv5n, backbone ends around layer 9 (P5 features).
        # We'll use a mid-level feature map for RoI extraction.
        if len(feature_maps) >= 10:
            return feature_maps[9]
        else:
            return feature_maps[-1]
