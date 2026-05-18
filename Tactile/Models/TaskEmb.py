"""
TACTILE — Task Embedding Module.

Manages the 14 x 128 task embedding matrix used for task scoring.
Two initialization strategies:
  1. Mean-pooled RoI features of task-preferred instances (no-train baseline)
  2. Learned contrastive embeddings (trained with contrastive loss)

For Stage 2A, strategy 1 is acceptable.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from ..Config.Tasks import NUM_TASKS


# Default save path.
TASK_EMB_PATH = Path(__file__).parent.parent / "Weights" / "TaskEmbeddings.npy"


class TaskEmbeddingTable(nn.Module):
    """
    A simple embedding table: 14 Tasks x 128 dimensions.
    Can be initialized randomly, from precomputed features, or learned.
    """

    def __init__(self, num_tasks: int = NUM_TASKS, emb_dim: int = 128):
        super().__init__()
        self.num_tasks = num_tasks
        self.emb_dim = emb_dim

        # Learnable embedding table.
        self.embeddings = nn.Embedding(num_tasks, emb_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, task_id: torch.Tensor) -> torch.Tensor:
        """
        Retrieve task embedding(s).

        Args:
            task_id: (B,) integer tensor

        Returns:
            (B, 128) task embedding vectors
        """
        return self.embeddings(task_id.long())

    def get_all_embeddings(self) -> torch.Tensor:
        """Return all 14 embeddings as (14, 128) tensor."""
        ids = torch.arange(self.num_tasks, device=self.embeddings.weight.device)
        return self.embeddings(ids)

    def init_from_features(self, task_features: dict):
        """
        Initialize embeddings from mean-pooled RoI features.

        Args:
            task_features: dict mapping task_id -> np.ndarray of shape (N, 128)
                          containing RoI features of preferred objects for that task.
        """
        with torch.no_grad():
            for task_id in range(self.num_tasks):
                if task_id in task_features and len(task_features[task_id]) > 0:
                    feats = task_features[task_id]
                    if isinstance(feats, np.ndarray):
                        feats = torch.from_numpy(feats)
                    mean_feat = feats.float().mean(dim=0)
                    # L2 normalize.
                    mean_feat = mean_feat / (mean_feat.norm() + 1e-8)
                    self.embeddings.weight[task_id] = mean_feat

    def save(self, path: str = None):
        """Save embeddings to disk as numpy array."""
        save_path = Path(path) if path else TASK_EMB_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), self.embeddings.weight.detach().cpu().numpy())
        print(f"[INFO] Saved task embeddings to {save_path}")

    def load(self, path: str = None):
        """Load embeddings from disk."""
        load_path = Path(path) if path else TASK_EMB_PATH
        if not load_path.exists():
            print(f"[WARNING] Task embeddings not found at {load_path}, using random init.")
            return
        emb_np = np.load(str(load_path))
        with torch.no_grad():
            self.embeddings.weight.copy_(torch.from_numpy(emb_np))
        print(f"[INFO] Loaded task embeddings from {load_path}")


class TaskProjectionHead(nn.Module):
    """
    Projection head: 256-D backbone features -> 128-D task-scoring features.
    Maps exactly to one systolic array pass on FPGA.
    """

    def __init__(self, in_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, 256) from GAP over RoI-aligned features

        Returns:
            (N, 128) projected features for task scoring
        """
        return self.proj(features)
