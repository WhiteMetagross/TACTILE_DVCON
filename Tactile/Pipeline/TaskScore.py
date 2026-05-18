"""
TACTILE — Task Scoring Module.

Computes task relevance scores for each proposal via dot product
between 128-D proposal features and 128-D task embeddings.

The scoring follows the VEGA software implementation:
  score_i = dot(feature_i[128], TaskEmb[task_id][128])
  Normalize to [0, 1] range for fused scoring.
"""

import torch
import numpy as np
from typing import List, Dict


def compute_task_scores(
    proposal_features: torch.Tensor,
    task_embedding: torch.Tensor,
) -> torch.Tensor:
    """
    Compute task relevance scores for each proposal.

    Args:
        proposal_features: (N, 128) float tensor — RoI features
        task_embedding: (128,) float tensor — task embedding for current task

    Returns:
        (N,) float tensor — task scores normalized to [0, 1]
    """
    if proposal_features.shape[0] == 0:
        return torch.tensor([], dtype=torch.float32)

    # Dot product.
    raw_scores = torch.matmul(proposal_features, task_embedding)  # (N,)

    # Normalize to [0, 1] using sigmoid.
    scores = torch.sigmoid(raw_scores)

    return scores


def compute_task_scores_int8(
    proposal_features: np.ndarray,
    task_embedding: np.ndarray,
) -> np.ndarray:
    """
    INT8-compatible task scoring (matches VEGA hardware implementation).

    Args:
        proposal_features: (N, 128) int8 array
        task_embedding: (128,) int8 array

    Returns:
        (N,) uint8 array — scores in [0, 255]
    """
    if proposal_features.shape[0] == 0:
        return np.array([], dtype=np.uint8)

    # Cast to int32 for accumulation (prevents overflow).
    features_i32 = proposal_features.astype(np.int32)
    emb_i32 = task_embedding.astype(np.int32)

    # Dot product.
    acc = features_i32 @ emb_i32  # (N,)

    # Right-shift by 14 and clamp to [0, 255].
    scores = np.clip(acc >> 14, 0, 255).astype(np.uint8)

    return scores
