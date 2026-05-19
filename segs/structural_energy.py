"""Structural energy helpers for PCA-guided sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def project_to_pca(feature: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project BCHW U-Net features into a stored PCA basis."""
    if feature.ndim != 4:
        raise ValueError(f"feature must have shape [B, C, H, W], got {tuple(feature.shape)}")

    _, channels, _, _ = feature.shape
    feature_2d = feature.permute(0, 2, 3, 1).reshape(-1, channels)
    return (feature_2d - mean.to(feature.device, feature.dtype)) @ basis.to(feature.device, feature.dtype)


def mean_teacher_features(
    teacher_feature: torch.Tensor,
    topk: int,
    tokens_per_sample: int | None = None,
) -> torch.Tensor:
    """Average top-k teacher features while preserving token order."""
    if topk <= 0:
        raise ValueError("topk must be positive")
    if teacher_feature.ndim != 2:
        raise ValueError(
            f"teacher_feature must have shape [topk * tokens, dim], got {tuple(teacher_feature.shape)}"
        )

    total_tokens, feature_dim = teacher_feature.shape
    if tokens_per_sample is None:
        if total_tokens % topk != 0:
            raise ValueError(f"teacher_feature token count {total_tokens} is not divisible by topk {topk}")
        tokens_per_sample = total_tokens // topk
    elif tokens_per_sample * topk != total_tokens:
        raise ValueError(
            f"topk * tokens_per_sample must equal teacher tokens: {topk} * {tokens_per_sample} != {total_tokens}"
        )

    return teacher_feature.view(topk, tokens_per_sample, feature_dim).mean(dim=0)


def structural_energy(
    feature: torch.Tensor,
    *,
    mean: torch.Tensor,
    basis: torch.Tensor,
    teacher_feature: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """MSE energy between current PCA-projected features and top-k teacher mean."""
    projected = project_to_pca(feature, mean, basis)
    teacher_mean = mean_teacher_features(
        teacher_feature.to(projected.device, projected.dtype),
        topk=topk,
        tokens_per_sample=projected.shape[0],
    )
    return F.mse_loss(projected, teacher_mean)

