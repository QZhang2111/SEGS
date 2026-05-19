"""View-bin helpers used by the SEGS prototype."""

from __future__ import annotations


def normalize_azimuth(azimuth: float) -> float:
    """Normalize degrees to [-180, 180]."""
    value = (float(azimuth) + 180.0) % 360.0 - 180.0
    if value == -180.0:
        return 180.0
    return value


def is_back_view(azimuth: float, threshold: float = 120.0) -> bool:
    """Return true for paper-aligned back-view bin."""
    return abs(normalize_azimuth(azimuth)) >= threshold


def structural_guidance_weight(azimuth: float, threshold: float = 120.0) -> float:
    """Ramp structural guidance from 0.5 at threshold to 1.0 at 180 degrees."""
    angle = abs(normalize_azimuth(azimuth))
    if angle < threshold:
        return 0.0
    if threshold >= 180.0:
        return 1.0
    return 0.5 + ((min(angle, 180.0) - threshold) / (180.0 - threshold)) * 0.5

