"""Small guidance schedules for the SEGS prototype."""

from __future__ import annotations


def brisque_quality_scale(low_quality_count: int, half_after: int = 5, stop_after: int = 10) -> float:
    """Scale guidance down when repeated BRISQUE checks indicate degraded views."""
    if low_quality_count >= stop_after:
        return 0.0
    if low_quality_count >= half_after:
        return 0.5
    return 1.0

