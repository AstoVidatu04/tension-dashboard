from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ConflictEstimate:
    p30: float
    p90: float
    note: str


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = lo
    return max(lo, min(hi, v))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def estimate_conflict_probability(
    scored: pd.DataFrame,
    dip_latest: float,
    mil_latest: float,
    econ_latest: float,
) -> ConflictEstimate:
    """
    Heuristic (NOT calibrated) estimate of direct US–Iran kinetic conflict probability.

    - Outputs: p30 and p90 as percentages (0..100).
    - Designed to be conservative: typically single digits.
    - Uses your subscores + short-term trends as inputs.

    IMPORTANT:
    This is a heuristic mapper from your indices to an estimated probability.
    Replace with a calibrated model when available.
    """

    # Base rates (heuristic): rare event.
    base30 = 2.0   # 2% / 30d baseline
    base90 = 6.0   # 6% / 90d baseline

    # Trend features from scored table (use last available 7d delta)
    d_dip_7 = 0.0
    d_mil_7 = 0.0
    d_econ_7 = 0.0
    if scored is not None and len(scored) >= 8:
        try:
            d_dip_7 = float(dip_latest - float(scored["dip_score"].iloc[-8]))
            d_mil_7 = float(mil_latest - float(scored["mil_score"].iloc[-8]))
            d_econ_7 = float(econ_latest - float(scored["econ_score"].iloc[-8]))
        except Exception:
            pass

    # Confirmation logic: military confirmation is required for high probabilities.
    mil_confirm = 1.0 if mil_latest >= 70 else 0.0
    dip_hot = 1.0 if dip_latest >= 80 else 0.0

    # Convert subscores into a small probability uplift using a bounded transform.
    # Map scores (0..100) into centered ranges (-1..+1) roughly.
    dip_c = (dip_latest - 50.0) / 35.0
    mil_c = (mil_latest - 50.0) / 30.0
    econ_c = (econ_latest - 50.0) / 40.0

    # Trend scaling
    dip_t = _clamp(d_dip_7 / 25.0, -1.0, 1.0)
    mil_t = _clamp(d_mil_7 / 20.0, -1.0, 1.0)
    econ_t = _clamp(d_econ_7 / 25.0, -1.0, 1.0)

    # Raw risk pressure (dimensionless). Military dominates for kinetic conflict.
    raw = (
        1.25 * mil_c +
        0.55 * dip_c +
        0.25 * econ_c +
        0.70 * mil_t +
        0.35 * dip_t +
        0.20 * econ_t
    )

    # Convert to a multiplier around 1.0, bounded.
    # sigmoid(raw) in (0,1) -> scale to (0.6..2.2)
    mult = 0.6 + 1.6 * _sigmoid(raw)

    # Apply conservative gating:
    # If military isn't confirming, cap p30 aggressively even if diplomatic is hot.
    p30 = base30 * mult
    if mil_latest < 55:
        p30 = min(p30, 8.0)
    elif mil_latest < 65:
        p30 = min(p30, 12.0)

    # If diplomatic is hot but military is low, keep it in "diplomatic crisis" band.
    if dip_hot and not mil_confirm:
        p30 = min(p30, 10.0)

    # 90d is higher but still conservative; use stronger dependency on military confirmation.
    p90 = base90 * (0.7 + 1.8 * _sigmoid(raw))
    if mil_latest < 55:
        p90 = min(p90, 18.0)
    elif mil_latest < 65:
        p90 = min(p90, 25.0)

    # Absolute clamps (avoid scary fake certainty)
    p30 = _clamp(p30, 0.0, 25.0)
    p90 = _clamp(p90, 0.0, 40.0)

    note = (
        "Heuristic estimate (not calibrated). Mostly driven by Military risk + trend; "
        "capped unless Military confirms."
    )
    return ConflictEstimate(p30=p30, p90=p90, note=note)
