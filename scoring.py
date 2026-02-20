from __future__ import annotations

import math
import numpy as np
import pandas as pd

from ui_components import clamp


def zscore_safe(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mu = float(s.mean(skipna=True)) if len(s) else 0.0
    std = float(s.std(ddof=0, skipna=True)) if len(s) else 0.0
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / std


def zscore_or_level(series: pd.Series, baseline: float, scale: float) -> pd.Series:
    """
    Use window z-score when there is variance.
    If variance collapses (single day / flat window), map absolute level to a
    pseudo-z score so indicators do not pin to 50 by construction.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    std = float(s.std(ddof=0, skipna=True)) if len(s) else 0.0
    if std > 1e-9 and not np.isnan(std):
        mu = float(s.mean(skipna=True))
        out = (s - mu) / std
    else:
        denom = max(float(scale), 1e-6)
        out = (s - float(baseline)) / denom
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def logistic_0_100(x: float) -> float:
    return float(100.0 * (1.0 / (1.0 + math.exp(-x))))


def logit_from_0_100(score_0_100: float) -> float:
    eps = 1e-6
    p = min(max(float(score_0_100) / 100.0, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def cap_term(x: pd.Series, m: float = 1.2) -> pd.Series:
    return np.clip(x.astype(float), -m, m)


def compute_subscores(
    daily: pd.DataFrame,
    smooth_days: int,
    w_tone: float = 0.7,
    w_intent: float = 1.0,
    w_conflict: float = 1.0,
    w_milrate: float = 1.0,
    w_econrate: float = 1.0,
    w_volume: float = 0.4,
    w_diplomacy_damp: float = 0.7,
    cap_m: float = 1.2,
) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame(columns=list(daily.columns) + ["dip_score","mil_score","econ_score","composite_score"])

    d = daily.copy()

    d["tone_sm"] = d["neg_tone_mean"].rolling(window=smooth_days, min_periods=1).mean()
    d["intent_sm"] = d["intent_net"].rolling(window=smooth_days, min_periods=1).mean()
    d["conflict_sm"] = d["conflict_share"].rolling(window=smooth_days, min_periods=1).mean()
    d["diplomacy_sm"] = d["diplomacy_share"].rolling(window=smooth_days, min_periods=1).mean()
    d["milrate_sm"] = d["mil_rate"].rolling(window=smooth_days, min_periods=1).mean()
    d["econrate_sm"] = d["econ_rate"].rolling(window=smooth_days, min_periods=1).mean()

    d["z_tone"] = zscore_or_level(d["tone_sm"], baseline=0.8, scale=0.8)
    d["z_intent"] = zscore_or_level(d["intent_sm"], baseline=0.0, scale=0.08)
    d["z_conflict"] = zscore_or_level(d["conflict_sm"], baseline=0.02, scale=0.05)
    d["z_diplomacy"] = zscore_or_level(d["diplomacy_sm"], baseline=0.08, scale=0.08)
    d["z_milrate"] = zscore_or_level(d["milrate_sm"], baseline=0.03, scale=0.07)
    d["z_econrate"] = zscore_or_level(d["econrate_sm"], baseline=0.03, scale=0.07)
    d["z_volume"] = zscore_or_level(np.log1p(d["articles"].astype(float)), baseline=2.2, scale=0.9)

    d["raw_dip"] = (
        cap_term(w_intent * d["z_intent"], cap_m) +
        cap_term(w_volume * d["z_volume"], cap_m) +
        cap_term(w_tone * d["z_tone"], cap_m) -
        cap_term(w_diplomacy_damp * d["z_diplomacy"], cap_m)
    )

    d["raw_mil"] = (
        cap_term(w_conflict * d["z_conflict"], cap_m) +
        cap_term(w_milrate * d["z_milrate"], cap_m) +
        cap_term(0.4 * d["z_tone"], cap_m)
    )

    d["raw_econ"] = (
        cap_term(w_econrate * d["z_econrate"], cap_m) +
        cap_term(0.5 * d["z_volume"], cap_m) +
        cap_term(0.3 * d["z_intent"], cap_m)
    )

    d["dip_score"] = d["raw_dip"].apply(lambda x: logistic_0_100(float(x)))
    d["mil_score"] = d["raw_mil"].apply(lambda x: logistic_0_100(float(x)))
    d["econ_score"] = d["raw_econ"].apply(lambda x: logistic_0_100(float(x)))

    d["composite_score"] = (
        0.45 * d["dip_score"] +
        0.35 * d["mil_score"] +
        0.20 * d["econ_score"]
    ).astype(float)

    for c in ["dip_score","mil_score","econ_score","composite_score"]:
        d[c] = d[c].apply(clamp)

    return d


def apply_diversity_multiplier(score_0_100: float, div_mult: float) -> float:
    if score_0_100 is None or (isinstance(score_0_100, float) and math.isnan(score_0_100)):
        return float("nan")
    raw = logit_from_0_100(clamp(score_0_100))
    raw *= float(div_mult)
    return clamp(logistic_0_100(raw))
