from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from config import (
    ESCALATION_KW,
    DEESCALATION_KW,
    MILITARY_KW,
    ECONOMIC_KW,
    CONFLICT_THEME_NEEDLES,
    DIPLO_THEME_NEEDLES,
)


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().strip().split())


def count_keywords_in_text(text: str, keywords: List[str]) -> int:
    t = normalize_text(text)
    if not t:
        return 0
    hits = 0
    for k in keywords:
        kk = k.lower()
        if kk and kk in t:
            hits += 1
    return hits


def themes_contains_any(themes, needles: List[str]) -> bool:
    if not isinstance(themes, list):
        return False
    for t in themes:
        if not isinstance(t, str):
            continue
        u = t.upper()
        for n in needles:
            if n in u:
                return True
    return False


def extract_tone_series(df: pd.DataFrame) -> pd.Series:
    """Robust extraction of tone from likely columns."""
    candidates = ["tone", "avgTone", "avgtone", "tone_score", "toneScore", "Tone"]
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if int(s.notna().sum()) > 0:
                return s.astype(float)

    if "tone" in df.columns:
        s = df["tone"].astype(str).str.extract(r"(-?\d+(\.\d+)?)")[0]
        s = pd.to_numeric(s, errors="coerce")
        return s.astype(float)

    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


def build_daily_features(df_articles: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date", "articles", "neg_tone_mean",
        "diplomacy_share", "conflict_share",
        "escal_hits", "deesc_hits", "intent_net",
        "mil_hits", "econ_hits", "mil_rate", "econ_rate",
    ]
    if df_articles is None or df_articles.empty:
        return pd.DataFrame(columns=cols)

    df = df_articles.copy()

    if "seendate" in df.columns:
        df["dt"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
    elif "datetime" in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    else:
        df["dt"] = pd.NaT

    df = df.dropna(subset=["dt"]).copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["date"] = df["dt"].dt.date

    df["tone_num"] = extract_tone_series(df).clip(-10, 10)
    df["neg_mag"] = df["tone_num"].apply(
        lambda x: float(-x) if pd.notna(x) and x < 0 else (0.0 if pd.notna(x) else np.nan)
    )

    title = df["title"] if "title" in df.columns else pd.Series([""] * len(df), index=df.index)
    extras = []
    for col in ["description", "snippet", "summary"]:
        if col in df.columns:
            extras.append(df[col].astype(str))
    if extras:
        text = title.astype(str) + " " + extras[0]
        for e in extras[1:]:
            text = text + " " + e
    else:
        text = title.astype(str)

    df["escal_hits_raw"] = text.apply(lambda t: count_keywords_in_text(t, ESCALATION_KW))
    df["deesc_hits_raw"] = text.apply(lambda t: count_keywords_in_text(t, DEESCALATION_KW))
    df["mil_hits_raw"] = text.apply(lambda t: count_keywords_in_text(t, MILITARY_KW))
    df["econ_hits_raw"] = text.apply(lambda t: count_keywords_in_text(t, ECONOMIC_KW))

    if "themes_list" in df.columns:
        df["is_diplomacy"] = df["themes_list"].apply(lambda x: 1 if themes_contains_any(x, DIPLO_THEME_NEEDLES) else 0)
        df["is_conflict"] = df["themes_list"].apply(lambda x: 1 if themes_contains_any(x, CONFLICT_THEME_NEEDLES) else 0)
    else:
        df["is_diplomacy"] = 0
        df["is_conflict"] = 0

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            articles=("url", "count") if "url" in df.columns else ("date", "count"),
            neg_tone_mean=("neg_mag", "mean"),
            diplomacy_share=("is_diplomacy", "mean"),
            conflict_share=("is_conflict", "mean"),
            escal_hits=("escal_hits_raw", "sum"),
            deesc_hits=("deesc_hits_raw", "sum"),
            mil_hits=("mil_hits_raw", "sum"),
            econ_hits=("econ_hits_raw", "sum"),
        )
        .sort_values("date")
    )

    daily["neg_tone_mean"] = daily["neg_tone_mean"].fillna(0.0).astype(float)
    daily["articles"] = daily["articles"].fillna(0).astype(int)

    for c in ["escal_hits", "deesc_hits", "mil_hits", "econ_hits"]:
        daily[c] = daily[c].fillna(0).astype(int)

    denom = daily["articles"].clip(lower=1)
    daily["intent_net"] = ((daily["escal_hits"] - daily["deesc_hits"]) / denom).astype(float)
    daily["mil_rate"] = (daily["mil_hits"] / denom).astype(float)
    daily["econ_rate"] = (daily["econ_hits"] / denom).astype(float)

    all_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    daily = daily.set_index("date").reindex(all_days).rename_axis("date").reset_index()

    for c in ["articles", "escal_hits", "deesc_hits", "mil_hits", "econ_hits"]:
        daily[c] = daily[c].fillna(0).astype(int)

    for c in ["neg_tone_mean", "diplomacy_share", "conflict_share", "intent_net", "mil_rate", "econ_rate"]:
        daily[c] = daily[c].fillna(0.0).astype(float)

    return daily
