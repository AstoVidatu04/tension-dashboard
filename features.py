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

MILITARY_INVENTORY_KW = {
    "Missiles/Rockets": ["missile", "ballistic", "cruise missile", "rocket", "launcher"],
    "Drones/UAV": ["drone", "uav", "loitering munition", "shahed", "recon drone"],
    "Air Defense": ["air defense", "sam", "interceptor", "s-300", "s-400", "patriot"],
    "Naval Assets": ["warship", "destroyer", "frigate", "submarine", "carrier", "naval task force"],
    "Aircraft": ["fighter jet", "sortie", "bomber", "f-16", "f-35", "combat aircraft"],
    "Ground Forces": ["troops", "tank", "artillery", "armor", "brigade", "mobilization"],
}

CONFLICT_SIGNAL_KW = {
    "Force mobilization": ["mobilization", "troop movement", "redeploy", "reserve call-up", "staging area"],
    "Missile activity": ["missile launch", "ballistic missile", "rocket fire", "cruise missile", "intercepted missile"],
    "Air defense activation": ["air defense", "interceptor", "radar lock", "sam battery", "anti-aircraft"],
    "Naval positioning": ["naval exercise", "warship", "carrier group", "strait transit", "maritime patrol"],
    "Proxy activity": ["militia", "proxy", "hezbollah", "houthis", "armed group"],
    "Command/alert posture": ["high alert", "combat readiness", "war footing", "evacuation order", "airspace closure"],
    "Direct clash reporting": ["exchange of fire", "airstrike", "retaliation", "cross-border strike", "casualties"],
}


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
    if isinstance(themes, str):
        # GDELT often returns semicolon/comma separated strings.
        themes = [t.strip() for t in themes.replace(",", ";").split(";") if t.strip()]
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

    themes_source = None
    if "themes_list" in df.columns:
        themes_source = df["themes_list"]
    elif "themes" in df.columns:
        themes_source = df["themes"]

    if themes_source is not None:
        df["is_diplomacy"] = themes_source.apply(lambda x: 1 if themes_contains_any(x, DIPLO_THEME_NEEDLES) else 0)
        df["is_conflict"] = themes_source.apply(lambda x: 1 if themes_contains_any(x, CONFLICT_THEME_NEEDLES) else 0)
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


def build_signal_intelligence(df_articles: pd.DataFrame) -> dict:
    inventory_cols = ["category", "articles_24h", "articles_7d", "articles_total", "mentions_total"]
    signal_cols = ["signal", "articles_24h", "articles_7d", "articles_total", "mentions_total"]
    recent_cols = ["dt", "title", "domain", "url", "signals"]

    if df_articles is None or df_articles.empty:
        return {
            "inventory": pd.DataFrame(columns=inventory_cols),
            "signals": pd.DataFrame(columns=signal_cols),
            "recent_signal_articles": pd.DataFrame(columns=recent_cols),
        }

    df = df_articles.copy()
    if "seendate" in df.columns:
        df["dt"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
    elif "datetime" in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    else:
        df["dt"] = pd.NaT

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
    df["_text"] = text.fillna("").astype(str)

    now_dt = pd.Timestamp.utcnow()
    is_24h = df["dt"].ge(now_dt - pd.Timedelta(hours=24)).fillna(False)
    is_7d = df["dt"].ge(now_dt - pd.Timedelta(days=7)).fillna(False)

    inv_rows = []
    for cat, kws in MILITARY_INVENTORY_KW.items():
        hits = df["_text"].apply(lambda t: count_keywords_in_text(t, kws)).astype(int)
        has_hit = hits > 0
        inv_rows.append(
            {
                "category": cat,
                "articles_24h": int((has_hit & is_24h).sum()),
                "articles_7d": int((has_hit & is_7d).sum()),
                "articles_total": int(has_hit.sum()),
                "mentions_total": int(hits.sum()),
            }
        )
    inventory_df = pd.DataFrame(inv_rows, columns=inventory_cols).sort_values(
        ["articles_7d", "mentions_total"], ascending=False
    )

    signal_rows = []
    signal_hit_cols = []
    for sig, kws in CONFLICT_SIGNAL_KW.items():
        col = f"_sig_{sig}"
        hits = df["_text"].apply(lambda t: count_keywords_in_text(t, kws)).astype(int)
        df[col] = hits
        signal_hit_cols.append(col)
        has_hit = hits > 0
        signal_rows.append(
            {
                "signal": sig,
                "articles_24h": int((has_hit & is_24h).sum()),
                "articles_7d": int((has_hit & is_7d).sum()),
                "articles_total": int(has_hit.sum()),
                "mentions_total": int(hits.sum()),
            }
        )
    signal_df = pd.DataFrame(signal_rows, columns=signal_cols).sort_values(
        ["articles_7d", "mentions_total"], ascending=False
    )

    def _signals_for_row(row: pd.Series) -> str:
        active = []
        for sig in CONFLICT_SIGNAL_KW.keys():
            if int(row.get(f"_sig_{sig}", 0)) > 0:
                active.append(sig)
        return ", ".join(active)

    if signal_hit_cols:
        has_any_signal = df[signal_hit_cols].sum(axis=1) > 0
    else:
        has_any_signal = pd.Series([False] * len(df), index=df.index)

    recent = df.loc[has_any_signal].copy()
    recent["signals"] = recent.apply(_signals_for_row, axis=1)
    if "domain" not in recent.columns:
        recent["domain"] = ""
    if "url" not in recent.columns:
        recent["url"] = ""
    recent_signal_articles = (
        recent.sort_values("dt", ascending=False)[["dt", "title", "domain", "url", "signals"]]
        .head(20)
        .reset_index(drop=True)
    )

    return {
        "inventory": inventory_df.reset_index(drop=True),
        "signals": signal_df.reset_index(drop=True),
        "recent_signal_articles": recent_signal_articles,
    }
