APP_BUILD = "2026-02-02-02"

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gdelt_structured import (
    fetch_gdelt_articles as fetch_gdelt_articles_structured,
    dedupe_syndication,
    source_diversity_factor,
)

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="USA–Iran Tension Dashboard (GDELT-only)", layout="wide")

DEFAULT_WINDOW_DAYS = 30
DEFAULT_MAXRECORDS = 250


# -----------------------------
# Utilities
# -----------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(lo, min(hi, v))


def logistic_0_100(x: float) -> float:
    # sigmoid -> 0..100
    return float(100.0 * (1.0 / (1.0 + math.exp(-x))))


def logit_from_0_100(score_0_100: float) -> float:
    eps = 1e-6
    p = min(max(float(score_0_100) / 100.0, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def zscore_safe(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mu = float(s.mean(skipna=True)) if len(s) else 0.0
    std = float(s.std(ddof=0, skipna=True)) if len(s) else 0.0
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / std


def risk_label_color(score: float) -> Tuple[str, str]:
    s = clamp(score)
    if s < 20:
        return "LOW", "#22c55e"
    if s < 40:
        return "GUARDED", "#eab308"
    if s < 60:
        return "ELEVATED", "#f97316"
    if s < 80:
        return "HIGH", "#ef4444"
    return "CRITICAL", "#7f1d1d"


def risk_meter(score: float, title: str) -> go.Figure:
    val = 0.0 if score is None or (isinstance(score, float) and math.isnan(score)) else float(score)
    label, color = risk_label_color(val)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": "/100", "font": {"size": 42}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 20], "color": "rgba(34,197,94,0.20)"},
                    {"range": [20, 40], "color": "rgba(234,179,8,0.18)"},
                    {"range": [40, 60], "color": "rgba(249,115,22,0.16)"},
                    {"range": [60, 80], "color": "rgba(239,68,68,0.16)"},
                    {"range": [80, 100], "color": "rgba(127,29,29,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def extract_tone_series(df: pd.DataFrame) -> pd.Series:
    """
    Robust extraction of tone from likely columns.
    Falls back to parsing the first float in a string (some feeds store tone like "-2.3,0.1,...").
    Returns float Series with NaN where unavailable.
    """
    candidates = ["tone", "avgTone", "avgtone", "tone_score", "toneScore", "Tone"]
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if int(s.notna().sum()) > 0:
                return s.astype(float)

    # Parse from string-like tone
    if "tone" in df.columns:
        s = df["tone"].astype(str).str.extract(r"(-?\d+(\.\d+)?)")[0]
        s = pd.to_numeric(s, errors="coerce")
        return s.astype(float)

    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().strip().split())


def count_keywords_in_text(text: str, keywords: List[str]) -> int:
    """
    Simple substring matching. For your use case this is fine and fast.
    """
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
    """
    themes may be list[str] (your code uses themes_list sometimes).
    """
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


# -----------------------------
# Keyword sets (GDELT-only intent model)
# -----------------------------
ESCALATION_KW = [
    "airstrike", "strike", "missile", "rocket", "drone", "ballistic",
    "retaliat", "revenge", "attack", "assault", "intercept", "downed",
    "irgc", "revolutionary guard", "sanction", "embargo",
    "detain", "arrest", "seiz", "confiscat",
    "naval", "warship", "destroyer", "carrier", "submarine",
    "proxy", "militia", "hezbollah", "houthis",
    "explosion", "blast", "killed", "casualt",
    "nuclear", "uranium", "enrichment",
    "strait of hormuz", "persian gulf",
]

DEESCALATION_KW = [
    "talk", "talks", "negotiat", "mediat", "diplomac", "peace", "ceasefire",
    "agreement", "deal", "backchannel", "dialogue", "de-escalat", "deescalat",
    "confidence-building", "summit",
]

CONFLICT_THEME_NEEDLES = [
    "ARMEDCONFLICT", "TERRORISM", "MILITARY", "VIOLENCE", "SECURITYSERVICES",
    "WEAPONS", "AIRSTRIKES", "MISSILES", "DRONE", "NAVY", "SANCTIONS",
]

DIPLO_THEME_NEEDLES = [
    "NEGOTIATIONS", "MEDIATION", "DIPLOMACY", "PEACE", "CEASEFIRE", "TREATY",
]


# -----------------------------
# Data fetch (cached)
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_fetch_structured(query: str, hours_back: int, max_records: int, cache_key: str) -> pd.DataFrame:
    _ = cache_key
    return fetch_gdelt_articles_structured(query=query, hours_back=hours_back, max_records=max_records)


# -----------------------------
# Feature engineering (daily)
# -----------------------------
def build_daily_features(df_articles: pd.DataFrame) -> pd.DataFrame:
    """
    Produces daily features:
      - articles
      - neg_tone_mean
      - diplomacy_share (theme-based)
      - conflict_share (theme-based)
      - escalation_hits_per_article
      - deesc_hits_per_article
      - intent_net (escalation - deesc per article)
    """
    if df_articles is None or df_articles.empty:
        return pd.DataFrame(columns=[
            "date", "articles", "neg_tone_mean",
            "diplomacy_share", "conflict_share",
            "escal_hits", "deesc_hits", "intent_net"
        ])

    df = df_articles.copy()

    # datetime
    if "seendate" in df.columns:
        df["dt"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
    elif "datetime" in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    else:
        df["dt"] = pd.NaT

    df = df.dropna(subset=["dt"]).copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "date", "articles", "neg_tone_mean",
            "diplomacy_share", "conflict_share",
            "escal_hits", "deesc_hits", "intent_net"
        ])

    df["date"] = df["dt"].dt.date

    # tone
    tone = extract_tone_series(df).clip(-10, 10)
    df["tone_num"] = tone

    # negativity magnitude (0 for non-negative tone)
    # If tone missing, NaN stays NaN; handled in aggregation.
    df["neg_mag"] = df["tone_num"].apply(lambda x: float(-x) if pd.notna(x) and x < 0 else (0.0 if pd.notna(x) else np.nan))

    # text for intent keywords (title + optional extras)
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

    # theme-based shares (if themes_list exists; otherwise 0)
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
        )
        .sort_values("date")
    )

    # Fill tone mean if missing (if tone not available, we want it to contribute 0, but NOT silently)
    daily["neg_tone_mean"] = daily["neg_tone_mean"].fillna(0.0).astype(float)

    # per-article rates
    daily["escal_hits"] = daily["escal_hits"].fillna(0).astype(int)
    daily["deesc_hits"] = daily["deesc_hits"].fillna(0).astype(int)
    daily["intent_net"] = ((daily["escal_hits"] - daily["deesc_hits"]) / daily["articles"].clip(lower=1)).astype(float)

    # reindex to full day range so rolling/z-score works consistently
    all_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    daily = daily.set_index("date").reindex(all_days).rename_axis("date").reset_index()

    daily["articles"] = daily["articles"].fillna(0).astype(int)
    daily["neg_tone_mean"] = daily["neg_tone_mean"].fillna(0.0).astype(float)
    daily["diplomacy_share"] = daily["diplomacy_share"].fillna(0.0).astype(float)
    daily["conflict_share"] = daily["conflict_share"].fillna(0.0).astype(float)
    daily["escal_hits"] = daily["escal_hits"].fillna(0).astype(int)
    daily["deesc_hits"] = daily["deesc_hits"].fillna(0).astype(int)
    daily["intent_net"] = daily["intent_net"].fillna(0.0).astype(float)

    return daily


def compute_score_series(
    daily: pd.DataFrame,
    smooth_days: int,
    w_tone: float,
    w_intent: float,
    w_conflict: float,
    w_volume: float,
    w_diplomacy: float,
) -> pd.DataFrame:
    """
    Combine multiple daily signals into one raw score then map to 0..100.

    Signals:
      - tone negativity (smoothed, z-scored)
      - intent_net (smoothed, z-scored)
      - conflict_share (smoothed, z-scored)
      - volume (articles, z-scored)
      - diplomacy_share (smoothed, z-scored) subtractive
    """
    if daily is None or daily.empty:
        return pd.DataFrame(columns=list(daily.columns) + ["raw", "score"])

    d = daily.copy()

    # Smooth
    d["tone_sm"] = d["neg_tone_mean"].rolling(window=smooth_days, min_periods=1).mean()
    d["intent_sm"] = d["intent_net"].rolling(window=smooth_days, min_periods=1).mean()
    d["conflict_sm"] = d["conflict_share"].rolling(window=smooth_days, min_periods=1).mean()
    d["diplomacy_sm"] = d["diplomacy_share"].rolling(window=smooth_days, min_periods=1).mean()

    # Normalize
    d["z_tone"] = zscore_safe(d["tone_sm"])
    d["z_intent"] = zscore_safe(d["intent_sm"])
    d["z_conflict"] = zscore_safe(d["conflict_sm"])
    d["z_volume"] = zscore_safe(d["articles"].astype(float))
    d["z_diplomacy"] = zscore_safe(d["diplomacy_sm"])

    # Raw equation (simple + explainable)
    d["raw"] = (
        (w_tone * d["z_tone"]) +
        (w_intent * d["z_intent"]) +
        (w_conflict * d["z_conflict"]) +
        (w_volume * d["z_volume"]) -
        (w_diplomacy * d["z_diplomacy"])
    ).astype(float)

    d["score"] = d["raw"].apply(lambda x: logistic_0_100(float(x)))
    return d


# -----------------------------
# UI
# -----------------------------
st.title("USA–Iran Tension Dashboard")
st.caption("GDELT-only: multi-signal indicator using tone, intent keywords, conflict themes, diplomacy themes, and volume anomalies.")

tab1, tab2 = st.tabs(["Tension dashboard", "Model & debug"])

with st.sidebar:
    st.header("GDELT")
    default_query = '(United States OR USA OR US OR Pentagon OR CENTCOM) (Iran OR Iranian OR Tehran OR IRGC OR "Strait of Hormuz" OR "Persian Gulf")'
    query = st.text_input("GDELT query", value=default_query)
    window_days = st.slider("Lookback (days)", 7, 180, DEFAULT_WINDOW_DAYS)
    maxrecords = st.slider("Max articles to fetch", 50, 500, DEFAULT_MAXRECORDS, step=25)

    st.header("Scoring")
    smooth_days = st.slider("Smoothing (days)", 1, 14, 3)

    st.subheader("Weights")
    w_tone = st.slider("Tone negativity weight", 0.0, 2.0, 0.9, 0.05)
    w_intent = st.slider("Intent keywords weight", 0.0, 2.0, 1.0, 0.05)
    w_conflict = st.slider("Conflict themes weight", 0.0, 2.0, 0.8, 0.05)
    w_volume = st.slider("Volume spike weight", 0.0, 2.0, 0.5, 0.05)
    w_diplomacy = st.slider("Diplomacy dampening", 0.0, 2.0, 0.6, 0.05)

    st.header("Controls")
    refresh = st.button("Refresh now")

# manual refresh cooldown (prevents accidental hammering of GDELT)
if "last_refresh_ts" not in st.session_state:
    st.session_state["last_refresh_ts"] = 0.0

cooldown_seconds = 20
cache_key = "stable"
if refresh:
    now_ts = time.time()
    if now_ts - st.session_state["last_refresh_ts"] < cooldown_seconds:
        st.warning(f"Please wait {cooldown_seconds} seconds between refreshes.")
    else:
        st.session_state["last_refresh_ts"] = now_ts
        cache_key = utc_now().strftime("%Y%m%d%H%M%S")

hours_back = int(window_days) * 24
end_dt = utc_now()

with st.spinner("Fetching GDELT articles…"):
    articles = cached_fetch_structured(query=query, hours_back=hours_back, max_records=maxrecords, cache_key=cache_key)

articles = dedupe_syndication(articles) if articles is not None else pd.DataFrame()

# source diversity multiplier (kept from your original design)
div_mult = source_diversity_factor(articles) if not articles.empty else 0.9

daily = build_daily_features(articles)
scored = compute_score_series(
    daily=daily,
    smooth_days=smooth_days,
    w_tone=w_tone,
    w_intent=w_intent,
    w_conflict=w_conflict,
    w_volume=w_volume,
    w_diplomacy=w_diplomacy,
)

base_latest_score = float(scored["score"].iloc[-1]) if len(scored) else float("nan")

# Apply diversity multiplier on logit (same concept as your older version)
adjusted_score = base_latest_score
if not math.isnan(adjusted_score):
    raw_logit = logit_from_0_100(adjusted_score)
    raw_logit *= float(div_mult)
    adjusted_score = logistic_0_100(raw_logit)
    adjusted_score = clamp(adjusted_score)

# quick “uncertainty band” based on recent raw volatility
low_band = high_band = None
if len(scored) >= 10 and not math.isnan(adjusted_score):
    recent = scored["score"].tail(14).astype(float)
    band = int(round(max(3.0, float(recent.std(ddof=0)))))  # simple band
    low_band = max(0, int(round(adjusted_score - band)))
    high_band = min(100, int(round(adjusted_score + band)))


# -----------------------------
# Tab 1: Dashboard
# -----------------------------
with tab1:
    st.info("GDELT cached (~1 hour). Score uses: tone negativity + intent keywords + conflict themes + volume spikes − diplomacy themes. Source diversity adjusts confidence.")

    c1, c2, c3, c4, c5 = st.columns([1.25, 1.25, 1, 1, 1])
    c1.plotly_chart(risk_meter(base_latest_score, "Base risk score (latest)"), use_container_width=True)
    c2.plotly_chart(risk_meter(adjusted_score, "Adjusted score (latest)"), use_container_width=True)
    c3.metric("Articles (deduped)", f"{len(articles)}")
    c4.metric("Window", f"{window_days} days")
    c5.metric("Updated (UTC)", end_dt.strftime("%Y-%m-%d %H:%M"))

    b1, b2, b3 = st.columns(3)
    b1.metric("Uncertainty band", "—" if low_band is None else f"{low_band}–{high_band}")
    b2.metric("Source diversity multiplier", f"{div_mult:.2f}")
    lbl, col = risk_label_color(adjusted_score if not math.isnan(adjusted_score) else 0.0)
    b3.metric("Risk label", lbl)

    st.divider()

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Risk score over time")
        if scored.empty:
            st.warning("No usable data. Try a longer window or loosen the query.")
        else:
            fig = px.line(scored, x="date", y="score")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily signals (normalized inputs)")
        if not scored.empty:
            plot_cols = ["z_tone", "z_intent", "z_conflict", "z_volume", "z_diplomacy"]
            melted = scored.melt(id_vars=["date"], value_vars=plot_cols, var_name="signal", value_name="z")
            fig2 = px.line(melted, x="date", y="z", color="signal")
            st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.subheader("Latest-day drivers")
        if scored.empty:
            st.write("—")
        else:
            last = scored.iloc[-1]
            drivers = pd.DataFrame(
                [
                    {"component": "Tone negativity (z)", "value": float(last["tone_sm"]), "z": float(last["z_tone"]), "weight": float(w_tone), "effect": float(w_tone) * float(last["z_tone"])},
                    {"component": "Intent net (z)", "value": float(last["intent_sm"]), "z": float(last["z_intent"]), "weight": float(w_intent), "effect": float(w_intent) * float(last["z_intent"])},
                    {"component": "Conflict themes (z)", "value": float(last["conflict_sm"]), "z": float(last["z_conflict"]), "weight": float(w_conflict), "effect": float(w_conflict) * float(last["z_conflict"])},
                    {"component": "Volume spike (z)", "value": int(last["articles"]), "z": float(last["z_volume"]), "weight": float(w_volume), "effect": float(w_volume) * float(last["z_volume"])},
                    {"component": "Diplomacy (z) [damp]", "value": float(last["diplomacy_sm"]), "z": float(last["z_diplomacy"]), "weight": float(w_diplomacy), "effect": -float(w_diplomacy) * float(last["z_diplomacy"])},
                    {"component": "Raw (sum)", "value": None, "z": None, "weight": None, "effect": float(last["raw"])},
                ]
            )
            st.dataframe(drivers, use_container_width=True, hide_index=True)

        st.subheader("Latest matching articles")
        if articles.empty:
            st.write("—")
        else:
            tmp = articles.copy()
            if "seendate" in tmp.columns:
                tmp["dt"] = pd.to_datetime(tmp["seendate"], errors="coerce", utc=True)
            elif "datetime" in tmp.columns:
                tmp["dt"] = pd.to_datetime(tmp["datetime"], errors="coerce", utc=True)
            else:
                tmp["dt"] = pd.NaT

            latest = tmp.sort_values("dt", ascending=False).head(15)
            for _, row in latest.iterrows():
                dt = row.get("dt")
                dt_str = dt.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(dt) else "—"
                title = row.get("title") or "(no title)"
                url = row.get("url") or ""
                domain = row.get("domain") or ""
                st.markdown(f"- [{title}]({url})  \n  *{dt_str} · {domain}*")

    st.divider()
    with st.expander("Raw daily table"):
        st.dataframe(scored, use_container_width=True)


# -----------------------------
# Tab 2: Model & Debug
# -----------------------------
with tab2:
    st.subheader("What the model uses (GDELT-only)")
    st.markdown(
        """
- **Tone negativity**: average magnitude of negative tone per article (smoothed + z-scored)  
- **Intent keywords**: escalation hits minus de-escalation hits (per article; smoothed + z-scored)  
- **Conflict themes share**: fraction of articles with conflict/military-related themes (smoothed + z-scored)  
- **Volume spike**: articles/day z-score (how unusual coverage volume is)  
- **Diplomacy share**: diplomacy-related themes z-score (subtracted as dampener)  
- **Source diversity**: scales the score’s logit (more diverse sources => stronger confidence)
        """
    )

    st.subheader("Debug: tone availability")
    if articles.empty:
        st.info("No articles loaded.")
    else:
        tone_s = extract_tone_series(articles)
        st.write("Article columns:", list(articles.columns))
        st.write("Tone non-null count:", int(tone_s.notna().sum()))
        st.write("Tone sample:", tone_s.dropna().head(10).tolist())

    st.subheader("Keyword sets (current)")
    st.write("Escalation keywords:", ESCALATION_KW)
    st.write("De-escalation keywords:", DEESCALATION_KW)
    st.write("Conflict theme needles:", CONFLICT_THEME_NEEDLES)
    st.write("Diplomacy theme needles:", DIPLO_THEME_NEEDLES)
