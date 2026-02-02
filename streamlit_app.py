import math
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import sqlite3
from pathlib import Path

from gdelt_structured import (
    fetch_gdelt_articles as fetch_gdelt_articles_structured,
    dedupe_syndication,
    source_diversity_factor,
    structured_signal_score,
)
from market_stress import market_amplifier


st.set_page_config(page_title="USA–Iran Tension Dashboard", layout="wide")

DEFAULT_WINDOW_DAYS = 30
DEFAULT_MAXRECORDS = 250
DEFAULT_SHIPPING_QUERY = (
    '("Red Sea" OR "Strait of Hormuz" OR tanker OR shipping OR container) '
    "AND (disruption OR attack OR reroute OR risk OR insurance)"
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    std = float(s.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - float(s.mean())) / std


def logistic_0_100(x: float) -> float:
    return float(100.0 * (1.0 / (1.0 + math.exp(-x))))


def logit_from_0_100(score_0_100: float) -> float:
    eps = 1e-6
    p = min(max(score_0_100 / 100.0, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def safe_is_json_response(r: requests.Response) -> bool:
    ctype = (r.headers.get("content-type") or "").lower()
    return "application/json" in ctype or "json" in ctype


def shift_detected(series: List[float], recent: int = 7, prior: int = 21, z: float = 1.2) -> bool:
    if series is None or len(series) < (recent + prior):
        return False
    s = np.array(series, dtype=float)
    r = s[-recent:]
    p = s[-(recent + prior) : -recent]
    if p.std() == 0:
        return False
    return abs(r.mean() - p.mean()) / p.std() >= z


def fmt_optional(x):
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except Exception:
        pass
    return x


def risk_meter(score: float, title: str):
    if score is None or (isinstance(score, float) and math.isnan(score)):
        val = 0.0
        suffix = ""
    else:
        val = float(score)
        suffix = "/100"

    def bar_color(v: float) -> str:
        if v < 25:
            return "#22c55e"
        if v < 50:
            return "#eab308"
        if v < 75:
            return "#f97316"
        return "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": suffix, "font": {"size": 42}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color(val)},
                "steps": [
                    {"range": [0, 25], "color": "rgba(34,197,94,0.25)"},
                    {"range": [25, 50], "color": "rgba(234,179,8,0.25)"},
                    {"range": [50, 75], "color": "rgba(249,115,22,0.25)"},
                    {"range": [75, 100], "color": "rgba(239,68,68,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=45, b=10))
    return fig


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_fetch_structured(query: str, hours_back: int, max_records: int, cache_key: str) -> pd.DataFrame:
    _ = cache_key
    return fetch_gdelt_articles_structured(query=query, hours_back=hours_back, max_records=max_records)


def build_daily_structured_features(df_articles: pd.DataFrame) -> pd.DataFrame:
    if df_articles is None or df_articles.empty:
        return pd.DataFrame(columns=["date", "articles", "tension_core", "diplomacy_share"])

    df = df_articles.copy()
    df["datetime"] = pd.to_datetime(df.get("seendate") or df.get("datetime"), errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df["date"] = df["datetime"].dt.date

    df["tone_num"] = pd.to_numeric(df.get("tone"), errors="coerce").clip(-10, 10)

    dip_keywords = {"NEGOTIATIONS", "MEDIATION", "DIPLOMACY", "PEACE", "CEASEFIRE", "TREATY"}

    def is_diplomatic(themes):
        if not isinstance(themes, list):
            return False
        return any(k in t.upper() for t in themes for k in dip_keywords)

    df["is_diplomatic"] = df["themes_list"].apply(is_diplomatic) if "themes_list" in df.columns else 0

    def tension_core(g):
        t = g["tone_num"].dropna()
        return float((-t[t < 0]).sum() / max(len(g), 1)) if not t.empty else 0.0

    daily = df.groupby("date", as_index=False).agg(
        articles=("url", "count"),
        diplomacy_share=("is_diplomatic", "mean"),
    )

    cores = df.groupby("date").apply(tension_core).reset_index(name="tension_core")
    daily = daily.merge(cores, on="date", how="left").fillna(0)

    return daily


def compute_structured_score_series(daily: pd.DataFrame, smooth_days: int, diplomacy_weight: float) -> pd.DataFrame:
    if daily.empty:
        return daily

    d = daily.copy()
    d["tension_sm"] = d["tension_core"].rolling(smooth_days, min_periods=1).mean()
    d["z_tension"] = zscore(d["tension_sm"])
    raw = d["z_tension"] - diplomacy_weight * d["diplomacy_share"]
    d["score"] = 100 / (1 + np.exp(-raw))
    return d


DB_PATH = Path(__file__).with_name("flights.db")


def db_conn():
    conn = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_flights_db():
    with db_conn() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS flight_samples (ts_utc TEXT PRIMARY KEY, airborne INTEGER NOT NULL)"
        )


def insert_flight_sample(ts: datetime, airborne: int):
    with db_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO flight_samples VALUES (?, ?)",
            (ts.replace(microsecond=0).isoformat(), int(airborne)),
        )


def read_flight_samples(hours_back=24):
    cutoff = (utc_now() - timedelta(hours=hours_back)).isoformat()
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT ts_utc, airborne FROM flight_samples WHERE ts_utc >= ? ORDER BY ts_utc",
            (cutoff,),
        ).fetchall()
    df = pd.DataFrame(rows, columns=["ts", "airborne"])
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


init_flights_db()


OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"


def get_secret(key: str) -> str:
    return st.secrets.get(key) or __import__("os").environ.get(key)


@st.cache_data(ttl=10 * 60)
def fetch_opensky_states_bbox(cache_key, lamin, lamax, lomin, lomax):
    _ = cache_key
    token = get_secret("OPENSKY_CLIENT_SECRET")
    if not token:
        return 0, "Missing OpenSky credentials"

    r = requests.get(
        OPENSKY_STATES_URL,
        params=dict(lamin=lamin, lamax=lamax, lomin=lomin, lomax=lomax),
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    states = r.json().get("states") or []
    airborne = sum(1 for s in states if isinstance(s, list) and len(s) > 8 and s[8] is False)
    return airborne, "OK"


st.title("USA–Iran Tension Dashboard")

tab1, tab2 = st.tabs(["Tension dashboard", "Air traffic signal"])

with st.sidebar:
    query = st.text_input("GDELT query", "(United States OR USA OR US) (Iran OR Iranian)")
    window_days = st.slider("Lookback (days)", 7, 180, DEFAULT_WINDOW_DAYS)
    maxrecords = st.slider("Max articles", 50, 250, DEFAULT_MAXRECORDS)
    smooth_days = st.slider("Smoothing", 1, 14, 3)
    diplomacy_weight = st.slider("Diplomacy dampening", 0.0, 2.0, 0.5)
    enable_market_amp = st.checkbox("Market stress amplifier", True)
    enable_air_signal = st.checkbox("Air traffic signal", True)
    refresh = st.button("Refresh")

end_dt = utc_now()
articles = cached_fetch_structured(query, window_days * 24, maxrecords, "k")
articles = dedupe_syndication(articles)
daily = build_daily_structured_features(articles)
scored = compute_structured_score_series(daily, smooth_days, diplomacy_weight)

base_score = float(scored["score"].iloc[-1]) if not scored.empty else float("nan")
adjusted_score = base_score * (market_amplifier() if enable_market_amp else 1.0)

with tab1:
    c1, c2 = st.columns(2)
    c1.plotly_chart(risk_meter(base_score, "Base risk score"), use_container_width=True)
    c2.plotly_chart(risk_meter(adjusted_score, "Adjusted risk score"), use_container_width=True)
