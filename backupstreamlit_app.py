import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlite3
import streamlit as st
import streamlit.components.v1 as components

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

# GDELT Events API (2.1)
GDELT_EVENTS_URL = "https://api.gdeltproject.org/api/v2/events/search"

# OpenSky
OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"

# Iran bbox
IR_LAMIN, IR_LAMAX = 25.29, 39.65
IR_LOMIN, IR_LOMAX = 44.77, 61.49


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


def classify_risk(score: float) -> Tuple[str, str]:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "NO DATA", "#6b7280"
    s = float(score)
    if s < 25:
        return "LOW RISK", "#22c55e"
    if s < 50:
        return "GUARDED", "#eab308"
    if s < 75:
        return "ELEVATED", "#f97316"
    return "HIGH RISK", "#ef4444"


def make_compact_gauge(score: float, color: str) -> go.Figure:
    val = 0.0 if score is None or (isinstance(score, float) and math.isnan(score)) else float(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=val,
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "visible": False},
                "bar": {"color": color, "thickness": 0.30},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [{"range": [0, 100], "color": "rgba(255,255,255,0.08)"}],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def risk_meter(score: float, title: str) -> go.Figure:
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


def radar_card(score: float, title: str, subtitle: str, updated_dt: datetime, next_update_seconds: int):
    label, color = classify_risk(score)
    mins_ago = int((utc_now() - updated_dt).total_seconds() // 60)
    mm = max(0, int(next_update_seconds // 60))
    ss = max(0, int(next_update_seconds % 60))
    countdown = f"{mm:02d}:{ss:02d}"
    has_data = not (score is None or (isinstance(score, float) and math.isnan(score)))
    pct = int(round(float(score))) if has_data else None

    pct_text = f"{pct}%" if pct is not None else "—"

    st.markdown(
        """
        <style>
          .sr-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 16px 16px 14px 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
          }
          .sr-pill {
            display:inline-block;
            padding:6px 14px;
            border-radius:999px;
            font-weight:800;
            font-size:12px;
            color:#0b0b0b;
          }
          .sr-title {
            font-size:13px;
            letter-spacing:0.10em;
            opacity:0.85;
            text-align:center;
          }
          .sr-sub {
            font-size:12px;
            opacity:0.65;
            margin-top:4px;
            text-align:center;
          }
          .sr-top {
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:12px;
            opacity:0.85;
            font-size:13px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sr-card">', unsafe_allow_html=True)

    top_l, top_r = st.columns([1, 1])
    with top_l:
        st.markdown(f'<div class="sr-top">Updated <b>{mins_ago} min ago</b> · UTC</div>', unsafe_allow_html=True)
    with top_r:
        st.markdown(
            f'<div class="sr-top" style="justify-content:flex-end;">Next update in <b>{countdown}</b></div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="sr-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sr-sub">{subtitle}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="text-align:center; margin-top:10px;"><span class="sr-pill" style="background:{color};">{label}</span></div>',
        unsafe_allow_html=True,
    )

    fig = make_compact_gauge(score, color)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:-120px;">
          <div style="font-size:56px; font-weight:900; color:{color}; line-height:1;">{pct}%</div>
          <div style="font-size:12px; opacity:0.65; margin-top:6px;">Composite alert score (0–100)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def live_tick(enable: bool):
    if not enable:
        return
    components.html(
        """
<script>
  setTimeout(() => {
    window.parent.postMessage({type: "streamlit:rerun"}, "*");
  }, 1000);
</script>
        """,
        height=0,
        width=0,
    )


# -----------------------------
# SQLite (flights + daily score history + labels)
# -----------------------------
DB_PATH = Path(__file__).with_name("risk_data.db")


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    conn = db_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS flight_samples (
                ts_utc TEXT PRIMARY KEY,
                airborne INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flight_samples_ts ON flight_samples(ts_utc)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_scores (
                date_utc TEXT PRIMARY KEY,
                tone_score REAL,
                events_score REAL,
                combined_score REAL,
                adjusted_score REAL,
                articles INTEGER,
                events INTEGER,
                label INTEGER
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_scores_date ON daily_scores(date_utc)")
        conn.commit()
    finally:
        conn.close()


def insert_flight_sample(ts: datetime, airborne: int) -> None:
    ts_utc = ts.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    conn = db_conn()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO flight_samples(ts_utc, airborne) VALUES (?, ?)",
            (ts_utc, int(airborne)),
        )
        conn.commit()
    finally:
        conn.close()


def read_flight_samples(hours_back: int = 24, limit: int = 5000) -> pd.DataFrame:
    cutoff = (utc_now() - timedelta(hours=hours_back)).replace(microsecond=0).isoformat()
    conn = db_conn()
    try:
        rows = conn.execute(
            """
            SELECT ts_utc, airborne
            FROM flight_samples
            WHERE ts_utc >= ?
            ORDER BY ts_utc ASC
            LIMIT ?
            """,
            (cutoff, int(limit)),
        ).fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=["ts", "airborne"])
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    df["airborne"] = pd.to_numeric(df["airborne"], errors="coerce").fillna(0).astype(int)
    return df


def prune_flights_db(keep_days: int = 120) -> int:
    cutoff = (utc_now() - timedelta(days=keep_days)).replace(microsecond=0).isoformat()
    conn = db_conn()
    try:
        cur = conn.execute("DELETE FROM flight_samples WHERE ts_utc < ?", (cutoff,))
        conn.commit()
        return int(cur.rowcount)
    finally:
        conn.close()


def upsert_daily_score(
    date_utc: str,
    tone_score: Optional[float],
    events_score: Optional[float],
    combined_score: Optional[float],
    adjusted_score: Optional[float],
    articles: int,
    events: int,
) -> None:
    conn = db_conn()
    try:
        conn.execute(
            """
            INSERT INTO daily_scores(date_utc, tone_score, events_score, combined_score, adjusted_score, articles, events)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(date_utc) DO UPDATE SET
              tone_score=excluded.tone_score,
              events_score=excluded.events_score,
              combined_score=excluded.combined_score,
              adjusted_score=excluded.adjusted_score,
              articles=excluded.articles,
              events=excluded.events
            """,
            (date_utc, tone_score, events_score, combined_score, adjusted_score, int(articles), int(events)),
        )
        conn.commit()
    finally:
        conn.close()


def set_label(date_utc: str, label: Optional[int]) -> None:
    conn = db_conn()
    try:
        if label is None:
            conn.execute("UPDATE daily_scores SET label=NULL WHERE date_utc=?", (date_utc,))
        else:
            conn.execute("UPDATE daily_scores SET label=? WHERE date_utc=?", (int(label), date_utc))
        conn.commit()
    finally:
        conn.close()


def read_daily_scores(days_back: int = 365) -> pd.DataFrame:
    cutoff = (utc_now() - timedelta(days=days_back)).date().isoformat()
    conn = db_conn()
    try:
        rows = conn.execute(
            """
            SELECT date_utc, tone_score, events_score, combined_score, adjusted_score, articles, events, label
            FROM daily_scores
            WHERE date_utc >= ?
            ORDER BY date_utc ASC
            """,
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(
        rows,
        columns=["date", "tone_score", "events_score", "combined_score", "adjusted_score", "articles", "events", "label"],
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


# -----------------------------
# GDELT structured articles (DOC 2.1) -> daily tone signal
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_fetch_structured(query: str, hours_back: int, max_records: int, cache_key: str) -> pd.DataFrame:
    _ = cache_key
    return fetch_gdelt_articles_structured(query=query, hours_back=hours_back, max_records=max_records)


def build_daily_tone_features(df_articles: pd.DataFrame) -> pd.DataFrame:
    if df_articles is None or df_articles.empty:
        return pd.DataFrame(columns=["date", "articles", "tension_core", "diplomacy_share"])

    df = df_articles.copy()
    if "seendate" in df.columns:
        df["datetime"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    else:
        df["datetime"] = pd.NaT

    df = df.dropna(subset=["datetime"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "articles", "tension_core", "diplomacy_share"])

    df["date"] = df["datetime"].dt.date
    df["tone_num"] = pd.to_numeric(df.get("tone", pd.Series([np.nan] * len(df))), errors="coerce").clip(-10, 10)

    dip_keywords = {"NEGOTIATIONS", "MEDIATION", "DIPLOMACY", "PEACE", "CEASEFIRE", "TREATY"}

    def is_diplomatic(themes):
        if not isinstance(themes, list):
            return False
        for t in themes:
            if not isinstance(t, str):
                continue
            u = t.upper()
            for k in dip_keywords:
                if k in u:
                    return True
        return False

    if "themes_list" in df.columns:
        df["is_diplomatic"] = df["themes_list"].apply(is_diplomatic).astype(int)
    else:
        df["is_diplomatic"] = 0

    def tension_core_for_group(g: pd.DataFrame) -> float:
        t = g["tone_num"].dropna()
        if t.empty:
            return 0.0
        return float((-t[t < 0]).sum() / max(len(g), 1))

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            articles=("url", "count"),
            diplomacy_share=("is_diplomatic", "mean"),
        )
        .sort_values("date")
    )

    cores = df.groupby("date").apply(tension_core_for_group).reset_index(name="tension_core")
    daily = daily.merge(cores, on="date", how="left")

    all_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    daily = daily.set_index("date").reindex(all_days).rename_axis("date").reset_index()

    daily["articles"] = daily["articles"].fillna(0).astype(int)
    daily["diplomacy_share"] = daily["diplomacy_share"].fillna(0.0).astype(float)
    daily["tension_core"] = daily["tension_core"].fillna(0.0).astype(float)
    return daily


def compute_tone_score_series(daily: pd.DataFrame, smooth_days: int, diplomacy_weight: float) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame(columns=list(daily.columns) + ["tone_score", "z_tone"])

    d = daily.copy()
    d["tension_sm"] = d["tension_core"].rolling(window=smooth_days, min_periods=1).mean()
    d["z_tone"] = zscore(d["tension_sm"])
    raw = d["z_tone"] - (diplomacy_weight * d["diplomacy_share"])
    d["tone_score"] = (100.0 * (1.0 / (1.0 + np.exp(-raw)))).astype(float)
    return d


# -----------------------------
# GDELT events (2.1) -> daily structured event signal
# -----------------------------
def _fmt_gdelt_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_fetch_events(query: str, hours_back: int, max_records: int, cache_key: str) -> pd.DataFrame:
    _ = cache_key
    end = utc_now()
    start = end - timedelta(hours=hours_back)
    params = {
        "query": query,
        "format": "json",
        "startdatetime": _fmt_gdelt_dt(start),
        "enddatetime": _fmt_gdelt_dt(end),
        "maxrecords": int(max_records),
    }
    try:
        r = requests.get(GDELT_EVENTS_URL, params=params, timeout=30)
    except requests.RequestException:
        return pd.DataFrame()

    if r.status_code != 200 or not safe_is_json_response(r):
        return pd.DataFrame()

    js = r.json() or {}
    events = js.get("events") or []
    if not isinstance(events, list) or not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    return df


def build_daily_events_features(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame(columns=["date", "events", "conflict_intensity", "coop_intensity"])

    df = df_events.copy()

    # GDELT Events commonly provides SQLDATE (YYYYMMDD) and GoldsteinScale
    if "SQLDATE" in df.columns:
        df["date"] = pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce").dt.date
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.date
    else:
        df["date"] = pd.NaT

    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "events", "conflict_intensity", "coop_intensity"])

    gs = pd.to_numeric(df.get("GoldsteinScale", pd.Series([np.nan] * len(df))), errors="coerce").fillna(0.0)
    df["gs"] = gs.clip(-10, 10)

    df["conflict_mass"] = (-df["gs"].clip(upper=0)).astype(float)  # negative becomes positive mass
    df["coop_mass"] = (df["gs"].clip(lower=0)).astype(float)

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            events=("gs", "size"),
            conflict_intensity=("conflict_mass", "mean"),
            coop_intensity=("coop_mass", "mean"),
        )
        .sort_values("date")
    )

    all_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    daily = daily.set_index("date").reindex(all_days).rename_axis("date").reset_index()
    daily["events"] = daily["events"].fillna(0).astype(int)
    daily["conflict_intensity"] = daily["conflict_intensity"].fillna(0.0).astype(float)
    daily["coop_intensity"] = daily["coop_intensity"].fillna(0.0).astype(float)
    return daily


def compute_events_score_series(daily_events: pd.DataFrame, smooth_days: int, coop_weight: float) -> pd.DataFrame:
    if daily_events is None or daily_events.empty:
        return pd.DataFrame(columns=list(daily_events.columns) + ["events_score", "z_events"])

    d = daily_events.copy()
    d["conflict_sm"] = d["conflict_intensity"].rolling(window=smooth_days, min_periods=1).mean()
    d["z_events"] = zscore(d["conflict_sm"])
    raw = d["z_events"] - (coop_weight * d["coop_intensity"])
    d["events_score"] = (100.0 * (1.0 / (1.0 + np.exp(-raw)))).astype(float)
    return d


# -----------------------------
# Flights anomaly (SQLite baseline)
# -----------------------------
def compute_flight_z_baselined(
    now_ts: datetime,
    baseline_days: int = 28,
    min_baseline_points: int = 30,
    lookback_hours_for_plot: int = 24,
) -> Tuple[Optional[float], pd.DataFrame]:
    hours_back = baseline_days * 24
    df = read_flight_samples(hours_back=hours_back)

    if df.empty or len(df) < 10:
        return None, read_flight_samples(hours_back=lookback_hours_for_plot)

    latest = df.iloc[-1]
    latest_ts = latest["ts"]
    latest_val = float(latest["airborne"])

    dow = int(latest_ts.dayofweek)
    hour = int(latest_ts.hour)

    df["dow"] = df["ts"].dt.dayofweek
    df["hour"] = df["ts"].dt.hour

    hist = df.iloc[:-1]
    base = hist[(hist["dow"] == dow) & (hist["hour"] == hour)].copy()

    if len(base) < min_baseline_points:
        df24 = read_flight_samples(hours_back=lookback_hours_for_plot)
        if len(df24) < 6:
            return None, df24
        z = float(zscore(df24["airborne"]).iloc[-1])
        return z, df24

    mu = float(base["airborne"].mean())
    sigma = float(base["airborne"].std(ddof=0))
    if sigma == 0.0 or np.isnan(sigma):
        return None, read_flight_samples(hours_back=lookback_hours_for_plot)

    z = (latest_val - mu) / sigma
    df24 = df[df["ts"] >= (now_ts - timedelta(hours=lookback_hours_for_plot))][["ts", "airborne"]].copy()
    return float(z), df24


# -----------------------------
# OpenSky OAuth and snapshot
# -----------------------------
def get_secret(key: str) -> str:
    if key in st.secrets:
        return str(st.secrets[key])
    import os

    v = os.environ.get(key)
    if v:
        return v
    raise RuntimeError(f"Missing secret: {key}")


@st.cache_data(ttl=50 * 60, show_spinner=False)
def fetch_opensky_token(cache_key: str) -> Tuple[Optional[str], str]:
    _ = cache_key
    try:
        client_id = get_secret("OPENSKY_CLIENT_ID")
        client_secret = get_secret("OPENSKY_CLIENT_SECRET")
    except Exception as e:
        return None, f"OpenSky credentials missing: {e}"

    try:
        r = requests.post(
            OPENSKY_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=30,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    except requests.RequestException as e:
        return None, f"OpenSky token request error: {e}"

    if r.status_code != 200:
        snippet = (r.text or "")[:220].replace("\n", " ").strip()
        return None, f"OpenSky token HTTP {r.status_code}. Snippet: {snippet}"

    if not safe_is_json_response(r):
        return None, "OpenSky token endpoint returned non-JSON."

    js = r.json()
    token = js.get("access_token")
    if not token:
        return None, "OpenSky token response missing access_token."
    return token, "OK"


@st.cache_data(ttl=10 * 60, show_spinner=False)
def fetch_opensky_states_bbox(
    cache_key: str,
    lamin: float,
    lamax: float,
    lomin: float,
    lomax: float,
) -> Tuple[int, str]:
    token, msg = fetch_opensky_token("token-" + cache_key)
    if not token:
        return 0, msg

    params = {"lamin": lamin, "lamax": lamax, "lomin": lomin, "lomax": lomax}
    try:
        r = requests.get(
            OPENSKY_STATES_URL,
            params=params,
            timeout=30,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "tension-dashboard/1.0"},
        )
    except requests.RequestException as e:
        return 0, f"OpenSky request error: {e}"

    if r.status_code != 200:
        snippet = (r.text or "")[:220].replace("\n", " ").strip()
        return 0, f"OpenSky HTTP {r.status_code}. Snippet: {snippet}"

    if not safe_is_json_response(r):
        return 0, "OpenSky returned non-JSON response."

    js = r.json()
    states = js.get("states") or []
    if not states:
        return 0, f"{msg} OpenSky returned 0 states in bbox."

    airborne = 0
    for s in states:
        if isinstance(s, list) and len(s) > 8:
            if s[8] is False:
                airborne += 1

    return airborne, msg


# -----------------------------
# Simple calibration (optional, uses labeled days)
# -----------------------------
def fit_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.05, steps: int = 800) -> Tuple[np.ndarray, float]:
    # X: (n, d), y: (n,)
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    for _ in range(steps):
        z = X @ w + b
        p = sigmoid(z)
        grad_w = (X.T @ (p - y)) / n
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-z))


# -----------------------------
# App UI + Update loop
# -----------------------------
init_db()

st.title("USA–Iran Tension Dashboard")
st.caption("Structured news indicator (GDELT DOC + GDELT Events) + aggregate air-traffic anomaly over Iran (OpenSky).")

tab1, tab2, tab3 = st.tabs(["Tension dashboard", "Air traffic signal", "Calibration"])

with st.sidebar:
    st.header("GDELT")
    default_query = "(United States OR USA OR US) (Iran OR Iranian)"
    query = st.text_input("GDELT query", value=default_query)
    window_days = st.slider("Lookback (days)", 7, 180, DEFAULT_WINDOW_DAYS)
    maxrecords = st.slider("Max records (articles/events)", 50, 500, DEFAULT_MAXRECORDS, step=25)

    st.header("Scoring")
    smooth_days = st.slider("Smoothing (days)", 1, 14, 3)
    diplomacy_weight = st.slider("Diplomacy dampening (tone)", 0.0, 2.0, 0.5, 0.1)
    coop_weight = st.slider("Cooperation dampening (events)", 0.0, 2.0, 0.4, 0.1)

    st.header("Blend")
    w_events = st.slider("Events weight", 0.0, 2.0, 1.0, 0.1)
    w_tone = st.slider("Tone weight", 0.0, 2.0, 1.0, 0.1)

    st.header("Amplifiers (latest only)")
    enable_market_amp = st.checkbox("Enable market stress amplifier", value=True)
    enable_shipping_amp = st.checkbox("Enable shipping disruption volume", value=False)
    shipping_query = st.text_input("Shipping query (GDELT DOC)", value=DEFAULT_SHIPPING_QUERY)

    st.header("Air traffic (latest only)")
    enable_air_signal = st.checkbox("Enable Iran air-traffic signal", value=True)
    w_air_traffic = st.slider("Air traffic impact (↑ when traffic drops)", 0.0, 3.0, 1.0, 0.1)

    st.header("Updates")
    auto_refresh = st.checkbox("Live countdown", value=True)
    update_interval = st.slider("Update interval (minutes)", 1, 15, 5)
    refresh = st.button("Refresh now")

    st.subheader("DB maintenance")
    keep_days = st.number_input("Keep flight samples (days)", min_value=14, max_value=365, value=120, step=7)
    if st.button("Prune flight DB"):
        deleted = prune_flights_db(keep_days=int(keep_days))
        st.success(f"Deleted {deleted} old flight samples (kept last {int(keep_days)} days).")

# session state for throttling
if "last_refresh_ts" not in st.session_state:
    st.session_state["last_refresh_ts"] = 0.0
if "last_data_update_dt" not in st.session_state:
    st.session_state["last_data_update_dt"] = utc_now()
if "snapshot" not in st.session_state:
    st.session_state["snapshot"] = None

cooldown_seconds = 30
update_seconds = int(update_interval) * 60

now_dt = utc_now()
elapsed_since_update = int((now_dt - st.session_state["last_data_update_dt"]).total_seconds())
remaining_to_update = max(0, update_seconds - elapsed_since_update)

force_update = False
cache_key = "stable"

if refresh:
    now_ts = time.time()
    if now_ts - st.session_state["last_refresh_ts"] < cooldown_seconds:
        st.warning(f"Please wait {cooldown_seconds} seconds between refreshes.")
    else:
        st.session_state["last_refresh_ts"] = now_ts
        force_update = True

due_update = remaining_to_update <= 0
if st.session_state["snapshot"] is None:
    force_update = True

# main update (fetches)
if force_update or due_update:
    cache_key = utc_now().strftime("%Y%m%d%H%M%S")
    end_dt = utc_now()
    hours_back = int(window_days) * 24

    with st.spinner("Fetching GDELT DOC (articles)…"):
        articles = cached_fetch_structured(query=query, hours_back=hours_back, max_records=maxrecords, cache_key=cache_key)
    articles = dedupe_syndication(articles) if articles is not None else pd.DataFrame()

    with st.spinner("Fetching GDELT Events…"):
        events_df = cached_fetch_events(query=query, hours_back=hours_back, max_records=maxrecords, cache_key="ev-" + cache_key)

    # Tone series
    div_mult = source_diversity_factor(articles) if not articles.empty else 0.8
    sig = (
        structured_signal_score(articles)
        if not articles.empty
        else {"tension_core": 0.0, "diplomacy_share": 0.0, "uncertainty": 1.0}
    )

    daily_tone = build_daily_tone_features(articles)
    tone_scored = compute_tone_score_series(daily_tone, smooth_days=smooth_days, diplomacy_weight=diplomacy_weight)

    # Events series
    daily_events = build_daily_events_features(events_df)
    events_scored = compute_events_score_series(daily_events, smooth_days=smooth_days, coop_weight=coop_weight)

    # Merge daily time series
    d = pd.DataFrame({"date": pd.date_range(end_dt.date() - timedelta(days=window_days - 1), end_dt.date(), freq="D").date})
    d = d.merge(tone_scored[["date", "tone_score"]], on="date", how="left")
    d = d.merge(events_scored[["date", "events_score"]], on="date", how="left")
    d["tone_score"] = d["tone_score"].fillna(method="ffill").fillna(50.0)
    d["events_score"] = d["events_score"].fillna(method="ffill").fillna(50.0)

    # Combine in logit space so weights behave smoothly
    def safe_logit(s):
        try:
            return logit_from_0_100(float(s))
        except Exception:
            return 0.0

    d["tone_logit"] = d["tone_score"].apply(safe_logit)
    d["events_logit"] = d["events_score"].apply(safe_logit)

    blend = (w_tone * d["tone_logit"]) + (w_events * d["events_logit"])
    # diversity multiplier nudges news component only (not events)
    blend = blend + (float(div_mult) - 1.0) * (0.6 * d["tone_logit"])

    d["combined_score"] = blend.apply(logistic_0_100)

    base_latest_score = float(d["combined_score"].iloc[-1]) if len(d) else float("nan")

    # Uncertainty band from DOC signal uncertainty (kept simple)
    band = int(round(8 * float(sig.get("uncertainty", 1.0))))
    low_band = None
    high_band = None
    if not math.isnan(base_latest_score):
        low_band = max(0, int(round(base_latest_score - band)))
        high_band = min(100, int(round(base_latest_score + band)))

    # Regime shift on combined series
    shift_flag = shift_detected(d["combined_score"].astype(float).tolist()) if not d.empty else False

    # Shipping amplifier (DOC volume)
    shipping_mult = 1.0
    shipping_articles_count = 0
    if enable_shipping_amp:
        with st.spinner("Fetching shipping disruption volume…"):
            ship_df = cached_fetch_structured(
                query=shipping_query, hours_back=72, max_records=250, cache_key="ship-" + cache_key
            )
        ship_df = dedupe_syndication(ship_df) if ship_df is not None else pd.DataFrame()
        shipping_articles_count = int(len(ship_df))
        if shipping_articles_count >= 40:
            shipping_mult = 1.05
        if shipping_articles_count >= 80:
            shipping_mult = 1.10

    # Market amplifier (helper)
    mkt_mult = float(market_amplifier()) if enable_market_amp else 1.0

    # Air signal (latest-only)
    airborne_over_iran = None
    air_msg = "Air traffic signal disabled."
    flight_z = None
    df24 = pd.DataFrame(columns=["ts", "airborne"])

    adjusted_latest_score = base_latest_score

    if enable_air_signal:
        with st.spinner("Fetching OpenSky air-traffic snapshot over Iran…"):
            bucket = end_dt.replace(minute=(end_dt.minute // 10) * 10, second=0, microsecond=0)
            os_cache_key = bucket.strftime("%Y%m%d%H%M")
            count, air_msg = fetch_opensky_states_bbox(
                cache_key=os_cache_key,
                lamin=IR_LAMIN,
                lamax=IR_LAMAX,
                lomin=IR_LOMIN,
                lomax=IR_LOMAX,
            )
            airborne_over_iran = int(count)
            insert_flight_sample(end_dt, airborne_over_iran)
            flight_z, df24 = compute_flight_z_baselined(
                now_ts=end_dt,
                baseline_days=28,
                min_baseline_points=30,
                lookback_hours_for_plot=24,
            )

        if flight_z is not None and not math.isnan(adjusted_latest_score):
            raw = logit_from_0_100(adjusted_latest_score)
            raw_adj = raw + (-w_air_traffic * float(flight_z))
            adjusted_latest_score = logistic_0_100(raw_adj)

    # Apply amplifiers
    if not math.isnan(adjusted_latest_score):
        adjusted_latest_score = float(adjusted_latest_score) * float(mkt_mult) * float(shipping_mult)
        adjusted_latest_score = float(max(0.0, min(100.0, adjusted_latest_score)))

    # Persist daily score history (latest day)
    today = end_dt.date().isoformat()
    upsert_daily_score(
        date_utc=today,
        tone_score=float(tone_scored["tone_score"].iloc[-1]) if len(tone_scored) else None,
        events_score=float(events_scored["events_score"].iloc[-1]) if len(events_scored) else None,
        combined_score=float(base_latest_score) if not math.isnan(base_latest_score) else None,
        adjusted_score=float(adjusted_latest_score) if not math.isnan(adjusted_latest_score) else None,
        articles=int(len(articles)),
        events=int(len(events_df)) if events_df is not None else 0,
    )

    st.session_state["snapshot"] = dict(
        end_dt=end_dt,
        articles=articles,
        events_df=events_df,
        daily_combined=d,
        tone_scored=tone_scored,
        events_scored=events_scored,
        base_latest_score=base_latest_score,
        adjusted_latest_score=adjusted_latest_score,
        div_mult=div_mult,
        sig=sig,
        low_band=low_band,
        high_band=high_band,
        shift_flag=shift_flag,
        shipping_mult=shipping_mult,
        shipping_articles_count=shipping_articles_count,
        mkt_mult=mkt_mult,
        airborne_over_iran=airborne_over_iran,
        air_msg=air_msg,
        flight_z=flight_z,
        df24=df24,
        diplomacy_weight=diplomacy_weight,
        coop_weight=coop_weight,
        w_air_traffic=w_air_traffic,
        w_events=w_events,
        w_tone=w_tone,
        window_days=window_days,
        enable_market_amp=enable_market_amp,
        enable_shipping_amp=enable_shipping_amp,
    )
    st.session_state["last_data_update_dt"] = end_dt

# snapshot view (cheap reruns)
snap = st.session_state["snapshot"]
end_dt = snap["end_dt"]
articles = snap["articles"]
events_df = snap["events_df"]
daily_combined = snap["daily_combined"]
tone_scored = snap["tone_scored"]
events_scored = snap["events_scored"]
base_latest_score = snap["base_latest_score"]
adjusted_latest_score = snap["adjusted_latest_score"]
div_mult = snap["div_mult"]
low_band = snap["low_band"]
high_band = snap["high_band"]
shift_flag = snap["shift_flag"]
shipping_mult = snap["shipping_mult"]
shipping_articles_count = snap["shipping_articles_count"]
mkt_mult = snap["mkt_mult"]
airborne_over_iran = snap["airborne_over_iran"]
air_msg = snap["air_msg"]
flight_z = snap["flight_z"]
df24 = snap["df24"]
w_air_traffic = snap["w_air_traffic"]
window_days = snap["window_days"]
enable_market_amp = snap["enable_market_amp"]
enable_shipping_amp = snap["enable_shipping_amp"]

now_dt = utc_now()
elapsed_since_update = int((now_dt - st.session_state["last_data_update_dt"]).total_seconds())
remaining_to_update = max(0, update_seconds - elapsed_since_update)

live_tick(auto_refresh)

# -----------------------------
# TAB 1: Dashboard
# -----------------------------
with tab1:
    radar_card(
        score=adjusted_latest_score,
        title="US–IRAN TENSION INDICATOR",
        subtitle="Composite alert score (GDELT DOC + GDELT Events + optional amplifiers)",
        updated_dt=st.session_state["last_data_update_dt"],
        next_update_seconds=remaining_to_update,
    )

    st.info(
        "GDELT is cached (1 hour). OpenSky snapshot is cached (~10 minutes). "
        "The live countdown reruns the UI every second but does not refetch unless due or manually refreshed."
    )

    if shift_flag:
        st.warning("Regime shift detected: the last week differs materially from the prior baseline.")

    c1, c2, c3, c4, c5 = st.columns([1.25, 1.25, 1, 1, 1])
    c1.plotly_chart(risk_meter(base_latest_score, title="Base score (latest)"), use_container_width=True)
    c2.plotly_chart(risk_meter(adjusted_latest_score, title="Adjusted score (latest)"), use_container_width=True)
    c3.metric("Articles (deduped)", f"{len(articles)}")
    c4.metric("Events (GDELT)", f"{0 if events_df is None else len(events_df)}")
    c5.metric("Updated (UTC)", end_dt.strftime("%Y-%m-%d %H:%M"))

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Uncertainty band", "—" if low_band is None else f"{low_band}–{high_band}")
    b2.metric("Source diversity multiplier", f"{div_mult:.2f}")
    b3.metric("Market amplifier", f"{mkt_mult:.2f}" if enable_market_amp else "off")
    b4.metric("Shipping amp", f"{shipping_mult:.2f} ({shipping_articles_count} arts)" if enable_shipping_amp else "off")

    st.caption(f"OpenSky: {air_msg}")

    a1, a2, a3 = st.columns(3)
    a1.metric("Airborne over Iran (snapshot)", "—" if airborne_over_iran is None else str(airborne_over_iran))
    a2.metric("Air-traffic z-score (baselined)", "—" if flight_z is None else f"{flight_z:+.2f}")
    a3.metric("Air traffic weight", f"{w_air_traffic:.1f}")

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Score over time")
        if daily_combined is None or daily_combined.empty:
            st.info("No usable data in this window.")
        else:
            fig = px.line(daily_combined, x="date", y="combined_score")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tone vs Events (daily)")
        if daily_combined is not None and not daily_combined.empty:
            tmp = daily_combined[["date", "tone_score", "events_score"]].melt(
                id_vars=["date"], var_name="signal", value_name="score"
            )
            fig2 = px.line(tmp, x="date", y="score", color="signal")
            st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.subheader("Drivers (latest)")
        st.caption("Hover column headers for definitions.")
        if daily_combined is None or daily_combined.empty:
            st.write("—")
        else:
            last_day = daily_combined.iloc[-1]
            drivers = pd.DataFrame(
                [
                    {"component": "tone_score", "value": float(last_day["tone_score"]), "effect": float(last_day["tone_logit"])},
                    {"component": "events_score", "value": float(last_day["events_score"]), "effect": float(last_day["events_logit"])},
                    {"component": "blend (base)", "value": float(last_day["combined_score"]), "effect": None},
                ]
            )
            drivers["effect"] = drivers["effect"].apply(fmt_optional)

            st.data_editor(
                drivers,
                use_container_width=True,
                hide_index=True,
                disabled=True,
                column_config={
                    "component": st.column_config.TextColumn(
                        "component",
                        help=(
                            "tone_score: derived from GDELT DOC tone + diplomacy themes\n"
                            "events_score: derived from GDELT Events Goldstein conflict/cooperation\n"
                            "blend (base): weighted combination mapped to 0–100"
                        ),
                    ),
                    "value": st.column_config.NumberColumn(
                        "value",
                        format="%.2f",
                        help="0–100 score value for the component.",
                    ),
                    "effect": st.column_config.NumberColumn(
                        "effect",
                        format="%.3f",
                        help="Internal logit-space contribution before mapping to 0–100 (for tone/events).",
                    ),
                },
            )

        st.subheader("Latest matching articles")
        if articles is None or articles.empty:
            st.write("—")
        else:
            tmp = articles.copy()
            if "seendate" in tmp.columns:
                tmp["dt"] = pd.to_datetime(tmp["seendate"], errors="coerce", utc=True)
            elif "datetime" in tmp.columns:
                tmp["dt"] = pd.to_datetime(tmp["datetime"], errors="coerce", utc=True)
            else:
                tmp["dt"] = pd.NaT

            latest = tmp.sort_values("dt", ascending=False).head(20)
            for _, row in latest.iterrows():
                dt = row.get("dt")
                dt_str = dt.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(dt) else "—"
                title = row.get("title") or "(no title)"
                url = row.get("url") or ""
                domain = row.get("domain") or ""
                st.markdown(f"- [{title}]({url})  \n  *{dt_str} · {domain}*")

    with st.expander("Raw daily table (current window)"):
        st.dataframe(daily_combined, use_container_width=True)


# -----------------------------
# TAB 2: Air traffic details
# -----------------------------
with tab2:
    st.subheader("Iran air-traffic signal (aggregated)")
    st.caption("SQLite persists aggregate airborne counts so anomaly detection uses a stable baseline across restarts.")
    st.write(f"Bounding box (Iran): lat {IR_LAMIN}–{IR_LAMAX}, lon {IR_LOMIN}–{IR_LOMAX}.")
    st.write(air_msg)

    if df24 is None or df24.empty:
        st.info("Not enough samples yet to compute a stable baseline. Refresh occasionally to collect samples.")
    else:
        plot_df = df24.copy()
        plot_df["ts"] = pd.to_datetime(plot_df["ts"], utc=True)
        fig = px.line(plot_df, x="ts", y="airborne", title="Airborne aircraft over Iran (SQLite samples, last 24h)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(plot_df.tail(50), use_container_width=True, hide_index=True)


# -----------------------------
# TAB 3: Calibration (manual labels + simple logistic fit)
# -----------------------------
with tab3:
    st.subheader("Calibration (optional)")
    st.caption(
        "If you label past days as escalation (1) / non-escalation (0), the app can fit a simple logistic model "
        "to map the score into an estimated probability. This is optional and only activates with enough labels."
    )

    hist = read_daily_scores(days_back=365)
    if hist.empty:
        st.info("No score history yet. Let the app run and refresh a few times.")
    else:
        # Label editor
        st.markdown("### Label days")
        st.caption("Labeling does not call any external API. It only writes to your local SQLite DB.")

        show = hist.tail(120).copy()
        show["label"] = show["label"].astype("Int64")

        edited = st.data_editor(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "label": st.column_config.SelectboxColumn(
                    "label",
                    help="1 = escalation, 0 = non-escalation, blank = unknown",
                    options=[None, 0, 1],
                )
            },
            disabled=["date", "tone_score", "events_score", "combined_score", "adjusted_score", "articles", "events"],
        )

        if st.button("Save labels"):
            changed = 0
            for _, row in edited.iterrows():
                dstr = row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"])
                lbl = row["label"]
                lbl_val = None if pd.isna(lbl) else int(lbl)
                old = hist.loc[hist["date"] == row["date"], "label"].iloc[0]
                old_val = None if pd.isna(old) else int(old)
                if lbl_val != old_val:
                    set_label(dstr, lbl_val)
                    changed += 1
            st.success(f"Saved {changed} label changes.")

        st.divider()
        st.markdown("### Model fit")

        hist2 = read_daily_scores(days_back=365)
        labeled = hist2.dropna(subset=["label", "combined_score", "events_score", "tone_score"]).copy()
        if len(labeled) < 25 or labeled["label"].nunique() < 2:
            st.info("Add at least ~25 labeled days across both classes (0 and 1) to enable calibration.")
        else:
            X = labeled[["combined_score", "events_score", "tone_score"]].astype(float).to_numpy()
            y = labeled["label"].astype(int).to_numpy()

            # standardize features
            mu = X.mean(axis=0)
            sig = X.std(axis=0)
            sig[sig == 0] = 1.0
            Xs = (X - mu) / sig

            w, b = fit_logistic_regression(Xs, y, lr=0.08, steps=900)

            st.write("Weights:", dict(zip(["combined", "events", "tone"], w.round(3))))
            st.write("Intercept:", round(float(b), 3))

            # probability over history (for display)
            Xall = hist2[["combined_score", "events_score", "tone_score"]].fillna(0.0).astype(float).to_numpy()
            Xall = (Xall - mu) / sig
            p = predict_proba(Xall, w, b)
            hist2["calibrated_p"] = p

            fig = px.line(hist2, x="date", y="calibrated_p", title="Calibrated probability estimate (from your labels)")
            st.plotly_chart(fig, use_container_width=True)

            # reliability buckets
            buckets = pd.cut(hist2.loc[hist2["label"].notna(), "calibrated_p"], bins=np.linspace(0, 1, 6))
            rel = (
                hist2.loc[hist2["label"].notna()]
                .groupby(buckets, observed=True)
                .agg(p_mean=("calibrated_p", "mean"), y_rate=("label", "mean"), n=("label", "count"))
                .reset_index(drop=True)
            )
            st.markdown("#### Reliability (labeled days)")
            st.dataframe(rel, use_container_width=True, hide_index=True)


