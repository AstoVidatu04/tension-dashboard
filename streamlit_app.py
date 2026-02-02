import math
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import sqlite3
from pathlib import Path

# New helpers
from gdelt_structured import (
    fetch_gdelt_articles as fetch_gdelt_articles_structured,
    dedupe_syndication,
    source_diversity_factor,
    structured_signal_score,
)
from market_stress import market_amplifier


# =============================
# App config
# =============================
st.set_page_config(page_title="USA–Iran Tension Dashboard", layout="wide")

DEFAULT_WINDOW_DAYS = 30
DEFAULT_MAXRECORDS = 250

DEFAULT_SHIPPING_QUERY = (
    '("Red Sea" OR "Strait of Hormuz" OR tanker OR shipping OR container) '
    "AND (disruption OR attack OR reroute OR risk OR insurance)"
)


# =============================
# Utilities
# =============================
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
    """Simple regime-shift detector: compare mean of last `recent` vs previous `prior`."""
    if series is None or len(series) < (recent + prior):
        return False
    s = np.array(series, dtype=float)
    r = s[-recent:]
    p = s[-(recent + prior) : -recent]
    if p.std() == 0:
        return False
    return abs(r.mean() - p.mean()) / p.std() >= z


def fmt_optional(x):
    """Convert NaN to None for nicer display in the tooltip table."""
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except Exception:
        pass
    return x


def risk_meter(score: float, title: str):
    """
    Plotly gauge meter for a 0–100 score with colored bands.
    """
    import plotly.graph_objects as go

    def _bar_color(v: float) -> str:
        if v < 25:
            return "#1f9d55"  # green
        if v < 50:
            return "#d4a106"  # yellow
        if v < 75:
            return "#cc6d1c"  # orange
        return "#c0392b"  # red

    if score is None or (isinstance(score, float) and math.isnan(score)):
        val = 0.0
        suffix = ""
    else:
        val = float(score)
        suffix = "/100"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": suffix, "font": {"size": 38}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": _bar_color(val)},
                "steps": [
                    {"range": [0, 25], "color": "#2ecc71"},
                    {"range": [25, 50], "color": "#f1c40f"},
                    {"range": [50, 75], "color": "#e67e22"},
                    {"range": [75, 100], "color": "#e74c3c"},
                ],
                "threshold": {
                    "line": {"color": "#111111", "width": 4},
                    "thickness": 0.75,
                    "value": val,
                },
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=55, b=10))
    return fig


# =============================
# GDELT Structured: caching wrapper
# =============================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_fetch_structured(query: str, hours_back: int, max_records: int, cache_key: str) -> pd.DataFrame:
    _ = cache_key
    df = fetch_gdelt_articles_structured(query=query, hours_back=hours_back, max_records=max_records)
    return df


def build_daily_structured_features(df_articles: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily time series using tone + (optional) themes.

    Output columns:
      date, articles, tension_core, diplomacy_share
    """
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

    tone = pd.to_numeric(df.get("tone", pd.Series([np.nan] * len(df))), errors="coerce").clip(-10, 10)
    df["tone_num"] = tone

    dip_keywords = {"NEGOTIATIONS", "MEDIATION", "DIPLOMACY", "PEACE", "CEASEFIRE", "TREATY"}

    def is_diplomatic(themes):
        if not isinstance(themes, list):
            return False
        tset = {t.strip().upper() for t in themes if isinstance(t, str) and t.strip()}
        return any(any(k in t for k in dip_keywords) for t in tset)

    if "themes_list" in df.columns:
        df["is_diplomatic"] = df["themes_list"].apply(is_diplomatic).astype(int)
    else:
        df["is_diplomatic"] = 0

    def tension_core_for_group(g: pd.DataFrame) -> float:
        t = g["tone_num"].dropna()
        if t.empty:
            return 0.0
        neg_mass = (-t[t < 0]).sum()
        n = max(len(g), 1)
        return float(neg_mass / n)

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


def compute_structured_score_series(daily: pd.DataFrame, smooth_days: int, diplomacy_weight: float) -> pd.DataFrame:
    """
    Convert daily tension_core (+ diplomacy_share adjustment) into a 0–100 score.
    We use z-scored smoothed tension_core as the base, then subtract diplomacy share.
    """
    if daily is None or daily.empty:
        return pd.DataFrame(columns=list(daily.columns) + ["score"])

    d = daily.copy()
    d["tension_sm"] = d["tension_core"].rolling(window=smooth_days, min_periods=1).mean()
    d["z_tension"] = zscore(d["tension_sm"])

    raw = d["z_tension"] - (diplomacy_weight * d["diplomacy_share"])
    d["score"] = (100.0 * (1.0 / (1.0 + np.exp(-raw)))).astype(float)

    return d


# =============================
# SQLite: persistent flight baseline (timestamp + aggregate count only)
# =============================
DB_PATH = Path(__file__).with_name("flights.db")


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_flights_db() -> None:
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


def compute_flight_z_baselined(
    now_ts: datetime,
    baseline_days: int = 28,
    min_baseline_points: int = 30,
    lookback_hours_for_plot: int = 24,
) -> Tuple[Optional[float], pd.DataFrame]:
    """
    Compute z-score of the latest airborne count relative to a baseline
    for the same hour-of-day + day-of-week over the last `baseline_days`.
    """
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


# IMPORTANT: init DB after functions are defined
init_flights_db()


# =============================
# OpenSky OAuth2 + Iran airspace aggregate
# =============================
OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"


def get_secret(key: str) -> str:
    if key in st.secrets:
        return str(st.secrets[key])
    import os

    v = os.environ.get(key)
    if v:
        return v
    raise RuntimeError(f"Missing secret: {key}. Add it in Streamlit Secrets.")


@st.cache_data(ttl=50 * 60, show_spinner=False)
def fetch_opensky_token(cache_key: str) -> Tuple[Optional[str], str]:
    _ = cache_key
    try:
        client_id = get_secret("OPENSKY_CLIENT_ID")
        client_secret = get_secret("OPENSKY_CLIENT_SECRET")
    except Exception as e:
        return None, f"OpenSky OAuth secrets missing: {e}"

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
        return None, f"OpenSky token request network error: {e}"

    if r.status_code != 200:
        snippet = (r.text or "")[:220].replace("\n", " ").strip()
        return None, f"OpenSky token HTTP {r.status_code}. Snippet: {snippet}"

    if not safe_is_json_response(r):
        return None, "OpenSky token endpoint returned non-JSON."

    js = r.json()
    token = js.get("access_token")
    if not token:
        return None, "OpenSky token response missing access_token."
    return token, "OpenSky token acquired."


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
        return 0, f"OpenSky network error: {e}"

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
            on_ground = s[8]
            if on_ground is False:
                airborne += 1

    return airborne, f"{msg} OpenSky bbox snapshot ok."


# =============================
# UI
# =============================
st.title("USA–Iran Tension Dashboard")
st.caption("Structured news-derived tension indicator (GDELT) + aggregate air-traffic signal over Iran (OpenSky).")

tab1, tab2 = st.tabs(["Tension dashboard", "Air traffic signal"])

# Sidebar controls
with st.sidebar:
    st.header("GDELT")
    default_query = "(United States OR USA OR US) (Iran OR Iranian)"
    query = st.text_input("GDELT query", value=default_query)

    window_days = st.slider("Lookback (days)", 7, 180, DEFAULT_WINDOW_DAYS)
    maxrecords = st.slider("Max articles to fetch", 50, 250, DEFAULT_MAXRECORDS, step=25)

    st.header("Structured scoring")
    smooth_days = st.slider("Smoothing (days)", 1, 14, 3)
    diplomacy_weight = st.slider("Diplomacy dampening", 0.0, 2.0, 0.5, 0.1)

    st.header("Amplifiers (latest score only)")
    enable_market_amp = st.checkbox("Enable market stress amplifier (oil/VIX)", value=True)
    enable_shipping_amp = st.checkbox("Enable shipping disruption news amplifier", value=False)
    shipping_query = st.text_input("Shipping query (GDELT)", value=DEFAULT_SHIPPING_QUERY)

    st.header("Air traffic signal")
    enable_air_signal = st.checkbox("Enable Iran air-traffic signal", value=True)
    w_air_traffic = st.slider("Air traffic impact (↑ when traffic drops)", 0.0, 3.0, 1.0, 0.1)
    st.caption("Air traffic uses an aggregate airborne count in an Iran bounding box (no per-aircraft listing).")

    st.header("Controls")
    refresh = st.button("Refresh now")

    st.subheader("Flight DB maintenance")
    keep_days = st.number_input("Keep flight samples (days)", min_value=14, max_value=365, value=120, step=7)
    if st.button("Prune flight DB"):
        deleted = prune_flights_db(keep_days=int(keep_days))
        st.success(f"Deleted {deleted} old flight samples (kept last {int(keep_days)} days).")

# Refresh cooldown
if "last_refresh_ts" not in st.session_state:
    st.session_state["last_refresh_ts"] = 0.0

cooldown_seconds = 30
cache_key = "stable"
if refresh:
    now_ts = time.time()
    if now_ts - st.session_state["last_refresh_ts"] < cooldown_seconds:
        st.warning(f"Please wait {cooldown_seconds} seconds between refreshes to avoid rate limits.")
    else:
        st.session_state["last_refresh_ts"] = now_ts
        cache_key = utc_now().strftime("%Y%m%d%H%M%S")

# --- Fetch GDELT structured ---
end_dt = utc_now()
hours_back = int(window_days) * 24

with st.spinner("Fetching GDELT articles…"):
    articles = cached_fetch_structured(query=query, hours_back=hours_back, max_records=maxrecords, cache_key=cache_key)

articles = dedupe_syndication(articles) if articles is not None else pd.DataFrame()

div_mult = source_diversity_factor(articles) if not articles.empty else 0.8
sig = (
    structured_signal_score(articles)
    if not articles.empty
    else {"tension_core": 0.0, "diplomacy_share": 0.0, "uncertainty": 1.0}
)

daily = build_daily_structured_features(articles)
scored = compute_structured_score_series(daily, smooth_days=smooth_days, diplomacy_weight=diplomacy_weight)

base_latest_score = float(scored["score"].iloc[-1]) if len(scored) else float("nan")

# Apply diversity multiplier in logit space (gentle)
if not math.isnan(base_latest_score):
    raw = logit_from_0_100(base_latest_score)
    raw *= float(div_mult)
    base_latest_score = logistic_0_100(raw)

# Uncertainty band (based on volume/diversity)
band = int(round(8 * float(sig.get("uncertainty", 1.0))))
low_band = None
high_band = None
if not math.isnan(base_latest_score):
    low_band = max(0, int(round(base_latest_score - band)))
    high_band = min(100, int(round(base_latest_score + band)))

# Regime shift
shift_flag = False
if not scored.empty:
    shift_flag = shift_detected(scored["score"].astype(float).tolist())

# --- Shipping disruption amplifier (latest-only) ---
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

# --- Market amplifier (latest-only) ---
mkt_mult = 1.0
if enable_market_amp:
    mkt_mult = float(market_amplifier())

# --- OpenSky Iran bbox aggregate (ONE call) ---
IR_LAMIN, IR_LAMAX = 25.29, 39.65
IR_LOMIN, IR_LOMAX = 44.77, 61.49

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
        raw_adj = raw + (-w_air_traffic * float(flight_z))  # traffic drop => higher risk
        adjusted_latest_score = logistic_0_100(raw_adj)

# Apply latest-only amplifiers after air adjustment
if not math.isnan(adjusted_latest_score):
    adjusted_latest_score = float(adjusted_latest_score) * float(mkt_mult) * float(shipping_mult)
    adjusted_latest_score = float(max(0.0, min(100.0, adjusted_latest_score)))


# =============================
# TAB 1: Main dashboard
# =============================
with tab1:
    st.info(
        "GDELT is cached (1 hour) to avoid rate limits. "
        "OpenSky air-traffic snapshot is cached (~10 minutes). "
        "Market/shipping amplifiers adjust the **latest** score only."
    )

    if shift_flag:
        st.warning("Regime shift detected: the last week differs materially from the prior baseline.")

    # KPIs: TWO meters (Base + Adjusted)
    c1, c2, c3, c4, c5 = st.columns([1.25, 1.25, 1, 1, 1])

    c1.plotly_chart(risk_meter(base_latest_score, title="Base risk score (latest)"), use_container_width=True)
    c2.plotly_chart(risk_meter(adjusted_latest_score, title="Adjusted score (latest)"), use_container_width=True)
    c3.metric("Articles (deduped)", f"{len(articles)}")
    c4.metric("Window", f"{window_days} days")
    c5.metric("Updated (UTC)", end_dt.strftime("%Y-%m-%d %H:%M"))

    # Uncertainty band + signal notes
    b1, b2, b3, b4 = st.columns(4)
    if low_band is None or high_band is None:
        b1.metric("Uncertainty band", "—")
    else:
        b1.metric("Uncertainty band", f"{low_band}–{high_band}")

    b2.metric("Source diversity multiplier", f"{div_mult:.2f}")
    b3.metric("Market amplifier", f"{mkt_mult:.2f}" if enable_market_amp else "off")
    if enable_shipping_amp:
        b4.metric("Shipping amp", f"{shipping_mult:.2f} ({shipping_articles_count} arts)")
    else:
        b4.metric("Shipping amp", "off")

    st.caption(f"OpenSky: {air_msg}")

    # Air traffic row
    a1, a2, a3 = st.columns(3)
    a1.metric("Airborne over Iran (snapshot)", "—" if airborne_over_iran is None else str(airborne_over_iran))
    a2.metric("Air-traffic z-score (baselined)", "—" if flight_z is None else f"{flight_z:+.2f}")
    a3.metric("Air traffic weight", f"{w_air_traffic:.1f}")

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Risk score over time (structured)")
        if scored.empty:
            st.info("No usable articles in this window. Try a longer window or loosen the query.")
        else:
            fig = px.line(scored, x="date", y="score")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Structured drivers (daily)")
        if not scored.empty:
            melted = scored.melt(
                id_vars=["date"],
                value_vars=["tension_core", "diplomacy_share", "articles"],
                var_name="signal",
                value_name="value",
            )
            fig2 = px.line(melted, x="date", y="value", color="signal")
            st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.subheader("Latest-day drivers (structured)")
        st.caption("Tip: hover the column headers for definitions.")
        if scored.empty:
            st.write("—")
        else:
            last = scored.iloc[-1]
            drivers = pd.DataFrame(
                [
                    {
                        "component": "tension_core (tone)",
                        "value": float(last["tension_core"]),
                        "z": float(last["z_tension"]),
                        "weight": 1.0,
                        "effect": float(last["z_tension"]) * 1.0,
                    },
                    {
                        "component": "diplomacy_share",
                        "value": float(last["diplomacy_share"]),
                        "z": None,
                        "weight": float(diplomacy_weight),
                        "effect": -float(diplomacy_weight) * float(last["diplomacy_share"]),
                    },
                    {
                        "component": "articles",
                        "value": int(last["articles"]),
                        "z": None,
                        "weight": None,
                        "effect": None,
                    },
                ]
            )

            drivers["z"] = drivers["z"].apply(fmt_optional)
            drivers["weight"] = drivers["weight"].apply(fmt_optional)
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
                            "Which signal this row represents.\n\n"
                            "- tension_core (tone): negativity in article tone (harder to game than keywords)\n"
                            "- diplomacy_share: fraction of coverage mentioning negotiation/peace themes (dampens risk)\n"
                            "- articles: number of matching articles after de-duplication (context / confidence)"
                        ),
                    ),
                    "value": st.column_config.NumberColumn(
                        "value",
                        format="%.3f",
                        help=(
                            "The raw value for this component on the latest day.\n\n"
                            "- tension_core: average negative-tone intensity (0 means neutral/positive overall)\n"
                            "- diplomacy_share: 0 to 1 (e.g., 0.25 means ~25% of articles looked diplomatic)\n"
                            "- articles: count of deduplicated articles for that day"
                        ),
                    ),
                    "z": st.column_config.NumberColumn(
                        "z (std dev)",
                        format="%.2f",
                        help=(
                            "Z-score = how unusual today's (smoothed) tension is versus recent history.\n\n"
                            "z = (today - mean) / std.\n"
                            "- z = 0: normal\n"
                            "- z = +1: noticeably higher than usual\n"
                            "- z = +2: very unusual / strong increase\n"
                            "- z < 0: lower than usual\n\n"
                            "Only computed for the tension signal in this version."
                        ),
                    ),
                    "weight": st.column_config.NumberColumn(
                        "weight",
                        format="%.2f",
                        help=(
                            "How strongly the component is applied.\n\n"
                            "- tension_core uses weight=1\n"
                            "- diplomacy_share uses your sidebar 'Diplomacy dampening' value\n"
                            "- articles has no weight because it’s context, not a direct score driver"
                        ),
                    ),
                    "effect": st.column_config.NumberColumn(
                        "effect",
                        format="%.3f",
                        help=(
                            "The contribution of this component to the score input (before converting to 0–100).\n\n"
                            "- tension_core effect ≈ z\n"
                            "- diplomacy_share effect = -weight × diplomacy_share (reduces risk)\n"
                            "- articles has no effect (used for uncertainty/confidence)"
                        ),
                    ),
                },
            )

        st.subheader("Latest matching articles")
        if articles.empty:
            st.write("—")
        else:
            if "seendate" in articles.columns:
                tmp = articles.copy()
                tmp["dt"] = pd.to_datetime(tmp["seendate"], errors="coerce", utc=True)
            else:
                tmp = articles.copy()
                tmp["dt"] = pd.NaT

            latest = tmp.sort_values("dt", ascending=False).head(20)
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


# =============================
# TAB 2: Air traffic signal details (aggregated)
# =============================
with tab2:
    st.subheader("Iran air-traffic signal (aggregated)")
    st.caption(
        "This tab shows the **persistent baseline** of airborne aircraft counts in an Iran bounding box "
        "and how it converts into a z-score used to adjust the latest tension score."
    )

    st.write(f"Bounding box (Iran): lat {IR_LAMIN}–{IR_LAMAX}, lon {IR_LOMIN}–{IR_LOMAX}.")
    st.write(air_msg)

    if df24.empty:
        st.info("Not enough samples yet to compute a stable baseline. Keep the app running / refresh occasionally.")
    else:
        plot_df = df24.copy()
        plot_df["ts"] = pd.to_datetime(plot_df["ts"], utc=True)
        fig = px.line(plot_df, x="ts", y="airborne", title="Airborne aircraft over Iran (SQLite samples, last 24h)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(plot_df.tail(50), use_container_width=True, hide_index=True)
