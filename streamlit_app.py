import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# =============================
# App config
# =============================
st.set_page_config(page_title="USA–Iran Tension Dashboard", layout="wide")

GDELT_DOCS_API = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_WINDOW_DAYS = 30
DEFAULT_MAXRECORDS = 250

EXPECTED_ARTICLE_COLS = ["datetime", "title", "url", "domain", "language", "sourceCountry"]


# =============================
# Utilities
# =============================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fmt_gdelt_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def safe_is_json_response(r: requests.Response) -> bool:
    ctype = (r.headers.get("content-type") or "").lower()
    return "application/json" in ctype or "json" in ctype


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    std = float(s.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - float(s.mean())) / std


def logistic_0_100(x: float) -> float:
    return float(100.0 * (1.0 / (1.0 + math.exp(-x))))


def logit_from_0_100(score_0_100: float) -> float:
    """
    Convert 0-100 score back to logit space so we can add extra signals.
    """
    eps = 1e-6
    p = min(max(score_0_100 / 100.0, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def empty_articles_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED_ARTICLE_COLS)


def parse_any_datetime(value: object) -> Optional[datetime]:
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    if s.isdigit() and len(s) >= 14:
        try:
            return datetime.strptime(s[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            pass

    try:
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def get_secret(key: str) -> str:
    if key in st.secrets:
        return str(st.secrets[key])

    import os
    v = os.environ.get(key)
    if v:
        return v

    raise RuntimeError(f"Missing secret: {key}. Add it in Streamlit Secrets.")


# =============================
# Lightweight headline classifier (GDELT)
# =============================
KEYWORDS: Dict[str, List[str]] = {
    "hostile": [
        "attack", "strike", "missile", "drone", "retaliat", "threat", "warn", "kill",
        "sanction", "terror", "proxy", "explosion", "intercept", "airstrike", "bomb",
        "assault", "raid", "escalat", "clash"
    ],
    "military": [
        "military", "navy", "fleet", "carrier", "bases", "troops", "pentagon",
        "deployment", "exercise", "drill", "air force", "marines", "submarine",
        "gulf", "strait of hormuz", "defense ministry", "weapon", "arms"
    ],
    "diplomacy": [
        "talks", "deal", "negotiat", "diplom", "envoy", "meeting", "agreement",
        "ceasefire", "de-escalat", "backchannel", "dialogue", "summit", "mediation"
    ],
}


def classify_title(title: str) -> Dict[str, int]:
    t = (title or "").lower()
    return {k: int(any(w in t for w in words)) for k, words in KEYWORDS.items()}


# =============================
# GDELT fetcher (robust + rate-limit aware)
# =============================
@dataclass
class FetchResult:
    df: pd.DataFrame
    status: str   # "ok" | "error"
    detail: str


def _gdelt_request(params: dict, timeout: int = 30) -> Tuple[Optional[dict], Optional[str]]:
    try:
        r = requests.get(
            GDELT_DOCS_API,
            params=params,
            timeout=timeout,
            headers={"User-Agent": "tension-dashboard/1.0"},
        )
    except requests.RequestException as e:
        return None, f"Network error: {e}"

    if r.status_code == 429:
        return None, "GDELT rate limit reached (HTTP 429). Please wait a moment and try again."

    if r.status_code != 200:
        snippet = (r.text or "")[:220].replace("\n", " ").strip()
        return None, f"GDELT HTTP {r.status_code}. Response snippet: {snippet}"

    if not safe_is_json_response(r):
        snippet = (r.text or "")[:220].replace("\n", " ").strip()
        return None, f"GDELT returned non-JSON content. Snippet: {snippet}"

    try:
        return r.json(), None
    except ValueError:
        snippet = (r.text or "")[:220].replace("\n", " ").strip()
        return None, f"Failed to parse JSON. Snippet: {snippet}"


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_gdelt_articles(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    maxrecords: int,
    cache_key: str,
) -> FetchResult:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": fmt_gdelt_dt(start_dt),
        "enddatetime": fmt_gdelt_dt(end_dt),
        "maxrecords": int(maxrecords),
        "sort": "HybridRel",
    }

    data, err = _gdelt_request(params)
    if err:
        if "429" not in err:
            time.sleep(0.7)
            data, err = _gdelt_request(params)
        if err:
            return FetchResult(empty_articles_df(), "error", err)

    articles = (data or {}).get("articles", [])
    if not articles:
        return FetchResult(empty_articles_df(), "ok", "No articles returned for that window/query.")

    ts_keys = ["seendate", "seenDate", "seen_date", "datetime", "date"]

    rows = []
    skipped_no_ts = 0
    skipped_unparseable_ts = 0

    for a in articles:
        ts_val = None
        for k in ts_keys:
            if a.get(k):
                ts_val = a.get(k)
                break

        if ts_val is None:
            skipped_no_ts += 1
            continue

        dt = parse_any_datetime(ts_val)
        if dt is None:
            skipped_unparseable_ts += 1
            continue

        rows.append(
            {
                "datetime": dt,
                "title": a.get("title") or "",
                "url": a.get("url") or "",
                "domain": a.get("domain") or "",
                "language": a.get("language") or "",
                "sourceCountry": a.get("sourceCountry") or a.get("sourcecountry") or "",
            }
        )

    if not rows:
        detail = (
            "GDELT returned matches, but none had a parseable timestamp. "
            f"Skipped: no timestamp field={skipped_no_ts}, unparseable timestamp={skipped_unparseable_ts}."
        )
        return FetchResult(empty_articles_df(), "ok", detail)

    df = pd.DataFrame(rows, columns=EXPECTED_ARTICLE_COLS).sort_values("datetime")
    detail = f"Fetched {len(df)} articles (skipped {skipped_no_ts + skipped_unparseable_ts})."
    return FetchResult(df, "ok", detail)


# =============================
# Feature engineering + scoring (GDELT)
# =============================
def build_daily_features(df_articles: pd.DataFrame) -> pd.DataFrame:
    if df_articles.empty:
        return pd.DataFrame(columns=["date", "articles", "hostile", "military", "diplomacy"])

    tmp = df_articles.copy()
    tmp["date"] = pd.to_datetime(tmp["datetime"], utc=True).dt.date

    labels = tmp["title"].apply(classify_title).apply(pd.Series)
    tmp = pd.concat([tmp, labels], axis=1)

    daily = (
        tmp.groupby("date", as_index=False)
        .agg(
            articles=("title", "count"),
            hostile=("hostile", "sum"),
            military=("military", "sum"),
            diplomacy=("diplomacy", "sum"),
        )
        .sort_values("date")
    )

    all_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    daily = daily.set_index("date").reindex(all_days).fillna(0).rename_axis("date").reset_index()

    for c in ["articles", "hostile", "military", "diplomacy"]:
        daily[c] = daily[c].astype(int)

    return daily


def compute_score(
    daily: pd.DataFrame,
    w_hostile: float,
    w_military: float,
    w_diplomacy: float,
    smooth_days: int,
) -> pd.DataFrame:
    if daily.empty:
        return daily.assign(score=np.nan)

    d = daily.copy()

    for c in ["hostile", "military", "diplomacy"]:
        d[f"{c}_sm"] = d[c].rolling(window=smooth_days, min_periods=1).mean()

    d["z_hostile"] = zscore(d["hostile_sm"])
    d["z_military"] = zscore(d["military_sm"])
    d["z_diplomacy"] = zscore(d["diplomacy_sm"])

    raw = (w_hostile * d["z_hostile"]) + (w_military * d["z_military"]) - (w_diplomacy * d["z_diplomacy"])
    # raw is a Series here, convert elementwise to 0-100
    d["score"] = (100.0 * (1.0 / (1.0 + np.exp(-raw)))).astype(float)

    return d


# =============================
# OpenSky OAuth2 + Iran airspace aggregate
# =============================
OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"


@st.cache_data(ttl=50 * 60, show_spinner=False)
def fetch_opensky_token(cache_key: str) -> Tuple[Optional[str], str]:
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

    try:
        js = r.json()
    except ValueError:
        return None, "OpenSky token endpoint returned non-JSON."

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
    """
    Fetch ONE snapshot within a bounding box and return an aggregate count:
    number of airborne state vectors in bbox.
    Bounding box filtering is supported by /states/all via lamin/lamax/lomin/lomax. :contentReference[oaicite:1]{index=1}
    """
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

    try:
        js = r.json()
    except ValueError:
        return 0, "OpenSky returned non-JSON response."

    states = js.get("states") or []
    if not states:
        return 0, f"{msg} OpenSky returned 0 states in bbox."

    # count airborne only (on_ground == False at index 8)
    airborne = 0
    for s in states:
        if isinstance(s, list) and len(s) > 8:
            on_ground = s[8]
            if on_ground is False:
                airborne += 1

    return airborne, f"{msg} OpenSky bbox snapshot ok."


def update_flight_history(sample_ts: datetime, airborne_count: int):
    """
    Store aggregate samples in session memory.
    """
    if "iran_flight_samples" not in st.session_state:
        st.session_state["iran_flight_samples"] = []

    st.session_state["iran_flight_samples"].append({"ts": sample_ts, "airborne": int(airborne_count)})

    # keep last ~48h worth if sampled every few minutes; cap list to stay safe
    st.session_state["iran_flight_samples"] = st.session_state["iran_flight_samples"][-500:]


def compute_flight_z_last_24h(now_ts: datetime) -> Tuple[Optional[float], pd.DataFrame]:
    """
    Compute z-score of latest airborne count vs last 24h of in-session samples.
    Returns (z, df_samples_24h)
    """
    samples = st.session_state.get("iran_flight_samples", [])
    if not samples:
        return None, pd.DataFrame(columns=["ts", "airborne"])

    df = pd.DataFrame(samples)
    if df.empty:
        return None, pd.DataFrame(columns=["ts", "airborne"])

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cutoff = now_ts - timedelta(hours=24)
    df24 = df[df["ts"] >= cutoff].sort_values("ts")
    if len(df24) < 6:
        return None, df24

    z = float(zscore(df24["airborne"]).iloc[-1])
    return z, df24


# =============================
# UI
# =============================
st.title("USA–Iran Tension Dashboard")
st.caption("News-derived tension indicator (GDELT) + aggregate air-traffic signal over Iran (OpenSky).")

tab1, tab2 = st.tabs(["Tension dashboard", "Air traffic signal"])

# Sidebar controls (shared)
with st.sidebar:
    st.header("GDELT")
    default_query = "(United States OR USA OR US) (Iran OR Iranian)"
    query = st.text_input("GDELT query", value=default_query)

    window_days = st.slider("Lookback (days)", 7, 180, DEFAULT_WINDOW_DAYS)
    maxrecords = st.slider("Max articles to fetch", 50, 250, DEFAULT_MAXRECORDS, step=25)

    st.header("Scoring weights")
    w_hostile = st.slider("Hostile (↑)", 0.0, 5.0, 2.0, 0.1)
    w_military = st.slider("Military (↑)", 0.0, 5.0, 1.5, 0.1)
    w_diplomacy = st.slider("Diplomacy (↓)", 0.0, 5.0, 1.0, 0.1)
    smooth_days = st.slider("Smoothing (days)", 1, 14, 3)

    st.header("Air traffic signal")
    enable_air_signal = st.checkbox("Enable Iran air-traffic signal", value=True)
    # Interpretation: if traffic DROPS vs baseline -> risk should INCREASE
    w_air_traffic = st.slider("Air traffic impact (↑ when traffic drops)", 0.0, 3.0, 1.0, 0.1)

    st.caption("This uses an aggregate count of airborne aircraft within an Iran bounding box (no per-aircraft listing).")

    st.header("Controls")
    refresh = st.button("Refresh now")


# Refresh cooldown (per session)
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


# --- Fetch GDELT ---
end_dt = utc_now()
start_dt = end_dt - timedelta(days=int(window_days))

with st.spinner("Fetching GDELT articles…"):
    fetch = fetch_gdelt_articles(
        query=query,
        start_dt=start_dt,
        end_dt=end_dt,
        maxrecords=maxrecords,
        cache_key=cache_key,
    )

if "last_good_df" not in st.session_state:
    st.session_state["last_good_df"] = empty_articles_df()

if fetch.status == "error":
    st.warning(fetch.detail)
    articles = st.session_state["last_good_df"]
    detail_msg = "Showing last cached data."
else:
    articles = fetch.df
    detail_msg = fetch.detail
    if not articles.empty:
        st.session_state["last_good_df"] = articles

daily = build_daily_features(articles)
scored = compute_score(daily, w_hostile, w_military, w_diplomacy, smooth_days=smooth_days)

base_latest_score = float(scored["score"].iloc[-1]) if len(scored) else float("nan")


# --- Fetch OpenSky Iran bbox aggregate (ONE call) ---
# A practical bbox covering Iran (city ranges roughly: lat 25.29–39.65, lon 44.77–61.49). :contentReference[oaicite:2]{index=2}
IR_LAMIN, IR_LAMAX = 25.29, 39.65
IR_LOMIN, IR_LOMAX = 44.77, 61.49

airborne_over_iran = None
air_msg = "Air traffic signal disabled."
flight_z = None
df24 = pd.DataFrame(columns=["ts", "airborne"])
adjusted_latest_score = base_latest_score

if enable_air_signal:
    with st.spinner("Fetching OpenSky air-traffic snapshot over Iran…"):
        # Round cache key to 10-minute buckets to reduce repeated token/state requests
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
        update_flight_history(end_dt, airborne_over_iran)
        flight_z, df24 = compute_flight_z_last_24h(end_dt)

    # Adjust only if we have enough history to compute a meaningful z-score
    if flight_z is not None and not math.isnan(base_latest_score):
        raw = logit_from_0_100(base_latest_score)
        # Traffic drop -> negative z. We want that to INCREASE risk => subtract z * weight.
        raw_adj = raw + (-w_air_traffic * float(flight_z))
        adjusted_latest_score = logistic_0_100(raw_adj)


# =============================
# TAB 1: Main dashboard
# =============================
with tab1:
    st.info(
        "GDELT is cached (1 hour) to avoid rate limits. "
        "OpenSky air-traffic snapshot is cached (~10 minutes). "
        "The air-traffic signal adjusts the **latest** score only (session baseline)."
    )

    if detail_msg:
        st.write(detail_msg)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Base risk score (latest)", "—" if math.isnan(base_latest_score) else f"{base_latest_score:.1f}/100")
    c2.metric("Adjusted score (latest)", "—" if math.isnan(adjusted_latest_score) else f"{adjusted_latest_score:.1f}/100")
    c3.metric("Articles fetched", f"{len(articles)}")
    c4.metric("Window", f"{window_days} days")
    c5.metric("Updated (UTC)", end_dt.strftime("%Y-%m-%d %H:%M"))

    st.caption(f"OpenSky: {air_msg}")

    # Extra KPI row for air traffic
    a1, a2, a3 = st.columns(3)
    a1.metric("Airborne over Iran (snapshot)", "—" if airborne_over_iran is None else str(airborne_over_iran))
    a2.metric("Air-traffic z-score (24h, session)", "—" if flight_z is None else f"{flight_z:+.2f}")
    a3.metric("Air traffic weight", f"{w_air_traffic:.1f}")

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Risk score over time (base)")
        if scored.empty:
            st.info("No usable articles in this window. Try a longer window or loosen the query.")
        else:
            fig = px.line(scored, x="date", y="score")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Signal volumes (daily)")
        if not scored.empty:
            melted = scored.melt(
                id_vars=["date"],
                value_vars=["hostile", "military", "diplomacy"],
                var_name="signal",
                value_name="count",
            )
            fig2 = px.line(melted, x="date", y="count", color="signal")
            st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.subheader("Latest-day drivers (base)")
        if scored.empty:
            st.write("—")
        else:
            last = scored.iloc[-1]
            drivers = pd.DataFrame(
                [
                    {"component": "hostile", "value": last["hostile"], "z": last["z_hostile"], "weight": w_hostile, "effect": w_hostile * last["z_hostile"]},
                    {"component": "military", "value": last["military"], "z": last["z_military"], "weight": w_military, "effect": w_military * last["z_military"]},
                    {"component": "diplomacy", "value": last["diplomacy"], "z": last["z_diplomacy"], "weight": w_diplomacy, "effect": -w_diplomacy * last["z_diplomacy"]},
                ]
            ).sort_values("effect", ascending=False)

            st.dataframe(drivers, use_container_width=True, hide_index=True)

        st.subheader("Latest matching articles")
        if articles.empty:
            st.write("—")
        else:
            latest = articles.sort_values("datetime", ascending=False).head(20)
            for _, row in latest.iterrows():
                dt_str = pd.to_datetime(row["datetime"], utc=True).strftime("%Y-%m-%d %H:%M UTC")
                title = row["title"] or "(no title)"
                url = row["url"] or ""
                domain = row.get("domain", "")
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
        "This tab shows the **session baseline** of airborne aircraft counts in an Iran bounding box "
        "and how it converts into a z-score used to adjust the latest tension score."
    )

    st.write(f"Bounding box (Iran): lat {IR_LAMIN}–{IR_LAMAX}, lon {IR_LOMIN}–{IR_LOMAX}.")
    st.write(air_msg)

    if df24.empty:
        st.info("Not enough samples yet to compute a stable 24h baseline. Leave the app running / refresh occasionally.")
    else:
        plot_df = df24.copy()
        plot_df["ts"] = pd.to_datetime(plot_df["ts"], utc=True)
        fig = px.line(plot_df, x="ts", y="airborne", title="Airborne aircraft over Iran (session samples, last 24h)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(plot_df.tail(50), use_container_width=True, hide_index=True)
