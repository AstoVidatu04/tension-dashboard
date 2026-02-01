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


def logistic_0_100(x: pd.Series) -> pd.Series:
    return 100.0 * (1.0 / (1.0 + np.exp(-x)))


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
    """
    Streamlit Secrets first, env var fallback.
    """
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
    d["score"] = logistic_0_100(raw)

    return d


# =============================
# OpenSky OAuth2 (aggregated-only)
# =============================
OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"


def empty_states_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "icao24", "callsign", "origin_country", "time_position", "last_contact",
            "lon", "lat", "baro_altitude", "on_ground", "velocity", "heading",
            "vertical_rate", "geo_altitude"
        ]
    )


@st.cache_data(ttl=50 * 60, show_spinner=False)
def fetch_opensky_token(cache_key: str) -> Tuple[Optional[str], str]:
    """
    OAuth2 client-credentials flow.
    """
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
def fetch_opensky_states(cache_key: str) -> Tuple[pd.DataFrame, str]:
    token, msg = fetch_opensky_token("token-" + cache_key)
    if not token:
        return empty_states_df(), msg

    try:
        r = requests.get(
            OPENSKY_STATES_URL,
            timeout=30,
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "tension-dashboard/1.0",
            },
        )
    except requests.RequestException as e:
        return empty_states_df(), f"OpenSky network error: {e}"

    if r.status_code != 200:
        snippet = (r.text or "")[:220].replace("\n", " ").strip()
        return empty_states_df(), f"OpenSky HTTP {r.status_code}. Snippet: {snippet}"

    try:
        js = r.json()
    except ValueError:
        return empty_states_df(), "OpenSky returned non-JSON response."

    states = js.get("states") or []
    if not states:
        return empty_states_df(), "OpenSky returned 0 states."

    rows = []
    for s in states:
        if not isinstance(s, list) or len(s) < 14:
            continue
        rows.append(
            {
                "icao24": (s[0] or "").lower().strip(),
                "callsign": (s[1] or "").strip(),
                "origin_country": s[2] or "",
                "time_position": s[3],
                "last_contact": s[4],
                "lon": s[5],
                "lat": s[6],
                "baro_altitude": s[7],
                "on_ground": s[8],
                "velocity": s[9],
                "heading": s[10],
                "vertical_rate": s[11],
                "geo_altitude": s[13],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return empty_states_df(), "OpenSky returned states but none were parseable."

    return df, f"{msg} OpenSky snapshot loaded: {len(df)} states."


def render_opensky_aggregates(states_df: pd.DataFrame):
    st.subheader("Flight activity (OpenSky) — aggregated snapshot")
    st.caption("Counts only (analysis view). No per-aircraft tracking list is shown by default.")

    if states_df.empty:
        st.info("No OpenSky data available right now.")
        return

    sdf = states_df.copy()
    sdf["air_status"] = np.where(sdf["on_ground"] == True, "on_ground", "airborne")
    sdf["callsign_prefix"] = sdf["callsign"].fillna("").astype(str).str.strip().str[:3].replace("", "UNK")

    with st.expander("Optional filters (aggregated only)", expanded=False):
        kw = st.text_input(
            "Filter by callsign keyword (optional)",
            value="",
            help="Example: an airline code like 'DLH' or 'BAW'.",
        )
        show_prefix_table = st.checkbox("Show callsign prefix table (aggregated)", value=True)

    if kw.strip():
        sdf = sdf[sdf["callsign"].astype(str).str.contains(kw.strip(), case=False, na=False)]

    by_country = (
        sdf.groupby("origin_country")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(25)
    )
    fig1 = px.bar(by_country, x="origin_country", y="count", title="Top origin countries (snapshot)")
    st.plotly_chart(fig1, use_container_width=True)
    st.dataframe(by_country, use_container_width=True, hide_index=True)

    by_status = (
        sdf.groupby("air_status")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    fig2 = px.bar(by_status, x="air_status", y="count", title="Airborne vs on-ground (snapshot)")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(by_status, use_container_width=True, hide_index=True)

    if show_prefix_table:
        by_prefix = (
            sdf.groupby("callsign_prefix")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(30)
        )
        fig3 = px.bar(by_prefix, x="callsign_prefix", y="count", title="Top callsign prefixes (snapshot)")
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(by_prefix, use_container_width=True, hide_index=True)


# =============================
# UI
# =============================
st.title("USA–Iran Tension Dashboard")
st.caption(
    "A transparent indicator built from public news signals (GDELT). "
    "Not a literal probability-of-war predictor."
)

tab1, tab2 = st.tabs(["Tension dashboard (GDELT)", "Flight activity (OpenSky)"])

with tab1:
    st.info(
        "⏱️ To avoid GDELT rate limits, data is cached for **1 hour**. "
        "Sliders update instantly using cached data. Use **Refresh now** only if needed."
    )

    with st.sidebar:
        st.header("Query & window")

        default_query = "(United States OR USA OR US) (Iran OR Iranian)"
        query = st.text_input("GDELT query", value=default_query)

        window_days = st.slider("Lookback (days)", 7, 180, DEFAULT_WINDOW_DAYS)
        maxrecords = st.slider("Max articles to fetch", 50, 250, DEFAULT_MAXRECORDS, step=25)

        st.header("Scoring weights")
        w_hostile = st.slider("Hostile (↑)", 0.0, 5.0, 2.0, 0.1)
        w_military = st.slider("Military (↑)", 0.0, 5.0, 1.5, 0.1)
        w_diplomacy = st.slider("Diplomacy (↓)", 0.0, 5.0, 1.0, 0.1)

        smooth_days = st.slider("Smoothing (days)", 1, 14, 3)

        st.header("Controls")
        refresh = st.button("Refresh now")

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

    if detail_msg:
        st.write(detail_msg)

    daily = build_daily_features(articles)
    scored = compute_score(daily, w_hostile, w_military, w_diplomacy, smooth_days=smooth_days)

    c1, c2, c3, c4 = st.columns(4)
    latest_score = float(scored["score"].iloc[-1]) if len(scored) else float("nan")

    c1.metric("Risk score (latest)", "—" if math.isnan(latest_score) else f"{latest_score:.1f}/100")
    c2.metric("Articles fetched", f"{len(articles)}")
    c3.metric("Window", f"{window_days} days")
    c4.metric("Updated (UTC)", end_dt.strftime("%Y-%m-%d %H:%M"))

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Risk score over time")
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
        st.subheader("Latest-day drivers")
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
            st.caption("Effects are based on smoothed z-scores and your weights.")

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

with tab2:
    st.info(
        "OpenSky data is shown as **aggregated counts** (analysis view). "
        "Snapshot is cached for 10 minutes."
    )

    colA, colB = st.columns([1, 2])
    with colA:
        opensky_refresh = st.button("Refresh OpenSky snapshot")
    with colB:
        st.caption("Tip: refresh only if needed.")

    opensky_cache_key = utc_now().strftime("%Y%m%d%H%M%S") if opensky_refresh else "stable"

    with st.spinner("Fetching OpenSky snapshot…"):
        states_df, msg = fetch_opensky_states(opensky_cache_key)

    st.write(msg)
    render_opensky_aggregates(states_df)
