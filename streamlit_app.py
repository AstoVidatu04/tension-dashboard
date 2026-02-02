import math
import time
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlite3
import streamlit as st
import streamlit.components.v1 as components

from gdelt_structured import (
    fetch_structured_articles,
    compute_base_risk_score,
    compute_adjusted_risk_score,
)

APP_TITLE = "USA–Iran Tension Dashboard"

st.set_page_config(page_title=APP_TITLE, layout="wide")


# -----------------------------
# Utilities
# -----------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def classify_risk(score: float) -> Tuple[str, str]:
    """
    Returns a label + hex color for a given risk score.
    """
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


def radar_gauge(score: float, color: str) -> go.Figure:
    """
    Semi-circular gauge.
    """
    s = clamp(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=s,
            number={"suffix": "/100", "font": {"size": 54}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.5)"},
                "bar": {"color": color, "thickness": 0.20},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20], "color": "rgba(34,197,94,0.20)"},
                    {"range": [20, 40], "color": "rgba(234,179,8,0.18)"},
                    {"range": [40, 60], "color": "rgba(249,115,22,0.16)"},
                    {"range": [60, 80], "color": "rgba(239,68,68,0.16)"},
                    {"range": [80, 100], "color": "rgba(127,29,29,0.18)"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    # Make it a semi-circle by hiding the lower half.
    fig.update_traces(gauge_axis_tickangle=0)
    fig.update_layout(
        annotations=[],
    )
    return fig


def small_gauge(score: float, title: str, color: str) -> go.Figure:
    s = clamp(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=s,
            number={"suffix": "/100", "font": {"size": 30}},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20], "color": "rgba(34,197,94,0.18)"},
                    {"range": [20, 40], "color": "rgba(234,179,8,0.16)"},
                    {"range": [40, 60], "color": "rgba(249,115,22,0.14)"},
                    {"range": [60, 80], "color": "rgba(239,68,68,0.14)"},
                    {"range": [80, 100], "color": "rgba(127,29,29,0.16)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


# -----------------------------
# UI Helpers
# -----------------------------
def radar_card(score: float, title: str, subtitle: str, updated_dt: datetime, next_update_seconds: int):
    """
    FIXED: Ensure the badge <div> does NOT start with leading spaces after a blank line,
    otherwise Streamlit Markdown prints it as code.
    """
    label, color = classify_risk(score)
    mins_ago = int((utc_now() - updated_dt).total_seconds() // 60)
    mm = max(0, int(next_update_seconds // 60))
    ss = max(0, int(next_update_seconds % 60))
    countdown = f"{mm:02d}:{ss:02d}"

    html = f"""
<div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 14px 16px 8px 16px;">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
    <div style="font-size:13px; opacity:0.85;">Updated <b>{mins_ago} min ago</b> · UTC</div>
    <div style="font-size:13px; opacity:0.85;">Next update in <b>{countdown}</b></div>
  </div>

  <div style="text-align:center; margin-top:12px;">
    <div style="font-size:13px; letter-spacing:0.10em; opacity:0.85;">{title}</div>
    <div style="font-size:12px; opacity:0.65; margin-top:4px;">{subtitle}</div>
<div style="display:inline-block; margin-top:10px; padding:6px 14px; border-radius:999px; background:{color}; color:#0b0b0b; font-weight:800; font-size:12px;">
  {label}
</div>
  </div>
</div>
"""
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)
    st.plotly_chart(radar_gauge(score, color), use_container_width=True)


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
# Data Fetching
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_fetch_structured(query: str, days: int) -> pd.DataFrame:
    return fetch_structured_articles(query=query, days=days)


# -----------------------------
# Layout
# -----------------------------
st.title(APP_TITLE)
st.caption("Structured news-derived indicator (GDELT) + aggregate air-traffic signal over Iran (OpenSky).")

tab1, tab2 = st.tabs(["Tension dashboard", "Air traffic signal"])

with st.sidebar:
    st.header("Controls")
    days = st.slider("Window (days)", min_value=7, max_value=60, value=30, step=1)
    refresh = st.checkbox("Auto-refresh (1s)", value=False)
    st.markdown("---")
    st.write("Data cache: GDELT cached (~1 hour). OpenSky snapshot cached (~10 minutes).")

live_tick(refresh)

# -----------------------------
# Main Dashboard
# -----------------------------
with tab1:
    # Query is defined inside gdelt_structured integration; keep this simple:
    query = "Iran OR Iranian OR Tehran OR IRGC OR Strait of Hormuz OR Persian Gulf OR USA OR United States OR Pentagon OR CENTCOM"

    try:
        df = cached_fetch_structured(query=query, days=days)
    except Exception as e:
        st.error(f"Failed to fetch structured articles: {e}")
        df = pd.DataFrame()

    if df.empty:
        st.warning("No articles returned for the current window.")
    else:
        base_score = compute_base_risk_score(df)
        adjusted_score = compute_adjusted_risk_score(df, base_score)

        updated_dt = utc_now()
        next_update_seconds = 5 * 60  # placeholder countdown; your app may set this based on cron/loop

        radar_card(
            score=adjusted_score,
            title="US–IRAN TENSION INDICATOR",
            subtitle="Composite alert level (news + optional amplifiers)",
            updated_dt=updated_dt,
            next_update_seconds=next_update_seconds,
        )

        st.info(
            "GDELT is cached (1 hour). OpenSky snapshot is cached (~10 minutes). "
            "Market/shipping amplifiers apply to the latest score only."
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            lbl, col = classify_risk(base_score)
            st.plotly_chart(small_gauge(base_score, "Base risk score (latest)", col), use_container_width=True)
        with c2:
            lbl, col = classify_risk(adjusted_score)
            st.plotly_chart(small_gauge(adjusted_score, "Adjusted score (latest)", col), use_container_width=True)
        with c3:
            st.metric("Articles (deduped)", f"{len(df):,}")
        with c4:
            st.metric("Window", f"{days} days")
        with c5:
            st.metric("Updated (UTC)", updated_dt.strftime("%Y-%m-%d %H:%M:%S"))

        st.markdown("---")
        st.subheader("Latest Articles")

        show_cols = [c for c in ["date", "title", "sourceCountry", "url"] if c in df.columns]
        if not show_cols:
            st.dataframe(df.head(100), use_container_width=True)
        else:
            st.dataframe(df[show_cols].head(100), use_container_width=True)

with tab2:
    st.subheader("Air Traffic Signal (placeholder)")
    st.write("If you have OpenSky ingestion code, paste it here and I will integrate it cleanly.")
