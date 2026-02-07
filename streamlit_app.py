from __future__ import annotations

import time
import math
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from config import APP_BUILD, DEFAULT_MAXRECORDS, DEFAULT_WINDOW_DAYS, DEFAULT_QUERY
from gdelt_client import fetch_and_dedupe
from features import build_daily_features
from scoring import compute_subscores, apply_diversity_multiplier
from ui_components import risk_meter, risk_label_color

from gdelt_structured import source_diversity_factor


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


st.set_page_config(page_title="USA–Iran Tension Dashboard (GDELT-only)", layout="wide")

st.title("USA–Iran Tension Dashboard")
st.caption(
    "GDELT-only multi-signal indicator with split risks: Diplomatic, Military, Economic. "
    f"Build {APP_BUILD}."
)

tab1, tab2 = st.tabs(["Tension dashboard", "Model & debug"])

DEFAULT_PIZZA_QUERY = (
    '(pizza OR pizzeria OR "pizza delivery") '
    '(Pentagon OR Arlington OR "Washington DC" OR "National Capital Region")'
)

with st.sidebar:
    st.header("GDELT")
    query = st.text_input("GDELT query", value=DEFAULT_QUERY)
    window_days = st.slider("Lookback (days)", 7, 180, DEFAULT_WINDOW_DAYS)
    maxrecords = st.slider("Max articles to fetch", 50, 500, DEFAULT_MAXRECORDS, step=25)

    st.header("Pentagon Pizza Index")
    pizza_query = st.text_input("Pizza query", value=DEFAULT_PIZZA_QUERY)
    pizza_window_days = st.slider("Pizza lookback (days)", 7, 90, 30)
    pizza_maxrecords = st.slider("Pizza max records", 50, 250, 150, step=25)

    st.header("Scoring")
    smooth_days = st.slider("Smoothing (days)", 1, 14, 3)

    st.header("Controls")
    refresh = st.button("Refresh now")


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
    articles = fetch_and_dedupe(query=query, hours_back=hours_back, max_records=maxrecords, cache_key=cache_key)

with st.spinner("Fetching Pentagon pizza signal…"):
    pizza_hours_back = int(pizza_window_days) * 24
    pizza_articles = fetch_and_dedupe(
        query=pizza_query,
        hours_back=pizza_hours_back,
        max_records=pizza_maxrecords,
        cache_key="pizza-" + cache_key,
    )

data_updated_dt = end_dt
if not articles.empty:
    if "seendate" in articles.columns:
        _dt = pd.to_datetime(articles["seendate"], errors="coerce", utc=True)
    elif "datetime" in articles.columns:
        _dt = pd.to_datetime(articles["datetime"], errors="coerce", utc=True)
    else:
        _dt = pd.Series([pd.NaT] * len(articles))
    if _dt.notna().any():
        data_updated_dt = _dt.max().to_pydatetime()
    data_age_minutes = int((end_dt - data_updated_dt).total_seconds() // 60) if data_updated_dt else None
else:
    data_age_minutes = None

div_mult = source_diversity_factor(articles) if not articles.empty else 0.9

daily = build_daily_features(articles)
scored = compute_subscores(daily, smooth_days=smooth_days)

dip_latest = float(scored["dip_score"].iloc[-1]) if len(scored) else float("nan")
mil_latest = float(scored["mil_score"].iloc[-1]) if len(scored) else float("nan")
econ_latest = float(scored["econ_score"].iloc[-1]) if len(scored) else float("nan")
comp_latest = float(scored["composite_score"].iloc[-1]) if len(scored) else float("nan")

comp_adj = apply_diversity_multiplier(comp_latest, div_mult) if not math.isnan(comp_latest) else float("nan")

pizza_daily = build_daily_features(pizza_articles)
pizza_scored = compute_subscores(pizza_daily, smooth_days=2)
pizza_latest = float(pizza_scored["composite_score"].iloc[-1]) if len(pizza_scored) else float("nan")

def pizza_status(score: float) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "No data"
    if score < 25:
        return "Nothing Ever Happens"
    if score < 45:
        return "Chatter"
    if score < 65:
        return "Noticeable"
    if score < 80:
        return "Busy Night"
    return "Red Alert"

low_band = high_band = None
if len(scored) >= 10 and not math.isnan(comp_adj):
    recent = scored["composite_score"].tail(14).astype(float)
    band = int(round(max(3.0, float(recent.std(ddof=0)))))
    low_band = max(0, int(round(comp_adj - band)))
    high_band = min(100, int(round(comp_adj + band)))

with tab1:
    st.info(
        "GDELT cached (~15 minutes). Scores: Diplomatic, Military, Economic + composite. "
        "Source diversity adjusts confidence for composite only."
    )

    st.subheader("Pentagon Pizza Index")
    p1, p2 = st.columns([1.2, 1])
    p1.plotly_chart(risk_meter(pizza_latest, "Pizza alarm"), use_container_width=True)
    p2.metric("Status", pizza_status(pizza_latest))
    p2.metric("Pizza articles (deduped)", f"{len(pizza_articles)}")

    c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1, 1])
    c1.plotly_chart(risk_meter(dip_latest, "Diplomatic risk"), use_container_width=True)
    c2.plotly_chart(risk_meter(mil_latest, "Military risk"), use_container_width=True)
    c3.plotly_chart(risk_meter(econ_latest, "Economic risk"), use_container_width=True)
    c4.metric("Articles (deduped)", f"{len(articles)}")
    c5.metric("Updated (UTC)", data_updated_dt.strftime("%Y-%m-%d %H:%M"))

    if data_age_minutes is not None and data_age_minutes > 180:
        st.warning(
            f"Latest article is {data_age_minutes} minutes old. "
            "Results may be stale; try Refresh or broaden the query."
        )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Composite (base)", "—" if math.isnan(comp_latest) else f"{comp_latest:.1f}/100")
    b2.metric("Composite (adjusted)", "—" if math.isnan(comp_adj) else f"{comp_adj:.1f}/100")
    b3.metric("Uncertainty band", "—" if low_band is None else f"{low_band}–{high_band}")
    b4.metric("Source diversity multiplier", f"{div_mult:.2f}")

    lbl, _ = risk_label_color(comp_adj if not math.isnan(comp_adj) else 0.0)
    st.metric("Composite label", lbl)

    st.divider()

    if not articles.empty:
        st.subheader("Source quality")
        dom = articles.get("domain", pd.Series(["unknown"] * len(articles))).fillna("unknown")
        top_domains = dom.value_counts().head(8)
        top_share = float(top_domains.iloc[0] / max(len(dom), 1)) if len(top_domains) else 0.0
        st.write(f"Top source share: {top_share:.1%}")
        top_domains_df = top_domains.rename_axis("domain").reset_index(name="count")
        st.dataframe(top_domains_df, hide_index=True)
        if top_share >= 0.45 and len(dom) >= 30:
            st.warning(
                "A single domain dominates the feed. Reliability may be lower; "
                "consider broadening the query or increasing max records."
            )

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Score over time")
        if scored.empty:
            st.warning("No usable data. Try a longer window or loosen the query.")
        else:
            series_choice = st.selectbox("Plot", ["Composite (base)", "Diplomatic", "Military", "Economic"], index=0)
            ycol = {
                "Composite (base)": "composite_score",
                "Diplomatic": "dip_score",
                "Military": "mil_score",
                "Economic": "econ_score",
            }[series_choice]

            fig = px.line(scored, x="date", y=ycol)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Latest-day drivers (debug)")
        if scored.empty:
            st.write("—")
        else:
            last = scored.iloc[-1]
            drivers = pd.DataFrame(
                [
                    {"subscore": "diplomatic", "raw": float(last.get("raw_dip", 0.0)), "z_intent": float(last.get("z_intent", 0.0)), "z_volume": float(last.get("z_volume", 0.0)), "z_diplomacy": float(last.get("z_diplomacy", 0.0))},
                    {"subscore": "military", "raw": float(last.get("raw_mil", 0.0)), "z_conflict": float(last.get("z_conflict", 0.0)), "z_milrate": float(last.get("z_milrate", 0.0)), "z_tone": float(last.get("z_tone", 0.0))},
                    {"subscore": "economic", "raw": float(last.get("raw_econ", 0.0)), "z_econrate": float(last.get("z_econrate", 0.0)), "z_volume": float(last.get("z_volume", 0.0)), "z_intent": float(last.get("z_intent", 0.0))},
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
                st.markdown(f"- [{title}]({url})  \\n  *{dt_str} · {domain}*")

    with st.expander("Raw daily table"):
        st.dataframe(scored, use_container_width=True)

with tab2:
    st.subheader("How it works")
    st.markdown(
        """
### Subscores
- **Diplomatic**: intent z + volume z + tone z − diplomacy z  
- **Military**: conflict themes z + military keyword rate z + tone z  
- **Economic**: economic keyword rate z + volume z + intent spillover z  

### Composite
`0.45*diplomatic + 0.35*military + 0.20*economic`

### Notes
- All components are z-scored over the selected window.
- Each term is capped (default ±1.2) to prevent one spike from dominating.
- Composite is adjusted by source diversity multiplier (logit-space scaling).
"""
    )

    st.subheader("Data diagnostics")
    if scored.empty:
        st.write("No data available.")
    else:
        nonzero_days = int((daily["articles"] > 0).sum()) if "articles" in daily.columns else 0
        date_min = str(daily["date"].min()) if "date" in daily.columns and len(daily) else "—"
        date_max = str(daily["date"].max()) if "date" in daily.columns and len(daily) else "—"

        diag = {
            "days_total": int(len(daily)),
            "days_with_articles": nonzero_days,
            "date_min": date_min,
            "date_max": date_max,
            "std_tone_sm": float(daily["neg_tone_mean"].std(ddof=0)) if "neg_tone_mean" in daily.columns else 0.0,
            "std_intent_net": float(daily["intent_net"].std(ddof=0)) if "intent_net" in daily.columns else 0.0,
            "std_conflict_share": float(daily["conflict_share"].std(ddof=0)) if "conflict_share" in daily.columns else 0.0,
            "std_mil_rate": float(daily["mil_rate"].std(ddof=0)) if "mil_rate" in daily.columns else 0.0,
            "std_econ_rate": float(daily["econ_rate"].std(ddof=0)) if "econ_rate" in daily.columns else 0.0,
            "std_articles": float(daily["articles"].astype(float).std(ddof=0)) if "articles" in daily.columns else 0.0,
        }
        st.json(diag)

        if nonzero_days <= 2 or all(v == 0.0 for k, v in diag.items() if k.startswith("std_")):
            st.warning(
                "Scores are likely ~50 because the underlying signals have near-zero variance "
                "or data exists on only 1–2 days. Increase lookback, broaden query, or use "
                "a higher max records to introduce variability."
            )
