from __future__ import annotations

import pandas as pd
import streamlit as st

from gdelt_structured import fetch_gdelt_articles as fetch_gdelt_articles_structured
from gdelt_structured import dedupe_syndication


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_articles_cached(query: str, hours_back: int, max_records: int, cache_key: str = "stable") -> pd.DataFrame:
    """Cached fetch to avoid hammering GDELT."""
    _ = cache_key
    return fetch_gdelt_articles_structured(query=query, hours_back=hours_back, max_records=max_records)


def fetch_and_dedupe(query: str, hours_back: int, max_records: int, cache_key: str = "stable") -> pd.DataFrame:
    df = fetch_articles_cached(query=query, hours_back=hours_back, max_records=max_records, cache_key=cache_key)
    if df is None or df.empty:
        return pd.DataFrame()
    return dedupe_syndication(df)
