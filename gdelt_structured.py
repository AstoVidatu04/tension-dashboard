import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

def _dt(dt: datetime) -> str:
    # GDELT uses YYYYMMDDHHMMSS (UTC)
    return dt.strftime("%Y%m%d%H%M%S")

def fetch_gdelt_articles(query: str, hours_back: int, max_records: int) -> pd.DataFrame:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    max_records = int(max_records)

    # GDELT DOC can be flaky at high maxrecords; clamp to reduce non-JSON responses.
    if max_records > 250:
        max_records = 250

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "formatdatetime": "true",
        "sort": "HybridRel",
        "timespan": f"{int(hours_back)}h",
    }

    try:
        r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "tension-dashboard/1.0"})
    except requests.RequestException:
        return pd.DataFrame()

    if r.status_code != 200:
        return pd.DataFrame()

    ctype = (r.headers.get("content-type") or "").lower()
    if "json" not in ctype:
        return pd.DataFrame()

    try:
        data = r.json()
    except ValueError:
        # Non-JSON or corrupt JSON
        return pd.DataFrame()

    arts = data.get("articles") or []
    if not isinstance(arts, list) or not arts:
        return pd.DataFrame()

    df = pd.DataFrame(arts)
    return df

def dedupe_syndication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light de-duplication:
    - Exact URL
    - Very similar titles within same day (cheap heuristic)
    """
    if df.empty:
        return df

    out = df.drop_duplicates(subset=["url"]).copy()

    # Title canonicalization
    t = (
        out["title"]
        .fillna("")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^a-z0-9 ]+", "", regex=True)
        .str.strip()
    )
    out["_tcanon"] = t
    out["_day"] = out["seendate"].dt.floor("D")

    out = out.drop_duplicates(subset=["_day", "_tcanon"]).drop(columns=["_tcanon", "_day"])
    return out


def source_diversity_factor(df: pd.DataFrame) -> float:
    """
    Penalize low domain diversity (many articles from few domains).
    Returns a multiplier ~ [0.6 .. 1.2]
    """
    if df.empty:
        return 0.8
    n = len(df)
    d = df["domain"].fillna("unknown").nunique()

    # domains per article ratio
    ratio = d / max(n, 1)

    # Heuristic mapping
    if d <= 3 and n >= 20:
        return 0.65
    if ratio >= 0.5 and d >= 10:
        return 1.15
    if ratio >= 0.35:
        return 1.05
    return 0.9


def structured_signal_score(df: pd.DataFrame) -> dict:
    """
    Output:
      - tension_core: based on negative tone intensity
      - diplomacy_share: share of diplomacy/negotiation themed coverage (if themes exist)
      - uncertainty: proxy based on volume + diversity
    """
    if df.empty:
        return {"tension_core": 0.0, "diplomacy_share": 0.0, "uncertainty": 1.0}

    # Core: negative tone mass (clip outliers)
    tone = df["tone"].dropna().clip(-10, 10)
    neg_mass = (-tone[tone < 0]).sum()  # bigger = more negative overall

    # Normalize by article count (avoid pure-volume domination)
    n = max(len(df), 1)
    tension_core = float(neg_mass / n)

    # Diplomacy share from themes (if present)
    # These are examples; tweak as you observe real outputs.
    dip_keywords = {
        "NEGOTIATIONS", "MEDIATION", "DIPLOMACY", "PEACE", "CEASEFIRE", "TREATY"
    }

    has_themes = df["themes_list"].apply(lambda x: isinstance(x, list) and len(x) > 0).any()
    if has_themes:
        def is_diplomatic(themes):
            tset = {t.strip().upper() for t in themes if t.strip()}
            return any(any(k in t for k in dip_keywords) for t in tset)

        dip = df["themes_list"].apply(is_diplomatic).sum()
        diplomacy_share = float(dip / n)
    else:
        diplomacy_share = 0.0

    # Uncertainty band proxy: fewer articles + low diversity => higher uncertainty
    div = df["domain"].fillna("unknown").nunique()
    # More data & diversity => lower uncertainty
    uncertainty = 1.0
    if n >= 80 and div >= 25:
        uncertainty = 0.5
    elif n >= 40 and div >= 15:
        uncertainty = 0.65
    elif n >= 20 and div >= 8:
        uncertainty = 0.8
    else:
        uncertainty = 1.1

    return {
        "tension_core": tension_core,
        "diplomacy_share": diplomacy_share,
        "uncertainty": uncertainty,
    }
