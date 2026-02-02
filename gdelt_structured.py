import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Any, Dict

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    ctype = (resp.headers.get("content-type") or "").lower()
    if "json" not in ctype:
        return {}
    try:
        return resp.json() or {}
    except ValueError:
        return {}


def _parse_seendate(series: pd.Series) -> pd.Series:
    """
    GDELT 'seendate' can be:
      - YYYYMMDDHHMMSS (string)
      - ISO datetime string
    Returns UTC pandas datetime.
    """
    if series is None or series.empty:
        return pd.to_datetime(pd.Series([], dtype="object"), utc=True, errors="coerce")

    s = series.astype(str).str.strip()

    # If numeric-like YYYYMMDDHHMMSS, parse via format first
    mask_num = s.str.fullmatch(r"\d{14}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")

    if mask_num.any():
        out.loc[mask_num] = pd.to_datetime(s.loc[mask_num], format="%Y%m%d%H%M%S", utc=True, errors="coerce")

    # Fallback parse for the rest
    if (~mask_num).any():
        out.loc[~mask_num] = pd.to_datetime(s.loc[~mask_num], utc=True, errors="coerce")

    return out


def fetch_gdelt_articles(query: str, hours_back: int, max_records: int) -> pd.DataFrame:
    """
    Fetch GDELT DOC 2.1 articles (ArtList).

    Notes:
      - GDELT DOC can return non-JSON under load/rate-limits.
      - Large maxrecords is often flaky; we clamp to 250 for stability.
    """
    max_records = int(max_records)
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
        r = requests.get(
            GDELT_DOC_ENDPOINT,
            params=params,
            timeout=30,
            headers={"User-Agent": "tension-dashboard/1.0"},
        )
    except requests.RequestException:
        return pd.DataFrame()

    if r.status_code != 200:
        return pd.DataFrame()

    data = _safe_json(r)
    arts = data.get("articles") or []
    if not isinstance(arts, list) or not arts:
        return pd.DataFrame()

    df = pd.DataFrame(arts)

    # Normalize expected columns so downstream code is stable
    if "seendate" in df.columns:
        df["seendate"] = _parse_seendate(df["seendate"])
    else:
        df["seendate"] = pd.NaT

    # Ensure these exist (avoid KeyError in later steps)
    for col in ["url", "title", "domain"]:
        if col not in df.columns:
            df[col] = None

    # tone may be missing or stringy
    if "tone" in df.columns:
        df["tone"] = pd.to_numeric(df["tone"], errors="coerce")
    else:
        df["tone"] = pd.NA

    # themes_list sometimes absent; keep as is if present
    if "themes_list" not in df.columns:
        df["themes_list"] = None

    return df


def dedupe_syndication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light de-duplication:
      - exact URL de-dupe
      - title de-dupe within same UTC day (cheap heuristic)
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    out = df.copy()

    # Ensure seendate is datetime
    if "seendate" in out.columns:
        if not pd.api.types.is_datetime64_any_dtype(out["seendate"]):
            out["seendate"] = _parse_seendate(out["seendate"])
    else:
        out["seendate"] = pd.NaT

    # Exact URL dedupe (if url exists)
    if "url" in out.columns:
        out = out.drop_duplicates(subset=["url"]).copy()

    # Title canonicalization
    title = out.get("title", pd.Series([""] * len(out))).fillna("").astype(str)
    tcanon = (
        title.str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^a-z0-9 ]+", "", regex=True)
        .str.strip()
    )
    out["_tcanon"] = tcanon

    # Day bucketing; if seendate missing -> treat all as same bucket
    if out["seendate"].notna().any():
        out["_day"] = out["seendate"].dt.floor("D")
    else:
        out["_day"] = pd.Timestamp.now(tz=timezone.utc).floor("D")

    out = out.drop_duplicates(subset=["_day", "_tcanon"]).drop(columns=["_tcanon", "_day"])
    return out


def source_diversity_factor(df: pd.DataFrame) -> float:
    """
    Penalize low domain diversity (many articles from few domains).
    Returns a multiplier ~ [0.6 .. 1.2]
    """
    if df is None or df.empty:
        return 0.8

    dom = df.get("domain", pd.Series(["unknown"] * len(df))).fillna("unknown")
    n = len(df)
    d = int(dom.nunique())

    ratio = d / max(n, 1)

    if d <= 3 and n >= 20:
        return 0.65
    if ratio >= 0.5 and d >= 10:
        return 1.15
    if ratio >= 0.35:
        return 1.05
    return 0.9


def structured_signal_score(df: pd.DataFrame) -> dict:
    """
    Outputs:
      - tension_core: average negative tone intensity (volume-normalized)
      - diplomacy_share: share of diplomacy/negotiation themed coverage
      - uncertainty: proxy based on volume + diversity
    """
    if df is None or df.empty:
        return {"tension_core": 0.0, "diplomacy_share": 0.0, "uncertainty": 1.0}

    n = max(len(df), 1)

    # Tone
    tone = pd.to_numeric(df.get("tone", pd.Series([pd.NA] * len(df))), errors="coerce").clip(-10, 10)
    neg_mass = float((-tone[tone < 0]).sum()) if tone.notna().any() else 0.0
    tension_core = float(neg_mass / n)

    # Diplomacy share from themes_list
    dip_keywords = {"NEGOTIATIONS", "MEDIATION", "DIPLOMACY", "PEACE", "CEASEFIRE", "TREATY"}
    themes = df.get("themes_list", pd.Series([None] * len(df)))

    def is_diplomatic(x):
        if not isinstance(x, list):
            return False
        for t in x:
            if not isinstance(t, str):
                continue
            u = t.upper()
            for k in dip_keywords:
                if k in u:
                    return True
        return False

    if themes.notna().any():
        diplomacy_share = float(themes.apply(is_diplomatic).sum() / n)
    else:
        diplomacy_share = 0.0

    # Uncertainty proxy
    dom = df.get("domain", pd.Series(["unknown"] * len(df))).fillna("unknown")
    div = int(dom.nunique())

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
        "uncertainty": float(uncertainty),
    }
