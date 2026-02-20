import requests
import pandas as pd
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


def fetch_gdelt_articles(query: str, hours_back: int, max_records: int) -> pd.DataFrame:
    max_records = int(max_records)
    if max_records > 250:
        max_records = 250

    def _fetch_once(sort: str, limit: int) -> pd.DataFrame:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": int(limit),
            "formatdatetime": "true",
            "sort": sort,
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
        return pd.DataFrame(arts)

    # Blend recency + relevancy to avoid single-day clustering that can pin scores near 50.
    per_call = max(1, max_records // 2)
    recent_df = _fetch_once(sort="DateDesc", limit=per_call)
    rel_df = _fetch_once(sort="HybridRel", limit=max_records - per_call)

    if recent_df.empty and rel_df.empty:
        return pd.DataFrame()

    df = pd.concat([recent_df, rel_df], ignore_index=True) if not recent_df.empty and not rel_df.empty else (
        recent_df if not recent_df.empty else rel_df
    )

    # If still too concentrated in a short time span, blend with oldest records as a fallback.
    se_probe = pd.to_datetime(df.get("seendate"), errors="coerce", utc=True)
    unique_days = int(se_probe.dt.floor("D").nunique()) if se_probe.notna().any() else 0
    if int(hours_back) >= 72 and unique_days < 3 and len(df) < max_records:
        older_df = _fetch_once(sort="DateAsc", limit=max_records - len(df))
        if not older_df.empty:
            df = pd.concat([df, older_df], ignore_index=True)

    # Ensure expected columns always exist
    for col in ["url", "title", "domain", "seendate", "tone", "themes_list"]:
        if col not in df.columns:
            df[col] = None

    # Parse seendate safely (handles numeric + ISO)
    sd = df["seendate"].astype(str).str.strip()
    is_num14 = sd.str.fullmatch(r"\d{14}")

    se = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    if is_num14.any():
        se.loc[is_num14] = pd.to_datetime(
            sd.loc[is_num14], format="%Y%m%d%H%M%S", utc=True, errors="coerce"
        )
    if (~is_num14).any():
        se.loc[~is_num14] = pd.to_datetime(
            sd.loc[~is_num14], utc=True, errors="coerce"
        )

    df["seendate"] = se
    df["tone"] = pd.to_numeric(df["tone"], errors="coerce")

    return df


def dedupe_syndication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe deduplication that NEVER calls .dt on non-datetime data.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    out = df.copy()

    # URL dedupe
    if "url" in out.columns:
        out = out.drop_duplicates(subset=["url"]).copy()

    # Title canonicalization
    if "title" not in out.columns:
        out["title"] = ""

    out["_tcanon"] = (
        out["title"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^a-z0-9 ]+", "", regex=True)
        .str.strip()
    )

    # FORCE datetime conversion here (this is the critical fix)
    se = pd.to_datetime(out.get("seendate"), utc=True, errors="coerce")

    if se.notna().any():
        out["_day"] = se.dt.floor("D")
        out = out.drop_duplicates(subset=["_day", "_tcanon"]).copy()
        out = out.drop(columns=["_day", "_tcanon"])
    else:
        out = out.drop_duplicates(subset=["_tcanon"]).drop(columns=["_tcanon"])

    return out


def source_diversity_factor(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.8

    n = len(df)
    d = int(df.get("domain", pd.Series(["unknown"] * n)).fillna("unknown").nunique())
    ratio = d / max(n, 1)

    if d <= 3 and n >= 20:
        return 0.65
    if ratio >= 0.5 and d >= 10:
        return 1.15
    if ratio >= 0.35:
        return 1.05
    return 0.9


def structured_signal_score(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"tension_core": 0.0, "diplomacy_share": 0.0, "uncertainty": 1.0}

    n = max(len(df), 1)
    tone = pd.to_numeric(df.get("tone"), errors="coerce").clip(-10, 10)
    neg_mass = float((-tone[tone < 0]).sum()) if tone.notna().any() else 0.0
    tension_core = neg_mass / n

    dip_keywords = {
        "NEGOTIATIONS",
        "MEDIATION",
        "DIPLOMACY",
        "PEACE",
        "CEASEFIRE",
        "TREATY",
    }

    themes = df.get("themes_list", pd.Series([None] * n))

    def is_diplomatic(x):
        if not isinstance(x, list):
            return False
        for t in x:
            if isinstance(t, str):
                u = t.upper()
                for k in dip_keywords:
                    if k in u:
                        return True
        return False

    diplomacy_share = float(themes.apply(is_diplomatic).sum() / n)

    div = int(df.get("domain", pd.Series(["unknown"] * n)).fillna("unknown").nunique())

    if n >= 80 and div >= 25:
        uncertainty = 0.5
    elif n >= 40 and div >= 15:
        uncertainty = 0.65
    elif n >= 20 and div >= 8:
        uncertainty = 0.8
    else:
        uncertainty = 1.1

    return {
        "tension_core": float(tension_core),
        "diplomacy_share": float(diplomacy_share),
        "uncertainty": float(uncertainty),
    }
