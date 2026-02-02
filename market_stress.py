import pandas as pd

def _stooq_csv(symbol: str) -> pd.DataFrame:
    # Example: https://stooq.com/q/d/l/?s=vix&i=d
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    return pd.read_csv(url)

def daily_return(symbol: str, lookback_days: int = 10) -> float | None:
    try:
        df = _stooq_csv(symbol)
        if df.empty or "Close" not in df.columns:
            return None
        df = df.tail(lookback_days + 1)
        if len(df) < 2:
            return None
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        if prev == 0:
            return None
        return (last / prev) - 1.0
    except Exception:
        return None

def market_amplifier() -> float:
    """
    Returns multiplier ~ [0.9 .. 1.2]
    - Oil jump up: amplifies
    - VIX jump up: amplifies
    """
    brent = daily_return("co1")   # Brent crude continuous on stooq (may vary)
    vix = daily_return("vix")

    amp = 1.0
    if brent is not None:
        if brent > 0.02: amp += 0.05
        if brent > 0.05: amp += 0.05
    if vix is not None:
        if vix > 0.03: amp += 0.05
        if vix > 0.07: amp += 0.05

    return max(0.9, min(1.2, amp))
