from __future__ import annotations

import math
import plotly.graph_objects as go
from typing import Tuple


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(lo, min(hi, v))


def risk_label_color(score: float) -> Tuple[str, str]:
    s = clamp(score)
    if s < 20:
        return "LOW", "#22c55e"
    if s < 40:
        return "GUARDED", "#eab308"
    if s < 60:
        return "ELEVATED", "#f97316"
    if s < 80:
        return "HIGH", "#ef4444"
    return "CRISIS", "#7f1d1d"


def risk_meter(score: float, title: str) -> go.Figure:
    val = 0.0 if score is None or (isinstance(score, float) and math.isnan(score)) else float(score)
    _, color = risk_label_color(val)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": "/100", "font": {"size": 42}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 20], "color": "rgba(34,197,94,0.20)"},
                    {"range": [20, 40], "color": "rgba(234,179,8,0.18)"},
                    {"range": [40, 60], "color": "rgba(249,115,22,0.16)"},
                    {"range": [60, 80], "color": "rgba(239,68,68,0.16)"},
                    {"range": [80, 100], "color": "rgba(127,29,29,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig
