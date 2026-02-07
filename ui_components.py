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
    label, color = risk_label_color(val)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=val,
            number={
                "suffix": "/100",
                "font": {"size": 40, "family": "IBM Plex Sans"},
            },
            delta={
                "reference": 50,
                "valueformat": ".0f",
                "increasing": {"color": "#ef4444"},
                "decreasing": {"color": "#22c55e"},
                "font": {"size": 14},
                "suffix": " vs 50",
            },
            title={"text": f"{title}<br><span style='font-size:12px;color:#6b7280'>{label}</span>", "font": {"size": 15}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "rgba(0,0,0,0.2)",
                    "tickfont": {"size": 10},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": "rgba(0,0,0,0.08)",
                "steps": [
                    {"range": [0, 20], "color": "rgba(34,197,94,0.18)"},
                    {"range": [20, 40], "color": "rgba(234,179,8,0.16)"},
                    {"range": [40, 60], "color": "rgba(249,115,22,0.14)"},
                    {"range": [60, 80], "color": "rgba(239,68,68,0.14)"},
                    {"range": [80, 100], "color": "rgba(127,29,29,0.16)"},
                ],
                "threshold": {
                    "line": {"color": "rgba(0,0,0,0.35)", "width": 2},
                    "thickness": 0.7,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=12, r=12, t=50, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "IBM Plex Sans"},
    )
    return fig
