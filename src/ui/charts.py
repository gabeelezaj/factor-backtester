"""Plotly figures.  Categorical hues in fixed order (validated palette): strategy blue, EW universe orange,
S&P aqua, VW market yellow, EW market magenta, oracle violet (dashed).  Text stays in ink colors."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COLORS = {"Strategy": "#2a78d6", "EW universe": "#eb6834", "S&P 500": "#1baf7a", "VW market": "#eda100",
          "EW market": "#e87ba4", "Oracle": "#4a3aa7"}
SERIES = {"net": "Strategy", "universe_ew": "EW universe", "sp500_total_return": "S&P 500",
          "market_vw_return": "VW market", "market_ew_return": "EW market"}
# no template / font color: Streamlit's theme supplies surface and ink colors for light and dark mode
LAYOUT = dict(margin=dict(l=40, r=20, t=90, b=40), hovermode="x unified",
              legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))


def _title(text: str) -> dict:
    return dict(text=text, y=0.97, yanchor="top", x=0, xanchor="left")


def cumulative_chart(monthly: pd.DataFrame, oracle: Optional[pd.Series] = None, split_date: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    for col, label in SERIES.items():
        if col not in monthly:
            continue
        w = (1 + monthly[col].fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=w.index, y=w, name=label, mode="lines",
                                 line=dict(color=COLORS[label], width=3 if label == "Strategy" else 1.5),
                                 hovertemplate="%{y:.2f}x"))
    if oracle is not None and len(oracle):
        w = (1 + oracle.reindex(monthly.index).fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=w.index, y=w, name="Oracle (upper bound)", mode="lines",
                                 line=dict(color=COLORS["Oracle"], width=1.5, dash="dash"), hovertemplate="%{y:.2f}x"))
    if split_date:
        fig.add_vline(x=pd.Timestamp(split_date).timestamp() * 1000, line=dict(color="#52514e", dash="dot", width=1))
        fig.add_annotation(x=pd.Timestamp(split_date), y=1, yref="paper", text="in-sample | out-of-sample",
                           showarrow=False, font=dict(color="#52514e", size=11), xanchor="left")
    fig.update_layout(title=_title("Growth of $1 (log scale, net of costs)"), yaxis_type="log", yaxis_title="wealth (×)", **LAYOUT)
    return fig


def drawdown_chart(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, label in (("net", "Strategy"), ("universe_ew", "EW universe"), ("sp500_total_return", "S&P 500")):
        if col not in monthly:
            continue
        w = (1 + monthly[col].fillna(0)).cumprod()
        dd = w / w.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=label, mode="lines",
                                 line=dict(color=COLORS[label], width=2 if label == "Strategy" else 1.2),
                                 fill="tozeroy" if label == "Strategy" else None,
                                 fillcolor="rgba(42,120,214,0.15)" if label == "Strategy" else None,
                                 hovertemplate="%{y:.1%}"))
    fig.update_layout(title=_title("Drawdown from peak"), yaxis_tickformat=".0%", **LAYOUT)
    return fig


def decile_chart(table: pd.DataFrame) -> go.Figure:
    t = table.reset_index()
    fig = go.Figure(go.Bar(x=[f"D{int(d)}" for d in t["decile"]], y=t["CAGR"], marker_color=COLORS["Strategy"],
                           text=[f"{v * 100:.1f}%" for v in t["CAGR"]], textposition="outside",
                           customdata=t["t vs EW universe"], hovertemplate="CAGR %{y:.1%}<br>t vs EW universe %{customdata:.2f}",
                           marker_line_width=0))
    fig.update_layout(title=_title("Annualized return by composite-score decile (1 = worst, 10 = best; EW, gross)"),
                      yaxis_tickformat=".0%", bargap=0.25, **{**LAYOUT, "hovermode": "closest"})
    return fig


def rolling_chart(rolling: pd.DataFrame, window_label: str) -> go.Figure:
    fig = go.Figure()
    for label in ("EW universe", "S&P 500"):
        if label in rolling:
            fig.add_trace(go.Scatter(x=rolling.index, y=rolling[label], name=f"vs {label}", mode="lines",
                                     line=dict(color=COLORS[label], width=1.8), hovertemplate="%{y:+.1%}/yr"))
    fig.add_hline(y=0, line=dict(color="#52514e", width=1))
    fig.update_layout(title=_title(f"Rolling {window_label} annualized excess return"), yaxis_tickformat="+.0%", **LAYOUT)
    return fig


def style_calendar(cal: pd.DataFrame):
    """Calendar-year table with outperformance vs each benchmark highlighted."""
    bench = [c for c in cal.columns if not c.startswith("vs ") and c not in ("Strategy", "months")]
    show = cal[["Strategy"] + bench].copy()
    styler = show.style.format("{:.1%}")

    def highlight(col):
        if col.name == "Strategy":
            return ["font-weight: 600"] * len(col)
        beat = cal[f"vs {col.name}"] > 0
        return ["background-color: #dff3e3; color: #0b0b0b" if b else "background-color: #fbe3e1; color: #0b0b0b" for b in beat]

    return styler.apply(highlight, axis=0)
