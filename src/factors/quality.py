"""Quality / accounting factors."""
from __future__ import annotations

import pandas as pd

from .base import HIGHER, LOWER, safe_div
from . import factor


@factor("gross_profitability", "Novy-Marx gross profitability", "gross profit TTM / total assets", HIGHER,
        requires=("gross_profit_ttm", "total_assets"), missing_policy="NaN when total assets <= 0",
        label="Gross profitability (gross profit / assets)", category="Quality")
def gross_profitability(s: pd.DataFrame) -> pd.Series:
    return safe_div(s["gross_profit_ttm"], s["total_assets"])


@factor("roe", "Return on equity", "net income TTM / book equity", HIGHER,
        requires=("net_income_ttm", "book_equity"),
        missing_policy="NaN when book equity <= 0 (a loss on negative equity would look like a high ROE)",
        label="Return on equity", category="Quality")
def roe(s: pd.DataFrame) -> pd.Series:
    return safe_div(s["net_income_ttm"], s["book_equity"])


@factor("accruals", "Sloan accruals (lower is better)", "(net income TTM - operating cash flow TTM) / total assets", LOWER,
        requires=("net_income_ttm", "operating_cash_flow_ttm", "total_assets"), missing_policy="NaN when total assets <= 0",
        label="Accruals (lower is better)", category="Quality")
def accruals(s: pd.DataFrame) -> pd.Series:
    return safe_div(s["net_income_ttm"] - s["operating_cash_flow_ttm"], s["total_assets"])


@factor("asset_growth", "Year-over-year growth in total assets (lower is better)", "total assets / total assets a year ago - 1", LOWER,
        requires=("total_assets", "total_assets_prior"),
        missing_policy="NaN when the prior-year point-in-time value is unavailable or <= 0",
        label="Asset growth (lower is better)", category="Quality")
def asset_growth(s: pd.DataFrame) -> pd.Series:
    return safe_div(s["total_assets"], s["total_assets_prior"]) - 1
