"""Greenblatt's two Magic Formula factors, using his definitions."""
from __future__ import annotations

import pandas as pd

from .base import HIGHER, safe_div
from . import factor


def net_working_capital(s: pd.DataFrame) -> pd.Series:
    """(current assets - cash) - (current liabilities - short-term debt), floored at zero."""
    return ((s["current_assets"] - s["cash"]) - (s["current_liabilities"] - s["short_term_debt"])).clip(lower=0)


def enterprise_value(s: pd.DataFrame) -> pd.Series:
    return (s["market_cap"] + s["long_term_debt"] + s["short_term_debt"]
            + s["preferred_stock"] + s["minority_interest"] - s["cash"])


@factor("return_on_capital", "Greenblatt return on capital", "EBIT / (max(NWC, 0) + net PP&E)", HIGHER,
        requires=("ebit_ttm", "current_assets", "cash", "current_liabilities", "short_term_debt", "net_ppe"),
        missing_policy="NWC floored at 0; NaN when tangible capital <= 0 or any input missing; negative EBIT kept (ranks worst)",
        label="Return on capital (EBIT / tangible capital)", category="Quality (Greenblatt)")
def return_on_capital(s: pd.DataFrame) -> pd.Series:
    return safe_div(s["ebit_ttm"], net_working_capital(s) + s["net_ppe"])


@factor("earnings_yield", "Greenblatt earnings yield", "EBIT / (market cap + debt + preferred + minority - cash)", HIGHER,
        requires=("ebit_ttm", "market_cap", "long_term_debt", "short_term_debt", "preferred_stock", "minority_interest", "cash"),
        missing_policy="NaN when enterprise value <= 0 (cash-rich shells) or any input missing; negative EBIT kept",
        label="Earnings yield (EBIT / enterprise value)", category="Value (Greenblatt)")
def earnings_yield(s: pd.DataFrame) -> pd.Series:
    return safe_div(s["ebit_ttm"], enterprise_value(s))
