"""Price-based value factors.  All use market cap at the formation date."""
from __future__ import annotations

import pandas as pd

from .base import HIGHER, safe_div
from . import factor


@factor("book_to_market", "Book equity to market cap", "book equity / market cap", HIGHER,
        requires=("book_equity", "market_cap"),
        missing_policy="NaN when book equity <= 0 (a negative-equity firm is not 'cheap') or market cap <= 0",
        label="Book-to-market (book equity / market cap)", category="Value")
def book_to_market(s: pd.DataFrame) -> pd.Series:
    return safe_div(s["book_equity"].where(s["book_equity"] > 0), s["market_cap"])


@factor("earnings_to_price", "Trailing net income to market cap", "net income TTM / market cap", HIGHER,
        requires=("net_income_ttm", "market_cap"),
        missing_policy="negative earnings kept (rank worst); NaN when market cap <= 0",
        label="Earnings-to-price (net income / market cap)", category="Value")
def earnings_to_price(s: pd.DataFrame) -> pd.Series:
    return safe_div(s["net_income_ttm"], s["market_cap"])


@factor("sales_to_price", "Trailing revenue to market cap", "revenue TTM / market cap", HIGHER,
        requires=("revenue_ttm", "market_cap"), missing_policy="NaN when market cap <= 0",
        label="Sales-to-price (revenue / market cap)", category="Value")
def sales_to_price(s: pd.DataFrame) -> pd.Series:
    return safe_div(s["revenue_ttm"], s["market_cap"])


@factor("fcf_yield", "Free cash flow yield", "(operating cash flow TTM - capex TTM) / market cap", HIGHER,
        requires=("operating_cash_flow_ttm", "capex_ttm", "market_cap"),
        missing_policy="negative FCF kept; NaN when market cap <= 0 or either flow missing",
        label="Free cash flow yield", category="Value")
def fcf_yield(s: pd.DataFrame) -> pd.Series:
    return safe_div(s["operating_cash_flow_ttm"] - s["capex_ttm"], s["market_cap"])


@factor("shareholder_yield", "Dividends plus net buybacks over market cap",
        "(dividends TTM + (shares diluted a year ago - shares diluted now) * price) / market cap", HIGHER,
        requires=("dividends_ttm", "shares_diluted", "shares_diluted_prior", "price", "market_cap"),
        missing_policy="net issuance makes it negative (kept); NaN when prior-year share count is unavailable",
        label="Shareholder yield (dividends + buybacks)", category="Value")
def shareholder_yield(s: pd.DataFrame) -> pd.Series:
    net_buyback = (s["shares_diluted_prior"] - s["shares_diluted"]) * s["price"]
    return safe_div(s["dividends_ttm"] + net_buyback, s["market_cap"])
