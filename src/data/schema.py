"""Canonical internal schemas that every DataSource must return.

Nothing outside ``src/data`` should ever need to know where the data came
from: the engine consumes exactly these three tables.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Returns: one row per security-month
# ---------------------------------------------------------------------------
RETURNS_COLUMNS: Dict[str, str] = {
    "security_id": "str",            # stable identifier (permno / ticker / synthetic id)
    "date": "datetime64[ns]",        # month end
    "total_return": "float64",       # monthly total return incl. delisting return, decimal
    "price": "float64",              # month-end price (used for min-price filter)
    "shares_outstanding": "float64", # in shares
    "market_cap": "float64",         # price * shares, in the same currency unit as fundamentals
    "exchange": "str",               # 'NYSE' | 'AMEX' | 'NASDAQ' | other
    "sic_code": "int64",             # 4-digit SIC
    "is_delisted_this_month": "bool",  # True on the final row of a security
}

# ---------------------------------------------------------------------------
# Fundamentals: one row per security-fiscal period (restatements = extra rows)
# ---------------------------------------------------------------------------
FUNDAMENTALS_COLUMNS: Dict[str, str] = {
    "security_id": "str",
    "fiscal_period_end": "datetime64[ns]",
    "available_date": "datetime64[ns]",  # THE point-in-time column: when the numbers became public
    # trailing-twelve-month flows
    "ebit_ttm": "float64",
    "revenue_ttm": "float64",
    "gross_profit_ttm": "float64",
    "net_income_ttm": "float64",
    "operating_cash_flow_ttm": "float64",
    "capex_ttm": "float64",              # positive number = cash spent
    "dividends_ttm": "float64",          # positive number = cash paid
    # balance sheet, latest quarter
    "current_assets": "float64",
    "cash": "float64",
    "current_liabilities": "float64",
    "short_term_debt": "float64",
    "long_term_debt": "float64",
    "net_ppe": "float64",
    "total_assets": "float64",
    "book_equity": "float64",
    "preferred_stock": "float64",
    "minority_interest": "float64",
    "shares_diluted": "float64",
}

# ---------------------------------------------------------------------------
# Benchmarks: one row per month
# ---------------------------------------------------------------------------
BENCHMARK_COLUMNS: Dict[str, str] = {
    "date": "datetime64[ns]",
    "sp500_total_return": "float64",
    "market_vw_return": "float64",
    "market_ew_return": "float64",
    "risk_free_rate": "float64",     # monthly, decimal
}

MAJOR_EXCHANGES = ("NYSE", "AMEX", "NASDAQ")


class SchemaError(ValueError):
    """Raised when a DataSource returns a frame that violates the canonical schema."""


def _check_columns(df: pd.DataFrame, spec: Dict[str, str], name: str) -> None:
    missing = [c for c in spec if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}")
    for col, dtype in spec.items():
        actual = df[col].dtype
        if dtype == "str":
            if not (actual == object or pd.api.types.is_string_dtype(actual)):
                raise SchemaError(f"{name}.{col}: expected string dtype, got {actual}")
        elif dtype.startswith("datetime64"):
            if not pd.api.types.is_datetime64_any_dtype(actual):
                raise SchemaError(f"{name}.{col}: expected datetime dtype, got {actual}")
        elif dtype == "float64":
            if not pd.api.types.is_float_dtype(actual):
                raise SchemaError(f"{name}.{col}: expected float dtype, got {actual}")
        elif dtype == "int64":
            if not pd.api.types.is_integer_dtype(actual):
                raise SchemaError(f"{name}.{col}: expected integer dtype, got {actual}")
        elif dtype == "bool":
            if not pd.api.types.is_bool_dtype(actual):
                raise SchemaError(f"{name}.{col}: expected bool dtype, got {actual}")


def _check_month_end(s: pd.Series, name: str) -> None:
    if len(s) and not bool(s.dt.is_month_end.all()):
        bad = s[~s.dt.is_month_end].head(3).tolist()
        raise SchemaError(f"{name}: dates must be month ends, e.g. {bad}")


def validate_returns(df: pd.DataFrame) -> pd.DataFrame:
    _check_columns(df, RETURNS_COLUMNS, "returns")
    _check_month_end(df["date"], "returns.date")
    if df.duplicated(["security_id", "date"]).any():
        raise SchemaError("returns: duplicate (security_id, date) rows")
    # A security has at most one delisting row and it must be its last row.
    d = df[df["is_delisted_this_month"]]
    if d.duplicated("security_id").any():
        raise SchemaError("returns: a security is delisted more than once")
    last = df.groupby("security_id")["date"].max()
    if len(d) and not (d.set_index("security_id")["date"] == last.reindex(d["security_id"]).values).all():
        raise SchemaError("returns: delisting row is not the security's final row")
    return df


def validate_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    _check_columns(df, FUNDAMENTALS_COLUMNS, "fundamentals")
    if (df["available_date"] < df["fiscal_period_end"]).any():
        raise SchemaError("fundamentals: available_date precedes fiscal_period_end (impossible)")
    if df.duplicated(["security_id", "fiscal_period_end", "available_date"]).any():
        raise SchemaError("fundamentals: duplicate (security_id, fiscal_period_end, available_date)")
    return df


def validate_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    _check_columns(df, BENCHMARK_COLUMNS, "benchmarks")
    _check_month_end(df["date"], "benchmarks.date")
    if df["date"].duplicated().any():
        raise SchemaError("benchmarks: duplicate dates")
    return df


def coerce_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort dtype coercion adapters can call before validation."""
    out = df.copy()
    out["security_id"] = out["security_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"])
    for c in ("total_return", "price", "shares_outstanding", "market_cap"):
        out[c] = out[c].astype("float64")
    out["exchange"] = out["exchange"].astype(str)
    out["sic_code"] = out["sic_code"].fillna(0).astype("int64")
    out["is_delisted_this_month"] = out["is_delisted_this_month"].astype(bool)
    return out[list(RETURNS_COLUMNS)]


def coerce_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["security_id"] = out["security_id"].astype(str)
    out["fiscal_period_end"] = pd.to_datetime(out["fiscal_period_end"])
    out["available_date"] = pd.to_datetime(out["available_date"])
    for c, t in FUNDAMENTALS_COLUMNS.items():
        if t == "float64":
            out[c] = out[c].astype("float64") if c in out else np.nan
    return out[list(FUNDAMENTALS_COLUMNS)]


def coerce_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for c in BENCHMARK_COLUMNS:
        if c != "date":
            out[c] = out[c].astype("float64")
    return out[list(BENCHMARK_COLUMNS)]


def empty_returns() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=("object" if t == "str" else t)) for c, t in RETURNS_COLUMNS.items()})


def empty_fundamentals() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=("object" if t == "str" else t)) for c, t in FUNDAMENTALS_COLUMNS.items()})


def empty_benchmarks() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in BENCHMARK_COLUMNS.items()})


def schema_columns() -> Dict[str, List[str]]:
    return {
        "returns": list(RETURNS_COLUMNS),
        "fundamentals": list(FUNDAMENTALS_COLUMNS),
        "benchmarks": list(BENCHMARK_COLUMNS),
    }
