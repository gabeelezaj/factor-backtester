"""A minimal adapter satisfies the DataSource contract; validators catch violations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    BENCHMARK_COLUMNS,
    FUNDAMENTALS_COLUMNS,
    RETURNS_COLUMNS,
    DataSource,
    SchemaError,
    check_contract,
    list_sources,
    validate_fundamentals,
    validate_returns,
)
from src.data.schema import coerce_benchmarks, coerce_fundamentals, coerce_returns


class DummySource(DataSource):
    """3 securities x 24 months, one of them delisted in month 18."""

    name = "dummy"

    def __init__(self):
        self.dates = pd.date_range("2020-01-31", periods=24, freq="ME")

    def get_monthly_returns(self, start, end):
        rows = []
        for sid, n in (("A", 24), ("B", 24), ("C", 18)):
            for i in range(n):
                rows.append(dict(
                    security_id=sid, date=self.dates[i], total_return=0.01 * (i % 3) - 0.01,
                    price=10.0 + i, shares_outstanding=1e6, market_cap=(10.0 + i) * 1e6,
                    exchange="NYSE", sic_code=3570, is_delisted_this_month=(sid == "C" and i == 17),
                ))
        return coerce_returns(pd.DataFrame(rows))

    def get_fundamentals(self, start, end):
        rows = []
        for sid in "ABC":
            for q in range(8):
                fpe = pd.Timestamp("2019-12-31") + pd.offsets.QuarterEnd(q)
                rows.append(dict(security_id=sid, fiscal_period_end=fpe, available_date=fpe + pd.Timedelta(days=60),
                                 **{c: 1.0 for c, t in FUNDAMENTALS_COLUMNS.items() if t == "float64"}))
        return coerce_fundamentals(pd.DataFrame(rows))

    def get_benchmarks(self, start, end):
        return coerce_benchmarks(pd.DataFrame({
            "date": self.dates, "sp500_total_return": 0.01, "market_vw_return": 0.01,
            "market_ew_return": 0.012, "risk_free_rate": 0.001,
        }))

    def get_metadata(self):
        return dict(name="dummy", display_name="Dummy", coverage_start="2020-01-31", coverage_end="2021-12-31",
                    is_simulated=True, survivorship_bias=False, caveats=["test double"])


def test_dummy_adapter_satisfies_contract():
    report = check_contract(DummySource(), "2020-01-31", "2021-12-31")
    assert report["returns_rows"] == 24 + 24 + 18
    assert report["returns_securities"] == 3
    assert report["fundamentals_rows"] == 24
    assert report["benchmark_rows"] == 24


def test_schema_column_lists_match_spec():
    assert list(RETURNS_COLUMNS) == [
        "security_id", "date", "total_return", "price", "shares_outstanding", "market_cap",
        "exchange", "sic_code", "is_delisted_this_month"]
    assert list(FUNDAMENTALS_COLUMNS) == [
        "security_id", "fiscal_period_end", "available_date", "ebit_ttm", "revenue_ttm", "gross_profit_ttm",
        "net_income_ttm", "operating_cash_flow_ttm", "capex_ttm", "dividends_ttm", "current_assets", "cash",
        "current_liabilities", "short_term_debt", "long_term_debt", "net_ppe", "total_assets", "book_equity",
        "preferred_stock", "minority_interest", "shares_diluted"]
    assert list(BENCHMARK_COLUMNS) == [
        "date", "sp500_total_return", "market_vw_return", "market_ew_return", "risk_free_rate"]


def test_validator_rejects_available_before_fiscal_end():
    f = DummySource().get_fundamentals(None, None)
    f.loc[0, "available_date"] = f.loc[0, "fiscal_period_end"] - pd.Timedelta(days=1)
    with pytest.raises(SchemaError):
        validate_fundamentals(f)


def test_validator_rejects_non_month_end_and_duplicates():
    r = DummySource().get_monthly_returns(None, None)
    bad = r.copy()
    bad.loc[0, "date"] = pd.Timestamp("2020-01-15")
    with pytest.raises(SchemaError):
        validate_returns(bad)
    dup = pd.concat([r, r.iloc[:1]])
    with pytest.raises(SchemaError):
        validate_returns(dup)


def test_validator_rejects_delisting_not_last_row():
    r = DummySource().get_monthly_returns(None, None)
    r.loc[(r.security_id == "A") & (r.date == r.date.min()), "is_delisted_this_month"] = True
    with pytest.raises(SchemaError):
        validate_returns(r)


def test_registry_names():
    assert set(list_sources()) == {"synthetic", "free", "wrds"}
