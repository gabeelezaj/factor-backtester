"""The one interface every market-data adapter implements.

The engine (factors, universe, ranking, backtest, analytics, UI) only ever
sees a ``DataSource``.  Switching providers is a config change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import pandas as pd

from .schema import validate_benchmarks, validate_fundamentals, validate_returns

DateLike = Union[str, pd.Timestamp]


class DataSource(ABC):
    """Abstract market-data provider.

    All three frame-returning methods must return the canonical schemas in
    :mod:`src.data.schema`.  ``start``/``end`` are inclusive month-end bounds
    (any parseable date is accepted; adapters should snap to month end).
    """

    #: short registry name, e.g. "synthetic"; set by subclasses
    name: str = "abstract"

    @abstractmethod
    def get_monthly_returns(self, start: DateLike, end: DateLike) -> pd.DataFrame:
        """One row per security-month, see ``RETURNS_COLUMNS``."""

    @abstractmethod
    def get_fundamentals(self, start: DateLike, end: DateLike) -> pd.DataFrame:
        """One row per security-fiscal period, see ``FUNDAMENTALS_COLUMNS``.

        ``start``/``end`` bound ``available_date`` so callers get every row
        that *became public* inside the window.  Adapters should include the
        last row available before ``start`` for each security (so a
        formation date early in the window still finds fundamentals).
        """

    @abstractmethod
    def get_benchmarks(self, start: DateLike, end: DateLike) -> pd.DataFrame:
        """One row per month, see ``BENCHMARK_COLUMNS``."""

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Describe the source honestly.  Required keys:

        name, display_name, coverage_start, coverage_end (ISO strings),
        is_simulated (bool), survivorship_bias (bool), caveats (list[str]).
        """

    # -- convenience wrappers the engine actually calls ----------------------
    def returns(self, start: DateLike, end: DateLike) -> pd.DataFrame:
        return validate_returns(self.get_monthly_returns(start, end))

    def fundamentals(self, start: DateLike, end: DateLike) -> pd.DataFrame:
        return validate_fundamentals(self.get_fundamentals(start, end))

    def benchmarks(self, start: DateLike, end: DateLike) -> pd.DataFrame:
        return validate_benchmarks(self.get_benchmarks(start, end))

    def describe(self) -> str:
        m = self.get_metadata()
        flag = "SIMULATED" if m.get("is_simulated") else ("SURVIVORSHIP-BIASED" if m.get("survivorship_bias") else "")
        return f"{m['display_name']} [{flag}] {m['coverage_start']}..{m['coverage_end']}"


REQUIRED_METADATA_KEYS = (
    "name", "display_name", "coverage_start", "coverage_end",
    "is_simulated", "survivorship_bias", "caveats",
)


def check_contract(source: DataSource, start: DateLike, end: DateLike) -> Dict[str, Any]:
    """Run every method of an adapter through the schema validators.

    Returns a small report dict; raises SchemaError / KeyError on violation.
    Used by tests and by the WRDS activation checklist.
    """
    meta = source.get_metadata()
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    if missing:
        raise KeyError(f"get_metadata() missing keys {missing}")
    r = source.returns(start, end)
    f = source.fundamentals(start, end)
    b = source.benchmarks(start, end)
    return {
        "metadata": meta,
        "returns_rows": len(r),
        "returns_securities": r["security_id"].nunique(),
        "fundamentals_rows": len(f),
        "fundamentals_securities": f["security_id"].nunique(),
        "benchmark_rows": len(b),
        "date_min": r["date"].min(),
        "date_max": r["date"].max(),
    }
