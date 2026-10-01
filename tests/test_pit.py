"""A ranking built at date T uses no fundamentals row with available_date after T."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import src.snapshot as snapshot_mod
from src.data.schema import FUNDAMENTALS_COLUMNS, coerce_fundamentals, coerce_returns
from src.pit import LookAheadWarning, asof_fundamentals
from src.snapshot import build_snapshot


def _fundamentals():
    rows = []
    for sid in ("A", "B"):
        for q in range(8):
            fpe = pd.Timestamp("2019-12-31") + pd.offsets.QuarterEnd(q)
            rows.append(dict(security_id=sid, fiscal_period_end=fpe, available_date=fpe + pd.Timedelta(days=60),
                             **{c: float(q + 1) for c, t in FUNDAMENTALS_COLUMNS.items() if t == "float64"}))
    return coerce_fundamentals(pd.DataFrame(rows))


def _returns():
    dates = pd.date_range("2019-01-31", periods=36, freq="ME")
    rows = [dict(security_id=sid, date=d, total_return=0.01, price=10.0, shares_outstanding=1e6, market_cap=1e7,
                 exchange="NYSE", sic_code=3570, is_delisted_this_month=False) for sid in ("A", "B") for d in dates]
    return coerce_returns(pd.DataFrame(rows))


def test_asof_picks_latest_public_period_and_ignores_future_rows():
    f = _fundamentals()
    # Q3-2020 (fpe 2020-09-30) is public 2020-11-29.  On 2020-11-30 it is the latest; on 2020-11-28 it is not.
    got = asof_fundamentals(f, "2020-11-30")
    assert (got["fiscal_period_end"] == pd.Timestamp("2020-09-30")).all()
    got = asof_fundamentals(f, "2020-11-28")
    assert (got["fiscal_period_end"] == pd.Timestamp("2020-06-30")).all()
    # a restatement of an older period published later does not override a newer period ...
    restated = f.iloc[[0]].copy()
    restated["available_date"] = pd.Timestamp("2021-06-01")
    restated["ebit_ttm"] = 999.0
    f2 = pd.concat([f, restated])
    got = asof_fundamentals(f2, "2021-06-30")
    assert got.loc[got.security_id == "A", "ebit_ttm"].item() == 6.0
    # ... but a restatement of the *latest* period is used once it is public, not before
    restated = f[(f.security_id == "A") & (f.fiscal_period_end == "2021-03-31")].copy()
    restated["available_date"] = pd.Timestamp("2021-07-15")
    restated["ebit_ttm"] = 999.0
    f3 = pd.concat([f, restated])
    assert asof_fundamentals(f3, "2021-07-14").query("security_id == 'A'")["ebit_ttm"].item() == 6.0
    assert asof_fundamentals(f3, "2021-07-15").query("security_id == 'A'")["ebit_ttm"].item() == 999.0


def test_snapshot_never_touches_rows_after_T(monkeypatch):
    """Spy on the choke point: every lookup is at or before T, on available_date, and the snapshot
    contains nothing dated after T."""
    f, r = _fundamentals(), _returns()
    T = pd.Timestamp("2020-11-30")
    # poison every row that is not yet public at T: if it leaks, a factor input becomes absurd
    poisoned = f.copy()
    poisoned.loc[poisoned["available_date"] > T, [c for c, t in FUNDAMENTALS_COLUMNS.items() if t == "float64"]] = 1e12
    calls = []
    real = snapshot_mod.asof_fundamentals

    def spy(fundamentals, formation_date, asof_col="available_date", max_staleness_days=548):
        calls.append((pd.Timestamp(formation_date), asof_col))
        return real(fundamentals, formation_date, asof_col, max_staleness_days)

    monkeypatch.setattr(snapshot_mod, "asof_fundamentals", spy)
    snap = build_snapshot(r, poisoned, T)
    assert calls and all(d <= T and col == "available_date" for d, col in calls)
    assert "available_date" not in snap.columns
    assert (snap["fiscal_period_end"] <= T).all()
    assert (snap["date"] == T).all()
    assert snap["ebit_ttm"].max() < 1e6                      # nothing poisoned leaked in
    assert (snap["fiscal_period_end"] == pd.Timestamp("2020-09-30")).all()


def test_snapshot_identical_with_future_data_removed():
    """Structural survivorship guard: the snapshot at T cannot depend on anything after T."""
    f, r = _fundamentals(), _returns()
    T = pd.Timestamp("2020-11-30")
    full = build_snapshot(r, f, T)
    truncated = build_snapshot(r[r["date"] <= T], f[f["available_date"] <= T], T)
    pd.testing.assert_frame_equal(full, truncated)


def test_broken_asof_column_warns():
    f = _fundamentals()
    with pytest.warns(LookAheadWarning):
        got = asof_fundamentals(f, "2020-09-30", asof_col="fiscal_period_end")
    assert (got["fiscal_period_end"] == pd.Timestamp("2020-09-30")).all()   # not public until 2020-11-29


def test_staleness_cutoff():
    f = _fundamentals()
    # last period ends 2021-09-30: 547 days later it is still usable, 577 days later it is stale
    assert len(asof_fundamentals(f, "2023-03-31")) == 2
    assert len(asof_fundamentals(f, "2023-04-30")) == 0
