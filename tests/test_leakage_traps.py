"""Permanent tripwires: deliberately broken variants must produce detectably inflated results.

Each test builds the *correct* composite and a *broken* one on the same panel and compares the
forward-return spread between the top and bottom decile (whole cross-section, quarterly formation,
3-month forward returns), which is precise enough to catch small leaks.  If a broken variant ever
stops looking inflated, either the generator lost its traps or the engine grew a leak of its own.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.backtest.schedule import rebalance_dates
from src.ranking import RankingConfig
from src.snapshot import build_snapshot, trailing_return_stats
from src.universe import select_universe
from tests.leakage_helpers import decile_spread, forward_returns, score_dates, top_decile_return

START, END = "1990-01-31", "2024-12-31"


@pytest.fixture(scope="module")
def panel(signal_source):
    r = signal_source.returns(START, END)
    f = signal_source.fundamentals(START, END)
    me = pd.DatetimeIndex(sorted(r["date"].unique()))
    return dict(r=r, f=f, dates=rebalance_dates(me, "quarterly"), fwd=forward_returns(r, 3))


@pytest.fixture(scope="module")
def correct(panel):
    return score_dates(panel["r"], panel["f"], panel["dates"])


def test_lookahead_trap_ranking_on_fiscal_period_end(panel, correct):
    """Ranking on fiscal_period_end sees each quarter 45-150 days before the market and books the
    announcement jumps.  This is the trap the whole point-in-time design exists to avoid."""
    r, f = panel["r"], panel["f"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broken = score_dates(r, f, panel["dates"], snapshot_fn=lambda T: build_snapshot(r, f, T, asof_col="fiscal_period_end"))
    good, bad = decile_spread(correct, panel["fwd"]), decile_spread(broken, panel["fwd"])
    assert bad > good * 1.25, (good, bad)
    assert top_decile_return(broken, panel["fwd"]) > top_decile_return(correct, panel["fwd"]) + 0.001


def test_forward_fill_trap_backfilling_missing_quarters(panel):
    """Filling a never-published quarter with the *next* period's numbers (on an on-time available_date)
    leaks the future for the names whose quarter went missing.

    The whole-cross-section spread barely moves (only ~2% of quarters are missing, and ranking on
    future values also mis-aligns with the published characteristic that drives expected returns), so
    this tripwire isolates the leak: among the stock-months whose snapshot actually changed, the sign of
    the leaked change must not predict forward returns.  It does, by ~1.8% per quarter."""
    r, f = panel["r"], panel["f"]
    first = f.drop_duplicates(["security_id", "fiscal_period_end"], keep="first")
    filled = []
    for sid, g in first.groupby("security_id"):
        g = g.set_index("fiscal_period_end").sort_index()
        gg = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="QE"))
        missing = gg["security_id"].isna()
        if not missing.any():
            continue
        gg = gg.bfill()                                             # <-- the bug: values from a later period
        gg.loc[missing, "available_date"] = gg.index[missing] + pd.Timedelta(days=60)
        filled.append(gg[missing].reset_index().rename(columns={"index": "fiscal_period_end"}))
    f_broken = pd.concat([f] + filled, ignore_index=True)
    assert len(f_broken) > len(f)
    me = pd.DatetimeIndex(sorted(r["date"].unique()))
    rows = []
    for T in rebalance_dates(me, "monthly"):
        good = select_universe(build_snapshot(r, f, T).set_index("security_id"))[["fiscal_period_end", "ebit_ttm"]]
        bad = build_snapshot(r, f_broken, T).set_index("security_id")[["fiscal_period_end", "ebit_ttm"]]
        j = good.join(bad, rsuffix="_broken")
        j = j[j["fiscal_period_end"] != j["fiscal_period_end_broken"]]
        rows.append(j.assign(date=T).reset_index())
    a = pd.concat(rows).merge(panel["fwd"], on=["security_id", "date"]).dropna()
    a["leak"] = np.log(a["ebit_ttm_broken"].clip(lower=1) / a["ebit_ttm"].clip(lower=1))
    up, down = a.loc[a["leak"] > 0, "fwd"], a.loc[a["leak"] < 0, "fwd"]
    assert len(up) > 500 and len(down) > 500
    assert up.mean() - down.mean() > 0.01, (up.mean(), down.mean())


def test_survivorship_trap_requiring_future_returns(panel, correct):
    """A universe rule 'must have 12 months of returns after T' silently deletes every bankruptcy."""
    r = panel["r"]
    last = r.groupby("security_id")["date"].max()

    def broken_universe(snap, T):
        uni = select_universe(snap)
        alive_next_year = last.reindex(uni.index) >= pd.Timestamp(T) + pd.DateOffset(months=12)
        return uni[alive_next_year.values]

    broken = score_dates(r, panel["f"], panel["dates"], universe_fn=broken_universe)
    assert len(broken) < len(correct)
    ew_good = correct.merge(panel["fwd"], on=["security_id", "date"])["fwd"].mean()
    ew_bad = broken.merge(panel["fwd"], on=["security_id", "date"])["fwd"].mean()
    assert ew_bad > ew_good + 0.001, (ew_good, ew_bad)             # inflated average stock return (~0.2%/quarter)
    # the bankruptcies are gone: no formation-date holding ever delists within the year
    held = broken.merge(r[r["is_delisted_this_month"]][["security_id", "date"]].rename(columns={"date": "death"}), on="security_id")
    assert not ((held["death"] > held["date"]) & (held["death"] < held["date"] + pd.DateOffset(months=12))).any()


def test_momentum_window_trap_extending_into_holding_period(panel):
    """Momentum whose window ends one month *after* the formation date contains the first holding
    month's return.  (In our convention formation is at month end T and returns start at T+1; a
    backtester that forms at the start of T and includes T's return in the signal has the same bug.)"""
    r, f = panel["r"], panel["f"]
    mom = RankingConfig({"momentum_12_1": 1.0})
    fwd1 = forward_returns(r, 1)

    def broken_snapshot(T):
        snap = build_snapshot(r, f, T)
        bad = trailing_return_stats(r, pd.Timestamp(T) + pd.offsets.MonthEnd(1), skip_last=0)   # window runs to T+1
        snap["mom_12_1"] = bad["mom_12_1"].reindex(snap["security_id"]).values
        return snap

    good = decile_spread(score_dates(r, f, panel["dates"], ranking=mom), fwd1)
    bad = decile_spread(score_dates(r, f, panel["dates"], ranking=mom, snapshot_fn=broken_snapshot), fwd1)
    assert abs(good) < 0.01                       # no momentum is planted, so the correct spread is ~0
    assert bad > 0.05, (good, bad)                # the broken one "predicts" the month it already contains

    # Including the formation month itself (12-0 instead of 12-1) is a definition choice, not a leak,
    # because returns are earned from T+1: it must NOT look inflated.
    def twelve_zero(T):
        snap = build_snapshot(r, f, T)
        snap["mom_12_1"] = trailing_return_stats(r, T, skip_last=0)["mom_12_1"].reindex(snap["security_id"]).values
        return snap

    same = decile_spread(score_dates(r, f, panel["dates"], ranking=mom, snapshot_fn=twelve_zero), fwd1)
    assert abs(same) < 0.01, same
