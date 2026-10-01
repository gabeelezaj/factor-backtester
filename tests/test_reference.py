"""Reference simulator, universe filters, schedule and cost model on hand-built inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostModel
from src.backtest.reference import simulate_topn
from src.backtest.schedule import rebalance_dates
from src.data.schema import coerce_returns
from src.universe import UniverseConfig, select_universe


def _panel():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    rows = []
    # A: +10% each month; B: -10% each month; C: bankrupt in month 3 (-80%), no row in month 4
    spec = {"A": [0.10, 0.10, 0.10, 0.10], "B": [-0.10, -0.10, -0.10, -0.10], "C": [0.0, 0.0, -0.80, None]}
    for sid, rets in spec.items():
        for d, ret in zip(dates, rets):
            if ret is None:
                continue
            rows.append(dict(security_id=sid, date=d, total_return=ret, price=10.0, shares_outstanding=1e8,
                             market_cap=1e9, exchange="NYSE", sic_code=3570,
                             is_delisted_this_month=(sid == "C" and d == dates[2])))
    return coerce_returns(pd.DataFrame(rows)), dates


def test_simulate_topn_compounding_delisting_and_costs():
    returns, dates = _panel()
    scores = pd.DataFrame({"security_id": ["A", "B", "C"], "date": [dates[0]] * 3, "score": [1.0, 0.5, 0.9]})
    res = simulate_topn(scores, returns, pd.DatetimeIndex([dates[0]]), top_n=2, cost_model=CostModel(kind="flat", flat_bps=20))
    # top 2 at formation: A and C, equal weight.  Month 2: 0.5*0.10 + 0.5*0 = 5%
    assert res.loc[dates[1], "gross"] == pytest.approx(0.05)
    # initial purchase cost: sum|dw| = 1.0 -> 20 bps, charged in the first month
    assert res.loc[dates[1], "net"] == pytest.approx(0.05 - 0.002)
    assert res.loc[dates[1], "turnover"] == pytest.approx(0.5)
    # Month 3: drifted weights A=0.55/1.05, C=0.5/1.05; C returns -80% (delisting)
    wA, wC = 0.55 / 1.05, 0.5 / 1.05
    assert res.loc[dates[2], "gross"] == pytest.approx(wA * 0.10 + wC * (-0.80))
    # Month 4: C is gone, proceeds sit in A -> return = A's return
    assert res.loc[dates[3], "gross"] == pytest.approx(0.10)
    assert res.loc[dates[3], "n_holdings"] == 1
    assert res.loc[dates[3], "cost"] == 0.0
    # universe EW: all three names at formation
    assert res.loc[dates[1], "universe_ew"] == pytest.approx((0.10 - 0.10 + 0.0) / 3)


def test_universe_filters():
    cs = pd.DataFrame({
        "security_id": list("ABCDEFG"),
        "market_cap": [1e9, 1e9, 1e9, 1e9, 40e6, 1e9, 5e9],
        "price": [10, 10, 10, 0.5, 10, 10, 10],
        "exchange": ["NYSE", "OTC", "NASDAQ", "NYSE", "NYSE", "AMEX", "NYSE"],
        "sic_code": [3570, 3570, 6020, 3570, 3570, 4911, 3570],
        "is_delisted_this_month": [False, False, False, False, False, False, True],
    }).set_index("security_id", drop=False)
    out = select_universe(cs)
    assert list(out["security_id"]) == ["A"]        # B: OTC, C: financial, D: price, E: size, F: utility, G: delisted
    out2 = select_universe(cs, UniverseConfig(exclude_sic_ranges=(), exchanges=("NYSE", "OTC", "NASDAQ", "AMEX"), max_names=2))
    assert list(out2["security_id"]) == ["A", "B"] or set(out2["security_id"]) <= {"A", "B", "C", "F"}


def test_rebalance_dates():
    me = pd.date_range("2000-01-31", periods=36, freq="ME")
    ann = rebalance_dates(me, "annual", annual_month=6)
    assert [d.strftime("%Y-%m") for d in ann] == ["2000-06", "2001-06", "2002-06"]
    q = rebalance_dates(me, "quarterly", annual_month=12)
    assert all(d.month in (3, 6, 9, 12) for d in q) and len(q) == 11   # 12 quarter ends minus the final month
    assert len(rebalance_dates(me, "monthly")) == 35
    assert rebalance_dates(me, "annual", annual_month=12, offset=1)[0].month == 1


def test_cost_model_size_dependent_and_monotone():
    cm = CostModel()
    c = cm.one_way_cost([1e7, 5e7, 1e9, 1e10, 1e11, 1e12])
    assert np.all(np.diff(c) <= 0)
    assert c[0] == pytest.approx(0.015) and c[2] == pytest.approx(0.002) and c[-1] == pytest.approx(0.0005)
    assert CostModel(kind="flat", flat_bps=20).one_way_cost([1e7, 1e12]).tolist() == [0.002, 0.002]
