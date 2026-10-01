"""Engine correctness (vs the reference simulator), signal recovery, placebo, engine-level traps."""
from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import pytest

import src.backtest.engine as engine_mod
from src.analytics.metrics import cagr, excess_cagr, t_stat_excess
from src.analytics.null_distribution import load_null, placement
from src.analytics.oracle import fraction_of_oracle, run_oracle
from src.backtest.engine import PreparedData, run_backtest
from src.backtest.reference import simulate_topn
from src.backtest.schedule import rebalance_dates
from src.config import BacktestConfig
from src.snapshot import build_snapshot

MAGIC = BacktestConfig.load("configs/magic_formula.yaml")


@pytest.fixture(scope="module")
def magic(signal_source):
    return run_backtest(MAGIC, signal_source)


@pytest.fixture(scope="module")
def oracle(signal_source):
    return run_oracle(signal_source, MAGIC.start, MAGIC.end, MAGIC.top_n, MAGIC.frequency, MAGIC.annual_month,
                      MAGIC.universe, MAGIC.costs, MAGIC.weighting)


def test_engine_matches_reference_simulator(magic, signal_source):
    """Same scores in -> same gross/net/turnover/cost out, month by month."""
    rk = magic.rankings
    scores = pd.DataFrame({"security_id": rk["security_id"], "date": rk["formation_date"],
                           "score": rk.groupby("formation_date")["rank"].transform("max") + 1 - rk["rank"]})
    returns = signal_source.returns(pd.Timestamp(MAGIC.start) - pd.DateOffset(months=13), MAGIC.end)
    ref = simulate_topn(scores, returns, magic.formation_dates, MAGIC.top_n, MAGIC.universe, MAGIC.costs)
    common = ref.index.intersection(magic.monthly.index)
    assert len(common) == len(magic.monthly)
    for col in ("gross", "net", "cost", "universe_ew"):
        np.testing.assert_allclose(magic.monthly.loc[common, col], ref.loc[common, col], atol=1e-12, err_msg=col)
    np.testing.assert_allclose(magic.monthly.loc[common, "turnover"].fillna(-1), ref.loc[common, "turnover"].fillna(-1), atol=1e-12)


def test_magic_formula_recovers_planted_signal(magic, oracle):
    m = magic.monthly
    ex = excess_cagr(m["net"], m["universe_ew"])
    assert ex > 0.01, ex
    assert t_stat_excess(m["net"], m["universe_ew"]) > 1.5
    null = load_null()
    assert null is not None and placement(ex, null)["percentile"] >= 95
    frac = fraction_of_oracle(m["net"], oracle["net"], oracle["universe_ew"])
    assert 0.6 < frac < 1.4, frac
    # strategy and oracle hold ~90% the same names; their monthly difference must be pure noise
    d = (m["net"] - oracle["net"].reindex(m.index)).dropna()
    assert abs(d.mean() / d.std() * np.sqrt(len(d))) < 2.5


def test_placebo_shows_no_reliable_outperformance(placebo_source):
    cfg = BacktestConfig.load("configs/placebo.yaml")
    res = run_backtest(cfg, placebo_source)
    m = res.monthly
    t = t_stat_excess(m["net"], m["universe_ew"])
    assert abs(t) < 2.0, t
    o = run_oracle(placebo_source, cfg.start, cfg.end)
    assert abs(excess_cagr(o["net"], o["universe_ew"])) < 0.02          # even the oracle has nothing to find
    assert np.isnan(fraction_of_oracle(m["net"], o["net"], o["universe_ew"]))  # ratio correctly refused


def test_lookahead_trap_in_engine(signal_source, monkeypatch):
    """Ranking on fiscal_period_end through the whole engine: inflated, and flagged by the oracle ratio.

    Quarterly rebalancing is used because the leak lives in the first 1-3 months after each formation
    (the announcement jumps); with annual holding it dilutes to ~+0.5%/yr, quarterly it is ~+1.5%/yr."""
    cfg = BacktestConfig.from_dict({**MAGIC.to_dict(), "frequency": "quarterly", "name": "lookahead_trap"})
    good = run_backtest(cfg, signal_source)
    o = run_oracle(signal_source, cfg.start, cfg.end, cfg.top_n, cfg.frequency, cfg.annual_month, cfg.universe, cfg.costs)
    monkeypatch.setattr(engine_mod, "build_snapshot", functools.partial(build_snapshot, asof_col="fiscal_period_end"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broken = run_backtest(cfg, signal_source)
    assert cagr(broken.monthly["net"]) > cagr(good.monthly["net"]) + 0.01
    f_good = fraction_of_oracle(good.monthly["net"], o["net"], o["universe_ew"])
    f_bad = fraction_of_oracle(broken.monthly["net"], o["net"], o["universe_ew"])
    assert f_bad > f_good + 0.4 and f_bad > 1.5, (f_good, f_bad)


def test_cap_timing_trap(magic, signal_source):
    """Market cap at the end of the formation month while also earning that month's return.

    For price-based factors this *deflates* the result: a stock that fell during month T has a higher
    earnings yield at T, gets selected, and the broken variant books the fall it was selected on.
    Either sign is a leak; the tripwire is that the result moves materially."""
    broken = run_backtest(MAGIC, signal_source, date_shift_months=-1)
    good, bad = cagr(magic.monthly["net"]), cagr(broken.monthly["net"])
    assert abs(bad - good) > 0.005, (good, bad)
    assert bad < good


def test_turnover_costs_and_holdings_log(magic):
    m, h = magic.monthly, magic.holdings
    first = m.index[0]
    assert m.loc[first, "turnover"] == pytest.approx(0.5)                # all buys, half of sum|dw|
    assert 0 < m.loc[first, "cost"] < 0.01                                 # size-dependent one-way cost on 1.0 traded
    assert (m["cost"] >= 0).all() and m["cost"].gt(0).sum() == len(magic.formation_dates)
    assert m["turnover"].dropna().between(0, 1).all()
    assert m["net"].notna().all() and (m["net"] <= m["gross"]).all()
    assert m["n_holdings"].max() <= MAGIC.top_n and m["n_holdings"].min() >= MAGIC.top_n - 5
    # holdings: top_n rows per formation date, weights sum to 1, factor values and ranks present
    per = h.groupby("formation_date")
    assert (per.size() == MAGIC.top_n).all()
    np.testing.assert_allclose(per["weight"].sum(), 1.0)
    for col in ("earnings_yield", "return_on_capital", "earnings_yield_rank", "return_on_capital_rank", "score_raw",
                "rank", "market_cap", "price", "sic_code", "fiscal_period_end", "holding_return"):
        assert col in h.columns and h[col].notna().mean() > 0.95, col
    assert (h["rank"] <= MAGIC.top_n).all()
    assert len(magic.rankings) > len(h) * 10                              # full universe ranked, not just the top
    assert (magic.universe_stats["n_universe"] >= magic.universe_stats["n_ranked"]).all()
    assert not h["sic_code"].between(6000, 6999).any()                    # financials excluded


def test_other_configurations_run(signal_source):
    base = dict(start="2000-12-31", end="2006-12-31", data_source=MAGIC.data_source)
    for kw in (dict(frequency="quarterly", sleeves=True), dict(frequency="monthly", top_n=50, weighting="cap"),
               dict(ranking_method="zscore", missing_policy="median",
                    factors={"earnings_yield": 1, "momentum_12_1": 1, "volatility_12m": 1, "asset_growth": 1}),
               dict(costs=MAGIC.costs.__class__(kind="flat", flat_bps=20), annual_month=6)):
        cfg = BacktestConfig(name="smoke", **base, **kw)
        res = run_backtest(cfg, signal_source)
        m = res.monthly
        assert len(m) >= 60 and m["net"].notna().all() and m["universe_ew"].notna().all()
        assert res.n_sleeves == (3 if kw.get("sleeves") else 1)
        w = res.holdings.groupby(["formation_date", "sleeve"])["weight"].sum()
        np.testing.assert_allclose(w, 1.0)
        if kw.get("weighting") == "cap":
            h = res.holdings_at(res.formation_dates[0])
            assert h.sort_values("market_cap")["weight"].is_monotonic_increasing


def test_unusable_window_raises_clear_error(signal_source):
    base = MAGIC.to_dict()
    with pytest.raises(ValueError, match="at least two years"):
        run_backtest(BacktestConfig.from_dict({**base, "start": "2000-01-31", "end": "2000-12-31"}), signal_source)
    # the shortest usable window runs
    res = run_backtest(BacktestConfig.from_dict({**base, "start": "2000-01-31", "end": "2001-01-31", "split_date": None}), signal_source)
    assert len(res.monthly) == 1 and res.monthly["n_holdings"].iloc[0] == MAGIC.top_n
