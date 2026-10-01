"""Metrics on series with known answers, decile analysis and the run log."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analytics import runlog
from src.analytics.deciles import decile_monthly_returns, decile_table
from src.analytics.metrics import (cagr, calendar_year_table, capm, hit_rates, max_drawdown, rolling_excess, sharpe,
                                   sortino, split_periods, summary_table, ann_vol)
from src.analytics.null_distribution import load_null
from src.analytics.report import run_report
from src.backtest.engine import run_backtest
from src.config import BacktestConfig


def _idx(n, start="2020-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


def test_cagr_vol_sharpe_sortino():
    r = pd.Series([0.01] * 24, index=_idx(24))
    assert cagr(r) == pytest.approx(1.01 ** 12 - 1)
    assert ann_vol(r) == 0.0
    rf = pd.Series(0.0, index=r.index)
    r2 = pd.Series([0.02, -0.01] * 12, index=_idx(24))
    assert sharpe(r2, rf) == pytest.approx(r2.mean() / r2.std(ddof=1) * np.sqrt(12))
    downside = np.sqrt((np.minimum(r2, 0) ** 2).mean())
    assert sortino(r2, rf) == pytest.approx(r2.mean() / downside * np.sqrt(12))


def test_max_drawdown_and_duration():
    r = pd.Series([0.10, -0.20, 0.05, 0.30, 0.01], index=_idx(5))
    mdd = max_drawdown(r)
    assert mdd["max_drawdown"] == pytest.approx(-0.20)        # peak 1.10 after month 1, trough 0.88
    assert mdd["peak"] == r.index[0] and mdd["trough"] == r.index[1]
    assert mdd["recovery"] == r.index[3]                       # 0.88 * 1.05 * 1.30 = 1.2012 > 1.10
    assert mdd["duration_months"] == 3 and mdd["recovered"]
    never = pd.Series([0.10, -0.30, 0.01, 0.01], index=_idx(4))
    m2 = max_drawdown(never)
    assert not m2["recovered"] and m2["duration_months"] == 3


def test_calendar_years_hit_rates_rolling():
    idx = _idx(24)
    m = pd.DataFrame({"net": 0.02, "universe_ew": 0.01, "market_ew_return": 0.01, "market_vw_return": 0.03,
                      "sp500_total_return": 0.03, "risk_free_rate": 0.0}, index=idx)
    cal = calendar_year_table(m)
    assert cal.loc[2020, "Strategy"] == pytest.approx(1.02 ** 12 - 1)
    assert cal.loc[2020, "vs EW universe"] == pytest.approx(1.02 ** 12 - 1.01 ** 12)
    hr = hit_rates(cal)
    assert hr["EW universe"] == 1.0 and hr["S&P 500"] == 0.0
    roll = rolling_excess(m, 12)
    assert len(roll) == 13
    assert roll["EW universe"].iloc[0] == pytest.approx(1.02 ** 12 - 1.01 ** 12)


def test_capm_recovers_known_alpha_beta():
    rng = np.random.default_rng(0)
    idx = _idx(240)
    rf = pd.Series(0.002, index=idx)
    mkt = pd.Series(rng.normal(0.006, 0.04, 240), index=idx) + rf
    strat = rf + 0.003 + 1.5 * (mkt - rf)              # exact: alpha 0.3%/month, beta 1.5
    m = pd.DataFrame({"net": strat, "market_vw_return": mkt, "risk_free_rate": rf})
    c = capm(m)
    assert c["beta"] == pytest.approx(1.5, abs=1e-9)
    assert c["alpha_annual"] == pytest.approx(0.036, abs=1e-9)
    assert c["r2"] == pytest.approx(1.0, abs=1e-9)


def test_split_periods_labels():
    idx = _idx(36)
    m = pd.DataFrame({"net": 0.01}, index=idx)
    p = split_periods(m, "2021-06-30")
    assert set(k.split(" (")[0] for k in p) == {"Full sample", "In-sample", "Out-of-sample"}
    assert len(p["In-sample (to 2021-06-30)"]) == 18 and len(p["Out-of-sample (after 2021-06-30)"]) == 18
    assert list(split_periods(m, None)) == ["Full sample"]


def test_summary_table_columns(signal_source):
    cfg = BacktestConfig.load("configs/magic_formula.yaml")
    res = run_backtest(cfg, signal_source)
    t = summary_table(res.monthly)
    assert list(t.index)[:2] == ["Strategy (net)", "Strategy (gross)"] and "S&P 500" in t.index
    assert t.loc["Strategy (net)", "CAGR"] < t.loc["Strategy (gross)", "CAGR"]


def test_decile_analysis_signal_vs_placebo(signal_source, placebo_source):
    cfg = BacktestConfig.load("configs/magic_formula.yaml")
    on = run_backtest(cfg, signal_source)
    d_on = decile_table(decile_monthly_returns(on, on.data.R, on.data.caps), on.monthly["universe_ew"])
    assert d_on["top_minus_bottom_annual"] > 0.03 and d_on["top_minus_bottom_t"] > 2.5
    assert d_on["table"]["CAGR"].idxmax() == 10 and d_on["table"]["CAGR"].idxmin() == 1
    off = run_backtest(BacktestConfig.load("configs/placebo.yaml"), placebo_source)
    d_off = decile_table(decile_monthly_returns(off, off.data.R, off.data.caps), off.monthly["universe_ew"])
    assert abs(d_off["top_minus_bottom_t"]) < 2


def test_run_log_and_multiple_testing(tmp_path, signal_source):
    log = tmp_path / "log.jsonl"
    cfg = BacktestConfig.load("configs/magic_formula.yaml")
    assert runlog.count(log) == 0
    n = runlog.log_run(cfg, {"cagr_net": 0.08, "excess_vs_universe_ew": 0.02, "null": {"percentile": 99}}, "x", log)
    assert n == 1 and runlog.count(log) == 1
    runlog.log_run(cfg, {"cagr_net": 0.08}, "x", log)                       # same config again
    cfg2 = BacktestConfig.from_dict({**cfg.to_dict(), "top_n": 50})
    runlog.log_run(cfg2, {"cagr_net": 0.07}, "x", log)
    assert runlog.count(log) == 2 and runlog.count(log, distinct=False) == 3
    null = load_null()
    bar1, bar20 = runlog.multiple_testing_bar(1, null), runlog.multiple_testing_bar(20, null)
    assert bar20["significance_bar_after_n"] > bar1["significance_bar_after_n"]
    assert bar20["expected_best_of_n"] > bar1["expected_best_of_n"]
    assert "20 configuration(s) tried" in bar20["warning"]


def test_full_report_runs(signal_source):
    cfg = BacktestConfig.load("configs/magic_formula.yaml")
    rep = run_report(run_backtest(cfg, signal_source), signal_source, log=False)
    for key in ("periods", "calendar", "hit_rates", "rolling_3y", "rolling_5y", "capm", "null", "deciles", "fraction_of_oracle", "run_log"):
        assert key in rep, key
    assert len(rep["periods"]) == 3


def test_verdict_statuses():
    from src.analytics.report import verdict
    good = {"excess_vs_universe_ew": 0.03, "t_vs_universe_ew": 2.5, "null": {"available": True, "percentile": 99, "n": 30},
            "deciles": {"top_minus_bottom_t": 3.0, "top_minus_bottom_annual": 0.05,
                        "table": pd.DataFrame({"CAGR": [0.02, 0.05, 0.09]}, index=[1, 5, 10])},
            "periods": {"Full": {"headline_excess": 0.03}, "In": {"headline_excess": 0.02}, "Out": {"headline_excess": 0.04}},
            "fraction_of_oracle": 1.0, "cost_drag_per_year": 0.003, "avg_turnover_per_year": 0.4}
    statuses = [s for s, _ in verdict(good, split=True)]
    assert statuses == ["good", "good", "good", "good", "good", "info"]
    bad = {**good, "excess_vs_universe_ew": -0.01, "t_vs_universe_ew": -0.5, "null": {"available": True, "percentile": 40, "n": 30},
           "deciles": {**good["deciles"], "top_minus_bottom_t": 0.3}, "fraction_of_oracle": 1.8,
           "periods": {"Full": {"headline_excess": -0.01}, "In": {"headline_excess": 0.02}, "Out": {"headline_excess": -0.03}},
           "cost_drag_per_year": 0.02}
    statuses = [s for s, _ in verdict(bad, split=True)]
    assert statuses == ["bad", "bad", "bad", "warn", "bad", "warn"]
    texts = " ".join(t for _, t in verdict(bad, split=True))
    assert "overfitting" in texts and "leaking" in texts
