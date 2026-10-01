"""The synthetic panel contains every trap and the planted signal behaves."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import check_contract


def _decile_spread(source):
    """Next-month EW return by decile of the knowable characteristic at the formation month end.

    true_alpha at date T uses only fundamentals with available_date <= T and the market cap at T,
    so no lag is needed: this is exactly what a point-in-time strategy could rank on.
    """
    ta = source.true_alpha().copy()
    ta["date"] = ta["date"] + pd.offsets.MonthEnd(1)
    r = source.returns("1990-01-31", "2024-12-31")[["security_id", "date", "total_return"]]
    d = ta.merge(r, on=["security_id", "date"])
    d["dec"] = d.groupby("date")["true_alpha"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    dec = d.groupby("dec")["total_return"].mean()
    return dec


def test_contract(signal_source):
    rep = check_contract(signal_source, "1990-01-31", "2024-12-31")
    assert rep["returns_securities"] > 1500
    assert rep["benchmark_rows"] == 420


def test_panel_is_unbalanced(signal_source):
    r = signal_source.returns("1990-01-31", "2024-12-31")
    per_year = r.groupby(r["date"].dt.year)["security_id"].nunique()
    assert per_year.between(600, 1200).all(), per_year
    n_rows = r.groupby("security_id").size()
    assert (n_rows < 420).mean() > 0.5            # most companies do not span the whole sample
    assert r["is_delisted_this_month"].sum() > 800  # plenty of deaths
    assert r["security_id"].nunique() > per_year.max() * 1.5


def test_delisting_returns_are_extreme_and_final(signal_source):
    r = signal_source.returns("1990-01-31", "2024-12-31")
    dl = r[r["is_delisted_this_month"]]
    assert (dl["total_return"] <= -0.7).mean() > 0.25   # bankruptcies
    assert (dl["total_return"] >= 0.10).mean() > 0.25   # acquisitions
    last = r.groupby("security_id")["date"].max()
    assert (dl.set_index("security_id")["date"] == last.loc[dl["security_id"]].values).all()


def test_reporting_lag_45_to_90_days_with_late_and_restated(signal_source):
    f = signal_source.fundamentals("1990-01-31", "2024-12-31")
    first = f.drop_duplicates(["security_id", "fiscal_period_end"], keep="first")
    lag = (first["available_date"] - first["fiscal_period_end"]).dt.days
    assert lag.min() >= 45
    assert 60 <= lag.median() <= 80
    assert (lag <= 90).mean() > 0.9           # ~3% late filers beyond 90 days
    assert lag.max() > 100
    per_company = (first["available_date"] - first["fiscal_period_end"]).dt.days.groupby(first["security_id"]).median()
    assert per_company.std() > 5              # lag varies by company
    restated = f.duplicated(["security_id", "fiscal_period_end"], keep=False)
    assert 0.01 < restated.mean() < 0.08
    # restatement rows are published later than the original
    g = f[restated].sort_values(["security_id", "fiscal_period_end", "available_date"])
    assert (g.groupby(["security_id", "fiscal_period_end"])["available_date"].diff().dropna() > pd.Timedelta(0)).all()


def test_missing_quarters_exist(signal_source):
    f = signal_source.fundamentals("1990-01-31", "2024-12-31")
    first = f.drop_duplicates(["security_id", "fiscal_period_end"])
    gaps = first.groupby("security_id")["fiscal_period_end"].apply(lambda s: (s.diff().dt.days > 100).sum())
    assert gaps.sum() > 500


def test_cross_section_realism(signal_source):
    f = signal_source.fundamentals("1990-01-31", "2024-12-31")
    r = signal_source.returns("1990-01-31", "2024-12-31")
    assert 0.02 < (f["ebit_ttm"] < 0).mean() < 0.15
    assert 0.005 < (f["book_equity"] < 0).mean() < 0.10
    caps = r["market_cap"]
    assert caps.quantile(0.99) / caps.median() > 20          # long right tail
    sic = r.drop_duplicates("security_id")["sic_code"]
    assert 0.08 < sic.between(6000, 6999).mean() < 0.25      # financials
    assert 0.02 < sic.between(4900, 4999).mean() < 0.10      # utilities
    assert r["exchange"].isin(["NYSE", "AMEX", "NASDAQ"]).mean() > 0.9
    assert (r["exchange"] == "OTC").any()
    assert (r["price"] < 1).any() and (caps < 50e6).any()
    # a few extreme return-on-capital outliers
    cap = (f["current_assets"] - f["cash"] - (f["current_liabilities"] - f["short_term_debt"])).clip(lower=0) + f["net_ppe"]
    roc = f["ebit_ttm"] / cap.replace(0, np.nan)
    assert (roc.abs() > 5).sum() > 50


def test_planted_signal_recovered_from_true_alpha(signal_source):
    """Calibrated to real-world factor spreads: 0.3-0.5%/month top-minus-bottom decile (see DECISIONS.md)."""
    dec = _decile_spread(signal_source)
    rho, _ = spearmanr(dec.index, dec.values)
    # At a realistic strength a single 35-year panel resolves the tails clearly; the middle deciles
    # differ by ~0.01%/month, below the ~0.05%/month standard error of a decile mean.
    assert rho > 0.5, dec
    assert dec.iloc[-1] > dec.iloc[1:-1].max() and dec.iloc[0] < dec.iloc[1:-1].min(), dec
    spread = dec.iloc[-1] - dec.iloc[0]
    assert 0.003 < spread < 0.006, dec


def test_placebo_has_no_signal(placebo_source):
    dec = _decile_spread(placebo_source)
    assert abs(dec.iloc[-1] - dec.iloc[0]) < 0.0015, dec
    rho, _ = spearmanr(dec.index, dec.values)
    assert abs(rho) < 0.7


def test_true_alpha_is_knowable_at_formation_date(signal_source):
    """true_alpha at T must be reproducible from fundamentals with available_date <= T only."""
    f = signal_source.fundamentals("1990-01-31", "2024-12-31")
    r = signal_source.returns("1990-01-31", "2024-12-31")
    ta = signal_source.true_alpha()
    T = pd.Timestamp("2005-12-31")
    pub = f[f["available_date"] <= T].sort_values(["security_id", "fiscal_period_end", "available_date"])
    first = pub.drop_duplicates(["security_id", "fiscal_period_end"], keep="first")   # generator uses as-first-reported
    latest = first.groupby("security_id").tail(1).set_index("security_id")
    cs = r[r["date"] == T].set_index("security_id")
    j = cs.join(latest, how="inner")
    ev = j["market_cap"] + j["short_term_debt"] + j["long_term_debt"] + j["preferred_stock"] + j["minority_interest"] - j["cash"]
    cap = (j["current_assets"] - j["cash"] - (j["current_liabilities"] - j["short_term_debt"])).clip(lower=0) + j["net_ppe"]
    ey = (j["ebit_ttm"] / ev).where(ev > 0)
    roc = (j["ebit_ttm"] / cap).where(cap > 0)
    comp = 0.5 * ey.rank(pct=True) + 0.5 * roc.rank(pct=True)
    got = ta[ta["date"] == T].set_index("security_id")["true_alpha"]
    both = pd.concat([comp.rename("comp"), got.rename("got")], axis=1, join="inner").dropna()
    assert len(both) > 500
    rho, _ = spearmanr(both["comp"], both["got"])
    assert rho > 0.97, rho


def test_hazard_tilt_direction(signal_source, placebo_source):
    """Unprofitable firms die more when the signal is on; with signal off deaths are unrelated to quality."""
    def bankruptcy_rate_by_roc_tercile(src):
        f = src.fundamentals("1990-01-31", "2024-12-31").drop_duplicates(["security_id", "fiscal_period_end"])
        cap = (f["current_assets"] - f["cash"] - (f["current_liabilities"] - f["short_term_debt"])).clip(lower=0) + f["net_ppe"]
        f = f.assign(roc=(f["ebit_ttm"] / cap).where(cap > 0))
        med = f.groupby("security_id")["roc"].median()
        c = src.companies().set_index("security_id")
        c["roc_tercile"] = pd.qcut(med.reindex(c.index).rank(method="first"), 3, labels=["low", "mid", "high"])
        return c.groupby("roc_tercile", observed=True)["death_type"].apply(lambda s: (s == "bankruptcy").mean())
    on = bankruptcy_rate_by_roc_tercile(signal_source)
    off = bankruptcy_rate_by_roc_tercile(placebo_source)
    assert on["low"] > 1.5 * on["high"], on
    assert abs(off["low"] - off["high"]) < 0.05, off


def test_value_trap_regime_raises_bankruptcies_among_cheap_stocks():
    from src.data import get_source
    base = get_source("synthetic", seed=42, cache=False)
    trap = get_source("synthetic", seed=42, value_trap=0.5, cache=False)

    def bankruptcies_by_ey_tercile(src):
        f = src.fundamentals("1990-01-31", "2024-12-31").drop_duplicates(["security_id", "fiscal_period_end"])
        f = f.merge(src.returns("1990-01-31", "2024-12-31")[["security_id", "date", "market_cap"]],
                    left_on=["security_id", "fiscal_period_end"], right_on=["security_id", "date"])
        ev = f["market_cap"] + f["short_term_debt"] + f["long_term_debt"] + f["preferred_stock"] + f["minority_interest"] - f["cash"]
        f["ey"] = (f["ebit_ttm"] / ev).where(ev > 0)
        med = f.groupby("security_id")["ey"].median()
        c = src.companies().set_index("security_id")
        c["ey_tercile"] = pd.qcut(med.reindex(c.index).rank(method="first"), 3, labels=["expensive", "mid", "cheap"])
        return c.groupby("ey_tercile", observed=True)["death_type"].apply(lambda s: (s == "bankruptcy").mean())
    b, t = bankruptcies_by_ey_tercile(base), bankruptcies_by_ey_tercile(trap)
    assert t["cheap"] > b["cheap"] * 1.3, (b, t)
    assert _decile_spread(trap).pipe(lambda d: d.iloc[-1] - d.iloc[0]) < _decile_spread(base).pipe(lambda d: d.iloc[-1] - d.iloc[0])


def test_missing_quarters_concentrate_in_distressed_firms(signal_source):
    f = signal_source.fundamentals("1990-01-31", "2024-12-31").drop_duplicates(["security_id", "fiscal_period_end"])
    f = f.sort_values(["security_id", "fiscal_period_end"])
    f["gap_before"] = f.groupby("security_id")["fiscal_period_end"].diff().dt.days > 100
    distressed = (f["ebit_ttm"] < 0) | (f["book_equity"] < 0)
    assert f.loc[distressed, "gap_before"].mean() > 2 * f.loc[~distressed, "gap_before"].mean()


def test_survivorship_trap_data_level(signal_source):
    """Requiring 12 months of future returns deletes every bankruptcy and inflates average returns."""
    r = signal_source.returns("1990-01-31", "2024-12-31")
    r = r.sort_values(["security_id", "date"])
    last = r.groupby("security_id")["date"].transform("max")
    has_future_year = (last - r["date"]).dt.days > 360
    all_ew = r.groupby("date")["total_return"].mean().mean()
    survivor_ew = r[has_future_year].groupby("date")["total_return"].mean().mean()
    assert survivor_ew - all_ew > 0.0005, (survivor_ew, all_ew)   # > 0.6%/yr inflation of the average stock
    assert not r.loc[has_future_year, "is_delisted_this_month"].any()   # every death silently deleted


def test_lookahead_trap_has_teeth(signal_source):
    """Returns in the month results become public move with the (not yet public) EBIT change.

    A backtest that keys on fiscal_period_end would 'know' the change before the market does.
    """
    f = signal_source.fundamentals("1990-01-31", "2024-12-31")
    f = f.drop_duplicates(["security_id", "fiscal_period_end"], keep="first").sort_values(["security_id", "fiscal_period_end"])
    f["surprise"] = f.groupby("security_id")["ebit_ttm"].diff() / f["revenue_ttm"].abs().clip(lower=1)
    f["ann_month"] = f["available_date"] + pd.offsets.MonthEnd(0)
    r = signal_source.returns("1990-01-31", "2024-12-31")[["security_id", "date", "total_return"]]
    ann = f.merge(r, left_on=["security_id", "ann_month"], right_on=["security_id", "date"]).dropna(subset=["surprise"])
    corr_ann = ann["surprise"].clip(-0.3, 0.3).corr(ann["total_return"])
    # control: the same surprise against the return two months after publication
    f["ctrl_month"] = f["ann_month"] + pd.offsets.MonthEnd(2)
    ctrl = f.merge(r, left_on=["security_id", "ctrl_month"], right_on=["security_id", "date"]).dropna(subset=["surprise"])
    corr_ctrl = ctrl["surprise"].clip(-0.3, 0.3).corr(ctrl["total_return"])
    assert corr_ann > 0.08, corr_ann
    assert corr_ann > corr_ctrl + 0.05, (corr_ann, corr_ctrl)


def test_seed_reproducibility():
    from src.data import get_source
    a = get_source("synthetic", seed=7, years=3, n_companies=100, cache=False).returns("1990-01-31", "1992-12-31")
    b = get_source("synthetic", seed=7, years=3, n_companies=100, cache=False).returns("1990-01-31", "1992-12-31")
    pd.testing.assert_frame_equal(a, b)
