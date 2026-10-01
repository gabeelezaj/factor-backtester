"""Backtest engine: snapshot -> universe -> factors -> composite -> top N -> simulate.

The scoring stage (``build_universe_snapshots`` + ``rank_snapshots``) is separate from the simulation
stage (``portfolio.simulate``) so the UI can cache snapshots and re-rank cheaply, and so tests can feed
deliberately broken inputs to either stage.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.portfolio import holding_period_returns, simulate, target_weights
from src.backtest.result import BacktestResult
from src.backtest.schedule import rebalance_dates, sleeve_count
from src.config import BacktestConfig
from src.data import DataSource, get_source
from src.factors import compute_factors
from src.ranking import RankingConfig, rank_stocks
from src.snapshot import build_snapshot
from src.universe import UniverseConfig, select_universe

LOOKBACK_MONTHS = 13   # extra history loaded before `start` so momentum / prior-year lookups work
SNAPSHOT_KEEP = ["price", "market_cap", "exchange", "sic_code", "fiscal_period_end", "fundamentals_age_days"]


class PreparedData:
    """Everything loaded once per (source, start, end)."""

    def __init__(self, source: DataSource, start, end):
        self.source = source
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        load_from = self.start - pd.DateOffset(months=LOOKBACK_MONTHS)
        self.returns = source.returns(load_from, self.end)
        self.fundamentals = source.fundamentals(load_from, self.end)
        self.benchmarks = source.benchmarks(self.start, self.end).set_index("date")
        self.month_ends = pd.DatetimeIndex(sorted(self.returns.loc[self.returns["date"] >= self.start, "date"].unique()))
        self.R = self.returns.pivot(index="date", columns="security_id", values="total_return")
        self.caps = self.returns.pivot(index="date", columns="security_id", values="market_cap")
        self.metadata = source.get_metadata()


def build_universe_snapshots(data: PreparedData, dates: Iterable[pd.Timestamp], universe_cfg: UniverseConfig,
                             max_staleness_days: int, cache: Optional[Dict] = None) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Formation-date snapshots restricted to the investable universe.  ``cache`` (any dict) is reused across runs."""
    out = {}
    for T in dates:
        key = (T, universe_cfg, max_staleness_days)
        if cache is not None and key in cache:
            out[T] = cache[key]
            continue
        snap = build_snapshot(data.returns, data.fundamentals, T, max_staleness_days=max_staleness_days)
        snap = snap.set_index("security_id", drop=False)
        snap["n_alive"] = len(snap)
        uni = select_universe(snap, universe_cfg)
        if cache is not None:
            cache[key] = uni
        out[T] = uni
    return out


def rank_snapshots(snapshots: Mapping[pd.Timestamp, pd.DataFrame], ranking: RankingConfig) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    """Composite ranking per formation date, plus a funnel table."""
    ranked, stats = {}, []
    for T, uni in snapshots.items():
        vals = compute_factors(uni, ranking.factor_names)
        r = rank_stocks(vals, ranking)
        r = r.join(uni[SNAPSHOT_KEEP])
        r.index.name = "security_id"
        r.insert(0, "formation_date", T)
        ranked[T] = r
        stats.append(dict(formation_date=T, n_alive=int(uni["n_alive"].iloc[0]) if len(uni) else 0,
                          n_universe=len(uni), n_ranked=len(r), n_dropped_missing=len(uni) - len(r)))
    return ranked, pd.DataFrame(stats)


def holdings_from_rankings(ranked: Mapping[pd.Timestamp, pd.DataFrame], top_n: int, weighting: str) -> Dict[pd.Timestamp, pd.Series]:
    return {T: target_weights(r.index[r["rank"] <= top_n], r["market_cap"], weighting) for T, r in ranked.items()}


def universe_weights(snapshots: Mapping[pd.Timestamp, pd.DataFrame]) -> Dict[pd.Timestamp, pd.Series]:
    return {T: target_weights(uni.index, uni["market_cap"], "equal") for T, uni in snapshots.items()}


def _combine_sleeves(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Average across sleeves that are invested each month (capital is deployed one sleeve at a time)."""
    if len(frames) == 1:
        return frames[0]
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.union(f.index)
    stacked = {c: pd.concat([f[c].reindex(idx) for f in frames], axis=1) for c in frames[0].columns}
    out = pd.DataFrame(index=idx)
    for c in ("gross", "net", "cost", "universe_ew"):
        if c in stacked:
            out[c] = stacked[c].mean(axis=1)
    if "turnover" in stacked:
        out["turnover"] = stacked["turnover"].sum(axis=1, min_count=1) / len(frames)   # per unit of total capital
    if "n_holdings" in stacked:
        out["n_holdings"] = stacked["n_holdings"].sum(axis=1, min_count=1)
    return out


def run_backtest(cfg: BacktestConfig, source: Optional[DataSource] = None, data: Optional[PreparedData] = None,
                 snapshot_cache: Optional[Dict] = None, date_shift_months: int = 0) -> BacktestResult:
    """Run ``cfg``.  ``date_shift_months`` exists only for the cap-timing tripwire test (see tests)."""
    if data is None:
        source = source or get_source(cfg.data_source.name, **cfg.data_source.params)
        data = PreparedData(source, cfg.start, cfg.end)
    ranking = cfg.ranking()
    n_sleeves = sleeve_count(cfg.frequency) if cfg.sleeves else 1
    if len(data.month_ends) < 13:
        raise ValueError(f"backtest window {cfg.start}..{cfg.end} has only {len(data.month_ends)} months of data; "
                         "use at least two years")
    if len(rebalance_dates(data.month_ends, cfg.frequency, cfg.annual_month, 0, data.start, data.end)) == 0:
        raise ValueError(f"no {cfg.frequency} rebalance date falls inside {cfg.start}..{cfg.end} before its last month; "
                         "extend the window or change the rebalance month")
    monthly_frames, uni_frames, holdings_rows, ranking_rows, stats_frames = [], [], [], [], []
    for sleeve in range(n_sleeves):
        dates = rebalance_dates(data.month_ends, cfg.frequency, cfg.annual_month, sleeve, data.start, data.end)
        snaps = build_universe_snapshots(data, dates, cfg.universe, cfg.max_staleness_days, snapshot_cache)
        ranked, stats = rank_snapshots(snaps, ranking)
        targets = holdings_from_rankings(ranked, cfg.top_n, cfg.weighting)
        uni_targets = universe_weights(snaps)
        sim_targets, sim_uni = targets, uni_targets
        if date_shift_months:   # tripwire only: apply the T-dated portfolio at another month end
            shift = lambda d: {T + pd.offsets.MonthEnd(date_shift_months): w for T, w in d.items()}
            sim_targets, sim_uni = shift(targets), shift(uni_targets)
        monthly_frames.append(simulate(sim_targets, data.R, data.caps, cfg.costs))
        uni_frames.append(simulate(sim_uni, data.R, data.caps, cfg.costs)[["gross"]].rename(columns={"gross": "universe_ew"}))
        hp = holding_period_returns(targets, data.R)
        for T, r in ranked.items():
            r = r.copy()
            r.insert(1, "sleeve", sleeve)
            r["weight"] = targets[T].reindex(r.index)
            r["holding_return"] = hp[T].reindex(r.index) if T in hp else np.nan
            ranking_rows.append(r.reset_index())
        stats["sleeve"] = sleeve
        stats_frames.append(stats)
    monthly = _combine_sleeves(monthly_frames)
    monthly["universe_ew"] = _combine_sleeves(uni_frames)["universe_ew"] if n_sleeves > 1 else uni_frames[0]["universe_ew"]
    monthly = monthly.join(data.benchmarks, how="left")
    rankings = pd.concat(ranking_rows, ignore_index=True)
    holdings = rankings[rankings["weight"].notna()].reset_index(drop=True)
    return BacktestResult(cfg, monthly, holdings, rankings, pd.concat(stats_frames, ignore_index=True),
                          data.metadata, n_sleeves, data)


def strategy_frame(cfg: BacktestConfig, source: DataSource) -> pd.DataFrame:
    """Minimal output (net + universe_ew) used by the null-distribution builder."""
    return run_backtest(cfg, source).monthly[["net", "universe_ew"]]
