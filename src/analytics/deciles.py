"""Decile analysis: sort the whole ranked universe into 10 equal-weight portfolios by composite score at
each formation date, hold to the next one.  A real signal rises fairly steadily from decile 1 (worst)
to decile 10 (best); a lucky top-30 does not."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analytics.metrics import cagr, t_stat_excess
from src.backtest.costs import CostModel
from src.backtest.portfolio import simulate, target_weights
from src.backtest.result import BacktestResult

N_DECILES = 10


def decile_assignments(rankings: pd.DataFrame, sleeve: int = 0) -> pd.DataFrame:
    r = rankings[rankings["sleeve"] == sleeve].copy()
    n = r.groupby("formation_date")["rank"].transform("max")
    # rank 1 = best -> decile 10; worst ranks -> decile 1
    r["decile"] = (N_DECILES - np.ceil(r["rank"] / n * N_DECILES) + 1).clip(1, N_DECILES).astype(int)
    return r


def decile_rankings(result: BacktestResult, frequency: str = "quarterly", snapshot_cache=None) -> pd.DataFrame:
    """Rankings at the decile cadence.  Reuses the strategy's own rankings when its calendar is at least
    as fine as ``frequency``; otherwise re-ranks the universe at the requested cadence (same universe,
    factors and method), because more formation dates give a far less noisy decile pattern."""
    from src.backtest.engine import build_universe_snapshots, rank_snapshots
    from src.backtest.schedule import rebalance_dates
    order = {"monthly": 0, "quarterly": 1, "annual": 2}
    cfg = result.config
    if order[cfg.frequency] <= order[frequency] or result.data is None:
        return result.rankings
    data = result.data
    dates = rebalance_dates(data.month_ends, frequency, cfg.annual_month, 0, data.start, data.end)
    snaps = build_universe_snapshots(data, dates, cfg.universe, cfg.max_staleness_days, snapshot_cache)
    ranked, _ = rank_snapshots(snaps, cfg.ranking())
    rows = []
    for T, r in ranked.items():
        r = r.copy()
        r.insert(1, "sleeve", 0)
        rows.append(r.reset_index())
    return pd.concat(rows, ignore_index=True)


def decile_monthly_returns(result: BacktestResult, R: pd.DataFrame, caps: pd.DataFrame, sleeve: int = 0,
                           frequency: str = "quarterly", snapshot_cache=None) -> pd.DataFrame:
    """Gross monthly return of each decile portfolio (no costs), columns 1..10."""
    a = decile_assignments(decile_rankings(result, frequency, snapshot_cache), sleeve)
    out = {}
    for d in range(1, N_DECILES + 1):
        sub = a[a["decile"] == d]
        targets = {T: target_weights(pd.Index(g["security_id"]), g.set_index("security_id")["market_cap"], "equal")
                   for T, g in sub.groupby("formation_date")}
        out[d] = simulate(targets, R, caps, CostModel(kind="flat", flat_bps=0.0))["gross"]
    return pd.DataFrame(out)


def decile_table(decile_returns: pd.DataFrame, universe_ew: pd.Series) -> Dict[str, object]:
    cols = list(decile_returns.columns)
    ann = pd.Series({d: cagr(decile_returns[d]) for d in cols}, name="CAGR")
    excess = pd.Series({d: cagr(decile_returns[d]) - cagr(universe_ew.reindex(decile_returns[d].dropna().index)) for d in cols}, name="Excess vs EW universe")
    tstat = pd.Series({d: t_stat_excess(decile_returns[d], universe_ew) for d in cols}, name="t vs EW universe")
    rho, _ = spearmanr(cols, ann.values)
    top, bottom = decile_returns[cols[-1]], decile_returns[cols[0]]
    return {
        "table": pd.DataFrame({"CAGR": ann, "Excess vs EW universe": excess, "t vs EW universe": tstat}).rename_axis("decile"),
        "spearman": float(rho),
        "top_minus_bottom_annual": float(cagr(top) - cagr(bottom)),
        "top_minus_bottom_t": t_stat_excess(top, bottom),
        "monotone_steps": int((ann.diff().dropna() > 0).sum()),
    }
