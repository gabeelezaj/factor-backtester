"""Shared machinery for the leakage tripwires: score the universe at formation dates, then measure
the forward return spread between the top and bottom decile of the composite.

The decile spread uses the whole cross-section, so it is far more precise than a 30-stock portfolio
and makes small leaks detectable.  ``forward_return`` compounds the next ``horizon`` monthly returns
(a delisted stock simply stops).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.factors import compute_factors
from src.ranking import RankingConfig, rank_stocks
from src.snapshot import build_snapshot
from src.universe import UniverseConfig, select_universe

MAGIC = RankingConfig({"earnings_yield": 0.5, "return_on_capital": 0.5})


def forward_returns(returns: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Per (security_id, date): compounded return over the next ``horizon`` months."""
    r = returns.sort_values(["security_id", "date"]).copy()
    r["l"] = np.log1p(r["total_return"].clip(lower=-0.999))
    g = r.groupby("security_id")["l"]
    fwd = sum(g.shift(-k) for k in range(1, horizon + 1))   # NaN if any of the next months is missing
    # allow a shorter window when the stock dies inside the horizon
    parts = [g.shift(-k) for k in range(1, horizon + 1)]
    stacked = pd.concat(parts, axis=1)
    fwd = np.expm1(stacked.sum(axis=1, min_count=1))
    return r.assign(fwd=fwd.values)[["security_id", "date", "fwd"]]


def score_dates(returns: pd.DataFrame, fundamentals: pd.DataFrame, dates, snapshot_fn: Optional[Callable] = None,
                ranking: RankingConfig = MAGIC, universe_cfg: UniverseConfig = UniverseConfig(),
                universe_fn: Optional[Callable] = None) -> pd.DataFrame:
    """Composite score (higher = better) for every universe stock at each formation date."""
    snapshot_fn = snapshot_fn or (lambda T: build_snapshot(returns, fundamentals, T))
    out = []
    for T in dates:
        snap = snapshot_fn(T).set_index("security_id", drop=False)
        uni = universe_fn(snap, T) if universe_fn else select_universe(snap, universe_cfg)
        vals = compute_factors(uni, ranking.factor_names)
        ranked = rank_stocks(vals, ranking)
        out.append(pd.DataFrame({"security_id": ranked.index, "date": pd.Timestamp(T),
                                 "score": (len(ranked) + 1 - ranked["rank"]).values}))
    return pd.concat(out, ignore_index=True)


def decile_spread(scores: pd.DataFrame, fwd: pd.DataFrame) -> float:
    """Mean forward return of the top decile minus the bottom decile (decimal)."""
    d = scores.merge(fwd, on=["security_id", "date"]).dropna(subset=["fwd"])
    d["dec"] = d.groupby("date")["score"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    by = d.groupby("dec")["fwd"].mean()
    return float(by.iloc[-1] - by.iloc[0])


def top_decile_return(scores: pd.DataFrame, fwd: pd.DataFrame) -> float:
    d = scores.merge(fwd, on=["security_id", "date"]).dropna(subset=["fwd"])
    d["dec"] = d.groupby("date")["score"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    return float(d[d["dec"] == 9]["fwd"].mean())
