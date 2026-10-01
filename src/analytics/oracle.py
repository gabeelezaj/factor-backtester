"""Oracle upper bound.

When a data source exposes a planted, point-in-time-knowable signal (``source.true_alpha()``),
the best any point-in-time strategy can do is to rank on that signal directly.  Every strategy
result is then expressed as a fraction of the oracle's excess return over the equal-weighted
universe: ~0 means the pipeline lost the signal (bug); >1 means information leaked.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.analytics.metrics import excess_cagr
from src.backtest.costs import CostModel
from src.backtest.reference import simulate_topn
from src.backtest.schedule import rebalance_dates
from src.universe import UniverseConfig


def oracle_available(source) -> bool:
    return callable(getattr(source, "true_alpha", None))


def run_oracle(source, start, end, top_n: int = 30, frequency: str = "annual", annual_month: int = 12,
               universe_cfg: UniverseConfig = UniverseConfig(), cost_model: Optional[CostModel] = None,
               weighting: str = "equal") -> Optional[pd.DataFrame]:
    """Top-N portfolio on the knowable planted signal, via the reference simulator.  None if unavailable."""
    if not oracle_available(source):
        return None
    scores = source.true_alpha().rename(columns={"true_alpha": "score"})
    returns = source.returns(start, end)
    month_ends = pd.DatetimeIndex(sorted(returns["date"].unique()))
    dates = rebalance_dates(month_ends, frequency, annual_month)
    return simulate_topn(scores, returns, dates, top_n, universe_cfg, cost_model, weighting)


MIN_ORACLE_EDGE = 0.005   # below 0.5%/yr the oracle itself is inside the luck band and the ratio is meaningless


def fraction_of_oracle(strategy_net: pd.Series, oracle_net: pd.Series, universe_ew: pd.Series) -> float:
    """Strategy excess CAGR over the EW universe divided by the oracle's.

    NaN when the oracle's own edge is below ``MIN_ORACLE_EDGE`` (e.g. a placebo panel): report the
    difference in %/yr instead.  Because strategy and oracle rank on nearly the same characteristic
    on the same panel, ratios within roughly 0.7-1.3 are noise; a ratio well above that is a leak.
    """
    o = excess_cagr(oracle_net, universe_ew)
    s = excess_cagr(strategy_net, universe_ew)
    return float(s / o) if o > MIN_ORACLE_EDGE else np.nan
