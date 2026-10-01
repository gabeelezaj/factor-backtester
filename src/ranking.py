"""Combine factor values into one composite ranking.

Two methods:

* ``rank_sum`` (Greenblatt): rank each factor with 1 = best, take the weighted sum of ranks, re-rank.
  ``score_raw`` uses weights scaled to sum to the number of factors, so with equal weights it is the
  plain sum of ranks (5th + 26th = 31); ``score`` uses weights that sum to 1 (a weighted average rank).
* ``zscore``: winsorize each factor at the 1st/99th percentile, standardize, flip lower-is-better
  factors, take the weighted sum, rank descending.

Missing values: ``drop`` (default) removes a stock that lacks any selected factor; ``median`` gives it
the median rank (rank_sum) or z = 0 (zscore).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from src.factors import get_factor

METHODS = ("rank_sum", "zscore")
MISSING_POLICIES = ("drop", "median")


@dataclass(frozen=True)
class RankingConfig:
    factors: Mapping[str, float]          # factor name -> weight (any positive scale)
    method: str = "rank_sum"
    missing_policy: str = "drop"
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.missing_policy not in MISSING_POLICIES:
            raise ValueError(f"missing_policy must be one of {MISSING_POLICIES}")
        if not self.factors:
            raise ValueError("at least one factor is required")
        if any(w <= 0 for w in self.factors.values()):
            raise ValueError("factor weights must be positive")
        for name in self.factors:
            get_factor(name)   # raises on unknown factor

    def normalized_weights(self) -> Dict[str, float]:
        total = float(sum(self.factors.values()))
        return {k: float(v) / total for k, v in self.factors.items()}

    @property
    def factor_names(self):
        return list(self.factors)


def rank_stocks(values: pd.DataFrame, cfg: RankingConfig) -> pd.DataFrame:
    """``values``: raw factor values (columns = factor names) indexed by security.

    Returns a frame sorted best-first with, per factor, ``<f>`` and ``<f>_rank`` (1 = best), plus
    ``score``, ``score_raw`` and ``rank`` (1 = best).  Stocks removed by the missing policy are absent.
    """
    w = cfg.normalized_weights()
    n_f = len(w)
    out = values[list(w)].copy()
    parts = {}
    if cfg.method == "rank_sum":
        for f in w:
            higher = get_factor(f).higher_is_better
            r = out[f].rank(ascending=not higher, method="average")
            if cfg.missing_policy == "median":
                r = r.fillna((r.count() + 1) / 2)
            out[f + "_rank"] = r
            parts[f] = r
        keep = pd.concat(parts, axis=1).notna().all(axis=1)
        out = out[keep]
        out["score"] = sum(w[f] * out[f + "_rank"] for f in w)
        out["score_raw"] = sum(w[f] * n_f * out[f + "_rank"] for f in w)
        ascending = True
    else:
        for f in w:
            x = out[f]
            lo, hi = x.quantile(cfg.winsor_lower), x.quantile(cfg.winsor_upper)
            xw = x.clip(lower=lo, upper=hi)
            sd = xw.std(ddof=1)
            z = (xw - xw.mean()) / sd if sd and sd > 0 else xw * 0.0
            if not get_factor(f).higher_is_better:
                z = -z
            if cfg.missing_policy == "median":
                z = z.fillna(0.0)
            out[f + "_z"] = z
            out[f + "_rank"] = z.rank(ascending=False, method="average")
            parts[f] = z
        keep = pd.concat(parts, axis=1).notna().all(axis=1)
        out = out[keep]
        out["score"] = sum(w[f] * out[f + "_z"] for f in w)
        out["score_raw"] = out["score"]
        ascending = False
    # deterministic tie-break: by score, then by index label
    out = out.sort_index().sort_values("score", ascending=ascending, kind="stable")
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def select_top(ranked: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return ranked[ranked["rank"] <= top_n]
