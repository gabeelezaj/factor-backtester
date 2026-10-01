"""Portfolio mechanics, vectorized.  Same conventions as the reference simulator (see DECISIONS.md):

* target weights set at formation date T are held from T+1;
* a name with no return row in a month is dropped at the start of that month (its final row already
  carried the delisting return) and its weight goes pro-rata to the survivors;
* weights drift with returns between rebalances;
* turnover = 0.5 * sum|target - drifted|, cost = sum|target - drifted| * one-way cost by market cap at T,
  charged in the first month after the rebalance (the initial purchase included).
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel


def target_weights(ids: pd.Index, market_caps: pd.Series, weighting: str) -> pd.Series:
    if len(ids) == 0:
        return pd.Series(dtype="float64")
    if weighting == "cap":
        w = market_caps.reindex(ids).astype(float)
        return w / w.sum()
    return pd.Series(1.0 / len(ids), index=ids)


def simulate(targets: Mapping[pd.Timestamp, pd.Series], R: pd.DataFrame, caps: pd.DataFrame,
             cost_model: CostModel) -> pd.DataFrame:
    """Run one sleeve.

    ``targets``: formation month end -> weights (index = security_id, sums to 1).
    ``R`` / ``caps``: wide monthly frames (index = month end, columns = security_id), NaN where not alive.
    Returns a frame indexed by month end with gross, net, turnover, cost, n_holdings.
    """
    cols = R.columns
    col_idx = {c: i for i, c in enumerate(cols)}
    Rm, Cm = R.to_numpy(dtype=float), caps.reindex(columns=cols).to_numpy(dtype=float)
    dates = R.index
    N = len(cols)
    w = np.zeros(N)
    pending_cost = 0.0
    invested = False
    out = []
    for i in range(1, len(dates)):
        t_form, t = dates[i - 1], dates[i]
        turnover = np.nan
        if t_form in targets:
            tw = np.zeros(N)
            s = targets[t_form]
            if len(s):
                tw[[col_idx[c] for c in s.index]] = s.to_numpy(dtype=float)
            delta = tw - w
            turnover = 0.5 * np.abs(delta).sum()
            pending_cost = float((np.abs(delta) * cost_model.one_way_cost(Cm[i - 1])).sum())
            w = tw
            invested = True
        if not invested:
            continue
        r = Rm[i]
        alive = ~np.isnan(r)
        w = np.where(alive, w, 0.0)
        tot = w.sum()
        if tot > 0:
            w = w / tot
        r0 = np.nan_to_num(r)
        gross = float((w * r0).sum())
        out.append(dict(date=t, gross=gross, net=gross - pending_cost, turnover=turnover, cost=pending_cost,
                        n_holdings=int(((w > 0) & alive).sum())))
        pending_cost = 0.0
        w = w * (1 + r0)
        tot = w.sum()
        if tot > 0:
            w = w / tot
    return pd.DataFrame(out).set_index("date") if out else pd.DataFrame(
        columns=["gross", "net", "turnover", "cost", "n_holdings"], index=pd.DatetimeIndex([], name="date"))


def holding_period_returns(targets: Mapping[pd.Timestamp, pd.Series], R: pd.DataFrame) -> Dict[pd.Timestamp, pd.Series]:
    """Per formation date: each holding's compounded return until the next formation date (or its death)."""
    dates = sorted(targets)
    idx = R.index
    out = {}
    for j, T in enumerate(dates):
        nxt = dates[j + 1] if j + 1 < len(dates) else idx[-1]
        window = R.loc[(idx > T) & (idx <= nxt), targets[T].index]
        out[T] = np.expm1(np.log1p(window.clip(lower=-0.999)).sum(axis=0, min_count=1))
    return out
