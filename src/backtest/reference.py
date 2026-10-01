"""Reference portfolio simulator: deliberately simple and slow, used to

* cross-check the production engine (same scores in -> same returns out),
* run the oracle portfolio (top N by the planted knowable characteristic),
* build the null distribution.

Conventions (shared with the engine, see DECISIONS.md):
* Formation at month end T uses only the cross-section at T; returns are earned from T+1.
* Weights drift with returns between rebalances.  A security that disappears (delisted: its final
  row carries the delisting return) has its weight removed at the start of the following month;
  proceeds are spread pro-rata over the survivors (equivalent to selling at the delisting price).
* Turnover = 0.5 * sum|w_new - w_drifted|.  Cost = sum|w_new - w_drifted| * one-way cost per name,
  charged in the first month after the rebalance.  The initial purchase is charged too.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel
from src.universe import UniverseConfig, select_universe


def _weights(ids: pd.Index, mcaps: pd.Series, weighting: str) -> pd.Series:
    if len(ids) == 0:
        return pd.Series(dtype="float64")
    if weighting == "cap":
        w = mcaps.loc[ids].astype(float)
        return w / w.sum()
    return pd.Series(1.0 / len(ids), index=ids)


def simulate_topn(scores: pd.DataFrame, returns: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, top_n: int = 30,
                  universe_cfg: UniverseConfig = UniverseConfig(), cost_model: Optional[CostModel] = None,
                  weighting: str = "equal", higher_is_better: bool = True) -> pd.DataFrame:
    """Simulate a top-N portfolio.  ``scores`` has columns security_id, date, score (date = formation month end).

    Returns a frame indexed by month end with columns: gross, net, turnover, cost, n_holdings, universe_ew.
    ``universe_ew`` is the equal-weighted return of the whole investable universe, rebalanced on the same dates.
    """
    cost_model = cost_model or CostModel()
    R = returns.pivot(index="date", columns="security_id", values="total_return")
    S = scores.pivot(index="date", columns="security_id", values="score")
    by_date = {d: g for d, g in returns.groupby("date")}
    dates = R.index
    rebal = set(pd.DatetimeIndex(rebalance_dates))
    w = pd.Series(dtype="float64")
    wu = pd.Series(dtype="float64")
    pending_cost = 0.0
    out = []
    for i in range(1, len(dates)):
        t_form, t = dates[i - 1], dates[i]
        turnover = np.nan
        if t_form in rebal:
            cs = by_date[t_form].set_index("security_id")
            uni = select_universe(cs, universe_cfg)
            sc = S.loc[t_form].reindex(uni.index).dropna() if t_form in S.index else pd.Series(dtype="float64")
            chosen = (sc.nlargest(top_n) if higher_is_better else sc.nsmallest(top_n)).index
            w_new = _weights(chosen, uni["market_cap"], weighting)
            wu_new = _weights(uni.index, uni["market_cap"], "equal")
            w_old = w.reindex(w_new.index.union(w.index)).fillna(0.0)
            delta = w_new.reindex(w_old.index).fillna(0.0) - w_old
            turnover = 0.5 * delta.abs().sum()
            pending_cost = float((delta.abs() * cost_model.one_way_cost(cs["market_cap"].reindex(delta.index))).sum())
            w, wu = w_new, wu_new
        if len(w) == 0:
            continue
        # names with no return row this month no longer exist (their final row carried the delisting
        # return): drop them now and spread the proceeds pro-rata over the survivors
        r = R.loc[t].reindex(w.index).dropna()
        w = w.reindex(r.index)
        w = w / w.sum() if w.sum() > 0 else w
        ru = R.loc[t].reindex(wu.index).dropna()
        wu = wu.reindex(ru.index)
        wu = wu / wu.sum() if wu.sum() > 0 else wu
        gross = float((w * r).sum())
        u_ret = float((wu * ru).sum())
        cost = pending_cost
        pending_cost = 0.0
        out.append(dict(date=t, gross=gross, net=gross - cost, turnover=turnover, cost=cost,
                        n_holdings=int(r.notna().sum()), universe_ew=u_ret))
        # drift
        w = w * (1 + r)
        w = w / w.sum() if w.sum() > 0 else w
        wu = wu * (1 + ru)
        wu = wu / wu.sum() if wu.sum() > 0 else wu
    return pd.DataFrame(out).set_index("date")
