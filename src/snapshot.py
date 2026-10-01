"""Formation-date snapshot: one row per security with everything a factor may use.

Only this module joins returns to fundamentals, and it does so through :mod:`src.pit`.  The
snapshot deliberately drops ``available_date`` so no factor can key on it.

Columns:
* from the returns cross-section at T: security_id, date, price, shares_outstanding, market_cap,
  exchange, sic_code, is_delisted_this_month
* latest point-in-time fundamentals (all FUNDAMENTALS_COLUMNS except available_date) plus
  ``fundamentals_age_days`` = T - fiscal_period_end
* ``total_assets_prior``, ``shares_diluted_prior``: the same lookup done at T - 12 months
* ``mom_12_1``: cumulative return over the 11 months ending the month *before* T (skips month T)
* ``vol_12m``: annualized std of the 12 monthly returns ending at T (needs >= 9 observations)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import FUNDAMENTALS_COLUMNS
from src.pit import ASOF_COLUMN, DEFAULT_MAX_STALENESS_DAYS, asof_fundamentals

PRIOR_COLUMNS = {"total_assets": "total_assets_prior", "shares_diluted": "shares_diluted_prior"}
MIN_MOMENTUM_MONTHS = 9
MIN_VOL_MONTHS = 9


def trailing_return_stats(returns: pd.DataFrame, end_date, lookback_months: int = 12, skip_last: int = 1,
                          ) -> pd.DataFrame:
    """Momentum and volatility from the ``lookback_months`` monthly returns ending at ``end_date``.

    Momentum skips the last ``skip_last`` months (12-1 momentum = 11 returns).  Volatility uses all
    ``lookback_months``.  The window never extends past ``end_date``; the engine always passes the
    formation date.
    """
    T = pd.Timestamp(end_date)
    start = T - pd.DateOffset(months=lookback_months) + pd.offsets.MonthEnd(0)
    w = returns[(returns["date"] > start) & (returns["date"] <= T)]
    mom_end = T - pd.DateOffset(months=skip_last) + pd.offsets.MonthEnd(0) if skip_last else T
    wm = w[w["date"] <= mom_end]
    logs = np.log1p(wm["total_return"].clip(lower=-0.999)).groupby(wm["security_id"])
    mom = np.expm1(logs.sum()).where(logs.count() >= min(MIN_MOMENTUM_MONTHS, lookback_months - skip_last))
    gv = w.groupby("security_id")["total_return"]
    vol = (gv.std(ddof=1) * np.sqrt(12)).where(gv.count() >= MIN_VOL_MONTHS)
    return pd.DataFrame({"mom_12_1": mom, "vol_12m": vol})


def build_snapshot(returns: pd.DataFrame, fundamentals: pd.DataFrame, formation_date, asof_col: str = ASOF_COLUMN,
                   max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS) -> pd.DataFrame:
    T = pd.Timestamp(formation_date)
    cs = returns[returns["date"] == T].set_index("security_id")
    if cs.empty:
        raise ValueError(f"no returns rows at formation date {T.date()} (must be a month end in the data)")
    pit = asof_fundamentals(fundamentals, T, asof_col, max_staleness_days).set_index("security_id")
    pit = pit.drop(columns=[ASOF_COLUMN])
    prior = asof_fundamentals(fundamentals, T - pd.DateOffset(months=12), asof_col, max_staleness_days)
    prior = prior.set_index("security_id")[list(PRIOR_COLUMNS)].rename(columns=PRIOR_COLUMNS)
    trailing = trailing_return_stats(returns, T)
    snap = cs.join(pit, how="left").join(prior, how="left").join(trailing, how="left")
    snap["fundamentals_age_days"] = (T - snap["fiscal_period_end"]).dt.days
    snap["formation_date"] = T
    return snap.reset_index()


SNAPSHOT_COLUMNS = (
    ["security_id", "date", "price", "shares_outstanding", "market_cap", "exchange", "sic_code", "is_delisted_this_month"]
    + [c for c in FUNDAMENTALS_COLUMNS if c not in ("security_id", ASOF_COLUMN)]
    + list(PRIOR_COLUMNS.values()) + ["mom_12_1", "vol_12m", "fundamentals_age_days", "formation_date"]
)
