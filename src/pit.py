"""THE point-in-time choke point.

Every fundamentals lookup in the project goes through :func:`asof_fundamentals`.  It keeps, for each
security, the latest fiscal period whose ``available_date`` is on or before the formation date
(restatements: the latest ``available_date`` for that period wins).  Factors never see
``available_date``; they receive the snapshot this produces.

``asof_col`` exists only so the test suite can build the deliberately broken look-ahead variant
(ranking on ``fiscal_period_end``).  Passing anything other than the default emits a loud warning.
"""
from __future__ import annotations

import warnings

import pandas as pd

ASOF_COLUMN = "available_date"
DEFAULT_MAX_STALENESS_DAYS = 548   # ~18 months: older fundamentals are treated as missing


class LookAheadWarning(UserWarning):
    pass


def asof_fundamentals(fundamentals: pd.DataFrame, formation_date, asof_col: str = ASOF_COLUMN,
                      max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS) -> pd.DataFrame:
    """One row per security: the latest fiscal period public as of ``formation_date``."""
    T = pd.Timestamp(formation_date)
    if asof_col != ASOF_COLUMN:
        warnings.warn(f"LOOK-AHEAD: fundamentals selected on {asof_col!r} instead of {ASOF_COLUMN!r}; "
                      "this is only valid inside the look-ahead trap test", LookAheadWarning, stacklevel=2)
    f = fundamentals[fundamentals[asof_col] <= T]
    if max_staleness_days is not None:
        f = f[f["fiscal_period_end"] >= T - pd.Timedelta(days=max_staleness_days)]
    f = f.sort_values(["security_id", "fiscal_period_end", ASOF_COLUMN])
    return f.groupby("security_id", sort=False).tail(1).reset_index(drop=True)
