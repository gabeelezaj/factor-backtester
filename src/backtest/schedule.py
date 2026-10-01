"""Rebalance calendars."""
from __future__ import annotations

from typing import Optional

import pandas as pd

FREQUENCIES = ("annual", "quarterly", "monthly")


def rebalance_dates(month_ends: pd.DatetimeIndex, frequency: str = "annual", annual_month: int = 12,
                    offset: int = 0, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None,
                    ) -> pd.DatetimeIndex:
    """Formation dates (month ends) at which the portfolio is rebuilt.

    ``offset`` shifts the calendar by whole months (used for staggered sleeves).
    The last month end is never a formation date: there is nothing to hold afterwards.
    """
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}")
    d = month_ends
    if start is not None:
        d = d[d >= pd.Timestamp(start)]
    if end is not None:
        d = d[d <= pd.Timestamp(end)]
    if frequency == "monthly":
        chosen = d
    elif frequency == "quarterly":
        chosen = d[(d.month - annual_month - offset) % 3 == 0]
    else:
        chosen = d[(d.month - annual_month - offset) % 12 == 0]
    return chosen[chosen < month_ends[-1]]


def sleeve_count(frequency: str) -> int:
    return {"annual": 12, "quarterly": 3, "monthly": 1}[frequency]
