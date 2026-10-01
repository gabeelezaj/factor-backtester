"""Return-based factors.  Inputs are computed by the snapshot builder from returns up to the formation date."""
from __future__ import annotations

import pandas as pd

from .base import HIGHER, LOWER
from . import factor


@factor("momentum_12_1", "12-month momentum skipping the most recent month", "cumulative return over months T-11..T-1", HIGHER,
        requires=("mom_12_1",), missing_policy="NaN with fewer than 9 of the 11 monthly returns",
        label="12-month momentum (skip last month)", category="Price")
def momentum_12_1(s: pd.DataFrame) -> pd.Series:
    return s["mom_12_1"]


@factor("volatility_12m", "Trailing 12-month return volatility (lower is better)", "annualized std of monthly returns T-11..T", LOWER,
        requires=("vol_12m",), missing_policy="NaN with fewer than 9 monthly returns",
        label="12-month volatility (lower is better)", category="Price")
def volatility_12m(s: pd.DataFrame) -> pd.Series:
    return s["vol_12m"]
