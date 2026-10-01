"""Investable universe at a formation date.

Operates on a *cross-section*: one row per security from the returns table at
the formation date.  It never sees anything after that date because the
snapshot builder only hands it rows with ``date == formation_date``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import pandas as pd

from src.data.schema import MAJOR_EXCHANGES


@dataclass(frozen=True)
class UniverseConfig:
    max_names: int = 5000
    min_market_cap: float = 50e6
    min_price: float = 1.0
    exchanges: Tuple[str, ...] = MAJOR_EXCHANGES
    # Greenblatt excludes financials and utilities: EBIT/EV and ROC are not meaningful for them
    exclude_sic_ranges: Tuple[Tuple[int, int], ...] = ((6000, 6999), (4900, 4999))


def select_universe(cross_section: pd.DataFrame, cfg: UniverseConfig = UniverseConfig()) -> pd.DataFrame:
    """Apply size / price / exchange / sector filters and keep the largest ``max_names`` by market cap.

    Securities delisted in the formation month are excluded: they have no next-month return to earn.
    """
    cs = cross_section
    keep = (
        ~cs["is_delisted_this_month"]
        & (cs["market_cap"] >= cfg.min_market_cap)
        & (cs["price"] >= cfg.min_price)
        & cs["exchange"].isin(cfg.exchanges)
    )
    for lo, hi in cfg.exclude_sic_ranges:
        keep &= ~cs["sic_code"].between(lo, hi)
    return cs[keep].nlargest(cfg.max_names, "market_cap")
