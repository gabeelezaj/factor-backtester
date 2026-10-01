"""Transaction cost models.  Costs are one-way, in decimal, per unit of traded weight."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

# (market cap in $, one-way cost in bps).  Half-spread plus impact for a patient, modest-size trader.
# Anchored on effective-spread evidence: mega caps ~5 bps, $1B names ~20 bps, $50M microcaps ~80-100 bps.
DEFAULT_SIZE_ANCHORS: Tuple[Tuple[float, float], ...] = (
    (1e7, 150.0), (5e7, 80.0), (2.5e8, 40.0), (1e9, 20.0), (1e10, 8.0), (1e11, 5.0),
)


@dataclass(frozen=True)
class CostModel:
    kind: str = "size"            # "size" (default) or "flat"
    flat_bps: float = 20.0        # used when kind == "flat"
    size_anchors: Tuple[Tuple[float, float], ...] = DEFAULT_SIZE_ANCHORS
    multiplier: float = 1.0       # scale the whole curve (stress test)

    def one_way_cost(self, market_cap) -> np.ndarray:
        mc = np.asarray(pd.Series(market_cap, dtype="float64").values)
        if self.kind == "flat":
            bps = np.full(mc.shape, self.flat_bps)
        elif self.kind == "size":
            xs = np.log10([a[0] for a in self.size_anchors])
            ys = [a[1] for a in self.size_anchors]
            bps = np.interp(np.log10(np.clip(np.nan_to_num(mc, nan=1e7), 1e5, None)), xs, ys)
        else:
            raise ValueError(f"unknown cost model kind {self.kind!r}")
        return bps * self.multiplier / 1e4

    def describe(self) -> str:
        if self.kind == "flat":
            return f"flat {self.flat_bps * self.multiplier:.0f} bps one-way"
        pts = ", ".join(f"${a[0] / 1e6:,.0f}M:{a[1] * self.multiplier:.0f}bps" for a in self.size_anchors)
        return f"size-dependent one-way ({pts})"
