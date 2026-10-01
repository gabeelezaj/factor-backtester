"""What a backtest produces."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import BacktestConfig


@dataclass
class BacktestResult:
    config: BacktestConfig
    monthly: pd.DataFrame          # index date: gross, net, turnover, cost, n_holdings, universe_ew, benchmarks, rf
    holdings: pd.DataFrame         # one row per (formation_date, sleeve, security) held, with factors/ranks/weights
    rankings: pd.DataFrame         # every ranked stock per formation date (for decile analysis)
    universe_stats: pd.DataFrame   # per formation date: counts through the funnel
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    n_sleeves: int = 1
    data: Any = field(default=None, repr=False, compare=False)   # PreparedData (wide returns/caps) for analytics

    @property
    def formation_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.holdings["formation_date"].unique()))

    def holdings_at(self, formation_date, sleeve: int = 0) -> pd.DataFrame:
        h = self.holdings
        return h[(h["formation_date"] == pd.Timestamp(formation_date)) & (h["sleeve"] == sleeve)].sort_values("rank")

    def to_csv(self, out_dir) -> Path:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.monthly.to_csv(d / "monthly.csv")
        self.holdings.to_csv(d / "holdings.csv", index=False)
        self.rankings.to_csv(d / "rankings.csv", index=False)
        self.universe_stats.to_csv(d / "universe_stats.csv", index=False)
        (d / "config.yaml").write_text(self.config.to_yaml())
        return d
