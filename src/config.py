"""Reproducible backtest configuration <-> YAML."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.backtest.costs import CostModel
from src.data import DEFAULT_SOURCE
from src.backtest.schedule import FREQUENCIES
from src.ranking import RankingConfig
from src.universe import UniverseConfig

WEIGHTINGS = ("equal", "cap")


@dataclass(frozen=True)
class DataSourceConfig:
    name: str = DEFAULT_SOURCE
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestConfig:
    name: str = "unnamed"
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    start: str = "1990-12-31"
    end: str = "2024-12-31"
    factors: Dict[str, float] = field(default_factory=lambda: {"return_on_capital": 0.5, "earnings_yield": 0.5})
    ranking_method: str = "rank_sum"
    missing_policy: str = "drop"
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    top_n: int = 30
    weighting: str = "equal"
    frequency: str = "annual"
    annual_month: int = 12
    sleeves: bool = False
    costs: CostModel = field(default_factory=CostModel)
    split_date: Optional[str] = None          # in-sample before, out-of-sample from this date
    max_staleness_days: int = 548
    notes: str = ""

    def __post_init__(self):
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")
        if self.frequency not in FREQUENCIES:
            raise ValueError(f"frequency must be one of {FREQUENCIES}")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        self.ranking()   # validates factors / method / policy

    def ranking(self) -> RankingConfig:
        return RankingConfig(self.factors, self.ranking_method, self.missing_policy)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["universe"]["exchanges"] = list(self.universe.exchanges)
        d["universe"]["exclude_sic_ranges"] = [list(x) for x in self.universe.exclude_sic_ranges]
        d["costs"]["size_anchors"] = [list(x) for x in self.costs.size_anchors]
        return d

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BacktestConfig":
        d = dict(d)
        ds = d.get("data_source", {})
        d["data_source"] = DataSourceConfig(ds.get("name", DEFAULT_SOURCE), dict(ds.get("params", {}) or {}))
        u = dict(d.get("universe", {}) or {})
        if "exchanges" in u:
            u["exchanges"] = tuple(u["exchanges"])
        if "exclude_sic_ranges" in u:
            u["exclude_sic_ranges"] = tuple(tuple(x) for x in u["exclude_sic_ranges"])
        d["universe"] = UniverseConfig(**u)
        c = dict(d.get("costs", {}) or {})
        if "size_anchors" in c:
            c["size_anchors"] = tuple(tuple(x) for x in c["size_anchors"])
        d["costs"] = CostModel(**c)
        return cls(**d)

    @classmethod
    def from_yaml(cls, text: str) -> "BacktestConfig":
        return cls.from_dict(yaml.safe_load(text) or {})

    @classmethod
    def load(cls, path) -> "BacktestConfig":
        return cls.from_yaml(Path(path).read_text())

    def save(self, path) -> None:
        Path(path).write_text(self.to_yaml())
