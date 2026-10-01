"""Factor definition: a small function plus metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
import pandas as pd

HIGHER, LOWER = "higher", "lower"


@dataclass(frozen=True)
class Factor:
    name: str
    description: str
    formula: str
    direction: str                    # "higher" = higher is better, "lower" = lower is better
    requires: Tuple[str, ...]         # snapshot columns the function reads
    fn: Callable[[pd.DataFrame], pd.Series]
    missing_policy: str = ""          # human-readable note on NaN / sign handling
    label: str = ""                   # human-readable name for the UI, e.g. "Earnings yield (EBIT / EV)"
    category: str = "Other"           # Value / Quality / Price / ...

    @property
    def higher_is_better(self) -> bool:
        return self.direction == HIGHER

    @property
    def direction_text(self) -> str:
        return "higher is better" if self.higher_is_better else "lower is better"

    @property
    def display_name(self) -> str:
        return self.label or self.name.replace("_", " ").capitalize()

    def compute(self, snapshot: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.requires if c not in snapshot.columns]
        if missing:
            raise KeyError(f"factor {self.name!r} needs snapshot columns {missing}")
        out = pd.Series(self.fn(snapshot), index=snapshot.index, dtype="float64", name=self.name)
        return out.replace([np.inf, -np.inf], np.nan)

    def to_dict(self) -> dict:
        return {"name": self.name, "label": self.display_name, "category": self.category, "description": self.description,
                "formula": self.formula, "direction": self.direction_text, "requires": list(self.requires),
                "missing_policy": self.missing_policy}


def safe_div(num: pd.Series, den: pd.Series, require_positive_den: bool = True) -> pd.Series:
    """num / den with NaN where den is missing, zero, or (optionally) non-positive."""
    bad = den.isna() | (den == 0) | ((den <= 0) if require_positive_den else False)
    return (num / den.where(~bad)).astype("float64")
