"""Factor registry.  Adding a factor is a decorated function in one of the modules below::

    @factor("my_factor", "one-line description", "formula", direction="higher",
            requires=("ebit_ttm", "market_cap"), missing_policy="NaN when market_cap <= 0")
    def my_factor(s):                      # s = snapshot DataFrame, return a Series
        return safe_div(s["ebit_ttm"], s["market_cap"])
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from .base import HIGHER, LOWER, Factor, safe_div

FACTOR_REGISTRY: Dict[str, Factor] = {}


def factor(name: str, description: str, formula: str, direction: str, requires: Iterable[str],
           missing_policy: str = "", label: str = "", category: str = "Other") -> Callable:
    if direction not in (HIGHER, LOWER):
        raise ValueError("direction must be 'higher' or 'lower'")

    def deco(fn: Callable[[pd.DataFrame], pd.Series]) -> Factor:
        f = Factor(name, description, formula, direction, tuple(requires), fn, missing_policy, label, category)
        if name in FACTOR_REGISTRY:
            raise ValueError(f"factor {name!r} already registered")
        FACTOR_REGISTRY[name] = f
        return f

    return deco


def get_factor(name: str) -> Factor:
    try:
        return FACTOR_REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown factor {name!r}; available: {list_factors()}") from None


def list_factors() -> List[str]:
    return list(FACTOR_REGISTRY)


def compute_factors(snapshot: pd.DataFrame, names: Iterable[str]) -> pd.DataFrame:
    """Factor values (raw, before any winsorizing/ranking), indexed like ``snapshot``."""
    return pd.DataFrame({n: get_factor(n).compute(snapshot) for n in names}, index=snapshot.index)


def factor_coverage(snapshot: pd.DataFrame, names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Missing-data rate per factor on a snapshot."""
    names = list(names) if names is not None else list_factors()
    vals = compute_factors(snapshot, names)
    return pd.DataFrame({
        "n_securities": len(snapshot),
        "n_available": vals.notna().sum(),
        "missing_rate": vals.isna().mean().round(4),
        "direction": [get_factor(n).direction for n in names],
    })


def factor_table() -> pd.DataFrame:
    return pd.DataFrame([f.to_dict() for f in FACTOR_REGISTRY.values()])


# register the built-in factors
from . import greenblatt, price, quality, value  # noqa: E402,F401

__all__ = ["Factor", "FACTOR_REGISTRY", "factor", "get_factor", "list_factors", "compute_factors",
           "factor_coverage", "factor_table", "safe_div", "HIGHER", "LOWER"]
