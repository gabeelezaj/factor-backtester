"""Data-source registry.  This is the ONLY place adapter names live.

Usage::

    from src.data import get_source
    src = get_source("synthetic", seed=42)

Adapters import their heavy optional dependencies lazily, so a missing
``wrds`` or ``yfinance`` package can never break another source.
"""
from __future__ import annotations

from typing import Any, Dict, List, Type

from .base import DataSource, check_contract
from .schema import (
    BENCHMARK_COLUMNS,
    FUNDAMENTALS_COLUMNS,
    RETURNS_COLUMNS,
    SchemaError,
    validate_benchmarks,
    validate_fundamentals,
    validate_returns,
)

DEFAULT_SOURCE = "synthetic"   # the only source that needs no credentials or network

_REGISTRY: Dict[str, str] = {
    # name -> "module:Class"; resolved lazily so optional deps stay optional
    "synthetic": "src.data.synthetic:SyntheticSource",
    "free": "src.data.free:FreeSource",
    "wrds": "src.data.wrds_source:WRDSSource",
}


def list_sources() -> List[str]:
    return list(_REGISTRY)


def source_class(name: str) -> Type[DataSource]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown data source {name!r}; known: {list_sources()}")
    module_path, cls_name = _REGISTRY[name].split(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def get_source(name: str, **params: Any) -> DataSource:
    return source_class(name)(**params)


def source_metadata_preview(name: str) -> Dict[str, Any]:
    """Metadata without constructing/loading data where the adapter supports it."""
    cls = source_class(name)
    fn = getattr(cls, "static_metadata", None)
    return fn() if fn else {"name": name, "display_name": name}


__all__ = [
    "DataSource", "check_contract", "get_source", "list_sources", "source_class", "DEFAULT_SOURCE",
    "RETURNS_COLUMNS", "FUNDAMENTALS_COLUMNS", "BENCHMARK_COLUMNS", "SchemaError",
    "validate_returns", "validate_fundamentals", "validate_benchmarks",
]
