"""Tiny Parquet cache used by adapters (never by the engine)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache"


def config_hash(params: Dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


class ParquetCache:
    def __init__(self, source_name: str, root: Optional[Path] = None):
        self.dir = (root or CACHE_ROOT) / source_name
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.parquet"

    def get(self, key: str) -> Optional[pd.DataFrame]:
        p = self.path(key)
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        df.to_parquet(self.path(key), index=False)

    def clear(self) -> None:
        for p in self.dir.glob("*.parquet"):
            p.unlink()
