"""Null distribution: what a zero-information strategy achieves by luck alone.

Built by running the complete pipeline on many independent zero-signal panels and recording the
annualized excess return over the equal-weighted universe.  Any strategy result is reported with
its percentile in this distribution.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.analytics.metrics import cagr, excess_cagr, sharpe, t_stat_excess

DEFAULT_NULL_PATH = Path(__file__).resolve().parents[2] / "validation" / "results" / "null_distribution.json"
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def run_null(source_factory: Callable[[int], Any], strategy: Callable[[Any], pd.DataFrame], seeds: Iterable[int],
             start, end, verbose: bool = True) -> pd.DataFrame:
    """``strategy(source)`` must return a frame indexed by date with columns ``net`` and ``universe_ew``."""
    rows = []
    for seed in seeds:
        src = source_factory(seed)
        res = strategy(src)
        b = src.benchmarks(start, end).set_index("date")
        row = dict(
            seed=seed,
            cagr=cagr(res["net"]),
            excess_vs_universe_ew=excess_cagr(res["net"], res["universe_ew"]),
            excess_vs_market_ew=excess_cagr(res["net"], b["market_ew_return"]),
            excess_vs_sp500=excess_cagr(res["net"], b["sp500_total_return"]),
            t_stat_vs_universe_ew=t_stat_excess(res["net"], res["universe_ew"]),
            sharpe=sharpe(res["net"], b["risk_free_rate"]),
        )
        rows.append(row)
        if verbose:
            print(f"seed {seed}: excess vs universe EW {row['excess_vs_universe_ew'] * 100:+.2f}%/yr  "
                  f"t={row['t_stat_vs_universe_ew']:+.2f}", flush=True)
    return pd.DataFrame(rows)


def summarize_null(df: pd.DataFrame, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": int(len(df)), "built": datetime.now(timezone.utc).isoformat(), "meta": meta or {}}
    for col in ("excess_vs_universe_ew", "excess_vs_market_ew", "excess_vs_sp500", "t_stat_vs_universe_ew", "sharpe"):
        v = df[col].dropna().values
        out[col] = {
            "values": [float(x) for x in v],
            "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
            "percentiles": {str(p): float(np.percentile(v, p)) for p in PERCENTILES},
        }
    return out


def save_null(summary: Dict[str, Any], path: Path = DEFAULT_NULL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1))
    return path


def load_null(path: Path = DEFAULT_NULL_PATH) -> Optional[Dict[str, Any]]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def placement(value: float, null: Optional[Dict[str, Any]], metric: str = "excess_vs_universe_ew") -> Dict[str, Any]:
    """Where ``value`` falls in the stored null: percentile rank (0-100), z-score, and a plain-English verdict."""
    if null is None or metric not in null or value is None or not np.isfinite(value):
        return {"available": False}
    vals = np.asarray(null[metric]["values"])
    pct = float((vals < value).mean() * 100)
    z = float((value - vals.mean()) / vals.std(ddof=1)) if len(vals) > 1 and vals.std(ddof=1) > 0 else np.nan
    if pct >= 99:
        verdict = "above the 99th percentile of luck"
    elif pct >= 95:
        verdict = "above the 95th percentile of luck"
    elif pct >= 90:
        verdict = "above the 90th percentile of luck (weak)"
    else:
        verdict = "inside the range of pure luck"
    meta = null.get("meta", {})
    return {"available": True, "percentile": pct, "z": z, "n": int(len(vals)), "verdict": verdict,
            "null_p95": null[metric]["percentiles"]["95"], "null_p99": null[metric]["percentiles"]["99"],
            "built_on": meta.get("panel", "unknown panel"), "strategy": meta.get("description", "")}
