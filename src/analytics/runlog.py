"""Run log: every configuration tried, with its headline result, so the number of trials is visible.

The best of N trials is overstated.  ``multiple_testing_bar`` uses the stored null distribution to say
where the *best* of N zero-information runs would be expected to land, and what excess return a result
would need to stay significant after N trials.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_LOG = Path(__file__).resolve().parents[2] / "runs" / "run_log.jsonl"


def append(entry: Dict[str, Any], path: Path = DEFAULT_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def read(path: Path = DEFAULT_LOG) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def count(path: Path = DEFAULT_LOG, distinct: bool = True) -> int:
    df = read(path)
    if df.empty:
        return 0
    return int(df["config_hash"].nunique()) if distinct else int(len(df))


def log_run(config, headline: Dict[str, Any], source_name: str, path: Path = DEFAULT_LOG) -> int:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": config.name, "config_hash": config.hash(), "source": source_name,
        "config": config.to_dict(),
        "cagr_net": headline.get("cagr_net"), "sharpe": headline.get("sharpe"),
        "excess_vs_universe_ew": headline.get("excess_vs_universe_ew"),
        "null_percentile": (headline.get("null") or {}).get("percentile"),
        "fraction_of_oracle": headline.get("fraction_of_oracle"),
    }
    append(entry, path)
    return count(path)


def multiple_testing_bar(n_trials: int, null: Optional[Dict[str, Any]], metric: str = "excess_vs_universe_ew") -> Dict[str, Any]:
    """Expected best-of-N luck and the bar for 5% significance after N trials (Sidak on the null)."""
    if not null or metric not in null or n_trials < 1:
        return {"available": False, "n_trials": n_trials}
    vals = np.asarray(null[metric]["values"])
    expected_best_pct = 100 * n_trials / (n_trials + 1)
    bar_pct = 100 * (1 - 0.05) ** (1 / n_trials)     # per-trial percentile so that P(any of N exceeds) = 5%
    return {
        "available": True, "n_trials": n_trials,
        "expected_best_of_n": float(np.percentile(vals, expected_best_pct)),
        "significance_bar_after_n": float(np.percentile(vals, min(bar_pct, 100))),
        "single_trial_bar": float(np.percentile(vals, 95)),
        "warning": (f"{n_trials} configuration(s) tried. The best of {n_trials} zero-information runs would be expected "
                    f"near {np.percentile(vals, expected_best_pct) * 100:+.2f}%/yr over the EW universe; after {n_trials} "
                    f"trials a result needs about {np.percentile(vals, min(bar_pct, 100)) * 100:+.2f}%/yr "
                    f"(instead of {np.percentile(vals, 95) * 100:+.2f}%) to stay significant at 5%."),
    }
