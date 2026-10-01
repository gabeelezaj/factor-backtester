#!/usr/bin/env python3
"""Run a backtest from a YAML config:  python run_backtest.py configs/magic_formula.yaml [--out runs/]"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.analytics.report import print_report, run_report
from src.backtest.engine import run_backtest
from src.config import BacktestConfig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", help="YAML config, e.g. configs/magic_formula.yaml")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-log", action="store_true", help="do not append to the run log")
    ap.add_argument("--factors", action="store_true", help="list the available factors and exit")
    args = ap.parse_args(argv)
    if args.factors:
        import pandas as pd
        from src.factors import factor_table
        pd.set_option("display.width", 220)
        pd.set_option("display.max_colwidth", 70)
        print(factor_table()[["name", "label", "direction", "formula", "missing_policy"]].to_string(index=False))
        return 0
    if not args.config:
        ap.error("a config path is required (or use --factors)")
    cfg = BacktestConfig.load(args.config)
    t0 = time.time()
    result = run_backtest(cfg)
    elapsed = time.time() - t0
    report = run_report(result, log=not args.no_log)
    out = result.to_csv(Path(args.out) / f"{cfg.name}_{cfg.hash()}")
    report["calendar"].to_csv(out / "calendar_years.csv")
    if "decile_returns" in report:
        report["decile_returns"].to_csv(out / "decile_returns.csv")
        report["deciles"]["table"].to_csv(out / "decile_table.csv")
    print_report(report, result)
    print(f"\nran in {elapsed:.1f}s; results written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
