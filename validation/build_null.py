"""Build the stored null distribution from independent zero-signal panels.

Usage: python -m validation.build_null [--seeds 30] [--strategy reference|engine]

The default strategy is the production engine running configs/magic_formula.yaml (the complete
pipeline).  ``--strategy reference`` uses the reference simulator on the planted characteristic
instead (the interim null used before the engine existed).  The JSON records which one was used.
"""
from __future__ import annotations

import argparse

from src.analytics.null_distribution import DEFAULT_NULL_PATH, run_null, save_null, summarize_null
from src.analytics.oracle import run_oracle
from src.data import get_source

START, END = "1990-01-31", "2024-12-31"


def reference_strategy(source):
    return run_oracle(source, START, END, top_n=30, frequency="annual", annual_month=12)


def engine_strategy(source):
    """The complete pipeline: PIT snapshot -> universe -> EY+ROC rank-sum -> top 30 EW -> annual, size costs."""
    from src.backtest.engine import strategy_frame
    from src.config import BacktestConfig
    return strategy_frame(BacktestConfig.load("configs/magic_formula.yaml"), source)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--strategy", choices=["reference", "engine"], default="engine")
    args = ap.parse_args()
    strategy = reference_strategy if args.strategy == "reference" else engine_strategy
    factory = lambda seed: get_source("synthetic", seed=seed, signal_strength=0.0, cache=False)
    df = run_null(factory, strategy, range(args.seed0, args.seed0 + args.seeds), START, END)
    meta = {
        "strategy": args.strategy,
        "description": "top 30 equal-weight, annual December rebalance, default universe filters, size-dependent costs",
        "panel": "synthetic 800 companies, 1990-2024, signal_strength=0",
        "seeds": f"{args.seed0}..{args.seed0 + args.seeds - 1}",
    }
    summary = summarize_null(df, meta)
    path = save_null(summary, DEFAULT_NULL_PATH)
    df.to_csv(path.with_suffix(".csv"), index=False)
    p = summary["excess_vs_universe_ew"]["percentiles"]
    print(f"\nsaved {path}  (n={summary['n']})")
    print("excess vs universe EW, %/yr:  " + "  ".join(f"p{k}={v * 100:+.2f}" for k, v in p.items()))
    print(f"mean {summary['excess_vs_universe_ew']['mean'] * 100:+.2f}  std {summary['excess_vs_universe_ew']['std'] * 100:.2f}")


if __name__ == "__main__":
    main()
