"""Signal-strength sweep: at which strength does the pipeline stop detecting the planted signal?

Usage: python -m validation.sweep [--strengths 0,0.25,0.5,1] [--value-trap 0,0.5]
Writes validation/results/sweep.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analytics.metrics import excess_cagr, t_stat_excess
from src.analytics.null_distribution import load_null, placement
from src.analytics.oracle import fraction_of_oracle, run_oracle
from src.backtest.engine import run_backtest
from src.config import BacktestConfig
from src.data import get_source

MAGIC = BacktestConfig.load(Path(__file__).resolve().parents[1] / "configs" / "magic_formula.yaml")

START, END = "1990-01-31", "2024-12-31"
OUT = Path(__file__).resolve().parent / "results" / "sweep.csv"


def decile_spread(source) -> float:
    ta = source.true_alpha().copy()
    ta["date"] = ta["date"] + pd.offsets.MonthEnd(1)
    r = source.returns(START, END)[["security_id", "date", "total_return"]]
    d = ta.merge(r, on=["security_id", "date"])
    d["dec"] = d.groupby("date")["true_alpha"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    dec = d.groupby("dec")["total_return"].mean()
    return float(dec.iloc[-1] - dec.iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strengths", default="0,0.25,0.5,1")
    ap.add_argument("--value-trap", default="0,0.5")
    ap.add_argument("--seeds", default="42,43,44,45,46", help="averaged; single-seed differences are mostly noise")
    args = ap.parse_args()
    null = load_null()
    rows = []
    for vt in [float(x) for x in args.value_trap.split(",")]:
        for s in [float(x) for x in args.strengths.split(",")]:
            for seed in [int(x) for x in args.seeds.split(",")]:
                src = get_source("synthetic", seed=seed, signal_strength=s, value_trap=vt, cache=False)
                res = run_oracle(src, START, END)
                ex = excess_cagr(res["net"], res["universe_ew"])
                eng = run_backtest(MAGIC, src).monthly
                ex_eng = excess_cagr(eng["net"], eng["universe_ew"])
                rows.append(dict(
                    value_trap=vt, signal_strength=s, seed=seed,
                    decile_spread_pct_per_month=decile_spread(src) * 100,
                    oracle_excess_pct_per_year=ex * 100,
                    oracle_t_stat=t_stat_excess(res["net"], res["universe_ew"]),
                    engine_excess_pct_per_year=ex_eng * 100,
                    engine_t_stat=t_stat_excess(eng["net"], eng["universe_ew"]),
                    engine_null_percentile=placement(ex_eng, null).get("percentile"),
                    fraction_of_oracle=fraction_of_oracle(eng["net"], res["net"], res["universe_ew"]),
                    null_percentile=placement(ex, null).get("percentile"),
                ))
                print(rows[-1], flush=True)
    raw = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    raw.to_csv(OUT.with_name("sweep_raw.csv"), index=False)
    g = raw.groupby(["value_trap", "signal_strength"])
    df = pd.DataFrame({
        "decile_spread_%/mo": g["decile_spread_pct_per_month"].mean(),
        "oracle_excess_%/yr": g["oracle_excess_pct_per_year"].mean(),
        "engine_excess_%/yr": g["engine_excess_pct_per_year"].mean(),
        "engine_excess_sd": g["engine_excess_pct_per_year"].std(),
        "engine_t_mean": g["engine_t_stat"].mean(),
        "engine_seeds_above_null_p95": g["engine_null_percentile"].apply(lambda x: (x >= 95).mean()),
        "fraction_of_oracle_mean": g["fraction_of_oracle"].mean(),
        "n_seeds": g.size(),
    }).reset_index()
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
