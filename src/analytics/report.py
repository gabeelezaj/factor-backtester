"""Full report for a backtest result: metrics per period, calendar years, rolling excess, CAPM, deciles,
null-distribution placement, oracle ratio and the run log's multiple-testing warning."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analytics import runlog
from src.analytics.deciles import decile_monthly_returns, decile_table
from src.analytics.metrics import (BENCHMARK_LABELS, calendar_year_table, capm, cagr, excess_cagr, hit_rates,
                                   rolling_excess, split_periods, summary_table, t_stat_excess, turnover_stats,
                                   ann_vol, sharpe)
from src.analytics.null_distribution import load_null, placement
from src.analytics.oracle import fraction_of_oracle, oracle_available, run_oracle
from src.backtest.result import BacktestResult

BENCHMARKS = BENCHMARK_LABELS


def headline(result: BacktestResult) -> Dict[str, Any]:
    m = result.monthly
    rf = m["risk_free_rate"]
    out: Dict[str, Any] = {"months": int(len(m)), "cagr_net": cagr(m["net"]), "cagr_gross": cagr(m["gross"]),
                           "ann_vol": ann_vol(m["net"]), "sharpe": sharpe(m["net"], rf)}
    out.update(turnover_stats(m))
    for col, label in BENCHMARKS.items():
        out[f"cagr_{col}"] = cagr(m[col])
        out[f"excess_vs_{col}"] = excess_cagr(m["net"], m[col])
        out[f"t_vs_{col}"] = t_stat_excess(m["net"], m[col])
    return out


def run_report(result: BacktestResult, source=None, log: bool = True, deciles: bool = True,
               decile_frequency: str = "quarterly", snapshot_cache=None) -> Dict[str, Any]:
    cfg = result.config
    m = result.monthly
    rep: Dict[str, Any] = {"headline": headline(result)}
    rep.update(rep["headline"])
    periods = split_periods(m, cfg.split_date)
    rep["periods"] = {label: {"summary": summary_table(sub), "capm": capm(sub), "headline_excess": excess_cagr(sub["net"], sub["universe_ew"]),
                              "t_vs_universe_ew": t_stat_excess(sub["net"], sub["universe_ew"]), "months": len(sub)}
                      for label, sub in periods.items() if len(sub) >= 12}
    rep["calendar"] = calendar_year_table(m)
    rep["hit_rates"] = hit_rates(rep["calendar"])
    rep["rolling_3y"] = rolling_excess(m, 36)
    rep["rolling_5y"] = rolling_excess(m, 60)
    rep["capm"] = capm(m)
    null = load_null()
    rep["null"] = placement(rep["excess_vs_universe_ew"], null)
    if deciles and result.data is not None:
        dr = decile_monthly_returns(result, result.data.R, result.data.caps, 0, decile_frequency, snapshot_cache)
        rep["deciles"] = decile_table(dr, m["universe_ew"])
        rep["deciles"]["frequency"] = decile_frequency
        rep["decile_returns"] = dr
    src = source
    if src is None:
        from src.data import get_source
        src = get_source(cfg.data_source.name, **cfg.data_source.params)
    if oracle_available(src):
        o = run_oracle(src, cfg.start, cfg.end, cfg.top_n, cfg.frequency, cfg.annual_month, cfg.universe, cfg.costs, cfg.weighting)
        common = o.index.intersection(m.index)
        rep["oracle_excess_vs_universe_ew"] = excess_cagr(o.loc[common, "net"], o.loc[common, "universe_ew"])
        rep["fraction_of_oracle"] = fraction_of_oracle(m.loc[common, "net"], o.loc[common, "net"], o.loc[common, "universe_ew"])
        rep["excess_vs_oracle"] = excess_cagr(m.loc[common, "net"], o.loc[common, "net"])
        rep["oracle_net"] = o.loc[common, "net"]
    n_trials = runlog.log_run(cfg, rep, result.source_metadata.get("name", "?")) if log else runlog.count()
    rep["run_log"] = runlog.multiple_testing_bar(max(n_trials, 1), null)
    return rep


def _pct(x: float) -> str:
    return "   n/a" if x is None or not np.isfinite(x) else f"{x * 100:6.2f}%"


def print_report(rep: Dict[str, Any], result: Optional[BacktestResult] = None) -> None:
    pd.set_option("display.width", 200)
    if result is not None:
        meta = result.source_metadata
        flag = "SIMULATED DATA" if meta.get("is_simulated") else ("SURVIVORSHIP-BIASED" if meta.get("survivorship_bias") else "")
        print(f"=== {result.config.name}  [{meta.get('display_name')}: {flag}]  "
              f"{result.monthly.index[0].date()}..{result.monthly.index[-1].date()}  ({result.n_sleeves} sleeve(s))")
    print(f"CAGR net {_pct(rep['cagr_net'])}  (gross {_pct(rep['cagr_gross'])}, cost drag {_pct(rep['cost_drag_per_year'])}/yr, "
          f"turnover {rep['avg_turnover_per_year']:.2f}/yr)   vol {_pct(rep['ann_vol'])}   Sharpe {rep['sharpe']:.2f}")
    for col, label in BENCHMARKS.items():
        print(f"  vs {label:12s} CAGR {_pct(rep[f'cagr_{col}'])}   excess {rep[f'excess_vs_{col}'] * 100:+6.2f}%/yr   t = {rep[f't_vs_{col}']:+.2f}")
    n = rep.get("null", {})
    if n.get("available"):
        print(f"  null: excess vs EW universe at the {n['percentile']:.0f}th percentile of {n['n']} zero-signal runs "
              f"(p95 {n['null_p95'] * 100:+.2f}%, p99 {n['null_p99'] * 100:+.2f}%) -> {n['verdict']}")
        if result is not None and result.source_metadata.get("name") not in n.get("built_on", ""):
            print(f"        (null built on: {n['built_on']} — a different universe than this run; treat as a rough bar)")
    if "fraction_of_oracle" in rep:
        frac = rep["fraction_of_oracle"]
        tail = (f"strategy captures {frac * 100:.0f}% of the oracle" if np.isfinite(frac)
                else "oracle has no edge on this panel, ratio not meaningful")
        print(f"  oracle: excess {rep['oracle_excess_vs_universe_ew'] * 100:+.2f}%/yr; strategy minus oracle "
              f"{rep['excess_vs_oracle'] * 100:+.2f}%/yr -> {tail}")
    for label, per in rep.get("periods", {}).items():
        c = per["capm"]
        print(f"\n--- {label}: {per['months']} months; excess vs EW universe {per['headline_excess'] * 100:+.2f}%/yr (t={per['t_vs_universe_ew']:+.2f}); "
              f"CAPM alpha {c['alpha_annual'] * 100:+.2f}%/yr (t={c['alpha_t']:+.2f}), beta {c['beta']:.2f}")
        t = per["summary"].copy()
        for col in ("CAGR", "Ann. vol", "Max drawdown", "Best month", "Worst month"):
            t[col] = t[col].map(lambda v: f"{v * 100:.1f}%")
        print(t.round(2).to_string())
    cal = rep["calendar"].copy()
    print("\n--- Calendar years (strategy vs benchmarks; * = beat)")
    show = cal[["Strategy"] + list(BENCHMARKS.values())].copy()
    for c in BENCHMARKS.values():
        show[c] = [f"{v * 100:6.1f}%{'*' if e > 0 else ' '}" for v, e in zip(cal[c], cal[f"vs {c}"])]
    show["Strategy"] = show["Strategy"].map(lambda v: f"{v * 100:6.1f}%")
    print(show.to_string())
    hr = rep["hit_rates"]
    print("years beaten: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in hr.items()))
    for w, key in (("3y", "rolling_3y"), ("5y", "rolling_5y")):
        r = rep[key]
        if len(r):
            print(f"rolling {w} excess vs EW universe: mean {r['EW universe'].mean() * 100:+.2f}%/yr, "
                  f"positive {(r['EW universe'] > 0).mean() * 100:.0f}% of windows, worst {r['EW universe'].min() * 100:+.2f}%/yr")
    if "deciles" in rep:
        d = rep["deciles"]
        print(f"\n--- Deciles by composite score (1 = worst, 10 = best), EW, gross, {d['frequency']} sorts")
        print("CAGR: " + "  ".join(f"D{i}:{v * 100:5.1f}%" for i, v in d["table"]["CAGR"].items()))
        print(f"top minus bottom {d['top_minus_bottom_annual'] * 100:+.2f}%/yr (t={d['top_minus_bottom_t']:+.2f}), "
              f"Spearman {d['spearman']:.2f}, {d['monotone_steps']}/9 steps up")
    print_verdict(rep, split=bool(result is not None and result.config.split_date))
    rl = rep.get("run_log", {})
    if rl.get("available"):
        print(f"\n!!! {rl['warning']}")


# ------------------------------------------------------------------ plain-English verdict
def verdict(rep: Dict[str, Any], split: bool = False) -> List[Tuple[str, str]]:
    """Answer the questions a reader should ask, as (status, sentence) pairs.  status: 'good' | 'warn' | 'bad' | 'info'."""
    out: List[Tuple[str, str]] = []
    ex, t = rep["excess_vs_universe_ew"], rep["t_vs_universe_ew"]
    if ex > 0 and t >= 2:
        out.append(("good", f"Beats the equal-weighted universe by {ex * 100:+.2f}%/yr with t = {t:.1f} — unlikely to be noise."))
    elif ex > 0:
        out.append(("warn", f"Beats the equal-weighted universe by {ex * 100:+.2f}%/yr, but t = {t:.1f}: this could easily be luck."))
    else:
        out.append(("bad", f"Does not beat the equal-weighted universe ({ex * 100:+.2f}%/yr). Any lead over the S&P 500 is a size or "
                           "weighting effect, not stock selection."))
    n = rep.get("null", {})
    if n.get("available"):
        p = n["percentile"]
        status = "good" if p >= 95 else ("warn" if p >= 80 else "bad")
        out.append((status, f"Luck check: at the {p:.0f}th percentile of {n['n']} zero-information runs "
                            f"({'clears' if p >= 95 else 'does not clear'} the 95th-percentile bar)."))
    d = rep.get("deciles")
    if d:
        tt, tb = d["top_minus_bottom_t"], d["top_minus_bottom_annual"]
        top_is_best = d["table"]["CAGR"].idxmax() == d["table"].index.max()
        status = "good" if (tt >= 2 and top_is_best) else ("warn" if tt >= 1 else "bad")
        out.append((status, f"Deciles: best minus worst {tb * 100:+.2f}%/yr (t = {tt:.1f}); "
                            f"{'the best decile is the top performer' if top_is_best else 'the best decile is NOT the top performer'} "
                            f"— {'the signal shows across the whole universe' if status == 'good' else 'the top picks may be lucky'}."))
    if split and len(rep.get("periods", {})) == 3:
        labels = list(rep["periods"])
        ins, oos = rep["periods"][labels[1]]["headline_excess"], rep["periods"][labels[2]]["headline_excess"]
        if ins > 0 and oos > 0:
            out.append(("good", f"Holds up out-of-sample: {ins * 100:+.2f}%/yr in-sample vs {oos * 100:+.2f}%/yr out-of-sample."))
        elif ins > 0:
            out.append(("warn", f"In-sample {ins * 100:+.2f}%/yr but out-of-sample {oos * 100:+.2f}%/yr — a sign of overfitting."))
        else:
            out.append(("info", f"In-sample {ins * 100:+.2f}%/yr, out-of-sample {oos * 100:+.2f}%/yr."))
    frac = rep.get("fraction_of_oracle")
    if frac is not None and np.isfinite(frac):
        status = "good" if 0.6 <= frac <= 1.3 else ("bad" if frac > 1.3 else "warn")
        out.append((status, f"Signal captured: {frac * 100:.0f}% of the oracle "
                            f"({'the pipeline recovers the planted signal' if status == 'good' else 'above 130% means information is leaking' if frac > 1.3 else 'most of the planted signal is being lost'})."))
    drag, turn = rep["cost_drag_per_year"], rep["avg_turnover_per_year"]
    out.append(("info" if drag < 0.01 else "warn",
                f"Trading costs take {drag * 100:.2f}%/yr at {turn * 100:.0f}% annual turnover"
                + (" — modest." if drag < 0.01 else " — a large share of the edge.")))
    return out


VERDICT_ICONS = {"good": "✅", "warn": "⚠️", "bad": "❌", "info": "ℹ️"}


def print_verdict(rep: Dict[str, Any], split: bool = False) -> None:
    print("\n--- Verdict")
    for status, text in verdict(rep, split):
        print(f"  {VERDICT_ICONS[status]} {text}")
