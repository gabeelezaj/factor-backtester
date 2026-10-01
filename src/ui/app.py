"""Streamlit front end.  Knows nothing about where data comes from: it lists sources from the registry and
reads every caveat from get_metadata()."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import runlog
from src.analytics.report import VERDICT_ICONS, run_report, verdict
from src.backtest.costs import CostModel
from src.backtest.engine import PreparedData, run_backtest
from src.config import BacktestConfig, DataSourceConfig
from src.data import get_source, list_sources, source_metadata_preview
from src.factors import factor_table, get_factor, list_factors
from src.ui.banner import render_banner
from src.ui.charts import cumulative_chart, decile_chart, drawdown_chart, rolling_chart, style_calendar
from src.universe import UniverseConfig

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

PRESETS: Dict[str, Dict[str, float]] = {
    "Magic Formula (Greenblatt)": {"return_on_capital": 50, "earnings_yield": 50},
    "Value only": {"earnings_yield": 50, "book_to_market": 25, "fcf_yield": 25},
    "Quality only": {"return_on_capital": 40, "gross_profitability": 40, "accruals": 20},
    "Value + quality + momentum": {"earnings_yield": 30, "gross_profitability": 30, "momentum_12_1": 25, "volatility_12m": 15},
    "Custom": {},
}

GLOSSARY = {
    "Factor": "A number computed for every stock at each rebalance date (e.g. earnings yield = EBIT ÷ enterprise value). "
              "Each factor has a direction: for some, *higher is better* (cheapness, profitability); for others, *lower is better* "
              "(volatility, accruals, asset growth). The ranking handles the direction for you.",
    "Composite ranking": "Greenblatt's method (*rank sum*): rank every stock on each factor with 1 = best, add the ranks up "
                         "(weighted), and rank the totals again. A stock that is 5th on one factor and 26th on another scores 31. "
                         "*Z-score* instead standardizes each factor (after trimming the 1% extremes) and adds the weighted z-scores.",
    "Point-in-time": "Fundamentals are only used from the date they became public (the filing or announcement date), never from "
                     "the fiscal period end. This is the single most common source of fake backtest returns.",
    "Equal-weighted universe": "The average return of *every* stock that passed the universe filters, rebalanced on the same dates, "
                               "no costs. This is the fair benchmark: a 30-stock equal-weight portfolio tilts towards smaller companies, "
                               "so beating the cap-weighted S&P 500 can be a size effect rather than skill.",
    "Excess return": "Strategy annual return minus the benchmark's annual return (both compounded).",
    "t-statistic": "How many standard errors the average monthly excess return is from zero. Roughly: below 2 could easily be luck.",
    "Luck check (null percentile)": "The same pipeline was run on 30 simulated markets where fundamentals predict *nothing*. "
                                    "The percentile says how many of those pure-luck runs the current result beats. "
                                    "Above the 95th is the usual bar; above the 99th is convincing.",
    "Signal captured (fraction of oracle)": "Only on simulated data: the best any point-in-time strategy could do is to rank on the "
                                            "planted signal directly (the *oracle*). 100% means the pipeline recovered the whole signal; "
                                            "near 0 means a bug lost it; well above 130% means information is leaking from the future.",
    "Sharpe / Sortino": "Return above the risk-free rate per unit of risk (all volatility / downside volatility only), annualized.",
    "Max drawdown": "Largest peak-to-trough loss, and the months from the peak until the old high was regained.",
    "CAPM alpha / beta": "Beta = sensitivity to the value-weighted market; alpha = annual return not explained by that exposure.",
    "Deciles": "The whole universe sorted into ten equal groups by composite score (1 = worst, 10 = best), each held equal-weighted. "
               "A real signal separates the tails; a lucky 30-stock portfolio does not.",
    "Configurations tried": "Every setting you try is logged. The best of many trials is always overstated, so the page shows "
                            "how high a result must be to still count after that many attempts.",
    "Staggered sleeves": "Split the capital into 12 (annual) or 3 (quarterly) portfolios, each rebalanced in a different month, "
                         "to remove the luck of picking one rebalance date.",
}


# ------------------------------------------------------------------ caching
@st.cache_resource(show_spinner="Loading data source…")
def cached_source(name: str, params_json: str):
    return get_source(name, **json.loads(params_json))


@st.cache_resource(show_spinner="Preparing panel…", max_entries=4)
def cached_data(name: str, params_json: str, start: str, end: str) -> PreparedData:
    return PreparedData(cached_source(name, params_json), start, end)


@st.cache_resource
def snapshot_cache(name: str, params_json: str) -> Dict:
    return {}


@st.cache_resource(show_spinner="Running backtest…", max_entries=24)
def cached_run(name: str, params_json: str, cfg_yaml: str, cfg_hash: str):
    cfg = BacktestConfig.from_yaml(cfg_yaml)
    data = cached_data(name, params_json, cfg.start, cfg.end)
    result = run_backtest(cfg, data=data, snapshot_cache=snapshot_cache(name, params_json))
    report = run_report(result, cached_source(name, params_json), log=True, snapshot_cache=snapshot_cache(name, params_json))
    return result, report


# ------------------------------------------------------------------ helpers
def fname(n: str) -> str:
    return get_factor(n).display_name


def fshort(n: str) -> str:
    return fname(n).split(" (")[0]


def _source_label(name: str) -> str:
    try:
        return source_metadata_preview(name).get("display_name", name)
    except Exception as e:  # adapter or its optional dependency unavailable
        return f"{name} (unavailable: {type(e).__name__})"


def _pct(x, signed=False):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:+.2f}%" if signed else f"{x * 100:.2f}%"


# ------------------------------------------------------------------ sidebar
def sidebar() -> Tuple[str, Dict[str, Any], BacktestConfig]:
    sb = st.sidebar
    sb.title("Build a strategy")
    sb.caption("Work top to bottom. Every control has a ? with an explanation. The page re-runs on each change.")

    with sb.expander("1 · Data", expanded=True):
        name = st.selectbox("Data source", list_sources(), format_func=_source_label, index=0,
                            help="Where prices and financial statements come from. The banner at the top of the page describes "
                                 "the chosen source and its limitations.")
        params: Dict[str, Any] = {}
        try:
            meta = source_metadata_preview(name)
        except Exception as e:
            st.error(f"{name} is not available: {e}")
            st.stop()
        for pname, spec in meta.get("ui_params", {}).items():
            label = spec.get("label", pname)
            if spec["type"] == "int":
                params[pname] = int(st.number_input(label, int(spec["min"]), int(spec["max"]), int(spec["default"]), int(spec["step"]), help=spec.get("help")))
            else:
                params[pname] = float(st.slider(label, float(spec["min"]), float(spec["max"]), float(spec["default"]), float(spec["step"]), help=spec.get("help")))
        cov_start = pd.Timestamp(meta.get("coverage_start", "1990-01-31")).year
        cov_end = pd.Timestamp(meta.get("coverage_end", "2024-12-31")).year
        rec_start = pd.Timestamp(meta["recommended_start"]).year if meta.get("recommended_start") else cov_start + 1
        y0, y1 = st.slider("Backtest years", cov_start, cov_end, (min(rec_start, cov_end - 2), cov_end),
                           help="First and last year of the backtest. The first year is used for the first portfolio's formation."
                                + (" This source's fundamentals only become usable around the suggested start." if meta.get("recommended_start") else ""))
        if y1 - y0 < 2:
            st.error("Choose a window of at least three years.")
            st.stop()
        use_split = st.checkbox("Report in-sample / out-of-sample separately", value=True,
                                help="Everything up to the split year is 'in-sample' (where you tuned the idea); everything after is "
                                     "'out-of-sample'. If the idea only works in-sample, it is probably overfitted.")
        split_year = st.slider("Split after year", y0, y1, (y0 + y1) // 2) if use_split else None

    with sb.expander("2 · Strategy", expanded=True):
        preset = st.selectbox("Preset", list(PRESETS), index=0,
                              help="Starting points. Pick one, then adjust the factors and weights below (that turns it into 'Custom').")
        if st.session_state.get("_preset") != preset:
            st.session_state["_preset"] = preset
            if PRESETS[preset]:
                st.session_state["factors_ms"] = list(PRESETS[preset])
                for n, w in PRESETS[preset].items():
                    st.session_state[f"w_{n}"] = w
        all_factors = list_factors()
        st.session_state.setdefault("factors_ms", list(PRESETS["Magic Formula (Greenblatt)"]))
        chosen: List[str] = st.multiselect("Factors", all_factors, key="factors_ms", format_func=fshort,
                                           help="Each factor ranks every stock; the ranks are combined with the weights below.")
        if not chosen:
            st.warning("Pick at least one factor.")
            st.stop()
        lower = [fshort(n) for n in chosen if not get_factor(n).higher_is_better]
        st.caption("Direction is handled for you: " + (f"{', '.join(lower)} count as *lower is better*; the rest as higher is better."
                                                       if lower else "for all selected factors, higher is better.")
                   + " Hover the ? on a weight slider for the formula.")
        st.markdown("**Weights** — normalized to sum to 100%")
        raw_w = {}
        for n in chosen:
            f = get_factor(n)
            st.session_state.setdefault(f"w_{n}", 100 // len(chosen))
            raw_w[n] = st.slider(fshort(n), 0, 100, key=f"w_{n}", step=5,
                                 help=f"{f.description}. Formula: {f.formula}. Direction: {f.direction_text}. "
                                      f"Missing data: {f.missing_policy}.")
        total = sum(raw_w.values()) or 1
        weights = {n: (w / total) for n, w in raw_w.items() if w > 0} or {chosen[0]: 1.0}
        modified = PRESETS[preset] and (set(chosen) != set(PRESETS[preset]) or any(raw_w.get(n) != w for n, w in PRESETS[preset].items()))
        st.caption(" · ".join(f"{fshort(n)} **{w * 100:.0f}%**" for n, w in weights.items())
                   + ("  — *modified from the preset*" if modified else ""))
        method = st.radio("How to combine factors", ["rank_sum", "zscore"], horizontal=True,
                          format_func=lambda m: {"rank_sum": "Rank sum (Greenblatt)", "zscore": "Z-score"}[m],
                          help="Rank sum: rank each factor 1 = best, add the weighted ranks, rank again — robust to outliers. "
                               "Z-score: trim the 1% extremes, standardize each factor, add the weighted z-scores — keeps magnitudes.")
        missing = st.radio("Stocks missing a factor", ["drop", "median"], horizontal=True,
                           format_func=lambda m: {"drop": "Exclude them", "median": "Give the median rank"}[m],
                           help="Missing fundamentals are more common for distressed companies, so 'exclude' quietly removes sick names.")

    with sb.expander("3 · Universe", expanded=False):
        st.caption("Which stocks are eligible at each rebalance date.")
        max_names = int(st.number_input("Largest N companies by market cap", 100, 10000, 5000, 100,
                                        help="Keep only the N largest companies (after the filters below)."))
        min_mcap = float(st.number_input("Minimum market cap ($ millions)", 0.0, 10000.0, 50.0, 10.0,
                                         help="Tiny companies are expensive to trade and their data is unreliable.")) * 1e6
        min_price = float(st.number_input("Minimum share price ($)", 0.0, 100.0, 1.0, 0.5,
                                          help="Penny stocks are excluded by default."))
        excl = []
        if st.checkbox("Exclude financials (SIC 6000–6999)", True,
                       help="Banks and insurers have no meaningful EBIT or working capital, so Greenblatt excluded them."):
            excl.append((6000, 6999))
        if st.checkbox("Exclude utilities (SIC 4900–4999)", True,
                       help="Regulated returns make utilities' ratios hard to compare with other companies."):
            excl.append((4900, 4999))
        extra = st.text_input("Extra industry exclusions (SIC ranges)", "", placeholder="e.g. 1300-1399, 8000-8099",
                              help="Comma-separated ranges of 4-digit SIC codes to leave out.")
        for part in [p.strip() for p in extra.split(",") if p.strip()]:
            try:
                lo, hi = part.split("-")
                excl.append((int(lo), int(hi)))
            except ValueError:
                st.warning(f"Ignored '{part}' — use the form 1300-1399.")
        exchanges = st.multiselect("Exchanges", ["NYSE", "AMEX", "NASDAQ", "OTC"], default=["NYSE", "AMEX", "NASDAQ"],
                                   help="Major exchanges only by default; OTC listings are thinly traded.")

    with sb.expander("4 · Portfolio", expanded=False):
        top_n = int(st.slider("Number of stocks to hold", 5, 200, 30, 5,
                              help="Greenblatt used 20–30. Fewer stocks = more luck in the result; more stocks = closer to the universe average."))
        weighting = st.radio("Position sizing", ["equal", "cap"], horizontal=True,
                             format_func=lambda w: {"equal": "Equal weight", "cap": "Market-cap weight"}[w],
                             help="Equal weight gives every pick the same money (tilts small). Cap weight sizes by company value.")
        frequency = st.radio("Rebalance", ["annual", "quarterly", "monthly"], horizontal=True,
                             format_func=str.capitalize,
                             help="How often the portfolio is rebuilt from scratch. More often = fresher information but more trading costs.")
        annual_month = 12
        if frequency == "annual":
            annual_month = MONTHS.index(st.selectbox("Rebalance month", MONTHS, index=11,
                                                     help="The month whose end is the formation date each year.")) + 1
        sleeves = False
        if frequency != "monthly":
            sleeves = st.checkbox("Staggered sleeves", False,
                                  help="Split the money into 12 (annual) or 3 (quarterly) portfolios, each rebalanced in a different month. "
                                       "Removes the luck of one particular rebalance date.")

    with sb.expander("5 · Trading costs", expanded=False):
        kind = st.radio("Cost model", ["size", "flat"], horizontal=True,
                        format_func=lambda k: {"size": "Depends on company size", "flat": "Same for every stock"}[k],
                        help="Size-dependent: from 5 bps (one-way) for mega caps to 150 bps for micro caps — small-cap tilts pay for it. "
                             "Flat: one number for everything.")
        flat_bps = float(st.number_input("One-way cost (basis points)", 0.0, 500.0, 20.0, 5.0)) if kind == "flat" else 20.0
        mult = 1.0
        if kind == "size":
            mult = float(st.slider("Cost multiplier", 0.0, 3.0, 1.0, 0.25, help="Stress test: 2.0 doubles every trading cost."))

    cfg = BacktestConfig(
        name="ui", data_source=DataSourceConfig(name, params),
        start=f"{y0}-01-31", end=f"{y1}-12-31", factors=weights, ranking_method=method, missing_policy=missing,
        universe=UniverseConfig(max_names, min_mcap, min_price, tuple(exchanges) or ("NYSE",), tuple(excl)),
        top_n=top_n, weighting=weighting, frequency=frequency, annual_month=annual_month, sleeves=sleeves,
        costs=CostModel(kind=kind, flat_bps=flat_bps, multiplier=mult),
        split_date=f"{split_year}-12-31" if split_year else None,
    )
    return name, params, cfg


# ------------------------------------------------------------------ main
def _strategy_sentence(cfg: BacktestConfig) -> str:
    facs = ", ".join(f"{fshort(n)} {w * 100:.0f}%" for n, w in cfg.factors.items())
    how = "rank-sum" if cfg.ranking_method == "rank_sum" else "z-score"
    when = {"annual": f"every {MONTHS[cfg.annual_month - 1]}", "quarterly": "every quarter", "monthly": "every month"}[cfg.frequency]
    return (f"Rank the universe on **{facs}** ({how}), buy the top **{cfg.top_n}** "
            f"{'equal-weighted' if cfg.weighting == 'equal' else 'cap-weighted'}, rebalance **{when}**"
            f"{' in staggered sleeves' if cfg.sleeves else ''}, {cfg.start[:4]}–{cfg.end[:4]}.")


def main() -> None:
    st.set_page_config(page_title="Factor Ranking Backtester", layout="wide", page_icon="📊")
    st.markdown("<style>section[data-testid='stSidebar'] {width: 400px !important;} "
                "section[data-testid='stSidebar'] > div {width: 400px !important;}</style>", unsafe_allow_html=True)
    name, params, cfg = sidebar()
    params_json = json.dumps(params, sort_keys=True)
    source = cached_source(name, params_json)
    meta = source.get_metadata()
    render_banner(meta)

    st.title("Multi-factor stock ranking backtester")
    st.markdown(_strategy_sentence(cfg))
    with st.expander("How to read this page (and a glossary)"):
        st.markdown(
            "**The idea.** At each rebalance date, every eligible stock is scored on the factors you chose, using only information "
            "that was public on that date. The best-scoring stocks are bought and held until the next rebalance. The chart and "
            "tables compare that portfolio with benchmarks — most importantly the *equal-weighted universe*, which is what you would "
            "get by buying every eligible stock instead of picking.\n\n"
            "**Three questions to ask of any result:** (1) Does it beat the equal-weighted universe, not just the S&P 500? "
            "(2) Does the *luck check* say it is outside what a zero-information strategy achieves? (3) Do the deciles rise from "
            "worst to best, or is only the top 30 lucky?")
        for term, text in GLOSSARY.items():
            st.markdown(f"- **{term}** — {text}")

    try:
        result, rep = cached_run(name, params_json, cfg.to_yaml(), cfg.hash())
    except ValueError as e:
        st.error(f"This configuration cannot run: {e}")
        st.stop()
    m = result.monthly
    null = rep.get("null", {})
    frac = rep.get("fraction_of_oracle", np.nan)
    full = rep["periods"]["Full sample"]["summary"]

    c = st.columns(3)
    c[0].metric("Annual return, net of costs", _pct(rep["cagr_net"]), f"before costs {_pct(rep['cagr_gross'])}", delta_color="off",
                help="Compound annual growth rate of the strategy after trading costs.")
    c[1].metric("vs equal-weighted universe", _pct(rep["excess_vs_universe_ew"], True) + " per year",
                f"t = {rep['t_vs_universe_ew']:+.2f} · universe earned {_pct(rep['cagr_universe_ew'])}", delta_color="off",
                help=GLOSSARY["Equal-weighted universe"] + " " + GLOSSARY["t-statistic"])
    c[2].metric("vs S&P 500", _pct(rep["excess_vs_sp500_total_return"], True) + " per year",
                f"t = {rep['t_vs_sp500_total_return']:+.2f} · S&P earned {_pct(rep['cagr_sp500_total_return'])}", delta_color="off",
                help="Beating the cap-weighted index is easier for a small-cap-tilted portfolio; compare with the universe number first.")
    c = st.columns(3)
    c[0].metric("Sharpe ratio", f"{rep['sharpe']:.2f}",
                f"volatility {_pct(rep['ann_vol'])} · max drawdown {_pct(full.loc['Strategy (net)', 'Max drawdown'])}", delta_color="off",
                help=GLOSSARY["Sharpe / Sortino"])
    c[1].metric("Luck check", f"{null['percentile']:.0f}th percentile" if null.get("available") else "n/a",
                null.get("verdict", "no null distribution available"), delta_color="off",
                help=GLOSSARY["Luck check (null percentile)"] + f" (Null built on: {null.get('built_on', '?')}.)")
    c[2].metric("Signal captured", f"{frac * 100:.0f}% of oracle" if np.isfinite(frac) else "n/a",
                ("only meaningful on simulated data" if "fraction_of_oracle" not in rep else
                 "oracle has no edge on this panel" if not np.isfinite(frac) else "70–130% is normal; more = leak"), delta_color="off",
                help=GLOSSARY["Signal captured (fraction of oracle)"])
    rl = rep.get("run_log", {})
    if rl.get("available"):
        st.warning(f"**{rl['n_trials']} configuration(s) tried so far.** {rl['warning'].split('. ', 1)[1]} "
                   f"({GLOSSARY['Configurations tried']})")

    st.subheader("Verdict")
    lines = verdict(rep, split=bool(cfg.split_date))
    st.markdown("\n".join(f"- {VERDICT_ICONS[status]} {text}" for status, text in lines))
    if meta.get("is_simulated"):
        st.caption("Simulated data: a good verdict here means the *engine* works, not that the strategy works in real markets.")
    elif meta.get("survivorship_bias"):
        st.caption("Survivorship-biased data: every line above is optimistic. Only the comparison with the equal-weighted "
                   "universe partly cancels the bias, because both sides share it.")

    tabs = st.tabs(["Performance", "Metrics", "Calendar years", "Deciles", "Holdings", "Config & export"])

    with tabs[0]:
        st.caption("Growth of $1 on a log scale, so equal vertical distances are equal percentage gains. The dashed line, when "
                   "present, is the oracle — the best a point-in-time strategy could do on simulated data. The dotted vertical "
                   "line is the in-sample / out-of-sample split.")
        st.plotly_chart(cumulative_chart(m, rep.get("oracle_net"), cfg.split_date))
        st.caption("Drawdown: how far below its previous peak the portfolio was at each point in time.")
        st.plotly_chart(drawdown_chart(m))
        st.caption("Rolling excess return: over every 3-year (and 5-year) window, the strategy's annualized return minus the "
                   "benchmark's. A robust strategy stays above zero most of the time, not just on average.")
        col1, col2 = st.columns(2)
        col1.plotly_chart(rolling_chart(rep["rolling_3y"], "3-year"))
        col2.plotly_chart(rolling_chart(rep["rolling_5y"], "5-year"))

    with tabs[1]:
        st.caption("CAGR = compound annual return. Ann. vol = annualized standard deviation of monthly returns. Sharpe and Sortino = "
                   "return above cash per unit of (all / downside) risk. Max drawdown = worst peak-to-trough loss and months to recover.")
        for label, per in rep["periods"].items():
            cp = per["capm"]
            st.subheader(label)
            st.caption(f"{per['months']} months · vs equal-weighted universe {_pct(per['headline_excess'], True)}/yr (t = {per['t_vs_universe_ew']:+.2f}) · "
                       f"CAPM alpha {_pct(cp['alpha_annual'], True)}/yr (t = {cp['alpha_t']:+.2f}), beta {cp['beta']:.2f}, R² {cp['r2']:.2f}")
            t = per["summary"].copy()
            st.dataframe(t.style.format({"CAGR": "{:.2%}", "Ann. vol": "{:.2%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
                                         "Max drawdown": "{:.1%}", "Drawdown months": "{:.0f}", "Best month": "{:.1%}", "Worst month": "{:.1%}"}),
                         )
        st.subheader("Turnover and costs")
        st.write(f"On average **{rep['avg_turnover_per_year'] * 100:.0f}% of the portfolio is replaced per year** (one-way). Trading costs "
                 f"reduce the annual return by **{_pct(rep['cost_drag_per_year'])}** ({cfg.costs.describe()}).")
        if "oracle_excess_vs_universe_ew" in rep:
            st.write(f"Oracle (top {cfg.top_n} on the planted signal): {_pct(rep['oracle_excess_vs_universe_ew'], True)}/yr over the universe; "
                     f"this strategy minus the oracle: {_pct(rep['excess_vs_oracle'], True)}/yr.")

    with tabs[2]:
        st.caption("Return in each calendar year. Green = the strategy beat that benchmark in that year, red = it did not. "
                   "The share of years beaten (full years only) is below the table.")
        st.dataframe(style_calendar(rep["calendar"]), height=min(800, 40 + 36 * len(rep["calendar"])))
        st.write("Years beaten: " + " · ".join(f"**{k}** {v * 100:.0f}%" for k, v in rep["hit_rates"].items()))

    with tabs[3]:
        d = rep.get("deciles")
        if d:
            st.caption(GLOSSARY["Deciles"] + f" Sorted {d['frequency']}, gross of costs. Each bar's return carries roughly ±2–3% per year "
                       "of sampling noise, so look at the tails and the top-minus-bottom test, not at whether every step is up.")
            st.plotly_chart(decile_chart(d["table"]))
            st.write(f"Best decile minus worst decile: **{_pct(d['top_minus_bottom_annual'], True)} per year** (t = {d['top_minus_bottom_t']:+.2f}). "
                     f"Rank correlation between decile and return: {d['spearman']:.2f}; {d['monotone_steps']} of 9 steps go up.")
            tbl = d["table"].rename(columns={"CAGR": "Annual return", "Excess vs EW universe": "vs EW universe", "t vs EW universe": "t-stat"})
            st.dataframe(tbl.style.format({"Annual return": "{:.2%}", "vs EW universe": "{:+.2%}", "t-stat": "{:+.2f}"}))

    with tabs[4]:
        st.caption("Exactly what was bought at each rebalance and why: every factor's value and rank, the combined rank, the weight, "
                   "and the return actually earned until the next rebalance.")
        dates = result.formation_dates
        col1, col2 = st.columns([3, 1])
        T = col1.select_slider("Rebalance date", options=list(dates), value=dates[-1], format_func=lambda dt: dt.strftime("%Y-%m-%d"))
        sleeve = col2.number_input("Sleeve", 0, result.n_sleeves - 1, 0) if result.n_sleeves > 1 else 0
        h = result.holdings_at(T, sleeve)
        stats = result.universe_stats
        srow = stats[(stats["formation_date"] == T) & (stats["sleeve"] == sleeve)].iloc[0]
        st.info(f"On {T.date()}: **{srow['n_alive']}** stocks existed → **{srow['n_universe']}** passed the universe filters → "
                f"**{srow['n_ranked']}** had every factor ({srow['n_dropped_missing']} dropped for missing data) → the top **{len(h)}** were bought.")
        uni_caps = result.rankings.loc[(result.rankings["formation_date"] == T) & (result.rankings["sleeve"] == sleeve), "market_cap"]
        if len(h) and len(uni_caps):
            med_h, med_u = h["market_cap"].median() / 1e6, uni_caps.median() / 1e6
            small = (h["market_cap"] < uni_caps.quantile(0.25)).mean()
            st.caption(f"Size tilt: median holding is **${med_h:,.0f}M** vs **${med_u:,.0f}M** for the universe; "
                       f"**{small * 100:.0f}%** of the picks are in the universe's smallest quarter. "
                       "A strong small tilt is why the equal-weighted universe, not the S&P 500, is the fair benchmark.")
        fcols = list(cfg.factors)
        cols = ["rank", "security_id", "score_raw"] + [c for f in fcols for c in (f, f + "_rank")] + \
               ["weight", "market_cap", "price", "sic_code", "exchange", "fiscal_period_end", "fundamentals_age_days", "holding_return"]
        show = h[cols].copy()
        show["market_cap"] = show["market_cap"] / 1e6
        rename = {"rank": "Rank", "security_id": "Stock", "score_raw": "Rank sum" if cfg.ranking_method == "rank_sum" else "Composite score",
                  "weight": "Weight", "market_cap": "Market cap ($M)", "price": "Price ($)", "sic_code": "SIC", "exchange": "Exchange",
                  "fiscal_period_end": "Financials as of", "fundamentals_age_days": "Data age (days)", "holding_return": "Return while held"}
        for f in fcols:
            rename[f] = fshort(f)
            rename[f + "_rank"] = f"{fshort(f)} rank"
        show = show.rename(columns=rename)
        fmt = {"Weight": "{:.1%}", "Market cap ($M)": "{:,.0f}", "Price ($)": "{:.2f}", "Return while held": "{:+.1%}",
               rename["score_raw"]: "{:.1f}", "Financials as of": lambda v: v.strftime("%Y-%m-%d") if pd.notna(v) else ""}
        fmt.update({fshort(f): "{:.3f}" for f in fcols})
        fmt.update({f"{fshort(f)} rank": "{:.0f}" for f in fcols})
        st.dataframe(show.style.format(fmt), hide_index=True, height=min(1000, 40 + 36 * len(show)))
        st.caption("Rank columns: 1 = best on that factor among all ranked stocks that day. Rank sum = weighted sum of the factor "
                   "ranks (lower is better); the overall Rank orders stocks by it. 'Financials as of' is the fiscal period whose "
                   "numbers were public on the rebalance date; 'Data age' is how old they were.")
        with st.expander("Full ranking on this date (top 300 of the universe)"):
            rk = result.rankings
            rk = rk[(rk["formation_date"] == T) & (rk["sleeve"] == sleeve)].sort_values("rank").head(300)
            rk = rk[["rank", "security_id", "score_raw"] + fcols + ["market_cap", "weight", "holding_return"]].rename(columns=rename)
            st.dataframe(rk, hide_index=True)

    with tabs[5]:
        st.caption("The YAML below reproduces this exact run from the command line: "
                   "`python3 run_backtest.py my_config.yaml`.")
        col1, col2 = st.columns(2)
        col1.download_button("Download config (YAML)", cfg.to_yaml(), f"backtest_{cfg.hash()}.yaml", "text/yaml")
        col1.code(cfg.to_yaml(), language="yaml")
        col2.markdown("**Results as CSV**")
        col2.download_button("Monthly returns and benchmarks", m.to_csv(), f"monthly_{cfg.hash()}.csv", "text/csv")
        col2.download_button("Holdings at every rebalance", result.holdings.to_csv(index=False), f"holdings_{cfg.hash()}.csv", "text/csv")
        col2.download_button("Full rankings at every rebalance", result.rankings.to_csv(index=False), f"rankings_{cfg.hash()}.csv", "text/csv")
        col2.download_button("Calendar-year table", rep["calendar"].to_csv(), f"calendar_{cfg.hash()}.csv", "text/csv")
        st.subheader("Factor library")
        st.caption("Every available factor, its formula, direction and how missing or negative inputs are treated.")
        ft = factor_table().rename(columns={"label": "Factor", "category": "Category", "description": "Description", "formula": "Formula",
                                            "direction": "Direction", "missing_policy": "Missing / negative data", "name": "Code name"})
        st.dataframe(ft[["Factor", "Category", "Formula", "Direction", "Missing / negative data", "Code name"]], hide_index=True)
        st.subheader("Run log")
        st.caption(GLOSSARY["Configurations tried"])
        log = runlog.read()
        if len(log):
            lg = log[["timestamp", "name", "config_hash", "source", "cagr_net", "excess_vs_universe_ew", "sharpe", "null_percentile"]].tail(50).iloc[::-1]
            lg = lg.rename(columns={"timestamp": "When", "name": "Name", "config_hash": "Config", "source": "Source", "cagr_net": "Annual return",
                                    "excess_vs_universe_ew": "vs EW universe", "sharpe": "Sharpe", "null_percentile": "Luck percentile"})
            st.dataframe(lg.style.format({"Annual return": "{:.2%}", "vs EW universe": "{:+.2%}", "Sharpe": "{:.2f}", "Luck percentile": "{:.0f}"}),
                         hide_index=True)
        st.subheader("Data source details")
        st.json(meta, expanded=False)


if __name__ == "__main__":
    main()
