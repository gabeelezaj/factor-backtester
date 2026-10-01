"""Performance statistics on monthly return series (decimal, indexed by month end)."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

BENCHMARK_LABELS: Dict[str, str] = {
    "universe_ew": "EW universe", "market_ew_return": "EW market",
    "market_vw_return": "VW market", "sp500_total_return": "S&P 500",
}


# ---------------------------------------------------------------- basics
def cagr(monthly: pd.Series) -> float:
    m = monthly.dropna()
    if len(m) == 0:
        return np.nan
    growth = float((1 + m).prod())
    return growth ** (12 / len(m)) - 1 if growth > 0 else -1.0


def ann_vol(monthly: pd.Series) -> float:
    m = monthly.dropna()
    return float(m.std(ddof=1) * np.sqrt(12)) if len(m) > 1 else np.nan


def sharpe(monthly: pd.Series, rf: pd.Series) -> float:
    ex = (monthly - rf.reindex(monthly.index)).dropna()
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(12)) if len(ex) > 1 and sd > 0 else np.nan


def sortino(monthly: pd.Series, rf: pd.Series) -> float:
    ex = (monthly - rf.reindex(monthly.index)).dropna()
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / downside * np.sqrt(12)) if len(ex) > 1 and downside > 0 else np.nan


def excess_cagr(strategy: pd.Series, benchmark: pd.Series) -> float:
    idx = strategy.dropna().index.intersection(benchmark.dropna().index)
    return cagr(strategy.loc[idx]) - cagr(benchmark.loc[idx])


def t_stat_excess(strategy: pd.Series, benchmark: pd.Series) -> float:
    d = (strategy - benchmark.reindex(strategy.index)).dropna()
    sd = d.std(ddof=1)
    return float(d.mean() / sd * np.sqrt(len(d))) if len(d) > 2 and sd > 0 else np.nan


# ------------------------------------------------------------- drawdowns
def wealth(monthly: pd.Series) -> pd.Series:
    return (1 + monthly.fillna(0)).cumprod()


def drawdown_series(monthly: pd.Series) -> pd.Series:
    w = wealth(monthly)
    return w / w.cummax() - 1


def max_drawdown(monthly: pd.Series) -> Dict[str, object]:
    """Deepest peak-to-trough loss, with peak / trough / recovery dates and duration in months
    (peak to recovery; to the end of the sample if never recovered)."""
    w = wealth(monthly)
    peak = w.cummax()
    dd = w / peak - 1
    if dd.empty:
        return {"max_drawdown": np.nan, "peak": None, "trough": None, "recovery": None, "duration_months": np.nan, "recovered": False}
    trough = dd.idxmin()
    peak_date = w.loc[:trough].idxmax()
    after = w.loc[trough:]
    rec = after[after >= peak.loc[trough]]
    recovery = rec.index[0] if len(rec) else None
    end = recovery if recovery is not None else w.index[-1]
    duration = int(((end.year - peak_date.year) * 12 + end.month - peak_date.month))
    return {"max_drawdown": float(dd.min()), "peak": peak_date, "trough": trough, "recovery": recovery,
            "duration_months": duration, "recovered": recovery is not None}


# ---------------------------------------------------------- tables
def summary_row(monthly: pd.Series, rf: pd.Series) -> Dict[str, float]:
    mdd = max_drawdown(monthly)
    return {
        "CAGR": cagr(monthly), "Ann. vol": ann_vol(monthly), "Sharpe": sharpe(monthly, rf), "Sortino": sortino(monthly, rf),
        "Max drawdown": mdd["max_drawdown"], "Drawdown months": mdd["duration_months"],
        "Best month": float(monthly.max()), "Worst month": float(monthly.min()),
    }


def summary_table(monthly: pd.DataFrame, strategy_col: str = "net", benchmarks: Mapping[str, str] = BENCHMARK_LABELS,
                  rf_col: str = "risk_free_rate") -> pd.DataFrame:
    rf = monthly[rf_col]
    rows = {"Strategy (net)": summary_row(monthly[strategy_col], rf)}
    if "gross" in monthly and strategy_col == "net":
        rows["Strategy (gross)"] = summary_row(monthly["gross"], rf)
    for col, label in benchmarks.items():
        if col in monthly:
            rows[label] = summary_row(monthly[col], rf)
    return pd.DataFrame(rows).T


def calendar_year_table(monthly: pd.DataFrame, strategy_col: str = "net",
                        benchmarks: Mapping[str, str] = BENCHMARK_LABELS) -> pd.DataFrame:
    """Calendar-year returns for the strategy and each benchmark, plus excess columns."""
    cols = [strategy_col] + [c for c in benchmarks if c in monthly]
    yearly = monthly[cols].groupby(monthly.index.year).apply(lambda g: (1 + g).prod() - 1)
    yearly = yearly.rename(columns={strategy_col: "Strategy", **{c: benchmarks[c] for c in cols[1:]}})
    for c in cols[1:]:
        yearly[f"vs {benchmarks[c]}"] = yearly["Strategy"] - yearly[benchmarks[c]]
    yearly["months"] = monthly[strategy_col].groupby(monthly.index.year).count()
    yearly.index.name = "year"
    return yearly


def hit_rates(calendar: pd.DataFrame) -> pd.Series:
    """Share of (full) calendar years in which the strategy beat each benchmark."""
    full = calendar[calendar["months"] == 12]
    cols = [c for c in full.columns if c.startswith("vs ")]
    return (full[cols] > 0).mean().rename(lambda c: c[3:])


def rolling_excess(monthly: pd.DataFrame, window_months: int, strategy_col: str = "net",
                   benchmarks: Mapping[str, str] = BENCHMARK_LABELS) -> pd.DataFrame:
    """Rolling annualized strategy return minus benchmark return."""
    def ann(s):
        return (1 + s).rolling(window_months).apply(lambda x: x.prod() ** (12 / window_months) - 1, raw=True)
    strat = ann(monthly[strategy_col])
    out = pd.DataFrame({benchmarks[c]: strat - ann(monthly[c]) for c in benchmarks if c in monthly})
    return out.dropna(how="all")


def capm(monthly: pd.DataFrame, strategy_col: str = "net", market_col: str = "market_vw_return",
         rf_col: str = "risk_free_rate") -> Dict[str, float]:
    """OLS of strategy excess return on market excess return: annualized alpha, beta, t-stats, R²."""
    d = monthly[[strategy_col, market_col, rf_col]].dropna()
    y = (d[strategy_col] - d[rf_col]).to_numpy()
    x = (d[market_col] - d[rf_col]).to_numpy()
    n = len(d)
    if n < 12:
        return {"alpha_annual": np.nan, "beta": np.nan, "alpha_t": np.nan, "beta_t": np.nan, "r2": np.nan, "n": n}
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = (resid ** 2).sum() / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"alpha_annual": float(coef[0] * 12), "beta": float(coef[1]), "alpha_t": float(coef[0] / se[0]),
            "beta_t": float(coef[1] / se[1]), "r2": float(r2), "n": n}


def turnover_stats(monthly: pd.DataFrame) -> Dict[str, float]:
    years = len(monthly) / 12
    return {
        "avg_turnover_per_year": float(monthly["turnover"].dropna().sum() / years) if years else np.nan,
        "cost_drag_per_year": cagr(monthly["gross"]) - cagr(monthly["net"]) if "gross" in monthly else np.nan,
        "total_cost_paid": float(monthly["cost"].sum()) if "cost" in monthly else np.nan,
    }


def split_periods(monthly: pd.DataFrame, split_date: Optional[str]) -> Dict[str, pd.DataFrame]:
    """Label rows as in-sample (before split) and out-of-sample (from split on)."""
    if not split_date:
        return {"Full sample": monthly}
    T = pd.Timestamp(split_date)
    return {"Full sample": monthly, f"In-sample (to {T.date()})": monthly[monthly.index <= T],
            f"Out-of-sample (after {T.date()})": monthly[monthly.index > T]}
