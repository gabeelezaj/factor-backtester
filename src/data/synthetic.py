"""Synthetic market simulator.

A deliberately trap-laden simulated market used to build and verify the engine
without any credentials.  Key design points (all logged in DECISIONS.md):

* Unbalanced panel: companies are born (IPO) and die (bankruptcy, acquisition,
  other delisting).  The final row of a dead company carries the delisting
  return in ``total_return`` and ``is_delisted_this_month=True``.
* Fundamentals are published with a company-specific lag of 45-90 days
  (``available_date``), a few are late, missing, or restated.
* The signal has two explicit components:
  1. ``true_alpha`` -- a persistent characteristic that is KNOWABLE at the
     formation date: the cross-sectional normal score of
     0.5*z(EBIT/EV) + 0.5*z(EBIT/(NWC+NetPPE)) computed from the latest
     quarter whose ``available_date`` <= formation date and the formation-date
     market cap.  Next month's expected return += signal_strength *
     signal_scale * true_alpha.  A point-in-time strategy can recover it
     exactly (up to restatement noise).
  2. An announcement surprise that is NOT knowable: in the month a quarter
     becomes public, the return jumps by ``announcement_effect`` x the quarter's
     own EBIT news.  Own news is iid, so it is unpredictable from public data,
     but a backtest that keys on ``fiscal_period_end`` sees it early.
  ``signal_strength`` scales every channel through which fundamentals affect
  returns (expected return, death hazard tilts, bankruptcy share), so
  ``signal_strength=0`` is a genuine placebo.
* Death hazard: low TRUE profitability raises the hazard (realistic).  The
  ``value_trap`` regime additionally makes low TRUE valuation (high EBIT/EV)
  raise the hazard, so cheapness partly reflects distress.  Missing quarters
  and exchange delistings are concentrated in distressed / sub-$1 names.
* Prices lead fundamentals: a share of last quarter's idiosyncratic price shock
  flows into next quarter's revenue, which keeps valuation ratios anchored
  without creating any predictability from public data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import DataSource
from .cache import ParquetCache, config_hash
from .schema import coerce_benchmarks, coerce_fundamentals, coerce_returns

_GEN_VERSION = 7  # bump to invalidate cached panels when the generator changes

# name, sic_lo, sic_hi, weight, ppe_frac, total_assets/revenue, ebit margin, ltd_frac
SECTORS = [
    ("Technology",             3570, 3579, 0.14, 0.12, 0.70, 0.13, 0.08),
    ("Healthcare",             2830, 2836, 0.10, 0.15, 0.80, 0.12, 0.12),
    ("Industrials",            3500, 3569, 0.12, 0.30, 0.85, 0.09, 0.20),
    ("Consumer Discretionary", 5200, 5999, 0.12, 0.25, 0.55, 0.07, 0.18),
    ("Consumer Staples",       2000, 2099, 0.07, 0.30, 0.70, 0.09, 0.22),
    ("Energy",                 1300, 1389, 0.07, 0.55, 1.20, 0.10, 0.25),
    ("Materials",              2800, 2829, 0.06, 0.50, 1.00, 0.08, 0.25),
    ("Telecom",                4800, 4899, 0.04, 0.45, 1.50, 0.12, 0.35),
    ("Financials",             6000, 6999, 0.15, 0.03, 6.00, 0.20, 0.30),
    ("Utilities",              4900, 4999, 0.05, 0.65, 2.50, 0.14, 0.40),
    ("Business Services",      7370, 7379, 0.08, 0.10, 0.60, 0.10, 0.10),
]
EXCHANGES = np.array(["NYSE", "NASDAQ", "AMEX", "OTC"])
EXCHANGE_P = [0.45, 0.45, 0.08, 0.02]
DEATH_TYPES = ("bankruptcy", "acquisition", "delisting")


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    n_companies: int = 800          # target number alive at the start (grows 0.6%/yr)
    start: str = "1990-01-31"
    years: int = 35
    signal_strength: float = 1.0    # 0 = placebo world; 1 = calibrated to real-world factor spreads
    signal_scale: float = 0.0005    # monthly expected return per 1 sd of knowable true_alpha at strength 1
    quality_hazard_tilt: float = 0.2  # hazard *= exp(-tilt * s * z(true ROC)): unprofitable firms die more
    value_trap: float = 0.0         # hazard *= exp(+value_trap * s * z(true EBIT/EV)): cheap firms die more
    announcement_effect: float = 0.8   # share of a quarter's own EBIT news absorbed by price in the month it goes public
    price_to_fundamentals: float = 0.9  # share of last quarter's price change that flows into next quarter's revenue
    size_premium: float = 0.0005    # monthly return per -1 sd of log market cap (small beats big), z clipped at +-2
    market_premium: float = 0.005   # monthly equity premium
    market_vol: float = 0.045
    death_hazard_annual: float = 0.035
    lag_min_days: int = 45
    lag_max_days: int = 90
    late_rate: float = 0.03
    missing_rate: float = 0.02
    restate_rate: float = 0.02


def _normal_scores(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Rank-based z-scores among ``mask`` rows; NaN / unmasked -> 0."""
    out = np.zeros(x.shape[0])
    ok = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
    n = int(ok.sum())
    if n < 2:
        return out
    vals = x[ok]
    ranks = np.empty(n)
    ranks[np.argsort(vals, kind="stable")] = np.arange(1, n + 1)
    out[ok] = norm.ppf((ranks - 0.5) / n)
    return out


class _Simulator:
    def __init__(self, cfg: SyntheticConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------ setup
    def _draw_companies(self, N: int) -> None:
        rng, cfg = self.rng, self.cfg
        weights = np.array([s[3] for s in SECTORS])
        sec = rng.choice(len(SECTORS), N, p=weights / weights.sum())
        lo = np.array([s[1] for s in SECTORS])[sec]
        hi = np.array([s[2] for s in SECTORS])[sec]
        self.sector = sec
        self.sic = rng.integers(lo, hi + 1)
        self.exchange = EXCHANGES[rng.choice(4, N, p=EXCHANGE_P)]
        self.beta = np.clip(rng.normal(1.0, 0.3, N), 0.3, 2.2)
        self.idio_vol = rng.uniform(0.06, 0.14, N)
        # scale: annual revenue at the start of the quarter grid, long right tail
        self.log_rev0 = rng.normal(np.log(5e8), 1.5, N)
        self.drift = np.clip(rng.normal(0.0, 0.003, N), -0.01, 0.01)        # per quarter, on top of price-led growth
        self.rev_sigma = rng.uniform(0.03, 0.09, N)
        self.margin_base = rng.normal(np.array([s[6] for s in SECTORS])[sec] - 0.02, 0.09)
        self.margin_sigma = rng.uniform(0.004, 0.012, N)
        self.gm_gap = rng.uniform(0.15, 0.45, N)
        self.payout = np.where(rng.random(N) < 0.4, 0.0, rng.uniform(0.1, 0.6, N))
        self.accr_bias = rng.normal(0.0, 0.015, N)
        self.ta_ratio = np.array([s[5] for s in SECTORS])[sec] * np.exp(rng.normal(0, 0.4, N))
        ppe = np.clip(np.array([s[4] for s in SECTORS])[sec] * np.exp(rng.normal(0, 0.35, N)), 0.005, 0.85)
        ppe[rng.random(N) < 0.01] = 0.003                       # asset-light -> extreme ROC outliers
        self.ppe_frac = ppe
        self.ca_frac = np.clip(rng.uniform(0.15, 0.55, N) * (1 - ppe), 0.03, 0.7)
        self.cash_frac = self.ca_frac * rng.uniform(0.05, 0.5, N)
        self.cl_frac = np.clip(self.ca_frac * rng.uniform(0.4, 1.3, N), 0.02, 0.6)
        self.std_frac = rng.uniform(0, 0.08, N) * (rng.random(N) < 0.6)
        ltd = np.clip(np.array([s[7] for s in SECTORS])[sec] * np.exp(rng.normal(0, 0.5, N)), 0, 0.7)
        distressed = rng.random(N) < 0.03
        ltd[distressed] = rng.uniform(0.6, 0.95, distressed.sum())   # -> some negative book equity
        self.ltd_frac = ltd
        self.pref_frac = np.where(rng.random(N) < 0.9, 0.0, rng.uniform(0.01, 0.06, N))
        self.mi_frac = np.where(rng.random(N) < 0.85, 0.0, rng.uniform(0.005, 0.04, N))
        self.other_liab_frac = rng.uniform(0.03, 0.15, N)
        self.mcap_to_ta = np.exp(rng.normal(np.log(1.2), 0.5, N))
        price0 = np.exp(rng.normal(np.log(20), 0.7, N))
        penny = rng.random(N) < 0.03
        price0[penny] = rng.uniform(0.3, 2.0, penny.sum())
        self.price0 = price0
        self.lag_base = rng.uniform(cfg.lag_min_days, cfg.lag_max_days - 5, N)

    def _init_quarters(self, N: int, Q: int) -> None:
        rng = self.rng
        self.own_rev = rng.standard_normal((N, Q))
        self.own_margin = rng.standard_normal((N, Q))
        self.ta_noise = rng.standard_normal((N, Q))
        self.accr_noise = rng.standard_normal((N, Q))
        self.capex_noise = rng.standard_normal((N, Q))
        self.pchg_q = np.zeros((N, Q))         # log price change (ex announcement jumps) summed within each quarter
        self.missing = np.zeros((N, Q), bool)  # quarter never published (drawn when the quarter is computed)
        z = lambda: np.zeros((N, Q))
        self.log_rev_q, self.margin_state = z(), z()
        self.q = {k: z() for k in (
            "rev_q", "ebit_q", "gp_q", "ni_q", "ocf_q", "capex_q", "div_q",
            "rev_ttm", "ebit_ttm", "gp_ttm", "ni_ttm", "ocf_ttm", "capex_ttm", "div_ttm",
            "ta", "ppe", "ca", "cash", "cl", "std", "ltd", "pref", "mi", "be")}

    def _compute_quarter(self, k: int) -> None:
        cfg, q = self.cfg, self.q
        if k == 0:
            self.log_rev_q[:, 0] = self.log_rev0 - np.log(4)
            self.margin_state[:, 0] = 0.0
        else:
            self.log_rev_q[:, k] = (self.log_rev_q[:, k - 1] + self.drift + self.rev_sigma * self.own_rev[:, k]
                                    + cfg.price_to_fundamentals * self.pchg_q[:, k - 1])
            self.margin_state[:, k] = 0.85 * self.margin_state[:, k - 1] + self.margin_sigma * self.own_margin[:, k]
        rev = np.exp(self.log_rev_q[:, k])
        margin = self.margin_base + self.margin_state[:, k]
        q["rev_q"][:, k] = rev
        q["ebit_q"][:, k] = margin * rev
        q["gp_q"][:, k] = (margin + self.gm_gap) * rev
        lo = max(0, k - 3)
        n = k - lo + 1
        for flow in ("rev", "ebit", "gp"):
            q[f"{flow}_ttm"][:, k] = q[f"{flow}_q"][:, lo:k + 1].sum(axis=1) * (4.0 / n)
        ta = self.ta_ratio * q["rev_ttm"][:, k] * np.exp(0.05 * self.ta_noise[:, k])
        q["ta"][:, k] = ta
        q["ppe"][:, k] = self.ppe_frac * ta
        q["ca"][:, k] = self.ca_frac * ta
        q["cash"][:, k] = self.cash_frac * ta
        q["cl"][:, k] = self.cl_frac * ta
        q["std"][:, k] = self.std_frac * ta
        q["ltd"][:, k] = self.ltd_frac * ta
        q["pref"][:, k] = self.pref_frac * ta
        q["mi"][:, k] = self.mi_frac * ta
        q["be"][:, k] = ta * (1 - self.cl_frac - self.ltd_frac - self.pref_frac - self.mi_frac - self.other_liab_frac)
        interest = 0.015 * (q["std"][:, k] + q["ltd"][:, k])
        ebit = q["ebit_q"][:, k]
        tax = 0.25 * np.maximum(ebit - interest, 0)
        ni = ebit - interest - tax
        dep = 0.025 * q["ppe"][:, k]
        q["ni_q"][:, k] = ni
        q["ocf_q"][:, k] = ni + dep + (self.accr_bias + 0.03 * self.accr_noise[:, k]) * rev
        q["capex_q"][:, k] = dep * np.exp(0.1 + 0.4 * self.capex_noise[:, k])
        q["div_q"][:, k] = self.payout * np.maximum(ni, 0)
        for flow in ("ni", "ocf", "capex", "div"):
            q[f"{flow}_ttm"][:, k] = q[f"{flow}_q"][:, lo:k + 1].sum(axis=1) * (4.0 / n)
        # missing quarters concentrate in distressed firms (negative EBIT or negative book equity)
        distress = (q["ebit_ttm"][:, k] < 0) | (q["be"][:, k] < 0)
        p_missing = cfg.missing_rate * np.where(distress, 5.0, 0.6)
        self.missing[:, k] = self.rng.random(len(distress)) < p_missing

    # -------------------------------------------------------------------- run
    def run(self) -> Dict[str, pd.DataFrame]:
        cfg, rng = self.cfg, self.rng
        M = cfg.years * 12
        month_ends = pd.date_range(pd.Timestamp(cfg.start) + pd.offsets.MonthEnd(0), periods=M, freq="ME")
        first_q = month_ends[0] - pd.DateOffset(months=24) + pd.offsets.QuarterEnd(0)
        fpe = pd.date_range(first_q, month_ends[-1] + pd.offsets.QuarterEnd(0), freq="QE")
        Q = len(fpe)
        q_true = np.searchsorted(fpe.values, month_ends.values, side="right") - 1   # last quarter ended <= month end
        q_contain = np.searchsorted(fpe.values, month_ends.values, side="left")     # quarter containing the month
        N = cfg.n_companies * 4
        self._draw_companies(N)
        self._init_quarters(N, Q)
        q = self.q

        # publication lags (days) for every company-quarter
        jitter = rng.uniform(-3, 8, (N, Q))
        lag = np.clip(self.lag_base[:, None] + jitter, cfg.lag_min_days, cfg.lag_max_days)
        late = rng.random((N, Q)) < cfg.late_rate
        lag = lag + late * rng.uniform(30, 60, (N, Q))
        lag_days = np.round(lag).astype(int)
        avail = fpe.values[None, :] + lag_days.astype("timedelta64[D]")
        # month index whose month end >= available_date: the quarter is public from that month end on
        ann_month = np.searchsorted(month_ends.values, avail.ravel(), side="left").reshape(N, Q)
        # own news in log-EBIT units: revenue innovation + margin innovation relative to the margin level
        surprise = np.clip(self.rev_sigma[:, None] * self.own_rev
                           + self.margin_sigma[:, None] * self.own_margin / np.maximum(np.abs(self.margin_base), 0.05)[:, None],
                           -0.3, 0.3)
        jump = np.zeros((N, M))
        ii, kk = np.nonzero(ann_month < M)
        np.add.at(jump, (ii, ann_month[ii, kk]), cfg.announcement_effect * surprise[ii, kk])
        # publication events bucketed by month, so the knowable characteristic can be tracked in the loop
        flat_ann = ann_month.ravel()
        order = np.argsort(flat_ann, kind="stable")
        bounds = np.searchsorted(flat_ann[order], np.arange(M + 1))
        pub_i, pub_k = np.divmod(order, Q)

        # exogenous market and risk-free paths
        rf = np.empty(M)
        rf[0] = 0.004
        for m in range(1, M):
            rf[m] = max(0.0, 0.0025 + 0.985 * (rf[m - 1] - 0.0025) + rng.normal(0, 0.0003))
        mkt = rf + cfg.market_premium + cfg.market_vol * rng.standard_t(5, M) / np.sqrt(5 / 3)

        # state
        alive = np.zeros(N, bool)
        born_m = np.full(N, -1)
        death_m = np.full(N, -1)
        death_type = np.full(N, -1)
        kpub = np.full(N, -1)                 # latest quarter public as of the previous month end
        mcap_prev = np.zeros(N)
        shares = np.zeros(N)
        R = np.full((N, M), np.nan)
        MCAP = np.full((N, M), np.nan)
        MCAP0 = np.full((N, M), np.nan)
        SHARES = np.full((N, M), np.nan)
        ALPHA = np.full((N, M), np.nan)       # knowable characteristic at the month end (formation date)
        DELIST = np.zeros((N, M), bool)
        last_k = -1
        next_pool = 0
        base_h = cfg.death_hazard_annual / 12
        s = cfg.signal_strength
        rows = np.arange(N)

        def publish(month: int) -> None:
            """Quarters whose available_date falls in ``month`` become public at that month end."""
            lo_, hi_ = bounds[month], bounds[month + 1]
            i_, k_ = pub_i[lo_:hi_], pub_k[lo_:hi_]
            ok = ~self.missing[i_, k_]
            np.maximum.at(kpub, i_[ok], k_[ok])

        def ratios(kidx: np.ndarray, mcap: np.ndarray):
            """(EBIT/EV, EBIT/(NWC+NetPPE)) using quarter ``kidx`` per company; NaN where undefined."""
            kk_ = np.maximum(kidx, 0)
            ebit = q["ebit_ttm"][rows, kk_]
            ev = mcap + q["std"][rows, kk_] + q["ltd"][rows, kk_] + q["pref"][rows, kk_] + q["mi"][rows, kk_] - q["cash"][rows, kk_]
            capital = np.maximum(q["ca"][rows, kk_] - q["cash"][rows, kk_] - (q["cl"][rows, kk_] - q["std"][rows, kk_]), 0) + q["ppe"][rows, kk_]
            ey_ = np.where((ev > 0) & (kidx >= 0), ebit / np.where(ev > 0, ev, 1), np.nan)
            roc_ = np.where((capital > 0) & (kidx >= 0), ebit / np.where(capital > 0, capital, 1), np.nan)
            return ey_, roc_

        def knowable(mask: np.ndarray, mcap: np.ndarray) -> np.ndarray:
            ey_, roc_ = ratios(kpub, mcap)
            comp = 0.5 * _normal_scores(ey_, mask) + 0.5 * _normal_scores(roc_, mask)
            return _normal_scores(np.where(mask, comp, np.nan), mask)

        for m in range(M):
            k = q_true[m]
            while last_k < k:
                last_k += 1
                self._compute_quarter(last_k)
            if m >= 1:
                publish(m - 1)
            alive_before = alive.copy()
            # births keep the population near target (slow growth)
            target = int(round(cfg.n_companies * 1.006 ** (m / 12)))
            n_births = max(0, target - int(alive.sum()))
            if next_pool + n_births > N:
                raise RuntimeError("synthetic company pool exhausted; raise n_companies multiplier")
            idx = np.arange(next_pool, next_pool + n_births)
            next_pool += n_births
            alive[idx] = True
            born_m[idx] = m
            mcap_prev[idx] = q["ta"][idx, k] * self.mcap_to_ta[idx]
            shares[idx] = mcap_prev[idx] / self.price0[idx]

            # 1. knowable characteristic as of the previous month end -> drives this month's expected return
            char = knowable(alive, mcap_prev)
            if m >= 1:
                ALPHA[alive_before, m - 1] = char[alive_before]
            # 2. true state (may not be public yet) -> distress / death hazard
            ey_t, roc_t = ratios(np.full(N, k), mcap_prev)
            z_ey_true = _normal_scores(ey_t, alive)
            z_roc_true = _normal_scores(roc_t, alive)
            size_z = _normal_scores(np.where(alive, np.log(np.maximum(mcap_prev, 1)), np.nan), alive)

            eps = self.idio_vol * rng.standard_t(5, N) / np.sqrt(5 / 3)
            r = (rf[m] + self.beta * (mkt[m] - rf[m]) - cfg.size_premium * np.clip(size_z, -2, 2)
                 + s * cfg.signal_scale * char + jump[:, m] + eps)
            r = np.maximum(r, -0.9)

            # deaths: unprofitable firms die more; under value_trap cheap firms die more; sub-$1 names get delisted
            age = m - born_m
            neg = (q["ebit_ttm"][:, k] < 0).astype(float)
            penny = (mcap_prev / np.maximum(shares, 1) < 1.0).astype(float)
            hz = (base_h * np.exp(-cfg.quality_hazard_tilt * s * np.clip(z_roc_true, -2.5, 2.5)
                                  + cfg.value_trap * s * np.clip(z_ey_true, -2.5, 2.5))
                  * (1 + s * neg) * (1 + 1.0 * penny))
            die = alive & (age >= 12) & (rng.random(N) < hz)
            p_bk = np.clip(0.25 - 0.15 * s * z_roc_true + 0.15 * cfg.value_trap * s * z_ey_true, 0.05, 0.85)
            p_other = np.minimum(0.15 + 0.5 * penny, 0.95 - p_bk)
            u = rng.random(N)
            dtype = np.where(u < p_bk, 0, np.where(u < p_bk + p_other, 2, 1))
            dl_ret = np.where(dtype == 0, rng.uniform(-0.95, -0.70, N),
                              np.where(dtype == 1, rng.uniform(0.10, 0.40, N), rng.uniform(-0.50, -0.20, N)))
            r = np.where(die, dl_ret, r)

            dy = np.clip(q["div_ttm"][:, k] / (12 * np.maximum(mcap_prev, 1)), 0, 0.03)
            mcap_new = mcap_prev * (1 + r - dy)
            issue = rng.random(N) < 0.01
            buyback = rng.random(N) < 0.008
            shares = shares * np.where(issue, 1 + np.clip(rng.normal(0.03, 0.04, N), 0, 0.2), 1.0)
            shares = shares * np.where(buyback, 1 - rng.uniform(0.01, 0.05, N), 1.0)

            R[alive, m] = r[alive]
            MCAP[alive, m] = mcap_new[alive]
            MCAP0[alive, m] = mcap_prev[alive]
            SHARES[alive, m] = shares[alive]
            DELIST[die, m] = True
            pch = np.log1p(np.maximum(r - dy, -0.89)) - jump[:, m]
            self.pchg_q[alive & ~die, q_contain[m]] += pch[alive & ~die]
            alive[die] = False
            death_m[die] = m
            death_type[die] = dtype[die]
            mcap_prev = mcap_new
        # characteristic at the final month end (no return follows, but a formation there is still valid)
        publish(M - 1)
        char = knowable(alive, mcap_prev)
        ALPHA[alive, M - 1] = char[alive]

        ids = np.array([f"S{i:05d}" for i in range(N)])
        born = born_m >= 0
        # ---------------------------------------------------------- returns
        ii, mm = np.nonzero(~np.isnan(R))
        returns = pd.DataFrame({
            "security_id": ids[ii], "date": month_ends.values[mm], "total_return": R[ii, mm],
            "price": MCAP[ii, mm] / SHARES[ii, mm], "shares_outstanding": SHARES[ii, mm],
            "market_cap": MCAP[ii, mm], "exchange": self.exchange[ii], "sic_code": self.sic[ii],
            "is_delisted_this_month": DELIST[ii, mm],
        })
        ai, am = np.nonzero(~np.isnan(ALPHA))
        true_alpha_df = pd.DataFrame({"security_id": ids[ai], "date": month_ends.values[am], "true_alpha": ALPHA[ai, am]})

        # ------------------------------------------------------ fundamentals
        end_m = np.where(death_m >= 0, death_m, M - 1)
        k_lo = np.where(born, q_true[np.maximum(born_m, 0)] - 2, Q)
        k_hi = np.where(born, q_true[end_m], -1)
        K = np.arange(Q)[None, :]
        fmask = (K >= k_lo[:, None]) & (K <= k_hi[:, None]) & ~self.missing
        fi, fk = np.nonzero(fmask)
        m_fpe = np.clip(np.searchsorted(month_ends.values, fpe.values[fk], side="left"), 0, M - 1)
        m_sh = np.clip(m_fpe, born_m[fi], end_m[fi])
        shares_at = SHARES[fi, m_sh]
        f = pd.DataFrame({
            "security_id": ids[fi], "fiscal_period_end": fpe.values[fk], "available_date": avail[fi, fk],
            "ebit_ttm": q["ebit_ttm"][fi, fk], "revenue_ttm": q["rev_ttm"][fi, fk], "gross_profit_ttm": q["gp_ttm"][fi, fk],
            "net_income_ttm": q["ni_ttm"][fi, fk], "operating_cash_flow_ttm": q["ocf_ttm"][fi, fk],
            "capex_ttm": q["capex_ttm"][fi, fk], "dividends_ttm": q["div_ttm"][fi, fk],
            "current_assets": q["ca"][fi, fk], "cash": q["cash"][fi, fk], "current_liabilities": q["cl"][fi, fk],
            "short_term_debt": q["std"][fi, fk], "long_term_debt": q["ltd"][fi, fk], "net_ppe": q["ppe"][fi, fk],
            "total_assets": q["ta"][fi, fk], "book_equity": q["be"][fi, fk], "preferred_stock": q["pref"][fi, fk],
            "minority_interest": q["mi"][fi, fk], "shares_diluted": shares_at * 1.02,
        })
        # restatements: a second row for the same fiscal period, later available_date, perturbed values
        rs = f.sample(frac=cfg.restate_rate, random_state=int(rng.integers(1 << 31))).copy()
        rs["available_date"] = rs["available_date"] + pd.to_timedelta(rng.integers(60, 180, len(rs)), unit="D")
        pert = 1 + rng.uniform(-0.15, 0.15, len(rs))
        rs["ebit_ttm"] *= pert
        rs["net_income_ttm"] *= pert
        rs["revenue_ttm"] *= 1 + rng.uniform(-0.05, 0.05, len(rs))
        fundamentals = pd.concat([f, rs], ignore_index=True).sort_values(
            ["security_id", "fiscal_period_end", "available_date"]).reset_index(drop=True)

        # -------------------------------------------------------- benchmarks
        w = np.nan_to_num(MCAP0)
        vw = np.nansum(w * np.nan_to_num(R), axis=0) / w.sum(axis=0)
        ew = np.nanmean(R, axis=0)
        sp = np.empty(M)
        for m in range(M):
            a = ~np.isnan(R[:, m])
            caps = np.where(a, MCAP0[:, m], -np.inf)
            top = np.argsort(caps)[-500:]
            top = top[a[top]]
            sp[m] = (MCAP0[top, m] * R[top, m]).sum() / MCAP0[top, m].sum() + rng.normal(0, 0.0015)
        benchmarks = pd.DataFrame({"date": month_ends, "sp500_total_return": sp, "market_vw_return": vw,
                                   "market_ew_return": ew, "risk_free_rate": rf})

        companies = pd.DataFrame({
            "security_id": ids[born], "sector": np.array([s[0] for s in SECTORS])[self.sector[born]],
            "birth": month_ends.values[born_m[born]],
            "death": np.where(death_m[born] >= 0, month_ends.values[np.maximum(death_m[born], 0)], np.datetime64("NaT")),
            "death_type": np.where(death_m[born] >= 0, np.array(DEATH_TYPES)[np.maximum(death_type[born], 0)], "alive"),
        })
        return {
            "returns": coerce_returns(returns),
            "fundamentals": coerce_fundamentals(fundamentals),
            "benchmarks": coerce_benchmarks(benchmarks),
            "true_alpha": true_alpha_df,
            "companies": companies,
        }


class SyntheticSource(DataSource):
    """Simulated market.  See module docstring.  Parameters: see ``SyntheticConfig``."""

    name = "synthetic"

    def __init__(self, cache: bool = True, cache_dir: Optional[str] = None, **params: Any):
        self.cfg = SyntheticConfig(**params)
        self._cache = ParquetCache("synthetic", Path(cache_dir) if cache_dir else None) if cache else None
        self._key = config_hash({**asdict(self.cfg), "_v": _GEN_VERSION})
        self._frames: Optional[Dict[str, pd.DataFrame]] = None

    # -- loading ----------------------------------------------------------
    def _load(self) -> Dict[str, pd.DataFrame]:
        if self._frames is not None:
            return self._frames
        names = ("returns", "fundamentals", "benchmarks", "true_alpha", "companies")
        if self._cache is not None:
            got = {n: self._cache.get(f"{self._key}_{n}") for n in names}
            if all(v is not None for v in got.values()):
                self._frames = got
                return got
        frames = _Simulator(self.cfg).run()
        if self._cache is not None:
            for n in names:
                self._cache.put(f"{self._key}_{n}", frames[n])
        self._frames = frames
        return frames

    @staticmethod
    def _bounds(start, end):
        return pd.Timestamp(start), pd.Timestamp(end)

    def get_monthly_returns(self, start, end) -> pd.DataFrame:
        s, e = self._bounds(start, end)
        r = self._load()["returns"]
        return r[(r["date"] >= s) & (r["date"] <= e)].reset_index(drop=True)

    def get_fundamentals(self, start, end) -> pd.DataFrame:
        s, e = self._bounds(start, end)
        f = self._load()["fundamentals"]
        lo = s - pd.DateOffset(months=18)   # keep the last rows published before the window
        return f[(f["available_date"] >= lo) & (f["available_date"] <= e)].reset_index(drop=True)

    def get_benchmarks(self, start, end) -> pd.DataFrame:
        s, e = self._bounds(start, end)
        b = self._load()["benchmarks"]
        return b[(b["date"] >= s) & (b["date"] <= e)].reset_index(drop=True)

    def get_metadata(self) -> Dict[str, Any]:
        cfg = self.cfg
        start = pd.Timestamp(cfg.start) + pd.offsets.MonthEnd(0)
        end = start + pd.DateOffset(months=cfg.years * 12 - 1) + pd.offsets.MonthEnd(0)
        return {
            "name": "synthetic",
            "display_name": "Synthetic simulated market",
            "coverage_start": start.strftime("%Y-%m-%d"),
            "coverage_end": end.strftime("%Y-%m-%d"),
            "is_simulated": True,
            "survivorship_bias": False,
            "caveats": [
                "SIMULATED DATA. Every company, price and financial statement is randomly generated. "
                "Nothing here is a historical result.",
                f"A signal is planted on purpose (signal_strength={cfg.signal_strength}); "
                "outperformance is expected by construction and says nothing about real markets.",
                "Use this source only to verify that the engine works.",
            ],
            "params": asdict(cfg),
            "ui_params": {
                "seed": {"label": "Random seed", "type": "int", "default": cfg.seed, "min": 0, "max": 99999, "step": 1,
                         "help": "Every seed is an independent simulated market. Try a few: the difference between seeds is pure luck, "
                                 "and it shows how much luck a single backtest contains."},
                "signal_strength": {"label": "Planted signal strength", "type": "float", "default": cfg.signal_strength, "min": 0.0, "max": 2.0, "step": 0.25,
                                    "help": "How strongly cheap, profitable companies outperform in this simulated market. "
                                            "0 = placebo (fundamentals predict nothing; any 'outperformance' is luck). "
                                            "1 = calibrated to real-world factor spreads (~0.45% per month between the best and worst decile)."},
                "value_trap": {"label": "Value-trap severity", "type": "float", "default": cfg.value_trap, "min": 0.0, "max": 1.0, "step": 0.25,
                               "help": "Above 0, cheap companies go bankrupt more often, so low valuations partly signal distress. "
                                       "Tests whether a value strategy survives the classic value trap."},
            },
            "planted_signal": "true_alpha (knowable at each month end from published fundamentals) = normal score of "
                              "0.5*z(EBIT/EV) + 0.5*z(EBIT/(NWC+NetPPE)); next month's expected return += "
                              "signal_strength * signal_scale * true_alpha. Announcement surprises are separate and unknowable.",
            "value_trap": cfg.value_trap,
        }

    @staticmethod
    def static_metadata() -> Dict[str, Any]:
        return SyntheticSource(cache=False).get_metadata()

    # -- extras for tests and the checkpoint (NOT part of the DataSource contract)
    def true_alpha(self) -> pd.DataFrame:
        """Knowable characteristic per (security_id, date=formation month end).

        Built only from fundamentals with available_date <= date and the market cap at date, so a
        point-in-time strategy could compute it.  It drives the expected return of the *following* month.
        """
        return self._load()["true_alpha"]

    def companies(self) -> pd.DataFrame:
        return self._load()["companies"]

    def summarize_panel(self) -> Dict[str, Any]:
        fr = self._load()
        r, f, c = fr["returns"], fr["fundamentals"], fr["companies"]
        yr = r["date"].dt.year
        alive_per_year = r.groupby(yr)["security_id"].nunique()
        deaths = c[c["death"].notna()].groupby([c["death"].dt.year, "death_type"]).size().unstack(fill_value=0)
        lag = (f["available_date"] - f["fiscal_period_end"]).dt.days
        first_pub = f.drop_duplicates(["security_id", "fiscal_period_end"], keep="first")
        lag_first = (first_pub["available_date"] - first_pub["fiscal_period_end"]).dt.days
        # missing quarters: gaps in each company's fiscal period sequence
        g = first_pub.groupby("security_id")["fiscal_period_end"]
        expected = ((g.max() - g.min()).dt.days / 91.3).round().astype(int) + 1
        missing_rate = 1 - first_pub.groupby("security_id").size().sum() / expected.sum()
        restate_rate = f.duplicated(["security_id", "fiscal_period_end"]).mean()
        caps = r["market_cap"]
        return {
            "companies_total": int(c.shape[0]),
            "companies_alive_per_year": alive_per_year,
            "deaths_per_year": deaths,
            "deaths_total_by_type": c["death_type"].value_counts().to_dict(),
            "returns_rows": int(len(r)),
            "fundamentals_rows": int(len(f)),
            "lag_days_first_publication": lag_first.describe(percentiles=[0.05, 0.5, 0.95]).round(1).to_dict(),
            "lag_days_all_rows_incl_restatements": lag.describe(percentiles=[0.05, 0.5, 0.95]).round(1).to_dict(),
            "missing_quarter_rate": float(missing_rate),
            "restatement_rate": float(restate_rate),
            "negative_ebit_rate": float((f["ebit_ttm"] < 0).mean()),
            "negative_book_equity_rate": float((f["book_equity"] < 0).mean()),
            "delisting_return_stats": r.loc[r["is_delisted_this_month"], "total_return"].describe().round(3).to_dict(),
            "market_cap_percentiles": caps.quantile([0.01, 0.1, 0.5, 0.9, 0.99, 1.0]).round(0).to_dict(),
            "sector_shares": c["sector"].value_counts(normalize=True).round(3).to_dict(),
            "exchange_shares": r.drop_duplicates("security_id")["exchange"].value_counts(normalize=True).round(3).to_dict(),
            "price_below_1_rate": float((r["price"] < 1).mean()),
        }


def _main() -> None:
    ap = argparse.ArgumentParser(description="Generate / summarize the synthetic panel")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--signal-strength", type=float, default=1.0)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    import time

    t0 = time.time()
    src = SyntheticSource(seed=args.seed, signal_strength=args.signal_strength)
    src._load()
    print(f"loaded in {time.time() - t0:.1f}s  [{src.describe()}]")
    if args.summary:
        pd.set_option("display.width", 160)
        for k, v in src.summarize_panel().items():
            print(f"\n== {k}")
            print(v.to_string() if isinstance(v, (pd.Series, pd.DataFrame)) else v)


if __name__ == "__main__":
    _main()
