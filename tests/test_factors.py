"""Factor arithmetic on a hand-built table.  Every expected number below is worked by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors import compute_factors, factor_coverage, get_factor, list_factors


def _snapshot() -> pd.DataFrame:
    # Stock A: ordinary.  B: negative NWC (floored).  C: cash-rich shell (EV<0), no capital, negative equity.
    # D: negative EBIT and net income.  E: missing fundamentals entirely.
    rows = {
        #                  A       B       C       D       E
        "market_cap":     [1000.,  500.,   100.,   800.,   300.],
        "price":          [10.,    5.,     1.,     8.,     3.],
        "ebit_ttm":       [100.,   40.,    5.,     -30.,   np.nan],
        "revenue_ttm":    [2000.,  600.,   50.,    900.,   np.nan],
        "gross_profit_ttm": [800., 200.,   20.,    300.,   np.nan],
        "net_income_ttm": [60.,    25.,    3.,     -50.,   np.nan],
        "operating_cash_flow_ttm": [90., 20., 4.,   -10.,   np.nan],
        "capex_ttm":      [30.,    10.,    1.,     20.,    np.nan],
        "dividends_ttm":  [20.,    0.,     0.,     8.,     np.nan],
        "current_assets": [300.,   100.,   520.,   400.,   np.nan],
        "cash":           [50.,    50.,    500.,   100.,   np.nan],
        "current_liabilities": [200., 300., 20.,   250.,   np.nan],
        "short_term_debt": [20.,   0.,     0.,     50.,    np.nan],
        "long_term_debt": [300.,   0.,     0.,     400.,   np.nan],
        "net_ppe":        [400.,   200.,   0.,     600.,   np.nan],
        "total_assets":   [1200.,  400.,   520.,   1500.,  np.nan],
        "book_equity":    [500.,   100.,   -30.,   200.,   np.nan],
        "preferred_stock": [0.,    0.,     0.,     50.,    np.nan],
        "minority_interest": [0.,  0.,     0.,     10.,    np.nan],
        "shares_diluted": [100.,   100.,   100.,   100.,   np.nan],
        "total_assets_prior": [1000., 500., np.nan, 1500., np.nan],
        "shares_diluted_prior": [110., 90.,  100.,  np.nan, np.nan],
        "mom_12_1":       [0.25,   -0.10,  np.nan, 0.0,    0.05],
        "vol_12m":        [0.20,   0.60,   np.nan, 0.30,   0.25],
    }
    return pd.DataFrame(rows, index=list("ABCDE"))


S = _snapshot()
F = compute_factors(S, list_factors())


def test_return_on_capital():
    # A: NWC = (300-50) - (200-20) = 70; capital = 70 + 400 = 470; 100/470
    assert F.loc["A", "return_on_capital"] == pytest.approx(100 / 470)
    # B: NWC = (100-50) - (300-0) = -250 -> floored to 0; capital = 200; 40/200
    assert F.loc["B", "return_on_capital"] == pytest.approx(0.20)
    # C: NWC = (520-500) - (20-0) = 0; PPE 0 -> capital 0 -> NaN
    assert np.isnan(F.loc["C", "return_on_capital"])
    # D: NWC = (400-100) - (250-50) = 100; capital 700; -30/700 (negative kept)
    assert F.loc["D", "return_on_capital"] == pytest.approx(-30 / 700)
    assert np.isnan(F.loc["E", "return_on_capital"])


def test_earnings_yield():
    # A: EV = 1000 + 300 + 20 + 0 + 0 - 50 = 1270
    assert F.loc["A", "earnings_yield"] == pytest.approx(100 / 1270)
    # B: EV = 500 - 50 = 450
    assert F.loc["B", "earnings_yield"] == pytest.approx(40 / 450)
    # C: EV = 100 - 500 = -400 -> NaN
    assert np.isnan(F.loc["C", "earnings_yield"])
    # D: EV = 800 + 400 + 50 + 50 + 10 - 100 = 1210; -30/1210
    assert F.loc["D", "earnings_yield"] == pytest.approx(-30 / 1210)


def test_value_factors():
    assert F.loc["A", "book_to_market"] == pytest.approx(0.5)
    assert np.isnan(F.loc["C", "book_to_market"])            # negative book equity
    assert F.loc["A", "earnings_to_price"] == pytest.approx(0.06)
    assert F.loc["D", "earnings_to_price"] == pytest.approx(-50 / 800)
    assert F.loc["A", "sales_to_price"] == pytest.approx(2.0)
    assert F.loc["A", "fcf_yield"] == pytest.approx((90 - 30) / 1000)
    assert F.loc["D", "fcf_yield"] == pytest.approx((-10 - 20) / 800)
    # shareholder yield A: dividends 20 + buyback (110-100)*10 = 100 -> 120/1000
    assert F.loc["A", "shareholder_yield"] == pytest.approx(0.12)
    # B: no dividends, issued 10 shares at $5 -> -50/500
    assert F.loc["B", "shareholder_yield"] == pytest.approx(-0.10)
    assert np.isnan(F.loc["D", "shareholder_yield"])         # no prior share count


def test_quality_factors():
    assert F.loc["A", "gross_profitability"] == pytest.approx(800 / 1200)
    assert F.loc["A", "roe"] == pytest.approx(60 / 500)
    assert np.isnan(F.loc["C", "roe"])                        # negative equity
    assert F.loc["A", "accruals"] == pytest.approx((60 - 90) / 1200)
    assert F.loc["D", "accruals"] == pytest.approx((-50 + 10) / 1500)
    assert F.loc["A", "asset_growth"] == pytest.approx(0.2)
    assert F.loc["B", "asset_growth"] == pytest.approx(-0.2)
    assert np.isnan(F.loc["C", "asset_growth"])


def test_price_factors_and_directions():
    assert F.loc["A", "momentum_12_1"] == 0.25 and np.isnan(F.loc["C", "momentum_12_1"])
    assert F.loc["B", "volatility_12m"] == 0.60
    assert get_factor("volatility_12m").direction == "lower"
    assert get_factor("accruals").direction == "lower"
    assert get_factor("asset_growth").direction == "lower"
    for n in ("return_on_capital", "earnings_yield", "book_to_market", "momentum_12_1", "shareholder_yield"):
        assert get_factor(n).direction == "higher"


def test_registry_and_metadata():
    assert len(list_factors()) == 13
    for n in list_factors():
        f = get_factor(n)
        assert f.description and f.formula and f.requires and f.missing_policy
    with pytest.raises(KeyError):
        get_factor("not_a_factor")
    with pytest.raises(KeyError):
        get_factor("earnings_yield").compute(S.drop(columns=["cash"]))


def test_coverage_report():
    cov = factor_coverage(S)
    assert cov.loc["earnings_yield", "n_available"] == 3          # A, B, D
    assert cov.loc["return_on_capital", "missing_rate"] == pytest.approx(0.4)
    assert cov.loc["shareholder_yield", "n_available"] == 3       # A, B, C
