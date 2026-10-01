"""Composite ranking arithmetic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ranking import RankingConfig, rank_stocks, select_top


def _values(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"earnings_yield": rng.normal(0.08, 0.05, n), "return_on_capital": rng.normal(0.2, 0.1, n)},
                        index=[f"S{i:02d}" for i in range(n)])


def test_rank_sum_fifth_plus_twenty_sixth_is_31():
    v = _values()
    # force S00 to be exactly 5th on earnings yield and 26th on return on capital
    v.loc["S00", "earnings_yield"] = v["earnings_yield"].drop("S00").sort_values(ascending=False).iloc[3] - 1e-9  # just below 4th
    v.loc["S00", "return_on_capital"] = v["return_on_capital"].drop("S00").sort_values(ascending=False).iloc[24] - 1e-9
    r = rank_stocks(v, RankingConfig({"earnings_yield": 0.5, "return_on_capital": 0.5}))
    assert r.loc["S00", "earnings_yield_rank"] == 5
    assert r.loc["S00", "return_on_capital_rank"] == 26
    assert r.loc["S00", "score_raw"] == 31            # sum of ranks before normalization
    assert r.loc["S00", "score"] == 15.5              # weighted average rank
    assert r["rank"].iloc[0] == 1 and r["score"].is_monotonic_increasing


def test_weights_are_normalized_and_matter():
    v = _values()
    a = rank_stocks(v, RankingConfig({"earnings_yield": 0.4, "return_on_capital": 0.6}))
    b = rank_stocks(v, RankingConfig({"earnings_yield": 2, "return_on_capital": 3}))
    pd.testing.assert_series_equal(a["rank"], b["rank"])
    only_ey = rank_stocks(v, RankingConfig({"earnings_yield": 1.0}))
    assert (only_ey["rank"] == only_ey["earnings_yield_rank"].rank(method="first")).all()
    assert not a["rank"].equals(only_ey["rank"].reindex(a.index))


def test_lower_is_better_factor_is_reversed():
    v = pd.DataFrame({"volatility_12m": [0.1, 0.5, 0.3]}, index=list("ABC"))
    r = rank_stocks(v, RankingConfig({"volatility_12m": 1.0}))
    assert r["volatility_12m_rank"].to_dict() == {"A": 1, "C": 2, "B": 3}
    z = rank_stocks(v, RankingConfig({"volatility_12m": 1.0}, method="zscore"))
    assert z["rank"].to_dict() == {"A": 1, "C": 2, "B": 3}   # low vol -> high score


def test_missing_policy_drop_vs_median():
    v = _values(10)
    v.loc["S03", "return_on_capital"] = np.nan
    dropped = rank_stocks(v, RankingConfig({"earnings_yield": 1, "return_on_capital": 1}))
    assert "S03" not in dropped.index and len(dropped) == 9
    med = rank_stocks(v, RankingConfig({"earnings_yield": 1, "return_on_capital": 1}, missing_policy="median"))
    assert "S03" in med.index and med.loc["S03", "return_on_capital_rank"] == 5.0   # (9 + 1) / 2
    z = rank_stocks(v, RankingConfig({"earnings_yield": 1, "return_on_capital": 1}, method="zscore", missing_policy="median"))
    assert z.loc["S03", "return_on_capital_z"] == 0.0


def test_zscore_winsorizes_outliers():
    v = _values(200)
    v.loc["S00", "earnings_yield"] = 1e6          # absurd outlier
    z = rank_stocks(v, RankingConfig({"earnings_yield": 1.0}, method="zscore"))
    assert z.loc["S00", "rank"] == 1
    assert z["earnings_yield_z"].max() < 4          # winsorized, not 14 sigma


def test_select_top_and_config_validation():
    r = rank_stocks(_values(), RankingConfig({"earnings_yield": 1, "return_on_capital": 1}))
    assert list(select_top(r, 5)["rank"]) == [1, 2, 3, 4, 5]
    with pytest.raises(KeyError):
        RankingConfig({"nope": 1.0})
    with pytest.raises(ValueError):
        RankingConfig({"earnings_yield": -1.0})
    with pytest.raises(ValueError):
        RankingConfig({"earnings_yield": 1.0}, method="magic")
