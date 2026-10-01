"""Free source: the SEC fact -> TTM logic on a hand-built companyfacts dict.  No network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import get_source
from src.data.free import build_company_fundamentals, shares_history, ttm_from_durations, _concept


def _fact(start, end, val, filed, form="10-Q"):
    d = {"end": end, "val": val, "filed": filed, "form": form, "fy": int(end[:4]), "fp": "Q1"}
    if start:
        d["start"] = start
    return d


def _facts():
    # fiscal year = calendar year.  FY2019 revenue 400 (Q 100 each); FY2020: Q1 110, Q2 120, Q3 130, Q4 140.
    rev = [
        _fact("2019-01-01", "2019-12-31", 400, "2020-02-15", "10-K"),
        _fact("2019-01-01", "2019-03-31", 100, "2019-05-01"), _fact("2019-01-01", "2019-06-30", 200, "2019-08-01"),
        _fact("2019-04-01", "2019-06-30", 100, "2019-08-01"), _fact("2019-01-01", "2019-09-30", 300, "2019-11-01"),
        _fact("2020-01-01", "2020-03-31", 110, "2020-05-01"), _fact("2020-01-01", "2020-06-30", 230, "2020-08-01"),
        _fact("2020-04-01", "2020-06-30", 120, "2020-08-01"), _fact("2020-01-01", "2020-09-30", 360, "2020-11-01"),
        _fact("2020-01-01", "2020-12-31", 500, "2021-02-15", "10-K"),
        # a restated FY2019 value in the FY2020 10-K comparatives: must be ignored (as first reported)
        _fact("2019-01-01", "2019-12-31", 999, "2021-02-15", "10-K"),
    ]
    assets = [_fact(None, e, v, f) for e, v, f in (("2019-12-31", 1000, "2020-02-15"), ("2020-03-31", 1010, "2020-05-01"),
                                                   ("2020-06-30", 1020, "2020-08-01"), ("2020-09-30", 1030, "2020-11-01"),
                                                   ("2020-12-31", 1040, "2021-02-15"))]
    op = [_fact("2019-01-01", "2019-12-31", 40, "2020-02-15", "10-K"), _fact("2020-01-01", "2020-06-30", 30, "2020-08-01"),
          _fact("2019-01-01", "2019-06-30", 20, "2019-08-01"), _fact("2020-01-01", "2020-12-31", 60, "2021-02-15", "10-K")]
    shares = [_fact(None, "2020-01-20", 1000, "2020-02-15"), _fact(None, "2020-07-20", 990, "2020-08-01")]
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": rev}}, "Assets": {"units": {"USD": assets}},
                                  "OperatingIncomeLoss": {"units": {"USD": op}}},
                      "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": shares}}}}}


def test_ttm_from_durations_rolls_correctly_and_ignores_restatements():
    facts = _facts()
    t = ttm_from_durations(_concept(facts, ["Revenues"])).set_index("end")
    assert t.loc["2019-12-31", "ttm"] == 400                                   # as first reported, not 999
    assert t.loc["2019-12-31", "filed"] == pd.Timestamp("2020-02-15")
    # TTM at 2020-06-30 = FY2019 400 + YTD-2020 230 - YTD-2019 200 = 430
    assert t.loc["2020-06-30", "ttm"] == 430 and t.loc["2020-06-30", "filed"] == pd.Timestamp("2020-08-01")
    assert t.loc["2020-09-30", "ttm"] == 400 + 360 - 300
    assert t.loc["2020-12-31", "ttm"] == 500
    assert t.loc["2020-03-31", "ttm"] == 400 + 110 - 100


def test_build_company_fundamentals_rows_and_availability():
    f = build_company_fundamentals(_facts(), "TEST").set_index("fiscal_period_end")
    assert list(f.index) == [pd.Timestamp(d) for d in ("2019-12-31", "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31")]
    assert f.loc["2020-06-30", "revenue_ttm"] == 430 and f.loc["2020-06-30", "total_assets"] == 1020
    assert f.loc["2020-06-30", "ebit_ttm"] == 40 + 30 - 20
    assert np.isnan(f.loc["2020-09-30", "ebit_ttm"])                             # no 9-month operating income reported
    assert (f["available_date"] >= f.index).all()
    assert f.loc["2020-06-30", "available_date"] == pd.Timestamp("2020-08-01")
    assert f.loc["2020-06-30", "preferred_stock"] == 0.0                         # absent tag -> 0 for optional items
    assert np.isnan(f.loc["2020-06-30", "net_ppe"])                              # absent required item -> NaN


def test_shares_history_and_metadata_without_network():
    sh = shares_history(_facts())
    assert sh["val"].tolist() == [1000, 990] and sh["filed"].iloc[1] == pd.Timestamp("2020-08-01")
    src = get_source("free", cache=False, max_tickers=5)
    m = src.get_metadata()
    assert m["survivorship_bias"] is True and m["is_simulated"] is False
    assert any("optimistic" in c for c in m["caveats"])
