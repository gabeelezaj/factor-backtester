"""WRDS adapter: the pure transforms, tested offline on hand-built frames.  No connection is made."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import get_source
from src.data.schema import validate_fundamentals, validate_returns
from src.data.wrds_source import (EXPECTED_TABLES, WRDSConfig, WRDSSource, availability_dates, build_ttm,
                                  compound_delisting_returns, fundq_to_canonical, legacy_to_returns,
                                  merge_quarterly_and_annual, resolve_links, ytd_to_quarterly)

CFG = WRDSConfig()


def _msf(permno, months, ret=0.01):
    dates = pd.date_range("2020-01-31", periods=months, freq="ME")
    return pd.DataFrame({"permno": permno, "date": dates, "ret": ret, "prc": -10.0, "shrout": 1000.0,
                         "shrcd": 10, "exchcd": 1, "siccd": 3570})


def test_compound_delisting_returns():
    msf = pd.concat([_msf(1, 6), _msf(2, 3), _msf(3, 4), _msf(4, 4), _msf(5, 6)], ignore_index=True)
    delist = pd.DataFrame({
        "permno": [1, 2, 3, 4],
        "dlstdt": [pd.Timestamp("2020-04-15"), pd.Timestamp("2020-04-10"), pd.Timestamp("2020-04-20"), pd.Timestamp("2020-04-20")],
        "dlret": [-0.5, 0.2, np.nan, np.nan],
        "dlstcd": [552, 231, 550, 233],        # 1: perf w/ dlret; 2: merger, no April row; 3: perf, missing; 4: merger, missing
    })
    out = compound_delisting_returns(msf, delist, missing_fill=-0.30)
    apr = pd.Timestamp("2020-04-30")
    a = out[(out.permno == 1) & (out.date == apr)].iloc[0]
    assert a["ret"] == pytest.approx((1 + 0.01) * (1 - 0.5) - 1) and a["is_delisted_this_month"]
    assert out[out.permno == 1]["date"].max() == apr                       # May, June dropped
    b = out[(out.permno == 2) & (out.date == apr)].iloc[0]
    assert b["ret"] == pytest.approx(0.2) and b["is_delisted_this_month"]  # row added
    c = out[(out.permno == 3) & (out.date == apr)].iloc[0]
    assert c["ret"] == pytest.approx((1 + 0.01) * (1 - 0.30) - 1)          # Shumway fill
    d = out[(out.permno == 4) & (out.date == apr)].iloc[0]
    assert d["ret"] == pytest.approx(0.01)                                  # non-performance: 0 fill
    assert not out[out.permno == 5]["is_delisted_this_month"].any() and len(out[out.permno == 5]) == 6


def test_legacy_to_returns_units_and_validation():
    msf = _msf(7, 3)
    msf["is_delisted_this_month"] = [False, False, True]
    r = legacy_to_returns(msf, CFG)
    validate_returns(r)
    assert (r["price"] == 10.0).all() and (r["shares_outstanding"] == 1e6).all() and (r["market_cap"] == 1e7).all()
    assert (r["exchange"] == "NYSE").all() and (r["security_id"] == "7").all()


def test_ytd_to_quarterly():
    f = pd.DataFrame({"gvkey": "1", "fyearq": 2020, "fqtr": [1, 2, 3, 4], "oancfy": [10.0, 25.0, 45.0, 70.0],
                      "capxy": [1.0, 2.0, 3.0, 4.0], "dvy": [0.0, 0.0, 0.0, 0.0]})
    q = ytd_to_quarterly(f)
    assert q["oancfq"].tolist() == [10.0, 15.0, 20.0, 25.0]
    gap = f[f.fqtr != 2].copy()                                             # Q2 missing -> Q3 difference undefined
    q2 = ytd_to_quarterly(gap)
    assert q2["oancfq"].tolist()[:1] == [10.0] and np.isnan(q2["oancfq"].iloc[1]) and q2["oancfq"].iloc[2] == 25.0


def test_build_ttm_requires_four_consecutive_quarters():
    dates = list(pd.date_range("2019-03-31", periods=6, freq="QE"))
    f = pd.DataFrame({"gvkey": "1", "datadate": dates, "x": [1.0, 2, 3, 4, 5, 6]})
    t = build_ttm(f, {"x": "x_ttm"})
    assert np.isnan(t["x_ttm"].iloc[2]) and t["x_ttm"].iloc[3] == 10 and t["x_ttm"].iloc[5] == 18
    g = f.drop(index=3)                                                     # a missing quarter breaks the window
    t2 = build_ttm(g, {"x": "x_ttm"})
    assert np.isnan(t2["x_ttm"].iloc[3]) and np.isnan(t2["x_ttm"].iloc[4])


def test_availability_dates():
    f = pd.DataFrame({"datadate": pd.to_datetime(["2020-03-31"] * 3),
                      "rdq": [pd.Timestamp("2020-05-05"), pd.NaT, pd.Timestamp("2020-03-01")]})
    a = availability_dates(f, CFG)
    assert a.iloc[0] == pd.Timestamp("2020-05-05")
    assert a.iloc[1] == pd.Timestamp("2020-03-31") + pd.Timedelta(days=90)
    assert a.iloc[2] == pd.Timestamp("2020-03-31") + pd.Timedelta(days=90)   # rdq before period end -> fallback
    assert availability_dates(f, CFG, annual=True).iloc[0] == pd.Timestamp("2020-07-31")


def test_resolve_links():
    fund = pd.DataFrame({"gvkey": ["1", "1", "2"], "datadate": pd.to_datetime(["2015-12-31", "2020-12-31", "2020-12-31"]), "v": [1, 2, 3]})
    link = pd.DataFrame({
        "gvkey": ["1", "1", "2", "2"], "lpermno": [100, 101, 200, 201],
        "linktype": ["LU", "LC", "LU", "LX"], "linkprim": ["P", "P", "C", "P"],
        "linkdt": pd.to_datetime(["2010-01-01", "2018-01-01", "2000-01-01", "2000-01-01"]),
        "linkenddt": [pd.Timestamp("2017-12-31"), pd.NaT, pd.NaT, pd.NaT],
    })
    m = resolve_links(fund, link)
    got = dict(zip(zip(m["gvkey"], m["datadate"]), m["permno"]))
    assert got[("1", pd.Timestamp("2015-12-31"))] == 100          # within first link range
    assert got[("1", pd.Timestamp("2020-12-31"))] == 101          # open-ended second link
    assert got[("2", pd.Timestamp("2020-12-31"))] == 200          # LX link ignored


def test_fundq_to_canonical_units_and_schema():
    dates = list(pd.date_range("2019-03-31", periods=8, freq="QE"))
    n = len(dates)
    fundq = pd.DataFrame({
        "gvkey": "1", "datadate": dates, "fyearq": [d.year for d in dates], "fqtr": [d.quarter for d in dates],
        "rdq": [d + pd.Timedelta(days=45) for d in dates], "indfmt": "INDL", "datafmt": "STD", "popsrc": "D", "consol": "C",
        "saleq": 100.0, "cogsq": 60.0, "oiadpq": 10.0, "niq": 6.0,
        "oancfy": [8.0, 16, 24, 32] * 2, "capxy": [2.0, 4, 6, 8] * 2, "dvy": [1.0, 2, 3, 4] * 2,
        "actq": 50.0, "cheq": 10.0, "lctq": 30.0, "dlcq": 5.0, "dlttq": 40.0, "ppentq": 80.0, "atq": 200.0,
        "ceqq": 90.0, "pstkq": 0.0, "mibq": 1.0, "txditcq": 2.0, "cshfdq": 10.0, "cshoq": 9.8,
    })
    link = pd.DataFrame({"gvkey": ["1"], "lpermno": [123], "linktype": ["LU"], "linkprim": ["P"],
                         "linkdt": [pd.Timestamp("2000-01-01")], "linkenddt": [pd.NaT]})
    out = fundq_to_canonical(fundq, link, CFG)
    validate_fundamentals(out)
    last = out.iloc[-1]
    assert last["security_id"] == "123" and last["available_date"] == dates[-1] + pd.Timedelta(days=45)
    assert last["ebit_ttm"] == pytest.approx(40e6) and last["revenue_ttm"] == pytest.approx(400e6)
    assert last["gross_profit_ttm"] == pytest.approx(160e6)
    assert last["operating_cash_flow_ttm"] == pytest.approx(32e6)     # quarterly 8 each, from YTD
    assert last["dividends_ttm"] == pytest.approx(4e6)
    assert last["book_equity"] == pytest.approx(92e6) and last["total_assets"] == pytest.approx(200e6)
    assert np.isnan(out["ebit_ttm"].iloc[0])                            # fewer than 4 quarters
    merged = merge_quarterly_and_annual(out, out.iloc[:1].assign(fiscal_period_end=pd.Timestamp("2018-12-31")))
    assert len(merged) == len(out) + 1


def test_registry_constructs_without_wrds_package_and_connect_fails_cleanly():
    src = get_source("wrds", cache=False)
    assert src.get_metadata()["is_simulated"] is False
    with pytest.raises(ImportError):
        src.connect()


def test_verify_schema_with_fake_connection(capsys):
    class FakeDB:
        def describe_table(self, library, table):
            if table == "msf":
                return pd.DataFrame({"name": ["permno", "date", "ret", "prc"]})   # shrout "missing"
            if table == "msedelist":
                return pd.DataFrame({"name": ["permno", "dlstdt", "dlret", "dlstcd", "dlpdt"]})
            raise RuntimeError("no such table")

    src = WRDSSource(cache=False)
    src._db = FakeDB()
    rep = src.verify_schema({"crsp.msf": EXPECTED_TABLES["crsp.msf"], "crsp.msedelist": EXPECTED_TABLES["crsp.msedelist"],
                             "comp.fundq": ["gvkey"]})
    missing = rep[rep["expected"] & ~rep["found"]]
    assert set(missing["column"]) == {"shrout", "*"}
    assert "dlpdt" in rep[rep["column"] == "(other columns present)"]["note"].iloc[1]
