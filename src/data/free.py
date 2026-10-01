"""Free real-data source: yfinance prices + SEC XBRL "companyfacts" fundamentals.

Honest labelling first:
* The universe is a static list of TODAY'S large/mid caps (``free_universe.py``) -> survivorship bias:
  results are optimistic.  A ticker that stops trading inside the window gets its last row flagged, but
  there is no delisting return.
* SEC companyfacts is genuinely point-in-time: every value carries the ``filed`` date of the filing that
  reported it, which becomes ``available_date``.  Values are taken as first reported (earliest filing per
  period); later comparatives/restatements are ignored.
* History is short: XBRL company facts start around 2009; the price download defaults to 2005.
* Shares outstanding come from the DEI cover-page fact (point-in-time, as of each filing); when absent,
  yfinance's *current* share count is used for every date (flagged in metadata).
* The S&P 500 benchmark is SPY's adjusted-close return (a proxy: ETF expenses, tracking).  VW/EW market
  benchmarks are computed from this same survivor universe.  Risk-free = 13-week T-bill yield (^IRX).

The SEC asks for a descriptive User-Agent with a contact: set ``SEC_USER_AGENT="Your Name you@example.com"``.
Optional dependencies (``yfinance``, ``requests``) are imported only inside this module's functions.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import DataSource
from .cache import CACHE_ROOT, ParquetCache, config_hash
from .free_universe import TICKERS
from .schema import coerce_benchmarks, coerce_fundamentals, coerce_returns

SEC_UA_DEFAULT = "factor-ranker research (set SEC_USER_AGENT to 'Your Name you@example.com')"
SEC_MIN_INTERVAL = 0.11        # SEC fair-access limit: 10 requests/second

# concept -> candidate us-gaap tags in priority order
FLOW_TAGS: Dict[str, List[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenuesNetOfInterestExpense"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "ebit": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock", "PaymentsOfOrdinaryDividends"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}
INSTANT_TAGS: Dict[str, List[str]] = {
    "current_assets": ["AssetsCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "short_term_debt": ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "net_ppe": ["PropertyPlantAndEquipmentNet"],
    "total_assets": ["Assets"],
    "book_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "preferred_stock": ["PreferredStockValue"],
    "minority_interest": ["MinorityInterest"],
}
SHARES_TAG = ("dei", "EntityCommonStockSharesOutstanding")
EXCHANGE_MAP = {"nasdaq": "NASDAQ", "nyse": "NYSE", "nyse american": "AMEX", "nyse arca": "OTHER", "cboe": "OTHER", "otc": "OTC"}


@dataclass(frozen=True)
class FreeConfig:
    start: str = "2005-01-31"
    end: str = "2024-12-31"
    tickers: Tuple[str, ...] = tuple(TICKERS)
    max_tickers: int = 0                  # 0 = all; useful for quick demos


# =========================================================================== SEC helpers
class SECClient:
    def __init__(self, cache_dir: Path, user_agent: Optional[str] = None):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ua = user_agent or os.environ.get("SEC_USER_AGENT") or SEC_UA_DEFAULT
        self._last = 0.0

    def get_json(self, url: str, name: str) -> Optional[Dict[str, Any]]:
        p = self.dir / name
        if p.exists():
            return json.loads(p.read_text())
        import requests

        wait = SEC_MIN_INTERVAL - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            try:
                r = requests.get(url, headers={"User-Agent": self.ua, "Accept-Encoding": "gzip, deflate"}, timeout=30)
                self._last = time.time()
                if r.status_code == 200:
                    p.write_text(r.text)
                    return r.json()
                if r.status_code in (403, 429):
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return None
            except Exception:
                time.sleep(1.0)
        return None

    def ticker_map(self) -> Dict[str, int]:
        js = self.get_json("https://www.sec.gov/files/company_tickers.json", "company_tickers.json") or {}
        return {v["ticker"].upper(): int(v["cik_str"]) for v in js.values()}

    def companyfacts(self, cik: int) -> Optional[Dict[str, Any]]:
        return self.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", f"facts_{cik}.json")

    def submissions(self, cik: int) -> Optional[Dict[str, Any]]:
        return self.get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", f"subs_{cik}.json")


def _entries(facts: Dict[str, Any], taxonomy: str, tag: str, unit_prefix: str = "USD") -> pd.DataFrame:
    """All reported values for one tag as a frame: start (NaT for instants), end, val, filed, form, fp, fy."""
    try:
        units = facts["facts"][taxonomy][tag]["units"]
    except KeyError:
        return pd.DataFrame()
    key = next((u for u in units if u == unit_prefix), None) or next((u for u in units if u.startswith(unit_prefix)), None)
    if key is None:
        return pd.DataFrame()
    df = pd.DataFrame(units[key])
    if df.empty:
        return df
    df["start"] = pd.to_datetime(df.get("start"), errors="coerce") if "start" in df else pd.NaT
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    return df.dropna(subset=["val"])


def _first_reported(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Earliest filing per period = as first reported."""
    return df.sort_values("filed").drop_duplicates(keys, keep="first")


def _concept(facts: Dict[str, Any], tags: List[str], taxonomy: str = "us-gaap", unit: str = "USD") -> pd.DataFrame:
    """Union of candidate tags, priority to the first tag that reports a given period."""
    frames = []
    for rank, tag in enumerate(tags):
        e = _entries(facts, taxonomy, tag, unit)
        if len(e):
            frames.append(e.assign(_rank=rank))
    if not frames:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "_rank"])
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["_rank", "filed"]).drop_duplicates(["start", "end"], keep="first")


def ttm_from_durations(d: pd.DataFrame) -> pd.DataFrame:
    """Trailing-twelve-month value at each period end from duration facts.

    TTM(D) = annual value if D is a fiscal year end, else annual(last FY end) + YTD(D) - YTD(D - 1 year),
    where YTD(D) is the longest sub-annual duration ending at D.  Returns end, ttm, filed (of the piece
    reported at D, i.e. when TTM(D) became computable).
    """
    if d.empty:
        return pd.DataFrame(columns=["end", "ttm", "filed"])
    d = d.dropna(subset=["start"]).copy()
    d["days"] = (d["end"] - d["start"]).dt.days
    annual = d[d["days"].between(350, 380)].sort_values("end")
    annual = _first_reported(annual, ["end"])
    ytd = d[d["days"].between(60, 300)].sort_values(["end", "days"]).drop_duplicates("end", keep="last")  # longest sub-annual per end
    ytd = _first_reported(ytd, ["end"]) if len(ytd) else ytd
    rows = []
    for _, a in annual.iterrows():
        rows.append(dict(end=a["end"], ttm=a["val"], filed=a["filed"]))
    for _, y in ytd.iterrows():
        prev_fy = annual[annual["end"] < y["end"]]
        if prev_fy.empty:
            continue
        a = prev_fy.iloc[-1]
        if (y["end"] - a["end"]).days > 300:
            continue                                                    # annual data too old (missing 10-K)
        target = y["end"] - pd.DateOffset(years=1)
        prior = ytd[(ytd["end"] - target).abs().dt.days <= 12]
        prior = prior[(prior["days"] - y["days"]).abs() <= 12]
        if prior.empty:
            continue
        rows.append(dict(end=y["end"], ttm=a["val"] + y["val"] - prior.iloc[0]["val"],
                         filed=max(y["filed"], a["filed"])))
    if not rows:
        return pd.DataFrame(columns=["end", "ttm", "filed"])
    return pd.DataFrame(rows).sort_values("end").drop_duplicates("end", keep="first")


def instants_asof(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=["end", "val", "filed"])
    inst = d[d["start"].isna()] if "start" in d else d
    inst = _first_reported(inst, ["end"])
    return inst[["end", "val", "filed"]].sort_values("end")


def build_company_fundamentals(facts: Dict[str, Any], security_id: str) -> pd.DataFrame:
    """Canonical fundamentals rows for one company from its companyfacts JSON."""
    flows = {}
    for concept, tags in FLOW_TAGS.items():
        unit = "shares" if concept == "shares_diluted" else "USD"
        flows[concept] = ttm_from_durations(_concept(facts, tags, unit=unit)).set_index("end")
    if flows["gross_profit"].empty and not flows["revenue"].empty and not flows["cogs"].empty:
        gp = flows["revenue"][["ttm"]].join(flows["cogs"][["ttm"]], lsuffix="_r", rsuffix="_c", how="inner")
        flows["gross_profit"] = pd.DataFrame({"ttm": gp["ttm_r"] - gp["ttm_c"],
                                              "filed": flows["revenue"]["filed"].reindex(gp.index)})
    inst = {c: instants_asof(_concept(facts, tags)).set_index("end") for c, tags in INSTANT_TAGS.items()}
    ends = sorted(set(inst["total_assets"].index) & set(flows["revenue"].index)) if len(flows["revenue"]) else []
    if not ends:
        return pd.DataFrame()
    rows = []
    for D in ends:
        row = {"security_id": security_id, "fiscal_period_end": D}
        filed = []
        for concept, col in (("revenue", "revenue_ttm"), ("gross_profit", "gross_profit_ttm"), ("ebit", "ebit_ttm"),
                             ("net_income", "net_income_ttm"), ("ocf", "operating_cash_flow_ttm"), ("capex", "capex_ttm"),
                             ("dividends", "dividends_ttm"), ("shares_diluted", "shares_diluted")):
            f = flows[concept]
            if D in f.index:
                row[col] = float(f.loc[D, "ttm"])
                filed.append(f.loc[D, "filed"])
            else:
                row[col] = np.nan
        for concept, f in inst.items():
            if D in f.index:
                row[concept] = float(f.loc[D, "val"])
                filed.append(f.loc[D, "filed"])
            else:
                row[concept] = 0.0 if concept in ("preferred_stock", "minority_interest", "short_term_debt") else np.nan
        row["available_date"] = max(filed) if filed else pd.NaT
        rows.append(row)
    out = pd.DataFrame(rows).dropna(subset=["available_date"])
    out["available_date"] = out["available_date"].where(out["available_date"] >= out["fiscal_period_end"], out["fiscal_period_end"] + pd.Timedelta(days=45))
    return out


def shares_history(facts: Dict[str, Any]) -> pd.DataFrame:
    e = _entries(facts, SHARES_TAG[0], SHARES_TAG[1], "shares")
    if e.empty:
        return e
    e = _first_reported(e, ["end"]).sort_values("end")
    return e[["end", "val", "filed"]]


# =========================================================================== the adapter
class FreeSource(DataSource):
    name = "free"

    def __init__(self, cache: bool = True, user_agent: Optional[str] = None, **params: Any):
        if "tickers" in params and params["tickers"] is not None:
            params["tickers"] = tuple(str(t).upper() for t in params["tickers"])
        self.cfg = FreeConfig(**params)
        self._cache = ParquetCache("free") if cache else None
        self._key = config_hash(asdict(self.cfg))
        self.sec = SECClient(CACHE_ROOT / "free" / "sec", user_agent)
        self._frames: Optional[Dict[str, pd.DataFrame]] = None
        self.failed: List[str] = []
        self.no_facts: List[str] = []
        self.current_shares_only: List[str] = []

    @property
    def tickers(self) -> List[str]:
        t = list(self.cfg.tickers)
        return t[: self.cfg.max_tickers] if self.cfg.max_tickers else t

    # -- prices ---------------------------------------------------------------------
    def _download_prices(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        import yfinance as yf

        raw = yf.download(self.tickers + ["SPY", "^IRX"], start=(pd.Timestamp(self.cfg.start) - pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                          end=(pd.Timestamp(self.cfg.end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
                          interval="1mo", auto_adjust=False, actions=False, group_by="column", threads=True, progress=False)
        adj, close = raw["Adj Close"].copy(), raw["Close"].copy()
        # Yahoo's batch endpoint drops tickers now and then; retry the empty ones individually, slowly
        missing = [t for t in self.tickers if t not in adj.columns or adj[t].dropna().empty]
        for t in missing:
            for attempt in range(2):
                try:
                    time.sleep(1.0 + attempt)
                    one = yf.download(t, start=adj.index.min().strftime("%Y-%m-%d"), end=(pd.Timestamp(self.cfg.end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
                                      interval="1mo", auto_adjust=False, actions=False, progress=False)
                    if one is not None and len(one.dropna(how="all")):
                        col = one["Adj Close"] if "Adj Close" in one else one["Close"]
                        adj[t] = (col.iloc[:, 0] if hasattr(col, "columns") else col).reindex(adj.index)
                        c2 = one["Close"]
                        close[t] = (c2.iloc[:, 0] if hasattr(c2, "columns") else c2).reindex(close.index)
                        break
                except Exception:
                    continue
        adj.index = pd.to_datetime(adj.index) + pd.offsets.MonthEnd(0)
        close.index = adj.index
        return adj, close

    def _info(self, ticker: str) -> Dict[str, Any]:
        p = self.sec.dir / f"yfinfo_{ticker}.json"
        if p.exists():
            return json.loads(p.read_text())
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info or {}
            keep = {k: info.get(k) for k in ("sharesOutstanding", "exchange", "sector", "industry", "longName")}
        except Exception:
            keep = {}
        p.write_text(json.dumps(keep))
        return keep

    # -- build ----------------------------------------------------------------------
    def _build(self) -> Dict[str, pd.DataFrame]:
        adj, close = self._download_prices()
        cik_map = self.sec.ticker_map()
        rows, frows, meta_rows = [], [], []
        for t in self.tickers:
            if t not in adj.columns or adj[t].dropna().empty:
                self.failed.append(t)
                continue
            cik = cik_map.get(t.replace("-", ""), cik_map.get(t))
            facts = self.sec.companyfacts(cik) if cik else None
            subs = self.sec.submissions(cik) if cik else None
            sic = int(subs.get("sic") or 0) if subs else 0
            exch_raw = (subs.get("exchanges") or [""])[0] if subs else ""
            exchange = EXCHANGE_MAP.get(str(exch_raw).lower(), "OTHER")
            sh = shares_history(facts) if facts else pd.DataFrame()
            px = pd.DataFrame({"adj": adj[t], "close": close[t]}).dropna()
            px = px[(px.index >= pd.Timestamp(self.cfg.start) - pd.DateOffset(months=1)) & (px.index <= pd.Timestamp(self.cfg.end))]
            px["total_return"] = px["adj"].pct_change()
            px = px.iloc[1:]
            if len(sh):
                s = sh.sort_values(["filed", "end"]).drop_duplicates("filed", keep="last").set_index("filed")["val"]
                px["shares"] = s.reindex(px.index, method="ffill").values          # shares as of the latest filing <= date
            else:
                px["shares"] = np.nan
            if px["shares"].isna().all():
                self.current_shares_only.append(t)
                px["shares"] = float(self._info(t).get("sharesOutstanding") or np.nan)
            px["shares"] = px["shares"].bfill()
            px = px.dropna(subset=["total_return", "shares"])
            if px.empty:
                self.failed.append(t)
                continue
            r = pd.DataFrame({"security_id": t, "date": px.index, "total_return": px["total_return"].values, "price": px["close"].values,
                              "shares_outstanding": px["shares"].values, "market_cap": (px["close"] * px["shares"]).values,
                              "exchange": exchange, "sic_code": sic, "is_delisted_this_month": False})
            if r["date"].max() < pd.Timestamp(self.cfg.end) - pd.DateOffset(months=2):
                r.loc[r.index[-1], "is_delisted_this_month"] = True         # stopped trading: no delisting return known
            rows.append(r)
            if facts:
                f = build_company_fundamentals(facts, t)
                if len(f):
                    frows.append(f)
                else:
                    self.no_facts.append(t)
            else:
                self.no_facts.append(t)
        returns = coerce_returns(pd.concat(rows, ignore_index=True)).sort_values(["security_id", "date"]).reset_index(drop=True)
        fundamentals = coerce_fundamentals(pd.concat(frows, ignore_index=True)) if frows else coerce_fundamentals(pd.DataFrame(columns=returns.columns))
        fundamentals = fundamentals.sort_values(["security_id", "fiscal_period_end", "available_date"]).reset_index(drop=True)
        # benchmarks
        spy = adj["SPY"].pct_change()
        irx = close["^IRX"]
        rf = ((1 + irx / 100) ** (1 / 12) - 1).reindex(spy.index).ffill().fillna(0.0)
        W = returns.pivot(index="date", columns="security_id", values="market_cap").shift(1)
        R = returns.pivot(index="date", columns="security_id", values="total_return")
        vw = (W * R).sum(axis=1) / W.where(R.notna()).sum(axis=1)
        ew = R.mean(axis=1)
        b = pd.DataFrame({"date": R.index, "sp500_total_return": spy.reindex(R.index).values, "market_vw_return": vw.values,
                          "market_ew_return": ew.values, "risk_free_rate": rf.reindex(R.index).values}).dropna(subset=["sp500_total_return"])
        benchmarks = coerce_benchmarks(b[(b["date"] >= pd.Timestamp(self.cfg.start)) & (b["date"] <= pd.Timestamp(self.cfg.end))])
        status = pd.DataFrame({"failed": [",".join(self.failed)], "no_facts": [",".join(self.no_facts)],
                               "current_shares_only": [",".join(self.current_shares_only)]})
        return {"returns": returns, "fundamentals": fundamentals, "benchmarks": benchmarks, "status": status}

    def _load(self) -> Dict[str, pd.DataFrame]:
        if self._frames is not None:
            return self._frames
        names = ("returns", "fundamentals", "benchmarks", "status")
        if self._cache is not None:
            got = {n: self._cache.get(f"{self._key}_{n}") for n in names}
            if all(v is not None for v in got.values()):
                self._frames = got
                st = got["status"].iloc[0]
                self.failed = [x for x in st["failed"].split(",") if x]
                self.no_facts = [x for x in st["no_facts"].split(",") if x]
                self.current_shares_only = [x for x in st["current_shares_only"].split(",") if x]
                return got
        frames = self._build()
        if self._cache is not None:
            for n in names:
                self._cache.put(f"{self._key}_{n}", frames[n])
        self._frames = frames
        return frames

    # -- interface ------------------------------------------------------------------
    def get_monthly_returns(self, start, end) -> pd.DataFrame:
        r = self._load()["returns"]
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return r[(r["date"] >= s) & (r["date"] <= e)].reset_index(drop=True)

    def get_fundamentals(self, start, end) -> pd.DataFrame:
        f = self._load()["fundamentals"]
        s, e = pd.Timestamp(start) - pd.DateOffset(months=18), pd.Timestamp(end)
        return f[(f["available_date"] >= s) & (f["available_date"] <= e)].reset_index(drop=True)

    def get_benchmarks(self, start, end) -> pd.DataFrame:
        b = self._load()["benchmarks"]
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return b[(b["date"] >= s) & (b["date"] <= e)].reset_index(drop=True)

    def get_metadata(self) -> Dict[str, Any]:
        m = self.static_metadata()
        m["coverage_start"], m["coverage_end"] = self.cfg.start, self.cfg.end
        m["params"] = {"start": self.cfg.start, "end": self.cfg.end, "n_tickers": len(self.tickers), "max_tickers": self.cfg.max_tickers}
        if self._frames is not None:
            m["failed_tickers"] = self.failed
            m["tickers_without_sec_facts"] = self.no_facts
            m["tickers_with_current_shares_only"] = self.current_shares_only
            m["caveats"] = m["caveats"] + [f"{len(self.failed)} ticker(s) failed to download, {len(self.no_facts)} have no usable SEC facts, "
                                           f"{len(self.current_shares_only)} use today's share count for every date."]
        return m

    @staticmethod
    def static_metadata() -> Dict[str, Any]:
        return {
            "name": "free",
            "display_name": "Free demo data (Yahoo Finance prices + SEC XBRL filings)",
            "coverage_start": FreeConfig.start, "coverage_end": FreeConfig.end,
            "is_simulated": False, "survivorship_bias": True,
            "recommended_start": "2011-01-31",   # XBRL fundamentals only become broadly available from 2009-2010
            "caveats": [
                "SURVIVORSHIP BIAS: the universe is a fixed list of today's large and mid caps. Companies that failed or were "
                "acquired are missing, so every result is optimistic. This is a demo, not research.",
                "Fundamentals history is short (XBRL filings start ~2009) and only US-GAAP filers are covered.",
                "Fundamentals are point-in-time (SEC filing dates) but taken as first reported; later restatements are ignored.",
                "S&P 500 is proxied by SPY's adjusted-close return; VW/EW 'market' benchmarks are computed from the same survivor universe.",
                "No delisting returns exist: a ticker that stops trading is flagged on its last month with its last price return only.",
            ],
            "ui_params": {
                "max_tickers": {"label": "Number of tickers (0 = all)", "type": "int", "default": 0, "min": 0, "max": 400, "step": 20,
                                "help": "0 = the whole list (~340 tickers, a few minutes on first download, instant afterwards); smaller for a quick demo."},
            },
        }
