"""WRDS adapter: CRSP monthly stock file + Compustat fundamentals + CCM link + Fama-French risk-free.

STATUS: written from the specification, NOT yet run against WRDS.  Every assumption about a table or
column that has not been checked against the live schema is marked ``# VERIFY:``.  Run
``python -m src.data.wrds_source --verify-schema`` once credentials work; it prints, for every table
and column this adapter expects, what the database actually has.  See ACTIVATE_WRDS.md.

Credentials: the ``wrds`` package reads ``~/.pgpass`` (create it once with ``wrds.Connection().create_pgpass_file()``).
The username may be given as a parameter or via the ``WRDS_USERNAME`` environment variable.  Nothing is
hard-coded and no password ever touches this file.

Design: SQL fetches are thin functions returning raw frames; all logic lives in pure pandas transforms
(``compound_delisting_returns``, ``ytd_to_quarterly``, ``build_ttm``, ``resolve_links`` ...) so it can
be unit-tested offline with hand-built frames (tests/test_wrds_offline.py).

Licensing: WRDS data may not be redistributed.  Everything this adapter downloads is cached under
``data/cache/wrds`` which is git-ignored.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import DataSource
from .cache import ParquetCache, config_hash
from .schema import coerce_benchmarks, coerce_fundamentals, coerce_returns, empty_fundamentals

# --------------------------------------------------------------------------- expected schema
# Everything the adapter reads, so verify_schema() can compare against the live database.
EXPECTED_TABLES: Dict[str, List[str]] = {
    # legacy CRSP (SAS-era layout)
    "crsp.msf": ["permno", "date", "ret", "prc", "shrout"],                       # VERIFY: ret is NULL (not -66/-77/-88/-99) for missing
    "crsp.msenames": ["permno", "namedt", "nameendt", "shrcd", "exchcd", "siccd"],  # VERIFY: nameendt NULL for current record?
    "crsp.msedelist": ["permno", "dlstdt", "dlret", "dlstcd"],
    # CRSP CIZ ("v2") layout
    "crsp.msf_v2": ["permno", "mthcaldt", "mthret", "mthprc", "shrout", "sharetype", "securitytype", "securitysubtype",
                    "usincflg", "issuertype", "primaryexch", "conditionaltype", "tradingstatusflg", "siccd"],
    # VERIFY: msf_v2 column names above (WRDS CIZ monthly stock file) and whether mthret already includes delisting returns
    "crsp.stkdelists": ["permno", "delistingdt", "delret", "delreasontype"],       # VERIFY: CIZ delisting table name/columns
    # indices
    "crsp.msi": ["date", "vwretd", "ewretd"],                                      # VERIFY: CRSP VW/EW market incl. dividends
    "crsp.msp500": ["caldt", "vwretd"],                                            # VERIFY: S&P 500 constituents VW total return
    # Compustat
    "comp.fundq": ["gvkey", "datadate", "fyearq", "fqtr", "rdq", "indfmt", "datafmt", "popsrc", "consol",
                   "saleq", "cogsq", "oiadpq", "niq", "oancfy", "capxy", "dvy", "actq", "cheq", "lctq", "dlcq", "dlttq",
                   "ppentq", "atq", "ceqq", "pstkq", "mibq", "txditcq", "cshfdq", "cshoq"],   # VERIFY: mibq vs mibtq; dvy exists in fundq
    "comp.funda": ["gvkey", "datadate", "fyear", "indfmt", "datafmt", "popsrc", "consol",
                   "sale", "cogs", "oiadp", "ni", "oancf", "capx", "dvt", "act", "che", "lct", "dlc", "dltt",
                   "ppent", "at", "ceq", "pstk", "mib", "txditc", "cshfd", "csho"],
    # link
    "crsp.ccmxpf_lnkhist": ["gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"],   # VERIFY: current name of the CCM link table
    # Fama-French
    "ff.factors_monthly": ["date", "rf"],                                          # VERIFY: rf in decimal per month; date convention
}

EXCHANGE_CODES = {1: "NYSE", 2: "AMEX", 3: "NASDAQ"}                    # legacy exchcd
CIZ_EXCHANGES = {"N": "NYSE", "A": "AMEX", "Q": "NASDAQ"}               # VERIFY: primaryexch codes
PERFORMANCE_DELIST_CODES = lambda c: (c == 500) | ((c >= 520) & (c <= 584))   # Shumway (1997)


@dataclass(frozen=True)
class WRDSConfig:
    start: str = "1990-01-31"
    end: str = "2024-12-31"
    quarterly_lag_days: int = 90          # available_date fallback when rdq is missing (quarterly)
    annual_lag_months: int = 4            # available_date fallback for annual (funda) rows
    missing_dlret_fill: float = -0.30     # Shumway: performance-related delisting with missing dlret
    prefer_ciz: bool = True               # use crsp.msf_v2 when present, else legacy msf + msedelist
    compustat_unit: float = 1e6           # Compustat reports $ millions
    crsp_shrout_unit: float = 1e3         # CRSP shrout is in thousands


# =========================================================================== pure transforms
def month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s) + pd.offsets.MonthEnd(0)


def compound_delisting_returns(msf: pd.DataFrame, delist: pd.DataFrame, missing_fill: float = -0.30) -> pd.DataFrame:
    """Legacy CRSP: fold ``dlret`` into the return of the delisting month and flag it.

    ``msf``: permno, date (month end), ret ...  ``delist``: permno, dlstdt, dlret, dlstcd.
    * If the stock has a return in the delisting month: (1+ret)(1+dlret) - 1.
    * If it has no row that month, a row is added with the delisting return alone.
    * Missing dlret: ``missing_fill`` for performance-related codes (500, 520-584), 0 otherwise.
    * Rows after the delisting month are dropped.
    """
    d = delist.dropna(subset=["dlstdt"]).copy()
    d["date"] = month_end(d["dlstdt"])
    d["dlret"] = d["dlret"].astype(float)
    perf = PERFORMANCE_DELIST_CODES(d["dlstcd"].fillna(0).astype(int))
    d["dlret"] = d["dlret"].where(d["dlret"].notna(), np.where(perf, missing_fill, 0.0))
    d = d.sort_values("dlstdt").drop_duplicates("permno", keep="last")[["permno", "date", "dlret"]]
    out = msf.merge(d, on=["permno", "date"], how="outer", indicator=True)
    # rows that exist only in the delist frame: the stock had no return that month
    only_dl = out["_merge"] == "right_only"
    out.loc[only_dl, "ret"] = out.loc[only_dl, "dlret"]
    both = out["_merge"] == "both"
    out.loc[both, "ret"] = (1 + out.loc[both, "ret"].fillna(0)) * (1 + out.loc[both, "dlret"]) - 1
    out["is_delisted_this_month"] = out["dlret"].notna()
    out = out.drop(columns=["_merge", "dlret"])
    # drop anything after the delisting month
    last = out.loc[out["is_delisted_this_month"], ["permno", "date"]].rename(columns={"date": "dl_date"})
    out = out.merge(last, on="permno", how="left")
    out = out[out["dl_date"].isna() | (out["date"] <= out["dl_date"])].drop(columns=["dl_date"])
    return out


def legacy_to_returns(msf: pd.DataFrame, cfg: WRDSConfig) -> pd.DataFrame:
    """msf rows (already filtered to common stocks on major exchanges, delisting folded in) -> canonical."""
    r = msf.copy()
    r["date"] = month_end(r["date"])
    r["price"] = r["prc"].abs()                                   # negative prc = bid/ask midpoint
    r["shares_outstanding"] = r["shrout"] * cfg.crsp_shrout_unit
    r["market_cap"] = r["price"] * r["shares_outstanding"]
    r["exchange"] = r["exchcd"].map(EXCHANGE_CODES).fillna("OTHER")
    r["security_id"] = r["permno"].astype(int).astype(str)
    r["total_return"] = r["ret"].astype(float)
    r["sic_code"] = r["siccd"].fillna(0).astype(int)
    r["is_delisted_this_month"] = r.get("is_delisted_this_month", False)
    r = r.dropna(subset=["total_return"])
    return coerce_returns(r)


def ciz_to_returns(msf2: pd.DataFrame, cfg: WRDSConfig) -> pd.DataFrame:
    r = msf2.copy()
    r["date"] = month_end(r["mthcaldt"])
    r["price"] = r["mthprc"].abs()
    r["shares_outstanding"] = r["shrout"] * cfg.crsp_shrout_unit      # VERIFY: shrout unit in msf_v2 (thousands?)
    r["market_cap"] = r["price"] * r["shares_outstanding"]
    r["exchange"] = r["primaryexch"].map(CIZ_EXCHANGES).fillna("OTHER")
    r["security_id"] = r["permno"].astype(int).astype(str)
    r["total_return"] = r["mthret"].astype(float)                   # VERIFY: includes delisting return in CIZ
    r["sic_code"] = r["siccd"].fillna(0).astype(int)
    # flag the final month of each permno that has a delisting record
    if "is_delisted_this_month" not in r:
        r["is_delisted_this_month"] = False
    r = r.dropna(subset=["total_return"])
    return coerce_returns(r)


def ytd_to_quarterly(fundq: pd.DataFrame, ytd_cols: Tuple[str, ...] = ("oancfy", "capxy", "dvy")) -> pd.DataFrame:
    """Compustat cash-flow items in fundq are fiscal-year-to-date: difference them within (gvkey, fyearq)."""
    f = fundq.sort_values(["gvkey", "fyearq", "fqtr"]).copy()
    for c in ytd_cols:
        prev = f.groupby(["gvkey", "fyearq"])[c].shift(1)
        q = f[c] - prev.fillna(0.0)
        q = q.where(f["fqtr"] > 1, f[c])                    # Q1: YTD == quarter
        # a gap in quarters inside the fiscal year makes the difference wrong -> NaN
        contiguous = f.groupby(["gvkey", "fyearq"])["fqtr"].diff().fillna(f["fqtr"]).eq(1) | (f["fqtr"] == 1)
        f[c[:-1] + "q"] = q.where(contiguous)
    return f


def build_ttm(fq: pd.DataFrame, flow_cols: Dict[str, str]) -> pd.DataFrame:
    """Trailing-twelve-month sums of quarterly flows over 4 consecutive fiscal quarters (else NaN)."""
    f = fq.sort_values(["gvkey", "datadate"]).copy()
    g = f.groupby("gvkey")
    # consecutive quarters: datadate spacing about 3 months for the last 3 steps
    gap = g["datadate"].diff().dt.days
    ok = gap.between(80, 100)
    ok3 = ok & ok.groupby(f["gvkey"]).shift(1, fill_value=False) & ok.groupby(f["gvkey"]).shift(2, fill_value=False)
    for src, dst in flow_cols.items():
        f[dst] = g[src].transform(lambda s: s.rolling(4, min_periods=4).sum()).where(ok3)
    return f


def availability_dates(f: pd.DataFrame, cfg: WRDSConfig, annual: bool = False) -> pd.Series:
    """available_date = rdq when present and not before datadate, else datadate + configurable lag."""
    fallback = (f["datadate"] + (pd.DateOffset(months=cfg.annual_lag_months) if annual else pd.Timedelta(days=cfg.quarterly_lag_days)))
    if "rdq" in f and not annual:
        rdq = pd.to_datetime(f["rdq"])
        return rdq.where(rdq.notna() & (rdq >= f["datadate"]), fallback)
    return fallback


def resolve_links(fund: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Attach permno to Compustat rows via CCM (linktype LU/LC, linkprim P/C, datadate within link range)."""
    l = link[link["linktype"].isin(["LU", "LC"]) & link["linkprim"].isin(["P", "C"])].copy()
    l["linkdt"] = pd.to_datetime(l["linkdt"])
    l["linkenddt"] = pd.to_datetime(l["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))   # VERIFY: NULL linkenddt = still active
    m = fund.merge(l[["gvkey", "lpermno", "linkdt", "linkenddt", "linkprim"]], on="gvkey", how="inner")
    m = m[(m["datadate"] >= m["linkdt"]) & (m["datadate"] <= m["linkenddt"])]
    # prefer the primary link when several match the same (gvkey, datadate)
    m["_prio"] = (m["linkprim"] != "P").astype(int)
    m = m.sort_values(["gvkey", "datadate", "_prio"]).drop_duplicates(["gvkey", "datadate"], keep="first")
    return m.drop(columns=["_prio", "linkdt", "linkenddt", "linkprim"]).rename(columns={"lpermno": "permno"})


QUARTERLY_MAP = {  # canonical column -> fundq expression after ytd_to_quarterly / build_ttm
    "ebit_ttm": "oiadpq_ttm", "revenue_ttm": "saleq_ttm", "gross_profit_ttm": "gpq_ttm", "net_income_ttm": "niq_ttm",
    "operating_cash_flow_ttm": "oancfq_ttm", "capex_ttm": "capxq_ttm", "dividends_ttm": "dvq_ttm",
    "current_assets": "actq", "cash": "cheq", "current_liabilities": "lctq", "short_term_debt": "dlcq",
    "long_term_debt": "dlttq", "net_ppe": "ppentq", "total_assets": "atq", "book_equity": "beq",
    "preferred_stock": "pstkq", "minority_interest": "mibq", "shares_diluted": "cshfdq",
}
ANNUAL_MAP = {
    "ebit_ttm": "oiadp", "revenue_ttm": "sale", "gross_profit_ttm": "gp", "net_income_ttm": "ni",
    "operating_cash_flow_ttm": "oancf", "capex_ttm": "capx", "dividends_ttm": "dvt",
    "current_assets": "act", "cash": "che", "current_liabilities": "lct", "short_term_debt": "dlc",
    "long_term_debt": "dltt", "net_ppe": "ppent", "total_assets": "at", "book_equity": "be",
    "preferred_stock": "pstk", "minority_interest": "mib", "shares_diluted": "cshfd",
}


def fundq_to_canonical(fundq: pd.DataFrame, link: pd.DataFrame, cfg: WRDSConfig) -> pd.DataFrame:
    """comp.fundq rows -> canonical fundamentals (quarterly path)."""
    f = fundq.copy()
    f["datadate"] = pd.to_datetime(f["datadate"])
    f = ytd_to_quarterly(f)                                           # -> oancfq, capxq, dvq
    f["gpq"] = f["saleq"] - f["cogsq"]
    f["beq"] = f["ceqq"] + f["txditcq"].fillna(0.0)                  # VERIFY: book equity = common equity + deferred taxes (FF-style)
    f = build_ttm(f, {"oiadpq": "oiadpq_ttm", "saleq": "saleq_ttm", "gpq": "gpq_ttm", "niq": "niq_ttm",
                      "oancfq": "oancfq_ttm", "capxq": "capxq_ttm", "dvq": "dvq_ttm"})
    f["available_date"] = availability_dates(f, cfg, annual=False)
    f = resolve_links(f, link)
    out = pd.DataFrame({"security_id": f["permno"].astype(int).astype(str), "fiscal_period_end": f["datadate"],
                        "available_date": f["available_date"]})
    for canon, col in QUARTERLY_MAP.items():
        out[canon] = f[col].astype(float) * (cfg.compustat_unit if canon != "shares_diluted" else cfg.compustat_unit)
    # VERIFY: cshfdq is in millions of shares (Compustat share counts are in millions) -> same unit multiplier
    return out


def funda_to_canonical(funda: pd.DataFrame, link: pd.DataFrame, cfg: WRDSConfig) -> pd.DataFrame:
    f = funda.copy()
    f["datadate"] = pd.to_datetime(f["datadate"])
    f["gp"] = f["sale"] - f["cogs"]
    f["be"] = f["ceq"] + f["txditc"].fillna(0.0)
    f["available_date"] = availability_dates(f, cfg, annual=True)
    f = resolve_links(f, link)
    out = pd.DataFrame({"security_id": f["permno"].astype(int).astype(str), "fiscal_period_end": f["datadate"],
                        "available_date": f["available_date"]})
    for canon, col in ANNUAL_MAP.items():
        out[canon] = f[col].astype(float) * cfg.compustat_unit
    return out


def merge_quarterly_and_annual(q: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """Use annual rows only for (security, fiscal period) pairs the quarterly path does not cover."""
    if a is None or a.empty:
        return q
    have = set(zip(q["security_id"], q["fiscal_period_end"]))
    extra = a[[(s, d) not in have for s, d in zip(a["security_id"], a["fiscal_period_end"])]]
    return pd.concat([q, extra], ignore_index=True).sort_values(["security_id", "fiscal_period_end", "available_date"])


# =========================================================================== the adapter
class WRDSSource(DataSource):
    """CRSP + Compustat via the ``wrds`` package.  Parameters: see ``WRDSConfig``, plus ``username``."""

    name = "wrds"

    def __init__(self, username: Optional[str] = None, cache: bool = True, **params: Any):
        self.cfg = WRDSConfig(**params)
        self.username = username or os.environ.get("WRDS_USERNAME")
        self._cache = ParquetCache("wrds") if cache else None
        self._key = config_hash(asdict(self.cfg))
        self._db = None
        self._layout: Optional[str] = None

    # -- connection -------------------------------------------------------------
    def connect(self):
        if self._db is None:
            import wrds  # optional dependency, imported only here

            kwargs = {"wrds_username": self.username} if self.username else {}
            self._db = wrds.Connection(**kwargs)    # reads ~/.pgpass; never pass a password here
        return self._db

    def detect_layout(self) -> str:
        """'ciz' if crsp.msf_v2 exists (and prefer_ciz), else 'legacy'."""
        if self._layout is None:
            db = self.connect()
            tables = set(db.list_tables(library="crsp"))      # VERIFY: list_tables signature
            self._layout = "ciz" if (self.cfg.prefer_ciz and "msf_v2" in tables) else "legacy"
        return self._layout

    def _sql(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        return self.connect().raw_sql(query, params=params, date_cols=None)   # VERIFY: raw_sql params kwarg

    # -- raw fetches ------------------------------------------------------------
    def fetch_legacy_msf(self) -> pd.DataFrame:
        q = """
            SELECT a.permno, a.date, a.ret, a.prc, a.shrout, b.shrcd, b.exchcd, b.siccd
            FROM crsp.msf a
            JOIN crsp.msenames b
              ON a.permno = b.permno AND b.namedt <= a.date AND a.date <= COALESCE(b.nameendt, DATE '2099-12-31')
            WHERE a.date BETWEEN %(start)s AND %(end)s
              AND b.shrcd IN (10, 11)          -- US common stock; excludes ADRs, ETFs, REITs, closed-end funds
              AND b.exchcd IN (1, 2, 3)        -- NYSE, AMEX, Nasdaq
        """
        return self._sql(q, {"start": self.cfg.start, "end": self.cfg.end})

    def fetch_legacy_delist(self) -> pd.DataFrame:
        return self._sql("SELECT permno, dlstdt, dlret, dlstcd FROM crsp.msedelist")

    def fetch_ciz_msf(self) -> pd.DataFrame:
        q = """
            SELECT permno, mthcaldt, mthret, mthprc, shrout, primaryexch, siccd
            FROM crsp.msf_v2
            WHERE mthcaldt BETWEEN %(start)s AND %(end)s
              AND sharetype = 'NS' AND securitytype = 'EQTY' AND securitysubtype = 'COM'
              AND usincflg = 'Y' AND issuertype IN ('ACOR', 'CORP')
              AND primaryexch IN ('N', 'A', 'Q') AND conditionaltype = 'RW' AND tradingstatusflg = 'A'
        """  # VERIFY: these are the WRDS-recommended CIZ equivalents of shrcd 10/11 + exchcd 1/2/3
        return self._sql(q, {"start": self.cfg.start, "end": self.cfg.end})

    def fetch_ciz_delist(self) -> pd.DataFrame:
        return self._sql("SELECT permno, delistingdt, delret, delreasontype FROM crsp.stkdelists")   # VERIFY

    def fetch_fundq(self) -> pd.DataFrame:
        cols = ", ".join(c for c in EXPECTED_TABLES["comp.fundq"])
        q = f"""
            SELECT {cols} FROM comp.fundq
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
              AND datadate BETWEEN %(start)s AND %(end)s
        """
        start = (pd.Timestamp(self.cfg.start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        return self._sql(q, {"start": start, "end": self.cfg.end})

    def fetch_funda(self) -> pd.DataFrame:
        cols = ", ".join(c for c in EXPECTED_TABLES["comp.funda"])
        q = f"""
            SELECT {cols} FROM comp.funda
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
              AND datadate BETWEEN %(start)s AND %(end)s
        """
        start = (pd.Timestamp(self.cfg.start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        return self._sql(q, {"start": start, "end": self.cfg.end})

    def fetch_link(self) -> pd.DataFrame:
        return self._sql("SELECT gvkey, lpermno, linktype, linkprim, linkdt, linkenddt FROM crsp.ccmxpf_lnkhist")

    def fetch_indices(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        msi = self._sql("SELECT date, vwretd, ewretd FROM crsp.msi WHERE date BETWEEN %(s)s AND %(e)s",
                        {"s": self.cfg.start, "e": self.cfg.end})
        sp = self._sql("SELECT caldt, vwretd FROM crsp.msp500 WHERE caldt BETWEEN %(s)s AND %(e)s",
                       {"s": self.cfg.start, "e": self.cfg.end})
        ff = self._sql("SELECT date, rf FROM ff.factors_monthly WHERE date BETWEEN %(s)s AND %(e)s",
                       {"s": (pd.Timestamp(self.cfg.start) - pd.DateOffset(months=1)).strftime("%Y-%m-%d"), "e": self.cfg.end})
        return msi, sp, ff

    # -- canonical frames (cached) ------------------------------------------------
    def _cached(self, name: str, builder):
        if self._cache is not None:
            got = self._cache.get(f"{self._key}_{name}")
            if got is not None:
                return got
        df = builder()
        if self._cache is not None:
            self._cache.put(f"{self._key}_{name}", df)
        return df

    def _build_returns(self) -> pd.DataFrame:
        if self.detect_layout() == "ciz":
            msf = self.fetch_ciz_msf()
            msf["mthcaldt"] = pd.to_datetime(msf["mthcaldt"])
            dl = self.fetch_ciz_delist()
            # VERIFY: in CIZ the delisting return is already inside mthret; we only flag the final month
            dl["date"] = month_end(dl["delistingdt"])
            flag = dl[["permno", "date"]].assign(is_delisted_this_month=True)
            msf["date"] = month_end(msf["mthcaldt"])
            msf = msf.merge(flag, on=["permno", "date"], how="left")
            msf["is_delisted_this_month"] = msf["is_delisted_this_month"].fillna(False).astype(bool)
            return ciz_to_returns(msf, self.cfg)
        msf = self.fetch_legacy_msf()
        msf["date"] = month_end(msf["date"])
        msf = compound_delisting_returns(msf, self.fetch_legacy_delist(), self.cfg.missing_dlret_fill)
        return legacy_to_returns(msf, self.cfg)

    def _build_fundamentals(self) -> pd.DataFrame:
        link = self.fetch_link()
        q = fundq_to_canonical(self.fetch_fundq(), link, self.cfg)
        a = funda_to_canonical(self.fetch_funda(), link, self.cfg)
        f = merge_quarterly_and_annual(q, a)
        f = f.dropna(subset=["fiscal_period_end", "available_date"])
        f = f.drop_duplicates(["security_id", "fiscal_period_end", "available_date"])
        return coerce_fundamentals(f)

    def _build_benchmarks(self) -> pd.DataFrame:
        msi, sp, ff = self.fetch_indices()
        msi["date"] = month_end(msi["date"])
        sp["date"] = month_end(sp["caldt"])
        ff["date"] = month_end(ff["date"])           # VERIFY: ff date is first-of-month; snapping to month end is correct
        b = (msi[["date", "vwretd", "ewretd"]].rename(columns={"vwretd": "market_vw_return", "ewretd": "market_ew_return"})
             .merge(sp[["date", "vwretd"]].rename(columns={"vwretd": "sp500_total_return"}), on="date", how="left")
             .merge(ff[["date", "rf"]].rename(columns={"rf": "risk_free_rate"}), on="date", how="left"))
        return coerce_benchmarks(b.sort_values("date"))

    # -- DataSource interface -----------------------------------------------------
    def get_monthly_returns(self, start, end) -> pd.DataFrame:
        r = self._cached("returns", self._build_returns)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return r[(r["date"] >= s) & (r["date"] <= e)].reset_index(drop=True)

    def get_fundamentals(self, start, end) -> pd.DataFrame:
        f = self._cached("fundamentals", self._build_fundamentals)
        s, e = pd.Timestamp(start) - pd.DateOffset(months=18), pd.Timestamp(end)
        return f[(f["available_date"] >= s) & (f["available_date"] <= e)].reset_index(drop=True)

    def get_benchmarks(self, start, end) -> pd.DataFrame:
        b = self._cached("benchmarks", self._build_benchmarks)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return b[(b["date"] >= s) & (b["date"] <= e)].reset_index(drop=True)

    def get_metadata(self) -> Dict[str, Any]:
        m = self.static_metadata()
        m["coverage_start"], m["coverage_end"] = self.cfg.start, self.cfg.end
        m["params"] = asdict(self.cfg)
        m["layout"] = self._layout or "unknown until connected"
        return m

    @staticmethod
    def static_metadata() -> Dict[str, Any]:
        return {
            "name": "wrds",
            "display_name": "WRDS: CRSP + Compustat",
            "coverage_start": WRDSConfig.start, "coverage_end": WRDSConfig.end,
            "is_simulated": False, "survivorship_bias": False,
            "caveats": [
                "Licensed data (WRDS). Not redistributable; caches stay in data/ (git-ignored).",
                "Standard Compustat carries some restated values, so even this backtest is mildly optimistic "
                "versus true as-first-reported data (use Compustat Snapshot/point-in-time for that).",
                "Missing delisting returns are filled with -30% for performance-related delistings (Shumway 1997).",
                "Where the earnings announcement date (rdq) is missing, availability is assumed 90 days after "
                "quarter end (4 months for annual data).",
                "Adapter status: UNVERIFIED until ACTIVATE_WRDS.md has been completed.",
            ],
        }

    # -- schema verification --------------------------------------------------------
    def verify_schema(self, expected: Dict[str, List[str]] = EXPECTED_TABLES) -> pd.DataFrame:
        """Compare every expected table/column with the live database.  Prints and returns a report."""
        db = self.connect()
        rows = []
        for full, cols in expected.items():
            lib, table = full.split(".")
            try:
                found = db.describe_table(library=lib, table=table)      # VERIFY: describe_table signature
                found_cols = set(found["name"].astype(str).str.lower())   # VERIFY: column called 'name'
                exists = True
            except Exception as e:                                        # table missing or no permission
                found_cols, exists = set(), False
                rows.append(dict(table=full, column="*", expected=True, found=False, note=f"table not found: {e}"))
                continue
            for c in cols:
                rows.append(dict(table=full, column=c, expected=True, found=c.lower() in found_cols, note=""))
            extra = sorted(found_cols - {c.lower() for c in cols})
            rows.append(dict(table=full, column="(other columns present)", expected=False, found=exists, note=", ".join(extra[:40])))
        report = pd.DataFrame(rows)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 120)
        print(report.to_string(index=False))
        missing = report[(report["expected"]) & (~report["found"])]
        print(f"\n{len(missing)} expected table/column(s) missing." if len(missing) else "\nAll expected tables and columns found.")
        return report


def _main() -> None:
    ap = argparse.ArgumentParser(description="WRDS adapter utilities (requires a working ~/.pgpass)")
    ap.add_argument("--verify-schema", action="store_true", help="compare expected tables/columns with the live database")
    ap.add_argument("--layout", action="store_true", help="print which CRSP layout (ciz/legacy) would be used")
    ap.add_argument("--contract", action="store_true", help="pull data and run the canonical-schema contract check")
    ap.add_argument("--username", default=None)
    args = ap.parse_args()
    src = WRDSSource(username=args.username)
    if args.verify_schema:
        src.verify_schema()
    if args.layout:
        print("CRSP layout:", src.detect_layout())
    if args.contract:
        from .base import check_contract

        rep = check_contract(src, src.cfg.start, src.cfg.end)
        for k, v in rep.items():
            if k != "metadata":
                print(f"{k}: {v}")


if __name__ == "__main__":
    _main()
