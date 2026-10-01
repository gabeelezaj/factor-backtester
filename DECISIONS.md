# Financial and data decisions

Running log of every judgment call that affects results. Newest at the bottom of each section.

## Schema / interface

- **`available_date` is the only point-in-time key.** Every fundamentals lookup goes through
  `src/pit.py::asof_fundamentals`, which filters `available_date <= formation_date`. Factors never see
  `available_date`; they receive a pre-filtered snapshot. `fiscal_period_end` is carried only for
  display and for the deliberately-broken look-ahead trap test.
- **Restatements are extra rows**, same `fiscal_period_end`, later `available_date`. The PIT lookup
  takes the row with the latest `available_date <= T`, so a restatement is used only after it is public.
- **`total_return` already includes the delisting return** in the security's final month
  (`is_delisted_this_month = True`). The engine does not need to know why a stock disappeared.
- **`capex_ttm` and `dividends_ttm` are positive when cash goes out**, matching Compustat sign conventions
  (`capx`, `dvt`). Free cash flow = OCF − capex.
- Money columns are in the same unit across returns and fundamentals (adapters must reconcile; Compustat
  reports $MM, CRSP market cap is price × shares in thousands — the WRDS adapter converts both to $).
- Fiscal periods bound by `available_date` in `get_fundamentals(start, end)`, and adapters include the last
  row published before `start` per security, so early formation dates still have data.

## Synthetic market — `src/data/synthetic.py`

### Signal: two explicit components
- **Knowable characteristic (`true_alpha`)**: at each month end T, the cross-sectional normal score of
  0.5·z(EBIT/EV) + 0.5·z(EBIT/(NWC+NetPPE)), using only quarters with `available_date ≤ T` (as first
  reported) and the market cap at T. Next month's expected return += `signal_strength × signal_scale × true_alpha`.
  A point-in-time strategy can reproduce it (test: Spearman > 0.97 against an independent recomputation).
  Restatements are *not* in the generator's version, so a strategy that uses restated values is slightly noisier.
- **Announcement surprise (unknowable)**: in the month a quarter becomes public, the return jumps by
  0.8 × the quarter's own EBIT news (log-EBIT units, clipped ±30%). Own news is iid → unpredictable from public
  data. A backtest keyed on `fiscal_period_end` sees it 45–150 days early, which is the look-ahead trap.
- Consequence: the recovery test sorts on `true_alpha` with **no lag**; the placebo shows −0.06%/mo with no lag.

### Calibration (real-world reference)
- **Target: 0.3–0.5%/month top-minus-bottom decile spread** for the composite. References: Fama–French HML
  factor ≈ 0.3%/month (Ken French data library, 1963–2023); Fama–French (1992) B/M decile spread ≈ 0.5–0.6%/month;
  Novy-Marx (2013) gross-profitability decile spread ≈ 0.31%/month; McLean & Pontiff (2016) find published
  anomalies lose ~58% out of sample. Greenblatt's own 1988–2004 numbers (~30% vs ~12%) are far above any of
  these and are not used as a calibration target.
- Default `signal_scale = 0.0005` per sd of `true_alpha` and `quality_hazard_tilt = 0.2`; realized spread
  0.44%/month on seed 42, 0.45–0.55 across seeds. Roughly half of the spread comes from the expected-return
  term and half from the bottom decile's extra bankruptcies (as in real data, Shumway 1997).
- `signal_strength` is a multiplier on *all* fundamentals→returns channels (expected return, hazard tilts,
  bankruptcy share). Sweep results live in `validation/results/sweep.csv`.
- At this strength, a single 35-year panel resolves the decile tails clearly but not the middle deciles (adjacent
  middle deciles differ by ~0.01%/month vs ~0.05%/month standard error). Tests assert tails + rank correlation > 0.5.

### Death hazard: direction and the value-trap regime
- **Profitability tilt** (default on): hazard ×= exp(−0.2 × s × z(true ROC)); unprofitable firms die more and
  deaths of unprofitable firms are more often bankruptcies. This *helps* a profitability strategy — deliberate,
  because it is true in the real world.
- **Valuation**: with `value_trap = 0` (default) cheapness has *no* effect on the hazard, i.e. the original
  version's tailwind ("cheap and profitable die less") is gone. With `value_trap > 0` the hazard ×= exp(+value_trap
  × s × z(true EBIT/EV)) and the bankruptcy share rises with cheapness: low valuation partly reflects distress.
  At `value_trap = 0.5` the decile spread falls ~40% and the oracle's excess return ~15–25%; at 1.0 the spread is gone.
- Neither death channel operates at `signal_strength = 0`.
- Sub-$1 names have double hazard and are mostly "other delisting" (−20…−50%). This is *not* scaled by
  `signal_strength` (it is a market-structure fact, and the default $1 price filter removes it).

### Missing data
- A quarter is never published with probability 0.6 × 2% for healthy firms and 5 × 2% for distressed firms
  (negative EBIT or negative book equity). Dropping stocks with missing factors therefore removes sick companies,
  as it does on real data.

### Other generator choices (unchanged)
- Reporting lag: per-company base U(45, 85) days + per-period jitter, clipped to [45, 90]; 3% late by +30–60 days;
  2% restated (second row, +60–180 days, EBIT/NI ±15%). No extra delay for fiscal Q4 (simplification).
- Prices lead fundamentals: 90% of last quarter's log price change (ex-announcement) flows into next quarter's
  revenue; keeps valuation dispersion stable over 35 years without public-data predictability.
- Deaths: base hazard 3.5%/yr; bankruptcy U(−95%, −70%), acquisition U(+10%, +40%), other U(−50%, −20%);
  no deaths in a company's first 12 months. Births track 800 × 1.006^years.
- Size premium +0.05%/month per −1 sd log cap (clipped ±2 sd), not scaled by `signal_strength`.
- Share count changes only via random issuance / buybacks; price = market cap / shares.
- Benchmarks: S&P proxy = VW of the 500 largest + 0.15% tracking noise; VW/EW of all alive; rf AR(1) ≈ 0.25%/mo.
- Money units are dollars throughout.

## Validation infrastructure

- **Reference simulator** (`src/backtest/reference.py`): top-N, formation at T uses only the cross-section at T,
  returns earned from T+1; weights drift; a delisted name's weight is removed at the start of the month after its
  final row (proceeds pro-rata to survivors = selling at the delisting price). Turnover = ½Σ|Δw| vs drifted
  weights; cost = Σ|Δw| × one-way cost, charged in the first month after the rebalance; the initial purchase is
  charged. The production engine must reproduce it (`test_engine_matches_reference_simulator`).
- **Transaction costs** (`src/backtest/costs.py`): default is size-dependent one-way cost interpolated in log
  market cap: $10M 150 bps, $50M 80, $250M 40, $1B 20, $10B 8, $100B 5. Anchored on effective-spread estimates
  (Novy-Marx & Velikov 2016; Frazzini, Israel & Moskowitz 2018: ~10–20 bps for large caps, 100+ bps for microcaps).
  `flat` 20 bps remains available for comparison. The oracle's realized drag is ≈ 0.3%/yr at 35% annual turnover.
- **Oracle** (`src/analytics/oracle.py`): top-N on `true_alpha` through the reference simulator. Every strategy is
  reported as *fraction of oracle* = strategy excess CAGR over the EW universe ÷ oracle's. ≈0 → pipeline lost the
  signal; >1 → leakage. Only available for sources that expose a planted signal (duck-typed, no source names).
- **Null distribution** (`src/analytics/null_distribution.py`, stored in `validation/results/null_distribution.json`):
  30 independent zero-signal panels (seeds 1000–1029), top-30 EW annual December rebalance, size-dependent costs;
  metric = annualized excess over the EW investable universe. Built with the reference simulator on the planted
  characteristic (which, at zero signal, is the same selection rule the pipeline applies); later regenerated
  with the production engine (`python -m validation.build_null`, see the Backtest engine section).
- **Universe** (`src/universe.py`): excludes names delisted in the formation month (no next-month return to earn).

## Factor library — `src/factors/`

Each factor is a registered function on the formation-date snapshot (`src/snapshot.py`), which is the only
place returns and fundamentals are joined, always through `src/pit.py`. The snapshot has no `available_date`
column, so a factor cannot key on it.

| Factor | Formula | Direction | Missing / sign treatment |
|---|---|---|---|
| return_on_capital | EBIT / (max(NWC,0) + net PP&E), NWC = (CA − cash) − (CL − STD) | higher | NaN when capital ≤ 0 or inputs missing; negative EBIT kept (ranks worst) |
| earnings_yield | EBIT / (mcap + LTD + STD + preferred + minority − cash) | higher | NaN when EV ≤ 0 (cash-rich shells); negative EBIT kept |
| book_to_market | book equity / mcap | higher | NaN when book equity ≤ 0 — a negative-equity firm is not "cheap" |
| earnings_to_price | net income TTM / mcap | higher | negative earnings kept |
| sales_to_price | revenue TTM / mcap | higher | — |
| fcf_yield | (OCF − capex) / mcap | higher | negative FCF kept |
| shareholder_yield | (dividends + (shares_prior − shares) × price) / mcap | higher | NaN without a prior-year share count; net issuance goes negative |
| gross_profitability | gross profit / total assets | higher | NaN when TA ≤ 0 |
| roe | net income / book equity | higher | NaN when book equity ≤ 0 (a loss on negative equity would look like high ROE) |
| accruals | (net income − OCF) / total assets | lower | NaN when TA ≤ 0 |
| asset_growth | TA / TA one year earlier − 1 | lower | NaN when the prior point-in-time value is unavailable |
| momentum_12_1 | cumulative return months T−11..T−1 | higher | NaN with < 9 of 11 months |
| volatility_12m | annualized std of months T−11..T | lower | NaN with < 9 months |

- All price-based ratios use market cap at the formation date T. Prior-year values (`*_prior`) are a second
  point-in-time lookup at T − 12 months, so asset growth and shareholder yield are also look-ahead free.
- **Staleness**: fundamentals older than 548 days (~18 months) at the formation date are treated as missing.
- Missing values propagate as NaN; the ranking stage decides whether to drop the stock or assign the median rank.
- On synthetic data missing rates are 0–6% per factor (higher for the prior-year and trailing-return factors);
  real data will be far worse, which is why the drop-vs-median policy is explicit.

## Universe and ranking — `src/universe.py`, `src/ranking.py`

- Universe at T: drop names delisted in month T, market cap ≥ $50M, price ≥ $1, NYSE/AMEX/NASDAQ only,
  exclude SIC 6000–6999 and 4900–4999 by default, then the largest N (default 5,000) by market cap.
- Rank-sum: each factor ranked with 1 = best by its own direction, ties get the average rank. `score` is the
  weighted *average* rank (weights sum to 1); `score_raw` uses weights × number of factors, so with equal weights
  it is the plain Greenblatt sum of ranks (5th + 26th = 31). Final rank sorts by `score`, ties broken by security id.
- Z-score: winsorize at the 1st/99th percentiles *within the universe at T*, standardize, negate lower-is-better
  factors, weighted sum, rank descending.
- Missing policy: `drop` (default) removes a stock lacking any selected factor; `median` assigns the median rank
  (rank-sum) or z = 0. Because missing quarters concentrate in distressed names, `drop` removes sick companies;
  this is visible in the holdings log, not hidden.

## Leakage tripwires (`tests/test_leakage_traps.py`, permanent)

| Trap | Construction | Detection on the synthetic panel |
|---|---|---|
| Rank on `fiscal_period_end` | `asof_col="fiscal_period_end"` (warns) | quarterly decile spread 1.42% → 2.08% |
| Back-fill missing quarters from the next period | grid reindex + `bfill`, on-time `available_date` | among stock-months whose snapshot changed, a leaked-up value is followed by +3.0% vs +1.2% (t≈3) |
| Universe requires 12 future months of returns | filter on last return date | average forward return +0.2%/quarter; every in-year death vanishes |
| Momentum window ends at T+1 | `trailing_return_stats(end=T+1, skip_last=0)` | momentum decile spread 0.3% → 15%+; 12-0 vs 12-1 (no overlap) shows no inflation |
| Market cap at T while earning month T | engine hold-window shifted by one month | engine level; see the Backtest engine section |

- The forward-fill trap taught us something about the generator: because the planted expected return keys on the
  *published* characteristic, ranking on future values misaligns with it and cancels most of the announcement-jump
  gain in the whole-cross-section spread. Leak detectors therefore compare *only* the observations a modification
  changed. Any modification that carries no information about the future must show zero relation between the
  change it induces and forward returns.

## Backtest engine — `src/backtest/engine.py`, `portfolio.py`, `src/config.py`

- **Timing convention**: formation at month end T uses the cross-section at T (market cap at T, fundamentals with
  `available_date ≤ T`, returns through T); the portfolio earns returns from T+1. The engine must reproduce the
  reference simulator month by month (`test_engine_matches_reference_simulator`, tolerance 1e-12).
- **Rebalance calendar**: annual (configurable month), quarterly, monthly. The final month end is never a formation
  date. Extra 13 months of history are loaded before `start` so momentum and prior-year lookups exist at the
  first formation date.
- **Weights**: equal (default) or market-cap at T. Between rebalances weights drift; a name that disappears is
  removed at the start of the following month with proceeds pro-rata to survivors.
- **Turnover** = ½ Σ|target − drifted| per rebalance; **cost** = Σ|target − drifted| × one-way cost by market cap at
  T, charged in the first month after the rebalance; the initial purchase is charged. Reported turnover per year is
  the sum over rebalances in the year.
- **Staggered sleeves**: 12 (annual) or 3 (quarterly) independent runs, offset by one month each; the combined
  return is the average over sleeves that are invested (capital deployed one sleeve at a time). Reported turnover
  is per unit of total capital.
- **Logs**: `rankings` = every ranked universe stock at every formation date (factor values, per-factor ranks,
  `score`, `score_raw`, `rank`, market cap, price, SIC, exchange, fiscal period used, fundamentals age, weight
  if held, realized holding-period return); `holdings` = the held subset; `universe_stats` = alive → universe →
  ranked → dropped-for-missing counts per date.
- **Fraction of oracle**: reported as NaN when the oracle's own excess is below 0.5%/yr (placebo panels), with the
  strategy-minus-oracle difference in %/yr shown instead. Strategy and oracle overlap ~90% in holdings on the
  default panel; ratios in roughly 0.7–1.3 are noise (their monthly difference has t ≈ 0.3). The look-ahead
  variant pushes the ratio to ~1.9 on a quarterly calendar.
- **Engine-level traps**: ranking on `fiscal_period_end` (+1.5%/yr CAGR quarterly, +0.5%/yr annual — the leak
  lives in the first months after each formation); applying the T-dated portfolio at T−1 so it earns month T
  (deflates value strategies by ~1.2%/yr annual, ~4%/yr quarterly — a leak in either direction is a leak).
- **Null distribution** is regenerated with the production engine (`python -m validation.build_null`), 30 seeds.
- Seed 42's zero-signal panel is a lucky one: the placebo strategy sits at the ~93rd percentile of the null with
  t ≈ 1.2. The engine-level placebo test therefore asserts |t| < 2 rather than "excess ≈ 0", and the report always
  prints the null percentile.

## Analytics — `src/analytics/`

- CAGR is geometric from monthly returns; volatility, Sharpe and Sortino are annualized with √12; Sharpe/Sortino use
  the monthly risk-free series from the data source. Sortino's denominator is the root mean square of negative
  excess returns (zeros included).
- Max drawdown duration = months from the peak to full recovery, or to the end of the sample if never recovered.
- Calendar-year table uses only full 12-month years for hit rates. Rolling excess = rolling annualized strategy
  return minus rolling annualized benchmark return (36 and 60 months).
- CAPM alpha/beta: OLS of monthly strategy excess return on VW-market excess return; alpha annualized ×12.
- **Benchmarks**: the equal-weighted *investable universe* (same filters, same formation dates, no costs) is the
  primary comparison — a 30-stock EW portfolio tilts small, so beating the cap-weighted S&P 500 can be the size
  effect. VW/EW market and S&P proxy from the data source are also shown.
- **Deciles** are a diagnostic of the signal, not of the strategy calendar: the universe is re-ranked at a quarterly
  cadence (same universe, factors, method), each decile held equal-weighted and gross of costs to the next sort.
  Decile 10 = best. At realistic signal strength only the tails separate in a single 35-year panel (each decile's
  CAGR has ~±2.5%/yr sampling noise), so the table also reports each decile's t-stat vs the EW universe and the
  top-minus-bottom t-stat, not just monotonicity.
- **In-sample / out-of-sample**: `split_date` in the config labels months ≤ split as in-sample and > split as
  out-of-sample; metrics are computed separately for each, plus the full sample.
- **Run log** (`runs/run_log.jsonl`): every run appends config, hash, headline numbers and null percentile. The
  report shows the distinct-configuration count, where the best of N zero-information runs would land
  (null percentile N/(N+1)), and the Sidak-adjusted 5% bar after N trials: null quantile (0.95)^(1/N).
- **Null percentile granularity**: with 30 null runs, percentiles above ~93 are coarse (the top value is the
  97th). The t-stat is reported alongside.

## WRDS adapter — `src/data/wrds_source.py` (written, not run)

- **Layout detection**: use `crsp.msf_v2` (CIZ) when it exists and `prefer_ciz` is on, else legacy `crsp.msf`.
  In CIZ the adapter assumes `mthret` already contains the delisting return and only flags the final month; in the
  legacy path `dlret` is compounded into the delisting month: (1+ret)(1+dlret)−1, or a new row if the month has no
  return. Rows after the delisting month are dropped.
- **Missing delisting returns**: −30% for performance-related codes (500, 520–584) per Shumway (1997), 0 otherwise.
  Configurable (`missing_dlret_fill`).
- **Common-stock filter**: legacy `shrcd ∈ {10, 11}` and `exchcd ∈ {1, 2, 3}` — this alone excludes ADRs (30/31),
  ETFs (73), REITs (18/48) and closed-end funds; CIZ uses the WRDS-recommended flag set (`sharetype='NS'`,
  `securitytype='EQTY'`, `securitysubtype='COM'`, `usincflg='Y'`, `issuertype ∈ {ACOR, CORP}`,
  `primaryexch ∈ {N, A, Q}`, `conditionaltype='RW'`, `tradingstatusflg='A'`). Share/exchange codes are taken from
  the `msenames` record valid at each date, so a stock that changes exchange is filtered month by month.
- **Prices**: `abs(prc)` (negative = bid/ask midpoint); market cap = price × shrout × 1,000.
- **Compustat**: `fundq` with `INDL/STD/D/C`; cash-flow items are fiscal-year-to-date and are differenced within
  (gvkey, fyearq) — a gap in the sequence makes the quarter NaN; TTM flows = sum of 4 consecutive quarters
  (~90-day spacing, else NaN); balance sheet from the latest quarter. EBIT = `oiadpq`; gross profit = `saleq − cogsq`;
  book equity = `ceqq + txditcq` (FF-style: common equity plus deferred taxes; preferred is a separate column).
  Dollar items × 1e6. Annual `funda` fills only fiscal periods the quarterly path does not cover.
- **`available_date`**: `rdq` when present and ≥ `datadate`; else `datadate + 90 days` (quarterly) or `+ 4 months`
  (annual). Both configurable.
- **Linking**: `crsp.ccmxpf_lnkhist`, `linktype ∈ {LU, LC}`, `linkprim ∈ {P, C}`, `datadate` within
  `[linkdt, linkenddt]` (NULL end = active), primary link preferred when several match.
- **Benchmarks**: `crsp.msi.vwretd/ewretd` (VW/EW market incl. dividends), `crsp.msp500.vwretd` (S&P 500 total
  return proxy), `ff.factors_monthly.rf`.
- **Known optimism**: standard Compustat carries restated values, so even a correct WRDS backtest is mildly
  optimistic versus true as-first-reported data (Compustat Snapshot would fix that).
- Licensed data: `data/` and all caches are git-ignored; nothing downloaded is ever committed.

## Free real-data source — `src/data/free.py`

- **Universe**: a static list of ~340 of today's US large/mid caps (`free_universe.py`). This is survivorship bias
  by construction and the metadata/banner say so. Tickers that fail to download are skipped and listed.
- **Prices**: Yahoo Finance monthly bars; `total_return` = adjusted-close percentage change (dividends and splits),
  `price` = unadjusted close. A ticker that stops trading before the sample end gets its final row flagged, but
  there is no delisting return (unknown), so even the flagged rows are optimistic.
- **Shares outstanding**: the DEI cover-page fact `EntityCommonStockSharesOutstanding` from SEC companyfacts,
  applied as of each filing date (point-in-time). If a company never reports it in that unit, today's yfinance
  share count is used for every date and the ticker is listed under `tickers_with_current_shares_only`.
- **Fundamentals**: SEC XBRL companyfacts. `available_date` = the `filed` date of the filing that reported the
  latest piece. Values are **as first reported** (earliest filing per period); comparatives in later filings are
  ignored, so restatements do not appear as extra rows (the point-in-time property is preserved either way).
- **TTM construction**: at a fiscal year end, the 10-K annual value; otherwise `annual(last FY) + YTD(this period)
  − YTD(same period last year)`, using the longest sub-annual duration ending at each date. Verified on Apple:
  FY2023 revenue 383,285 and operating income 114,301 ($M) match the 10-K; June-2024 TTM revenue 385,603 =
  383,285 + 296,105 − 293,787.
- **Tag map**: revenue = Revenues / RevenueFromContractWithCustomerExcludingAssessedTax / SalesRevenueNet;
  EBIT = OperatingIncomeLoss (missing for ~20% of firm-quarters, notably banks); gross profit = GrossProfit or
  revenue − CostOfRevenue; net income = NetIncomeLoss; OCF = NetCashProvidedByUsedInOperatingActivities;
  capex = PaymentsToAcquirePropertyPlantAndEquipment; dividends = PaymentsOfDividends(CommonStock);
  balance sheet = AssetsCurrent, CashAndCashEquivalentsAtCarryingValue, LiabilitiesCurrent, DebtCurrent /
  LongTermDebtCurrent, LongTermDebtNoncurrent / LongTermDebt, PropertyPlantAndEquipmentNet, Assets,
  StockholdersEquity, PreferredStockValue, MinorityInterest. Preferred, minority interest and short-term debt
  default to 0 when untagged; every other missing item is NaN (and the ranking's missing policy applies).
- **SIC and exchange** from the SEC submissions endpoint.
- **Benchmarks**: S&P 500 = SPY adjusted-close return (an ETF proxy); VW/EW "market" = the survivor universe
  itself; risk-free from the 13-week T-bill yield (^IRX), de-annualized.
- **SEC fair access**: 10 requests/second, descriptive User-Agent. Set `SEC_USER_AGENT="Your Name you@example.com"`;
  nothing personal is hard-coded. All downloads are cached under `data/cache/free/` (git-ignored).
- The stored null distribution was built on the synthetic panel; the report and UI say so when another source
  is in use. Regenerate it per universe if you rely on it.
- Demo result, 2011–2024, Magic Formula on this universe: +7.1%/yr over the S&P 500 (t = 4.8) but only +1.8%/yr
  over the equal-weighted survivor universe (t = 1.2). That gap *is* the survivorship bias plus equal weighting.

## Notes and known limitations

- **Sale cost on a name that delists in the formation month**: at a rebalance, a holding whose final row is the
  formation month is still in the drifted weights, so its removal is charged a one-way cost as if sold. Real-world you
  could not sell it; the effect is a few basis points of extra drag per such event, i.e. conservative.
- **Free source share counts before the first XBRL filing**: for months before a company's first DEI share-count
  filing, the first known count is carried backwards. Market cap on those early months is therefore slightly
  wrong-dated; it only affects the size filter and cap weights, never the factor values.
- **Short windows**: the engine refuses windows under two years (no meaningful momentum window or prior-year values)
  with an explicit error instead of an empty result; the UI enforces three years.
- Sensitivity, restated: at realistic signal strength a 30-stock, 35-year backtest clears the 95th percentile of luck
  in roughly 40% of simulated markets; the decile analysis on the whole cross-section is the reliable detector.
