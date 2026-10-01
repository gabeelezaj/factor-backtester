# Multi-factor stock ranking backtester

A generalisation of Greenblatt's Magic Formula: pick any factors, weight them, rank every stock, buy the top N,
rebalance, and compare against benchmarks — with the plumbing that keeps a backtest honest (point-in-time
fundamentals, delisting returns, transaction costs, a null distribution, an oracle upper bound, and permanent
leakage tripwires).

All market data sits behind one interface (`src/data/base.py::DataSource`). Three implementations:

| Source | Status | What it is |
|---|---|---|
| `synthetic` | working, tested | 800-company simulated market with a planted, calibrated signal. Verifies the engine. |
| `free` | working demo | Yahoo Finance prices + SEC XBRL filings for ~340 of today's large caps. **Survivorship-biased.** |
| `wrds` | written, unrun | CRSP + Compustat. Activate with [ACTIVATE_WRDS.md](ACTIVATE_WRDS.md). |

## Setup

Python 3.9+ (written 3.9-compatible, runs on 3.11+).

```bash
pip3 install -r requirements.txt
```

Optional: `pip3 install yfinance requests` for the free source, `pip3 install wrds` for WRDS.
The free source downloads from the SEC, which asks for a contact in the User-Agent:
`export SEC_USER_AGENT="Your Name you@example.com"`.

## Run

One command runs a backtest from a YAML config:

```bash
python3 run_backtest.py configs/magic_formula.yaml
```

One command launches the UI (or, on a Mac, double-click **`Launch Backtester.command`** in the project folder —
it opens Terminal, starts the app and opens your browser at http://localhost:8501; close that Terminal window to stop):

```bash
streamlit run app.py
```

Tests (≈1 minute; includes the recovery, placebo and leakage-trap tests):

```bash
python3 -m pytest -q
```

Other configs: `configs/placebo.yaml` (same strategy on a zero-signal panel), `configs/example_multifactor.yaml`
(five factors, z-score, quarterly, sleeves), `configs/free_demo.yaml` (real tickers, survivorship-biased).

## Switching data sources

Edit the `data_source` block of a config — nothing else changes:

```yaml
data_source:
  name: synthetic        # or: free, wrds
  params:
    seed: 42             # source-specific; see the adapter's config dataclass
    signal_strength: 1.0
```

In the UI, pick the source in the sidebar. The banner at the top of the page is generated from the source's
`get_metadata()` and always states what you are looking at (simulated / survivorship-biased / licensed).

Adding a source: subclass `DataSource`, return the three canonical frames (`src/data/schema.py`), register the
class in `src/data/__init__.py`. `tests/test_no_leak.py` fails if any source name appears outside `src/data/`.

## Adding a factor

A decorated function in `src/factors/` (any of the existing modules, or a new one imported from
`src/factors/__init__.py`):

```python
from src.factors import factor, safe_div

@factor("ebitda_to_ev", "EBITDA over enterprise value", "EBITDA / EV", direction="higher",
        requires=("ebit_ttm", "market_cap", "cash", "long_term_debt", "short_term_debt"),
        missing_policy="NaN when EV <= 0")
def ebitda_to_ev(s):                      # s = the formation-date snapshot, one row per security
    ev = s["market_cap"] + s["long_term_debt"] + s["short_term_debt"] - s["cash"]
    return safe_div(s["ebit_ttm"], ev)
```

The snapshot (`src/snapshot.py`) already contains point-in-time fundamentals, prior-year values, trailing
momentum/volatility and the market data at the formation date; it never contains `available_date`, so a factor
cannot look ahead. Add a hand-checked case to `tests/test_factors.py`.

## How correctness is checked

- **Point-in-time**: every fundamentals lookup goes through `src/pit.py::asof_fundamentals` (`available_date <= T`).
  A spy test proves a snapshot at T touches no later row.
- **Recovery / placebo**: the synthetic market plants a signal calibrated to real-world factor spreads
  (~0.45%/month decile spread). The pipeline recovers it (≈100% of the oracle on the default panel, ~80% across
  seeds); with `signal_strength: 0` it finds nothing.
- **Null distribution**: 30 zero-signal panels through the real pipeline give the luck distribution of the
  headline number (`validation/results/null_distribution.json`); every result is reported with its percentile.
- **Oracle**: on synthetic data, the best any point-in-time strategy could do; results are shown as a fraction of it.
- **Tripwires** (`tests/test_leakage_traps.py`, `tests/test_backtest.py`): ranking on `fiscal_period_end`,
  back-filled quarters, a universe that requires future returns, a momentum window into the holding period, and
  market cap at T while earning month T — each must keep looking broken.
- **Engine vs reference**: the vectorised engine reproduces a deliberately simple simulator to 1e-12.
- **Fresh-clone check**: the suite passes in an empty clone with no caches (the synthetic panel regenerates in
  under a second; the null distribution is committed under `validation/results/`).

Every financial and data judgement is logged in [DECISIONS.md](DECISIONS.md).

## Layout

```
run_backtest.py  app.py            entry points
configs/                            reproducible YAML configs
src/data/                           DataSource interface, schema, synthetic / free / wrds adapters (only place source names live)
src/pit.py  src/snapshot.py         point-in-time lookup and formation-date snapshot
src/factors/  src/universe.py  src/ranking.py
src/backtest/                       engine, portfolio mechanics, costs, schedule, reference simulator
src/analytics/                      metrics, deciles, oracle, null distribution, run log, report
src/ui/                             Streamlit app
validation/                         null-distribution builder and signal-strength sweep (+ stored results)
tests/
```

Runs land in `runs/<name>_<confighash>/` (monthly returns, holdings, full rankings, calendar years, deciles,
config). `runs/run_log.jsonl` counts every configuration tried; the report shows the count and how much a
result needs to clear after that many trials.

`data/` and `runs/` are git-ignored: WRDS data is licensed and must never be committed.
