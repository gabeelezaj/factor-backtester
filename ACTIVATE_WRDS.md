# Activating the WRDS data source

Checklist for the day the WRDS account is approved. Budget: about an hour, most of it waiting for queries.
Nothing outside `src/data/wrds_source.py` changes.

## 1. Install and authenticate (once)

```bash
pip3 install wrds
```

```bash
python3 -c "import wrds; db = wrds.Connection(); db.create_pgpass_file(); print(db.list_libraries()[:10])"
```

The first call prompts for your WRDS username and password **in the terminal** and writes `~/.pgpass`
(mode 600). After that no prompt appears and no credential is ever stored in this repository.
Optionally `export WRDS_USERNAME=<your username>` so the adapter does not need to ask.

## 2. Compare the adapter's assumptions with the live schema

```bash
python3 -m src.data.wrds_source --verify-schema
```

Prints every table and column the adapter expects (`EXPECTED_TABLES` in `wrds_source.py`), whether it exists,
and the other columns each table actually has. Then:

```bash
python3 -m src.data.wrds_source --layout
```

tells you whether the CIZ (`crsp.msf_v2`) or legacy (`crsp.msf` + `crsp.msedelist`) path will be used.

## 3. Resolve every `# VERIFY:` comment

```bash
grep -n "VERIFY" src/data/wrds_source.py
```

Each one is a specific assumption. The ones most likely to need attention:

| Assumption | Where to check |
|---|---|
| `crsp.msf_v2` column names (`mthcaldt`, `mthret`, `mthprc`, `shrout`, `primaryexch`, `sharetype`, …) and the CIZ common-stock filter values | `--verify-schema` output; WRDS "CRSP CIZ migration" guide |
| Whether `mthret` in CIZ already includes the delisting return (adapter assumes **yes**; if not, port `compound_delisting_returns` to `crsp.stkdelists`) | CRSP CIZ documentation |
| `shrout` unit in `msf_v2` (adapter assumes thousands, as in legacy) | compare `mthprc * shrout * 1000` with `mthcap` if present |
| `crsp.msp500.vwretd` as the S&P 500 total return; `crsp.msi.vwretd/ewretd` as VW/EW market | `--verify-schema`; spot-check a known month |
| `ff.factors_monthly.date` is first-of-month and `rf` is a monthly decimal | inspect a few rows |
| `comp.fundq`: `mibq` vs `mibtq`; `dvy` present; `txditcq` present | `--verify-schema` |
| Compustat share count unit (millions) for `cshfdq` / `cshfd` | Compustat manual |
| `crsp.ccmxpf_lnkhist` still the CCM link table; `linkenddt` NULL means active | `--verify-schema` |
| `wrds.Connection.raw_sql(query, params=...)` / `describe_table` / `list_tables` signatures | `help(wrds.Connection)` |

Delete each `# VERIFY:` comment as it is confirmed, or fix the code and note the change in `DECISIONS.md`.

## 4. Pull the data and run the contract test

```bash
python3 -m src.data.wrds_source --contract
```

This downloads returns, fundamentals and benchmarks for 1990–2024 (cached to `data/cache/wrds/`, git-ignored),
runs the canonical-schema validators, and prints row and security counts. Sanity numbers to expect: roughly
4,000–7,000 securities per year, month-end dates only, `available_date >= fiscal_period_end` everywhere, and a few
thousand rows with `is_delisted_this_month = True`.

Then run the offline transform tests, which still apply:

```bash
python3 -m pytest tests/test_wrds_offline.py -q
```

## 5. Spot checks before trusting anything

- Pick one well-known company; compare `revenue_ttm` and `ebit_ttm` at a fiscal year end with its 10-K (remember
  Compustat is in $ millions; the adapter multiplies by 1e6).
- Confirm a famous bankruptcy shows a large negative `total_return` in its final month with the flag set.
- Confirm the reporting lag distribution (`available_date - fiscal_period_end`) is mostly 30–90 days.

## 6. Switch the config and rerun

In `configs/magic_formula.yaml`:

```yaml
data_source:
  name: wrds
  params: {}
```

```bash
python3 run_backtest.py configs/magic_formula.yaml
```

The report's oracle line disappears (no planted signal on real data); the null-distribution percentile still
applies as a rough luck bar but was built on the synthetic panel — regenerate it for the real universe size if
you rely on it. The UI picks the source from the sidebar; the banner will read "WRDS: CRSP + Compustat".

## 7. Remember

- WRDS data is licensed: never commit `data/` or any cache, never share exported CSVs outside the license.
- Standard Compustat contains restated values; results are mildly optimistic versus as-first-reported data.
