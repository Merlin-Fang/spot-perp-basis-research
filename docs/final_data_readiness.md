# Final data acquisition and structural readiness

**FINAL_DATA_STRUCTURAL_QA_READY** — `final-inputs-v1`, following Task 3's
raw-input-only authorization. All required source partitions were acquired or
reused, checksum-verified, parsed, built and independently reconciled. Known
missing/invalid/inactive minutes remain explicit. This is structural readiness;
final statistical support and model performance have not been measured.

Baseline `pre-final-data-v1` resolves to `29dec9b7315f39001ac6c9f0a1476ab661f6c410`.
The frozen `final-survival-evaluation-v1.0.0` and its disabled evaluation flags
are unchanged. [Reproduction](final_data_reproduction.md),
[compact verification receipt](final_data_verification.json).

## Actual source coverage

There are **432 final-period monthly ZIPs plus 12 December-2022 warmup ZIPs**.
All **444** match their official Binance SHA256 companions. **228** were reused
(216 BTC/ETH final-period ZIPs and all 12 warmup ZIPs); **216** DOGE/ADA final-period
ZIPs were downloaded. Reused bytes: 298,227,408; downloaded ZIP bytes: 241,914,992.
The official EFFR JSON response adds one source, 218,136 bytes. No duplicate ZIP
transfer, conflict, corrupt archive, unresolved transfer or missing upstream
partition occurred. The initial inventory classified 228 existing inputs as
unverified and 216 as missing; existence was not treated as validation.

Every cell below has 12 verified monthly files for each source. Funding counts
are read from actual events, rather than generated from a nominal schedule.
Missing/invalid/inactive counts describe the canonical minute decision clock.

| Asset | Year | Spot ZIPs | USDT-M ZIPs | Funding ZIPs | Actual funding events | Missing Spot minutes | Invalid Spot minutes | Valid paired inactive minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 2023 | 12 | 12 | 12 | 1,095 | 80 | 1 | 189 |
| BTCUSDT | 2024 | 12 | 12 | 12 | 1,098 | 0 | 0 | 89 |
| BTCUSDT | 2025 | 12 | 12 | 12 | 1,095 | 0 | 0 | 37 |
| ETHUSDT | 2023 | 12 | 12 | 12 | 1,095 | 80 | 1 | 93 |
| ETHUSDT | 2024 | 12 | 12 | 12 | 1,098 | 0 | 0 | 89 |
| ETHUSDT | 2025 | 12 | 12 | 12 | 1,095 | 0 | 0 | 38 |
| DOGEUSDT | 2023 | 12 | 12 | 12 | 1,095 | 80 | 1 | 200 |
| DOGEUSDT | 2024 | 12 | 12 | 12 | 1,098 | 0 | 0 | 90 |
| DOGEUSDT | 2025 | 12 | 12 | 12 | 1,095 | 0 | 0 | 39 |
| ADAUSDT | 2023 | 12 | 12 | 12 | 1,095 | 80 | 1 | 891 |
| ADAUSDT | 2024 | 12 | 12 | 12 | 1,098 | 0 | 0 | 143 |
| ADAUSDT | 2025 | 12 | 12 | 12 | 1,095 | 0 | 0 | 40 |

All perpetual source minutes are observed and valid. The 2023 Spot gap is
`[2023-03-24 12:40, 14:00)` UTC in **bar-opening time**, in all four assets.
The preceding 12:39 bar closes at 12:39:41.646 rather than 12:39:59.999, so the
original close-time rule flags it invalid. No correction, interpolation,
forward fill or dropping of clock rows was applied. Thus each asset has 525,519
valid paired minute endpoints in 2023, versus 527,040 in leap-2024 and 525,600 in
2025. These source anomalies are handled missingness, not parser failures or
permission to relax later survival censoring. No causal explanation of the
exchange interruption was inferred from market values.

## Canonical inputs and independent checks

The isolated ignored `data/final_data_v1/` contains **165 canonical Parquet
tables**, 818,435,652 bytes: 148 minute partitions, 12 annual hourly tables,
four event tables and one EFFR table. The final clock has **6,312,960 minute rows**
and **105,216 hourly input rows**, including missing/inactive rows. These are
input observations, not labels, forecasts, opportunities or independent trials.

| Asset | 2023 active hourly inputs / clock | 2024 active hourly inputs / clock | 2025 active hourly inputs / clock |
| --- | --- | --- | --- |
| BTCUSDT | 8,756 / 8,760 | 8,783 / 8,784 | 8,760 / 8,760 |
| ETHUSDT | 8,757 / 8,760 | 8,783 / 8,784 | 8,760 / 8,760 |
| DOGEUSDT | 8,753 / 8,760 | 8,783 / 8,784 | 8,760 / 8,760 |
| ADAUSDT | 8,736 / 8,760 | 8,780 / 8,784 | 8,760 / 8,760 |

Independent CSV reads reconcile **12,625,600 retained raw venue rows**, every
OHLCV/quote/trade/taker field, close timestamp and observation key, and **13,524
funding events** including warmup. Duplicate keys: zero. Invalid OHLC/activity
records: zero; invalid close-time records: the four partial bars above. The
existing 10% minute-price and 100bps cross-market diagnostic flags were retained;
197 paired cross-market flags were not used to remove observations. There are
1,938 valid paired inactive minute endpoints across the full four-asset clock.

Actual per-file timestamp inspection found 148 perpetual kline files in ms,
100 Spot files in ms and 48 2025 Spot files in us; funding uses ms. All canonical
timestamps normalize to UTC ns. Both sides of the 2024–2025 transition, annual
midnights, February 28/29 and March 1, 2024, and the terminal cutoff passed.
The 28 explicit per-asset boundary checks preserve the immediately completed
minute. January 2023 midnight comes from December 2022; annual hourly tables
include it. Minute partitions remain keyed by opening month, so consumers must
include the preceding month when selecting decision-year endpoints.

Canonical decision inputs end strictly before January 2026. The last raw
December-2025 bar remains in its ZIP but supplies no in-window decision endpoint.
No 2026 market file was accessed. All four assets were listed before this window;
there is no structural prelisting or launch quarantine within these sources.

## Funding and financing

The final period contains **13,152 actual funding events** plus 372 warmup events.
Archive metadata reports eight-hour intervals throughout; there are no interval
changes, duplicate settlements or cadence gaps at the existing one-second
diagnostic tolerance. Millisecond jitter is preserved: for example January 2025's
first settlement is 00:00:00.015 UTC, not rounded to midnight. This is an observed
history, not a promise about future scheduling. Signed decimal rates are retained.
Settlement +1min publication, backward as-of and maximum settlement age 12h
produce **zero unknown funding inputs** on the minute and hourly clocks.
No future funding labels or cash payments were constructed.

The official [NYFed EFFR source](https://www.newyorkfed.org/markets/reference-rates/effr)
returned **772 current-vintage observations**, effective December 1, 2022 through
December 31, 2025: 21/250/251/250 by effective year. Duplicate and weekend records:
zero; maximum effective-date gap: four calendar days. Nonpublication days are
handled by the frozen backward join. Percentage/100, effective-date +7 calendar
days and zero financing remain separate columns. Independent as-of reconciliation
found **zero unknown EFFR inputs**, including annual boundaries. The publisher
provides no companion checksum: provenance is the official HTTPS response,
retrieval metadata and frozen local SHA256. Current-vintage provenance is not an
original Aave or vintage-as-published history.

Raw schemas, exact official endpoint templates, fields, cache semantics and
commands are in [the reproduction guide](final_data_reproduction.md). Exhaustive
inventories, source-specific retrieval times/URLs/hashes, missing intervals and
QA evidence are in ignored `inventory.json`, `build_receipt.json` and
`qa_receipt.json` under the data root. Public content-set digests in the compact
receipt distinguish exact reproduction from a later revised official vintage.

## Verification and preservation

**286 portable tests passed** (all previous 262 plus 24 focused tests), in
185.865 seconds; the focused suite separately passed in 2.347 seconds. The
source-only export test used Git HEAD plus exactly the proposed maintained
modules/CLI/config, 444 synthetic official-format source partitions, an external
temporary data root and denied network connections. It verified the offline CLI
and a month of ingestion/storage/independent QA without legacy untracked code.
This is fixture portability, not a second full real-data download.

A real offline rerun of `--verify --build --qa` passed in 2.36 seconds, with
140,912 KiB maximum RSS. All **1,503** source/table/inventory/receipt files retained
their content, size and mtime. Completed build/QA stages validate hashes and reuse
their prior verified receipts; they do not recompute the full raw-row replay on
each idempotent invocation. A clean raw-only reconstruction does perform that
replay. The full first build processes one month at a time; its peak RSS was not
separately instrumented. Raw ZIPs and canonical tables together occupy about
1.36 GB before small inventories/logs; no capacity failure occurred.

All **26,595** pre-existing file states and **21,138** protected content hashes
are unchanged. The scientific protocol, YAML and freeze receipt match their
verified hashes. HEAD, index and refs are unchanged; no staging, commit, push or
tag occurred. Existing untracked research files and the modified README draft
were preserved; an optional README pointer patch is proposed in the guide.

No final-period first-passage labels, risk trials, model fits, calibration,
forecasts, Brier/predictive metrics, trading cash or economic outcomes were
constructed or inspected. The source-only guard rejects calls to those numerical
modules. Portable regressions use their existing synthetic/development fixtures.

## Publication recommendation and remaining gates

Recommend **only these eight new files** for the next reviewed data-preparation
commit; nothing has been staged:

- `src/spot_perp_basis/data/final_sources.py`
- `src/spot_perp_basis/data/final_inputs.py`
- `scripts/prepare_final_data.py`
- `configs/final_data_v1.json`
- `tests/test_final_data.py`
- `docs/final_data_reproduction.md`
- `docs/final_data_readiness.md`
- `docs/final_data_verification.json`

Keep the entire `data/final_data_v1/` ignored: downloaded/imported ZIPs,
checksums, raw HTTP receipts/JSON, Parquet tables, exhaustive inventories,
preservation snapshots, test/runtime logs and generated coverage/QA receipts.
Existing data, results, experiment archives, environments, caches and unrelated
untracked files remain excluded. No `.gitignore` change is needed.

There is **no unresolved acquisition or structural-build blocker**. Missing
Spot minutes, partial close times, activity exclusions, price-proxy conventions
and current-vintage financing remain explicit limitations. A later authorized
adapter must consume validity/availability masks and the full decision clock;
this task does not certify complete survival paths or statistical model support.
The shortest next step is review/publish this source milestone, then separately
authorize the bounded final input/forecast integration and registered final
statistical execution. Development-only fitting/evaluation guards and disabled
frozen execution flags have not been relaxed.
