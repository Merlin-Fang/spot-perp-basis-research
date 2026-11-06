# Final data-to-forecasting input integration preflight

**FINAL_FORECASTING_INPUT_INTEGRATION_READY.** All 24 requested
asset × evaluation-year × financing input cells are structurally available.
Historical fitting/calibration inputs, warmup, source identities, completed-bar
clocks and validity masks are accounted for. Statistical support is unmeasured.
No real final-period features, labels, risk sets, fitting, calibration, forecasts,
scores, intervals or economic outcomes were generated.

The final-data milestone really is committed: `a487769fe8e9079b699707da20595152c49f9a97`,
`feat: add reproducible final data acquisition and QA`, contains the eight Task 3
publication files. Task 4A starts from that commit; it does not assume the earlier
publication recommendation was a commit. The frozen protocol, disabled evaluation
flags and all existing development-only numerical guards remain unchanged.

## Small maintained interface

`data/forecast_inputs.py:adapt_hourly` converts either historical five-asset
hourly input fields or new final canonical fields into the existing annual
base-input contract. It returns the unchanged hourly clock in `origin_time_utc`,
symbol/control, float64 Spot/perpetual closes and annual financing, boolean
`theory_available`, bar opening/availability and funding-known/settlement/
publication fields. It returns separate venue validity/state masks and the
signed last funding rate as an auxiliary input. It does not compute rho or
extend the registered feature set with funding rates.

Invalid/missing prices are masked in the new model-input view; original source
values and flags remain in their unchanged canonical tables. Valid inactive
marks stay present while their primary active-input mask is false. Missing,
invalid, inactive and active states are distinct, and no clock rows are removed
or prices filled. The active mask follows the original positive volume **and**
trade-count rule on both valid venues; complete trailing hours/paths are not
added as requirements. No future label/status determines input issuance.

`data/input_preflight.py:inspect_inputs` verifies pinned input catalogs, old/new
source handoffs, hourly schema and calendar coverage. Its production path reuses
the source-only `structural_only` guard, existing funding join, historical EFFR
conversion/as-of and pure annual-window/origin-mask functions. It inspects the
annual schema/model registry and feature order statically, without importing an
execution runner. `BASE_INPUT`/`OPTIONAL_INPUT` compatibility is tested against
the maintained checkpoint interface; the real feature builder and risk-trial
interface are exercised only with pre-2023 synthetic observations. No model is
trained by the new focused tests.

The existing six features remain `rho, abs_rho, proposed_direction,
low_fee_margin, rho_change1, rho_minus_mean24`, in that order. Synthetic tests
exercise their causal right-closed 24h/minimum-12 history, change over the prior
scheduled hour, gap censoring and future-mutation invariance through the existing
implementations. No second feature, label, likelihood or model implementation
was introduced. Existing portable regressions retain their prior synthetic/
development fitting exercises; they never receive real final data.

## Historical inputs and provenance

The required historical source window is December 2020 through December 2022.
The first retained context hour is December 31, 2020 00:00 UTC; this supplies 24h
of warmup before the first fitting origin, without admitting earlier training
origins. **100** four-asset historical monthly hourly partitions and **300**
official source ZIPs/checksum companions were verified, plus four actual-event
tables and the original official 2020–2022 EFFR source/receipt. The pinned
`five-asset-observations-v1` catalog supplies original processing/configuration
and source hashes. The preflight reads selected raw input columns, not archived
forecast or experiment-result columns. BNB is not loaded.

Historical fitting coverage has genuine missingness. In the first 2023 fit
window each asset has 17 missing/invalid hourly pairs; valid inactive counts
are 3/3/4/3 for BTC/ETH/DOGE/ADA. September–December 2022 calibration has no
missing/invalid hourly pairs and 0/0/1/1 valid inactive pairs. Those hours remain
in the clock. These are input-quality counts, not supported labels or fitted
sample sizes. All historical and final checked stages have zero unknown
funding/reference-financing inputs.

A real inspection initially exposed a neutral compatibility defect: historical
DOGE/ADA May 2021 and November 2022 archives have local cache-name prefixes.
Their source bytes and official hashes agree, but the first adapter compared
the checksum's official filename to the local alias. The fix compares the
official URL basename while still verifying the exact URL, archive SHA256 and
checksum SHA256. The focused alias test covers this case. No archive was moved,
renamed, repaired or rebuilt. The initial diagnostic remains in ignored results;
the successful record is `verified_input_readiness.json`.

All **444** Task 3 final/warmup archive receipts/checksums were checked offline,
without download or dataset reconstruction. The new source/table content-set
pins match the published Task 3 receipt. Final hourly, reference, event and
March-interval table hashes are verified before bounded reads. December 2022
event overlap reconciles exactly: 93 events per asset, 372 total. The 21
overlapping December EFFR effective-date/rate/publication records agree across
the historical and final source vintages. Shared warmup records are compared
before the explicit historical/final handoff, never silently deduplicated.

## Annual calendars and four-asset readiness

All windows use the registered UTC 20/4/12-month schedule. The table contains
input-clock counts before any feature/outcome eligibility. The parenthesized
fit/calibration counts apply only the frozen strict `t+16h<C` calendar rule.
Actual trial/label publications must later be strictly before C as well; they
were not constructed to test support here.

| Evaluation | Fit window | Fit clock (calendar-permitted) | Calibration window | Calibration clock (calendar-permitted) | Evaluation clock |
| --- | --- | --- | --- | --- | --- |
| 2023 | [2021-01-01, 2022-09-01) | 14,592 (14,576) | [2022-09-01, 2023-01-01) | 2,928 (2,912) | 8,760 |
| 2024 | [2022-01-01, 2023-09-01) | 14,592 (14,576) | [2023-09-01, 2024-01-01) | 2,928 (2,912) | 8,784 |
| 2025 | [2023-01-01, 2024-09-01) | 14,616 (14,600) | [2024-09-01, 2025-01-01) | 2,928 (2,912) | 8,760 |

| Asset | 2023 zero / EFFR inputs | 2024 zero / EFFR inputs | 2025 zero / EFFR inputs |
| --- | --- | --- | --- |
| BTCUSDT | available / available | available / available | available / available |
| ETHUSDT | available / available | available / available | available / available |
| DOGEUSDT | available / available | available / available | available / available |
| ADAUSDT | available / available | available / available | available / available |

Every 24h warmup clock is present, including December 2022 context for the 2025
fit. Monthly historical partitions are selected by bar-opening month, including
the previous month, before filtering on decision timestamps in the Parquet
reader. January 2023 midnight comes from Task 3's December 2022 final bar;
old historical observations stop before 2023. All 12 asset/annual midnight
handoffs are exact and unduplicated. Leap 2024 retains all 8,784 scheduled hours.
No evaluation-origin 16h deletion is applied. The final endpoint is strictly
before January 1, 2026; no 2026 market input is required or read.

The real March interruption remains 81 explicit minute endpoints per asset:
one invalid partial Spot bar at decision 12:40 and 80 missing Spot observations
at decisions 12:41–14:00, March 24, 2023 UTC. Perpetual observations remain
present. The two affected hourly origins, 13:00 and 14:00, are unavailable as
active paired inputs. Synthetic pre-2023 tests reproduce the gap, invalidate
the next hour's rho change, and verify first-passage censoring before the gap
with publication at the first missing hour. No real March outcomes were made.
The frozen hourly target remains distinct from minute-path risk.

Funding preserves signed rates and exact subsecond settlement/publication
timestamps. The existing join is checked at settlement +1min, with age <=12h
from settlement. A false known flag cannot hide future publication metadata.
Historical EFFR uses the existing guarded percent/100, effective-date +7day
conversion/as-of; final EFFR values are copied from the verified canonical input
while its most recent permitted effective/publication clock is checked. Zero
financing is a separate constant, with no new funding or reference-age rule.
No financing/funding cash is calculated.

## Reproduction and data-bundle requirements

Use the existing locked Python environment described in
[final_data_reproduction.md](final_data_reproduction.md); no dependencies were
added. From the repository root, with the verified final dataset and original
historical inputs present:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.preflight_final_integration \
  --data-root data/final_data_v1 --history-root . \
  --output results/final_forecasting_integration_preflight/verified_input_readiness.json

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -p test_final_input_integration.py -v
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.test_portable
```

A fresh source checkout must be supplied with the verified **input data**, not
old untracked Python scripts. `--history-root` can point to an external bundle
retaining the relative paths in `configs/forecast_input_integration_v1.json` and
its two historical manifests. The exact **708** historical input dependencies
and hashes are emitted as `required_history_inputs` in the ignored successful
report: 300 ZIPs, 300 checksums, 100 hourly tables, four event tables, two
manifests, EFFR JSON and its provenance receipt. No historical builder or archived
forecast/trading output is required. Do not copy whole experimental result
directories; only the explicitly cataloged source files are needed.

The final data root uses Task 3's portable layout and pins. `--public-root` may
locate the pinned public receipt outside the source checkout; `--input-catalog`
can describe a relocated verified input bundle or a synthetic fixture. Catalogs
contain paths/hashes only, with no scientific or execution options. The default
committed catalog pins the actual verified inputs. Wrong schema, hashes,
instrument, duplicate clocks, units or handoffs fail explicitly. Global catalog
failure retains 24 unavailable cells; a missing/bad partition conservatively
invalidates that asset's six cells rather than substituting inputs. Individual
observed missing/inactive rows remain handled missingness inside present tables.

The CLI has no acquire, build, fit, label, forecast or score action. It writes
only an optional new immutable metadata report; omit `--output` to inspect
without saving one. Repeating a completed report preserves bytes and mtime.
The clean-source test exports committed HEAD, overlays only the four proposed
integration source/config files and runs this same CLI with an external
synthetic bundle and denied network. No real holdout is used in that test.

## Verification, limits and publication

See [final_forecasting_integration_verification.json](final_forecasting_integration_verification.json)
for final test counts, source/config pins, repeatability, preservation and measured
resources. **317 portable tests passed**, retaining all prior 286 and adding
31 focused integration tests; the focused suite separately passed in 57.783s.
The complete portable run took 255.673s. Clean-source CLI reproduction and
explicit prohibited-operation tests passed. All 28,116 prior file states and
21,146 prior content hashes remain unchanged, including archived experiments,
the scientific freeze and README draft; HEAD/index/refs are unchanged.
The successful real preflight took 10.78s with 402,800 KiB maximum
RSS. It streamed archive hashes, read 112 small hourly partitions and four
bounded March minute intervals, and did not materialize three years of minute
inputs. Its 243KB exhaustive input/dependency report remains ignored.
A real repeat took 10.73s with 411,908 KiB maximum RSS, returned the same 24
available input cells, and preserved the completed report's content and mtime.

Recommend only these seven new files for the next reviewed commit:

- `src/spot_perp_basis/data/forecast_inputs.py`
- `src/spot_perp_basis/data/input_preflight.py`
- `scripts/preflight_final_integration.py`
- `configs/forecast_input_integration_v1.json`
- `tests/test_final_input_integration.py`
- `docs/final_forecasting_integration_preflight.md`
- `docs/final_forecasting_integration_verification.json`

Keep `results/final_forecasting_integration_preflight/`, all datasets, source
archives/manifests, full dependency inventories, checkpoint/forecast blobs and
runtime logs ignored. Existing untracked research files and the modified README
draft are outside this change. Nothing is staged, committed, pushed or tagged.

There is **no unresolved input-integration blocker**. This result does not
establish trainability, optimizer convergence, calibration/class support,
survival/IPCW availability or out-of-sample skill. Final numerical execution
remains separately gated: existing annual fit/checkpoint/forecast APIs still
reject final years, and this task creates no final numerical runner. A later
explicit authorization must enable only the registered final numerical path;
no data/readiness result grants that permission. Financial replay remains
outside this forecasting input task.
