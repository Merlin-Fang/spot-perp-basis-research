# Final raw inputs: reproduction and structural QA

Task 3 separately authorizes raw-input acquisition and structural QA for BTCUSDT,
ETHUSDT, DOGEUSDT and ADAUSDT during UTC `[2023-01-01, 2026-01-01)`.
December 2022 is the sole extra source month: it supplies the first completed
minute, past funding and EFFR lookback. This permission does not authorize event
labels, fitting, calibration, forecasts, scores or trading. The scientific YAML
and all development-only numerical access guards remain unchanged.

Use a clean checkout containing the publication files listed in
[final_data_readiness.md](final_data_readiness.md). No legacy untracked script,
private manifest or historical experiment is needed. From its root, create an
environment with Python >=3.12 and the existing hashed locks:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-build-lock.txt
.venv/bin/python -m pip install --require-hashes -r requirements-lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

Tested here: Python 3.14.7, NumPy 2.5.4, pandas 3.0.6, PyArrow 26.0.0,
PyYAML 6.0.3, matplotlib 3.11.2 and setuptools 84.0.0. No dependencies were added.

Select an isolated, writable data-version directory with sufficient free space.
Paths are explicit; there is no developer-home or private-cache default. The
actual run used the ignored repository-relative `data/final_data_v1` directory.
Never select an existing historical dataset directory as the new version root.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.prepare_final_data \
  --data-root data/final_data_v1 --acquire --verify --build --qa
```

An optional `--reuse-root data/raw/binance` scans an existing raw Binance tree
by exact authorized file names and venue directories before downloading. It
does not require that tree or its old manifests. Every imported archive must
match a newly retrieved official checksum; identical duplicate copies are
recorded. Conflicting candidates remain untouched and receive an ignored
quarantine record, blocking that partition. The original run used this option.

The offline command makes no network requests. Supply the version directory's
raw archives, `.CHECKSUM` and `.receipt.json` files and `inventory.json`; omit
canonical tables/build receipts for a clean reconstruction. A completed version
is checked, never silently rebuilt under different source/code identities.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.prepare_final_data \
  --data-root data/final_data_v1 --offline --verify --build --qa
```

Actions must be explicit. Calling the CLI without an action is an error;
`--offline --acquire` is forbidden. Incomplete inventories return exit code 2
with exact source keys and transfer errors in `inventory.json`. Acquisition is
partition-resumable with up to three retries per HTTP request and four workers.
Only checksum-verified bodies are atomically published; interrupted transfers
are retried as a whole monthly file. Completed sources are immutable. Rerunning
a version rechecks local checksums without silently refreshing its vintage.
Use a reviewed new data version for changed upstream inputs or processing code.

After a fresh acquisition/build, compare the source and canonical table content
sets with the published compact receipt. HTTP retrieval timestamps legitimately
change between checkouts and are excluded from these pins. Run from the repo
root (replace the data-directory argument if needed):

```bash
.venv/bin/python -B - data/final_data_v1 <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = json.loads(pathlib.Path('docs/final_data_verification.json').read_text())
inventory = json.loads((root/'inventory.json').read_text())
sources = [[r['key'], r['receipt']['sha256']] for r in inventory['files']]
sources.append(['NYFED/EFFR/2022-12_2025-12', inventory['effr']['receipt']['sha256']])
tables = [[r['path'], r['sha256']] for r in json.loads((root/'build_receipt.json').read_text())['artifacts']]
for key, rows in [('canonical_source_set_sha256', sources), ('canonical_table_set_sha256', tables)]:
    actual = hashlib.sha256(json.dumps(sorted(rows), separators=(',', ':')).encode()).hexdigest()
    assert actual == expected[key], f'{key}: revised source vintage or different processing output'
print('Published input and table content identities match')
PY
```

Official archives/rates can be revised. A different valid official checksum or
current-vintage response does not establish reproduction of this snapshot.
Preserve the verified raw snapshot for exact offline reproduction, report pin
mismatches and review a new version; do not edit expected pins to force a pass.

## Sources and conventions

[Binance Public Data](https://github.com/binance/binance-public-data) documents
the monthly kline schema, official SHA256 companions and the January 2025 Spot
transition to microseconds. Templates are:

```text
https://data.binance.vision/data/spot/monthly/klines/SYMBOL/1m/SYMBOL-1m-YYYY-MM.zip
https://data.binance.vision/data/futures/um/monthly/klines/SYMBOL/1m/SYMBOL-1m-YYYY-MM.zip
https://data.binance.vision/data/futures/um/monthly/fundingRate/SYMBOL/SYMBOL-fundingRate-YYYY-MM.zip
```

Each ZIP has an independently fetched `.CHECKSUM`. The parser inspects each
actual file, supporting s/ms/us/ns timestamps and normalizing UTC nanoseconds.
It rejects duplicate rows, mixed units, unexpected members/headers, wrong-month
observations and non-minute bar openings. Official archive funding was available
for the entire requested scope, so an API fallback/pagination path was unnecessary.

EFFR comes from the [New York Fed](https://www.newyorkfed.org/markets/reference-rates/effr)
through its date-bounded official endpoint:

```text
https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json?startDate=2022-12-01&endDate=2025-12-31
```

The current-vintage JSON response is frozen with retrieval metadata and a local
SHA256. The publisher provides no companion checksum; this is official HTTPS
provenance and local integrity verification, not publisher-signed authentication.
Keep effective dates and reported percentages. Divide by 100, assume availability
at effective-date +7 calendar days, use backward as-of with exact matches, and
retain ACT/365 simple annual-decimal treatment without a new age threshold.
Do not carry an old 2022-only history indefinitely in place of this source.
No credentials were required. Future access restrictions must be reported as
source failures, not bypassed or replaced with another instrument.

## Canonical storage contract

All generated files below live inside the explicit data root and stay out of Git.

| Path | Meaning and keys |
| --- | --- |
| `inventory_before.json`, `inventory.json` | Initial metadata inventory and exhaustive source status/receipt inventory; 444 ZIPs including 12 warmup ZIPs, plus EFFR |
| `raw/{spot,usdm_perpetual,funding}/SYMBOL/YYYY-MM/` | Official ZIP, exact checksum text, HTTP retrieval/provenance receipt, source size/SHA256 |
| `raw/effr/2022-12_2025-12.json` and receipt | Frozen official current-vintage EFFR response |
| `canonical/minute/SYMBOL/YYYY-MM.parquet` | Full decision-minute grid, keyed by **source bar-opening month**, including explicit missing/invalid/inactive rows |
| `canonical/hourly/SYMBOL/YEAR.parquet` | Exact hourly-origin samples of immediately completed minute inputs; keyed by decision year |
| `canonical/funding/SYMBOL.parquet` | Actual events, source month/SHA, interval metadata, publication/cadence flags and explicit warmup role |
| `canonical/effr.parquet` | Effective date, reported percentage, annual decimal, assumed availability and source |
| `build_receipt.json`, `qa_receipt.json` | Processing/configuration identities, output hashes, per-source audits, missing intervals, independent source reconciliation and QA summary |
| `conflicts/` | Source conflict records; conflicting originals are retained |

Minute/hourly index is unique UTC-nanosecond `decision_time_utc` with `symbol`
and `bar_open_time_utc = decision_time_utc −1min`. Per venue the tables retain
open/high/low/close, base/quote volume, trade count, taker base/quote volume,
ignore, normalized close time, observation/validity/error flags and unchanged/
large-movement flags. `pair_observed`, `pair_valid` and `pair_active` are distinct.
Validity and activity must be respected by a later authorized adapter; raw
invalid values remain visible for audit. No missing price is forward-filled.
Inactive means either venue has zero volume or zero trades. Existing 10% minute
log-price and 100bps cross-market flags are diagnostics, never exclusion rules.

Past funding fields retain last actual settlement, assumed publication time,
signed rate, past interval, age, stale/known flags. Predictor availability is
settlement +1 minute and maximum age is 12h **from settlement**, with exact
availability matches allowed. Actual event records remain separate from these
predictor joins; no funding receipts or cash ownership are calculated. All
source months are after the four assets' established listings; the final-period
quarantine flag is therefore false, not a relaxation of launch rules.

The raw Dec-2022 last bar supplies Jan-2023 midnight. Every later opening month
can similarly supply its next month's midnight. Never filter these endpoints
by opening year. Canonical decision inputs exclude `2026-01-01T00:00Z`; the last
Dec-2025 raw bar is retained in the official ZIP but contributes no in-window
decision endpoint. No 2026 market source is requested. Leap 2024 has 527,040
minute /8,784 hourly origins per asset; 2023 and 2025 each have 525,600/8,760.
These are input-clock counts, not survival-label or statistical-support counts.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -p test_final_data.py -v
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.test_portable
```

The fixture suite creates official-format ZIP/checksum/receipt inputs, including
the 2024–2025 format transition, missing/inactive bars, exact publication joins,
duplicate/conflict failures and calendar edges. Its clean-source test exports
Git HEAD, overlays only the proposed new source/config files, removes dependence
on the working checkout through `PYTHONPATH`, denies network connections, verifies
all 444 synthetic cached partitions through the CLI, then builds/stores/reconciles
a synthetic month through the same maintained interfaces. No real source is
used by portable tests. This tests code portability; the full real download is
not repeated inside that export.

Real QA independently reads all retained raw market rows with a second CSV
reader, comparing every OHLCV/activity field, close timestamp and observation
key to canonical tables. It independently checks every funding timestamp/rate/
interval, EFFR as-of joins, full calendar identity and stored hourly/minute
identity. Build and QA receipts bind source inventory and processing hashes.
Existing completed-version reruns verify these immutable artifacts and preserve
content/mtime. The compact [verification record](final_data_verification.json)
records actual results; exhaustive source/QA inventories remain ignored.

Statistical support is deliberately unmeasured. Gaps, inactivity, invalid marks,
funding cadence and asynchronous trade-price proxies may affect later labels and
model support; no future first-passage targets were built to assess that here.

The pre-existing README draft was preserved byte-for-byte. Proposed optional
entry point for review with that draft (not applied during this task):

```diff
@@ -1,1 +1,3 @@
 # Funding-aware spot–perpetual carry research
+
+Final raw-input acquisition and structural QA: [reproduction guide](docs/final_data_reproduction.md). Final model evaluation requires separate authorization.
```
