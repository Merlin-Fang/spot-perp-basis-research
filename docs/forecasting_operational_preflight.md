# Forecasting operational integration preflight

**FORECASTING_OPERATIONAL_PREFLIGHT_VERIFIED — engineering-complete for the bounded development forecasting path. Final-data access and evaluation remain disabled.**

`training/operational.py` reuses the existing annual forecaster, first-passage/KM functions, checkpoint validator and canonical scoring. It adds an observed-prefix cursor, immutable Parquet tables with versioned JSON manifests, and a durable progress record linking the cursor to every already-committed output chunk. Model/calibration/source/runtime identities and exact origins are pinned.

Full hourly causal context is retained and replayed at chunk boundaries. This preserves 90-day terminal-record history, ongoing paths and rolling-feature numerical history without implementing another label/KM engine. A temporary administrative terminal record at the current prefix end has publication equal to its last hour and cannot enter a prediction at or before that hour under the registered strict-past rule. Later chunks reconstruct it with newly observed inputs. Only supplied prefixes enter the chunk API; no future file or future outcome screen selects prediction rows.

Each table manifest binds its file SHA256, logical frame/dtypes/index identity, source/runtime, snapshot, year/control, model registry, requested interval and optional label-as-of cutoff. Load validates every asset/model/origin key and restores original UTC object/NaT representations and hourly-index frequency metadata. Missing/duplicate keys, stale/tampered history, wrong provenance and conflicts fail explicitly. Files and manifests publish atomically without clobbering; the progress manifest commits only after referenced output chunks and context are durable. Orphan blobs after an interrupted write are harmless; retry uses the same hashes and duplicate validation.

One necessary fix in `training/checkpoints.py` excludes `scoring_status` from the cross-model equality check on shared label copies. Scoring availability depends on each model, and its existing per-row check remains in force. No feature, label, model, calibration or scoring calculation changes.

The end-to-end fixture uses original synthetic 2020–2022 inputs: fitting 2020-01-01–2021-09-01, calibration 2021-09-01–2022-01-01, and the full 2022 evaluation year, UTC half-open. Its fitted snapshot exactly matches the prior verified fixture. A supported model checkpoint is saved/reloaded, followed by an uninterrupted reference and twelve monthly forecast chunks. At July 1 the in-memory cursor and completed-output list are discarded and restored from the persisted progress pin. Focused tests separately exercise a fresh Python-process resume.

**Exact parity:** 315,360 forecast rows covering 35,040 asset-origins, four assets and all nine series. Uninterrupted, resumed and saved/reloaded forecasts agree exactly, including probabilities, timestamps, statuses, row ordering and metadata. The separately attached annual outcome table also round-trips exactly. Canonical point scores, paired keys/losses, equal-asset means and shared-calendar uncertainty agree exactly on 24,749 primary scoring origins. These are engineering checks on synthetic data, not independent forecasting-performance evidence.

Unsupported annual dispositions are explicit for all 36 asset/model cells. A separate named 72-hour optimizer-failure injection makes ADA raw matched/calibrated matched and DOGE calibrated tree unavailable: all 2,592 requested rows remain, affected probabilities are null, diagnostics survive resume and table reload, and the complete four-asset mean is unavailable. The registered causal rolling reference remains dynamic; a frozen-fallback failure does not invent a fitted rolling curve.

The runner records reproducible pins in `results/forecasting_operational_preflight/summary.json`. Peak RSS and elapsed time are measured separately in its RESOURCE log and the [verification record](forecasting_operational_verification.json), keeping canonical outputs deterministic. Full history is practical for this fixed hourly four-asset study; repeated prefix reconstruction adds CPU and yearly scoring materializes the full forecast table. This is bounded batch processing, not a constant-memory streaming service. The harness holds reference, resumed and labeled tables simultaneously, so its peak RSS exceeds that of context retention alone. Context retention avoids unsafe pruning of long ongoing paths. Nanosecond observation clocks are tested against the unchanged annual decision-clock representation.

Reproduce from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -p test_annual_operational.py -v
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.test_portable
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.preflight_forecasting_operations --write-report
```

**A. Forecasting engineering:** complete for the requested development-only operational path, subject to the verification receipt. No general-purpose framework, separate incremental KM engine, database, distributed queue, live service, or mid-optimizer resume is required for the registered annual batch protocol. Those features are omitted. The existing pre-2023 guards remain intact; this does not certify final-data coverage, real-year support or final-mode execution.

**B. Access/authorization tasks:** obtain explicit authorization for final-data acquisition/access and structural QA; verify four-asset spot/perpetual/funding history and EFFR reference coverage, availability clocks, source hashes, warmup and boundary handoffs. Any final-period IO activation must be narrowly authorized and checked against the frozen windows, including leap 2024; the current development guards do not automatically permit it. Final labels/performance and execution require their separately authorized stage. Financial replay remains separate from this forecasting-only path. No real 2023–2025 contents were opened, hashed or evaluated.

**C. Shortest safe path:** pin this verified code/artifacts; authorize bounded acquisition/structural QA; connect only the authorized frozen data partitions and validate IO/calendar/source boundaries without selecting on forecast outcomes; then obtain final-execution authorization and run the registered annual forecasts/storage/scoring with the existing unavailable-cell rules. Do not insert a new model search or build another forecasting framework. No such next task ran here.
