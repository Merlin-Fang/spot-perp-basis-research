# Annual checkpoint serialization and IO preflight

**ANNUAL_CHECKPOINT_IO_READY_FOR_BOUNDED_PREFLIGHT. Final holdout access and evaluation remain disabled.**

The new `training/checkpoints.py` wraps the existing immutable annual snapshot in canonical JSON. No existing scientific module, experiment, checkpoint or report is edited. It saves tree states/cuts, matched/coarse logistic coefficients/scalers, calibration maps, annual/magnitude/censor KM curves, fitting/calibration metadata and source identities without numerical conversion.

`checkpoint_provenance` binds only pre-evaluation fitting/calibration inputs, the frozen scientific documents, the fixed source-code allowlist and Python/NumPy/pandas versions. `save_checkpoint` uses atomic no-clobber publication; identical saves preserve file bytes and mtime. `load_checkpoint` requires a caller-pinned file SHA256, expected year/control and expected provenance. Unknown versions, duplicate JSON keys, nonfinite numbers, corruption, missing fields, changed definitions and mismatched identities fail explicitly. An embedded digest is not treated as independent provenance; the expected pin must come from a trusted run receipt.

`validate_contexts` enforces the four-asset UTC hourly causal-input schema, explicit missing hours, typed marks/rates/availability and original publication rules. `validate_forecasts` requires all requested asset × model × origin keys, including unavailable rows; it checks probabilities, raw/calibrated parent relationships, cutoffs, support/failure statuses and rolling-publication timing. Optional ex-post outcomes are checked separately against an explicit observed-prefix cutoff, with late-year censoring retained.

The existing output uses UTC Timestamp objects mixed with NaT for rolling publications and sometimes timezone-naive all-NaT columns for pending outcomes. Validation accepts these existing nullable representations; it rejects every non-null naive/string timestamp and never localizes an unknown clock.

Strict save/load requires the complete registered fitted state. A genuine unsupported fit/map raises `CheckpointUnsupportedError` with asset/model/status diagnostics. It does not substitute a different model or hide unavailable cells. The output validator separately accepts explicitly unavailable rows from the existing coordinator. A later orchestrator must translate a failed annual checkpoint into the registered unavailable annual cells and retain diagnostics; that orchestration is not implemented here.

The bounded demonstration uses the unchanged synthetic 2020–2022 fixture: fit 2020-01-01–2021-09-01, calibration 2021-09-01–2022-01-01 and evaluation 2022-01-01–2023-01-01, UTC half-open. All four assets and nine series are exercised under zero financing. This is not evidence about real future-year model support or performance, and it is not an EFFR annual run.

**Exact parity:** 315,360 forecast/model rows (35,040 scheduled asset-origins), maximum probability difference **0**. Original and loaded snapshots and full forecast frames agree exactly, including nulls, statuses, metadata and row ordering. Canonical scoring also agrees exactly on 24,749 paired asset-origins. No refit or recalibration occurs during loading/issuance.

Checkpoint: `results/annual_checkpoint_io_preflight/checkpoints/e5d1c690f3f74866e4149519ae98eeb4eb44317289681bf55ba3fdbcfebfd9fd/annual_2022_zero_rate.checkpoint.json` (476,981 bytes). SHA256 `8a4728ebfc2ee7150e882f7daceb479249f33ce217c000ba36bad7c6bd332f06`. Development checkpoints use source-binding version directories; prior versions remain intact. Only checkpoint state and compact summaries are retained; forecast tables stay in memory. Cross-directory serialization is byte-identical and repeated save preserves mtime. The verification record also checks a complete runner rerun.

Focused tests exercise corrupted/truncated/reformatted/oversized files; missing/conflicting/symlink targets and failed atomic publication; unsupported states; altered model/calibration/KM metadata; future/naive timestamps; incomplete scopes; parent-map mismatch; and pending/censored outcomes. See [verification](annual_checkpoint_io_verification.json) for final test counts, preservation and rerun receipts.

Reproduce from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -p test_annual_checkpoint_io.py -v
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.test_portable
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.preflight_annual_checkpoint_io --write-report
```

Remaining forecasting IO gates: connect an explicitly authorized, bounded data-source adapter; verify restored source identities/warmup at operational boundaries; define resumable rolling-terminal record/context persistence; preserve row populations through a versioned forecast-table storage adapter; and implement annual failure-to-unavailable reporting without fallback. The model checkpoint does not claim to persist live data history or a rolling estimator ledger. Current same-runtime JSON parity does not certify cross-version numerical compatibility.

Final data acquisition/QA (including future EFFR and four-asset coverage), final execution authorization and continuous financial replay/checkpoint engineering remain separate, unpassed gates. Existing hour-versus-minute and last-trade measurement limitations remain. No real 2023–2025 contents, new market downloads, financial backtests, staging, commits or pushes were used.

Recommended next bounded preflight: synthetic/pre-2023 chunked context and rolling-record resume parity, followed by forecast-table storage round-trips with exact key/status preservation. Do not begin holdout preparation or evaluation under this task.
