# Annual forecasting engineering preflight

**ANNUAL_FORECASTING_ADAPTERS_READY_FOR_FINAL_PREFLIGHT. Final evaluation remains disabled.**

This is an engineering check on simulated 2020–2022 prices and allowlisted historical development artifacts. No real 2023–2025 market, funding, labels, forecasts or outcomes were opened or hashed. The frozen scientific protocol/configuration/receipt remain byte-identical. No trading-policy experiment ran.

`data/windows.py` provides pure `annual_windows` metadata and strict `origin_mask`. `training/fit.py:fit_window` reuses the matched risk trials, unchanged optimizers, two-feature historical KM, six-feature censor KM and coarse logistic. `training/annual.py` adds causal validation, `fit_development_year`, immutable `AnnualForecastSnapshot`, `issue_forecasts`, `attach_outcomes` and `evaluate_forecasts`. `evaluation/statistics.py` adds strict complete-scope annual intervals and equal-cell points.

The exact registered 2023/2024/2025 calendars are tested as metadata. Numerical entry points reject final-period execution before using contexts. Existing development guards and historical checkpoint results/tests are preserved. 2024 has 366 days/8,784 hours; no 2026 outcome can complete a 2025 label.

**Verification: 24 new focused tests and all 189 prior portable tests pass (213 total).** Boundary tests cover strict purges/publications, training-only transforms, causal rolling records, future-mutation invariance, pooled calibration failure, unavailable fits/censor support, full-scope aggregation and one shared bootstrap draw matrix across four assets. The unchanged checkpoint reproduces all eight asset/control cases; its source/results/tests remain preserved.

The separate synthetic demonstration uses fit **2020-01-01–2021-09-01**, calibration **2021-09-01–2022-01-01**, evaluation **2022-01-01–2023-01-01**, all UTC half-open. It exercises the same 20/4/full-year arithmetic at compatible development dates; it is not a final-year run. Sixteen-hour strict purges apply to fit/calibration, with warmup outside origin windows. There is no evaluation-origin purge: late-year origins remain administratively censored.

Four assets produce **35,040 scheduled origins** and **315,360 model/origin rows** over nine registered forecast variants. Predictions are issued before labels are attached. Pending, missing-feature, unsupported and early-censored rows remain explicit. Only compact summaries are saved; no bulk fits/forecasts are persisted.

| Synthetic asset | Fitting origins / trials | KM / censor origins | Calibration samples (tree / matched / coarse) |
| --- | --- | --- | --- |
| ADAUSDT | 14553 / 150970 | 14588 / 14565 | 2065 / 2065 / 2065 |
| BTCUSDT | 14547 / 119738 | 14586 / 14561 | 2033 / 2033 / 2033 |
| DOGEUSDT | 14550 / 142536 | 14587 / 14563 | 2058 / 2058 / 2058 |
| ETHUSDT | 14550 / 132320 | 14587 / 14563 | 2060 / 2060 / 2060 |

Individual monotone Platt is attempted first; pooled four-asset fitting is lazy and occurs only after an individual failure, within the same model/control. Maps, scaler/cuts and annual references freeze before evaluation; rolling KM alone updates from terminal records published strictly before each origin. The verification JSON records all numerical/support failures, map sources and sample counts.

| Synthetic asset | Primary paired origins | Logistic minus tree IPCW Brier |
| --- | --- | --- |
| BTCUSDT | 6196 | 0.181628408 |
| ETHUSDT | 6187 | 0.198366385 |
| DOGEUSDT | 6176 | 0.190108803 |
| ADAUSDT | 6190 | 0.188474827 |

Synthetic equal-asset gain: **0.189644606**. Joint 14-day interval: **[0.186036539, 0.192904235]**, 27 occupied blocks, frozen seed 20261102 and 1,000 shared draws. These fixture values validate execution, not financial predictability. The 12 final-year cells and their aggregate remain unavailable. No cross-year confidence interval is constructed.

Every point, primary difference and bootstrap input uses the canonical censor-KM guards: fitted maximum followup, G(8) and event G(T−1) ≥0.05. Early-censored rows contribute zero while remaining in the denominator. Named empirical panels have exact common keys; an unavailable model or asset cannot silently disappear.

Historical source parity separately refits BTC zero-rate on the original May-2021 design and compares 2022 predictions against archived Experiment 3.7. Tree state, risk-matrix/response identities, KM curves, logistic transforms/coefficients and five raw forecast variants reconcile at 1e−13 tolerance. The unchanged four-asset November checkpoint is also rerun independently. These are previously inspected development results, not independent holdout performance.

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.test_portable
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.preflight_annual_forecasting --write-report
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.checkpoint_survival_baseline --write-report
```

See [verification](annual_forecasting_verification.json) for test, preservation and idempotence receipts; `results/annual_forecasting_preflight/summary.json` contains compact support/scoring/source identities.

Remaining gates are separate: authorized final-data acquisition/structural QA and bounded IO adapters; annual final-mode integration and operational preflight; financial continuous/checkpoint state and accounting parity; and explicit final execution authorization. None is passed by synthetic success. The hour-versus-minute crossing and last-trade measurement limitations remain unchanged. Small IPCW weights rely on the registered censoring assumptions; the synthetic fixture cannot establish real final-year calibration support or drift robustness. EFFR stays a separate control, with no new rate inputs.

The smallest next step is a development-only annual checkpoint/serialization round-trip and operational IO contract review, preserving the same final access gates. Do not begin final data preparation or evaluation without its separate authorization.
