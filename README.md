# Spot–Perpetual Basis Research

Research code for a time-series machine learning study of **cryptocurrency spot–perpetual basis convergence**, with a focus on causal survival forecasting and the economic value of forecast-informed trading decisions.

**Course:** ESE 4380 — Time Series Machine Learning (Fall 2026)  
**Project status:** November 2026; baseline and forecasting research in progress.  
**Scope:** BTCUSDT, ETHUSDT, DOGEUSDT, ADAUSDT; hourly forecast origins; 8-hour primary convergence horizon.

> This is an academic research project, not an executable trading service or an investment recommendation. Prices and fills in the research are historical/proxy measurements rather than simultaneous live executable quotes.

## Research question

Does a small nonlinear discrete-time hazard model predict **eight-hour theoretical spot–perpetual basis convergence** more accurately than a fairly matched linear model and historical survival baselines? A separate economic study asks whether any forecasting skill improves a fixed spot–perpetual strategy's **net** trading outcome and risk relative to simpler participation rules.

At each hourly decision time, features use only completed, causally available observations. An origin-fixed position direction defines the target: whether the theoretical basis discrepancy first reaches its directional zero-crossing boundary within the next eight scheduled hourly observations. Unobserved futures are **censored**, not labeled as negatives.

**Primary metric:** eight-hour inverse-probability-of-censoring-weighted (IPCW) Brier score; lower is better.

**Models:** raw minimal histogram-boosted survival Tree; feature- and risk-set-matched hourly Logistic hazard; unconditional/magnitude Kaplan–Meier references. Calibration and other registered controls are secondary.

**Financing contexts:** zero rate (primary) and NY Fed EFFR with a seven-calendar-day assumed release lag (secondary).

## November 9 baseline checkpoint (2022 development data)

The course baseline checkpoint and its independently checked numerical evidence are at:

- [Baseline checkpoint report](docs/checkpoints/2026-11-09_baseline.md)
- [Verification receipt](docs/checkpoints/2026-11-09_verification.json)
- [Baseline reproduction script](scripts/checkpoint_survival_baseline.py)
- [Baseline-specific tests](tests/test_survival_checkpoint.py)

The following are actual **2022 development** results, not fresh final-test findings. All models in each row are evaluated on the same scored forecast origins, and all errors are IPCW Brier scores.

| Asset | Scored origins | Unconditional KM | Matched Logistic | Minimal Tree | Logistic − Tree |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 210 | 0.263538 | 0.248652 | 0.263239 | −0.014588 |
| ETHUSDT | 320 | 0.253381 | 0.307592 | 0.286623 | +0.020969 |
| DOGEUSDT | 1,912 | 0.347116 | 0.206268 | 0.154677 | +0.051591 |
| ADAUSDT | 2,458 | 0.446123 | 0.243615 | 0.152103 | +0.091512 |

*Zero-rate primary context. A positive difference favors the Tree. BTC and ETH alone show mixed evidence; the larger four-asset development mean reflects DOGE/ADA and was already examined during model development. No independent confirmation or economic-profit claim follows from this table.*

Historical model-selection window: **Jan–Apr 2021**; refit: **Jan 2020–Apr 2021**, subject to launch-aware availability; 2022 development evaluation. A separate pre-registered annual prequential design governs 2023–2025. The repository's earlier checkpoint report was prepared before the 2023–2025 forecast results were inspected; those results have since been opened in November 2026 and must not be described as a still-sealed test set in new submissions.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/spot_perp_basis/data/` | Market input alignment, causal features, censoring labels and annual windows |
| `src/spot_perp_basis/models/` | Kaplan–Meier and survival Tree/Logistic estimators |
| `src/spot_perp_basis/training/` | Fitting, annual snapshots and deterministic checkpoints |
| `src/spot_perp_basis/evaluation/` | IPCW Brier scoring, paired comparisons and audits |
| `src/spot_perp_basis/backtest/` | Research trading/accounting components (economic results require independent validation) |
| `scripts/` | Small documented experiment, verification and data preparation entry points |
| `tests/` | Unit, synthetic and portable regression checks |
| `docs/` | Scientific protocol, data provenance, baseline and verification reports |
| `configs/` | Frozen experiment/data specifications |

The research protocol is [`final-survival-evaluation-v1.0.0`](docs/final_evaluation_protocol.md), with its [frozen configuration](configs/final_evaluation.yaml). The checkpoint's historical May-2021 refit is **not** the later annual final-test schedule.

## Environment

Python **3.12 or later** is required by `pyproject.toml`; the published data-processing environment was tested with Python **3.14.7**. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-build-lock.txt
.venv/bin/python -m pip install --require-hashes -r requirements-lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

Run the portable regression checks:

```bash
.venv/bin/python -B -m scripts.test_portable
```

The portable runner intentionally omits three documented archival-workspace-preservation integration tests. See the checkpoint verification receipt for the exact test status at the time of that report.

### Reproducing the November 9 historical checkpoint

**Input requirement:** this script reuses the verified **pre-2023 historical experiment artifacts** (archived May-2021 fits, causal input Parquet, historical risk sets/labels and pre-2023 minute diagnostics) under the relative paths validated by its explicit source allowlist. Those bulky inputs live in local `experiments/` and `data/` directories and are **not included in this Git repository**.

With the verified historical bundle restored to the expected paths:

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_survival_checkpoint.py' -v
.venv/bin/python -B -m scripts.checkpoint_survival_baseline --write-report
```

Compact results are written to the ignored `results/checkpoints/2026-11-09/` directory. The command **recomputes predictions/scores from archived historical fits** and does not train a new model or access final-period outcomes. The checkpoint report and verification JSON publish the resulting measurements independently of access to the bulky inputs.

**Reproducibility limitation:** a clean clone can install the software and run portable tests, but cannot reproduce this **historical numeric checkpoint command** without the separately preserved historical bundle. The older research artifact archive is not yet packaged for standalone public distribution; do not interpret the command above as a clean-clone, one-command reproduction claim.

## Data sources and data access

- [Binance Public Data](https://github.com/binance/binance-public-data): official historical Spot and USDT-M perpetual one-minute OHLCV plus separate actual funding events, with checksum companions.
- [Federal Reserve Bank of New York — EFFR](https://www.newyorkfed.org/markets/reference-rates/effr): reference financing-rate sensitivity; the original underlying borrowing contract/rate is not available.

The committed [final-period data reproduction guide](docs/final_data_reproduction.md) documents **2023–2025 source acquisition, checksum verification, UTC timestamp normalization, minute/hourly canonical tables and known data-quality limitations**. For that separately implemented pipeline:

```bash
.venv/bin/python -B -m scripts.prepare_final_data \
  --data-root data/final_data_v1 --acquire --verify --build --qa
```

This prepares **final-period input data**, **not** the older 2020–2022 checkpoint experiment bundle. Output data, historical experimental artifacts and bulk model forecasts remain excluded from Git. Data access is subject to original providers' terms.

## Data integrity and scientific limitations

- Raw one-minute bars become available **after** their interval closes; hourly origins use the last completed bar. No future or partial bar is an input.
- Funding predictor data are treated as published at settlement +1 minute and retained only while sufficiently recent; EFFR has an explicit **assumed** seven-day lag.
- Times are UTC; missing, invalid and inactive observations are distinct. No interpolation or fabricated prices repair source gaps.
- Risk labels are direction-fixed first crossings; unknown outcomes are censored. Scoring uses a fitting-only reverse censoring KM and support checks.
- Repeated/overlapping hourly outcomes, financing copies and correlated assets are **not independent replications**.
- Theory relies on conditional paper parameters and last-trade price proxies. Neither prediction quality nor proxy financial comparisons establish live executable profit or verified margin safety.

## Project milestones

The **November 9 submission** uses the verified 2022 development checkpoint above. Final 2023–2025 forecasting has subsequently been executed locally under a separately authorized frozen protocol; its newer code and numerical report **may not yet be published in this branch**. Do not treat a missing Git report as evidence that it never ran, and do not infer economic outcomes from forecast scores.

**Remaining work:** (1) verify continuous trading state/carry-in and cash accounting, (2) execute separately authorized 2023–2025 economic replay, (3) compare Tree against He, matched Logistic and simpler participation controls, (4) write the final paper, and (5) complete historical-data bundle packaging and end-to-end repository reproduction.

See the scientific protocol and verified checkpoint documents for exact definitions. No credentials or raw restricted data are required to read the source.
