# Forecasting milestone publication scope

This local publication packages the verified `FORECASTING_OPERATIONAL_PREFLIGHT_VERIFIED` development milestone. It does not authorize holdout access or execution. The scientific baseline is protocol `final-survival-evaluation-v1.0.0`; the pre-publication Git HEAD was `702a95d8d11b00b24194f68e208df720f35715cc`.

The index initially contained only `configs/final_evaluation.yaml`, already part of the protocol group. No unrelated staged content was present. Explicit file lists are used throughout; existing work is not reset or discarded.

The current milestone passed 262 portable tests (243 preserved plus 19 operational checks) and exact uninterrupted/resumed/storage/scoring parity for 315,360 synthetic forecast/model rows. Before local commits, the portable suite is repeated in the working tree and a clean source-only export. Generated datasets, forecasts, checkpoint blobs, runtime outputs, caches and experimental archives are excluded.

The six older regression modules, their three static configurations, collateral helper and four imported script dependencies are included in group B because omitting them would make the full 262-test suite unavailable in a clean checkout. Those scripts are retained source; no acquisition, modeling or financial research driver is executed by this publication task.

Older phase documentation, unrelated drivers/configurations, generated manifests and the pre-existing README draft remain untouched outside this milestone. Some legacy drivers have unresolved migration/publication dependencies; they are not certified by the current forecasting preflight. Historical archive-dependent reproductions still require separately restored ignored inputs. The synthetic preflights and portable tests are available from committed source.

The frozen protocol/YAML/freeze receipt remain byte-identical to their verified versions. Older verification receipts remain historical evidence: the current operational receipt documents the necessary model-specific `scoring_status` consistency correction and its source hash change. Receipt hashes are not rewritten to conceal implementation history.

## A — docs: freeze final survival evaluation protocol

- `docs/final_evaluation_protocol.md`
- `configs/final_evaluation.yaml`
- `docs/final_evaluation_freeze_receipt.md`

## B — test: verify four-asset survival forecasting baseline

- `scripts/checkpoint_survival_baseline.py`
- `src/spot_perp_basis/evaluation/survival_audit.py`
- `tests/test_survival_checkpoint.py`
- `docs/checkpoints/2026-11-09_baseline.md`
- `docs/checkpoints/2026-11-09_verification.json`
- `tests/test_carry_pilot.py`
- `tests/test_collateral_scenario.py`
- `tests/test_economics.py`
- `tests/test_pilot_acquisition.py`
- `tests/test_reuse_integrity.py`
- `tests/test_target_audit.py`
- `src/spot_perp_basis/backtest/collateral.py`
- `scripts/download_pilot.py`
- `scripts/reuse_spot_archives.py`
- `scripts/run_carry_pilot.py`
- `scripts/run_target_audit.py`
- `configs/carry_pilot.yaml`
- `configs/collateral_scenario.yaml`
- `configs/target_audit.yaml`

## C — feat: implement annual rolling forecasting pipeline

- `src/spot_perp_basis/data/windows.py`
- `src/spot_perp_basis/evaluation/statistics.py`
- `src/spot_perp_basis/training/fit.py`
- `src/spot_perp_basis/training/annual.py`
- `scripts/preflight_annual_forecasting.py`
- `tests/test_annual_forecasting.py`
- `docs/annual_forecasting_preflight.md`
- `docs/annual_forecasting_verification.json`

## D — feat: add verified forecasting checkpoints and resume

- `src/spot_perp_basis/training/checkpoints.py`
- `src/spot_perp_basis/training/operational.py`
- `scripts/preflight_annual_checkpoint_io.py`
- `scripts/preflight_forecasting_operations.py`
- `tests/test_annual_checkpoint_io.py`
- `tests/test_annual_operational.py`
- `docs/annual_checkpoint_io_preflight.md`
- `docs/annual_checkpoint_io_verification.json`
- `docs/forecasting_operational_preflight.md`
- `docs/forecasting_operational_verification.json`
- `docs/forecasting_milestone_publication.md`

## Remaining pre-existing work outside these commits

The following inventory was captured before staging. These paths are not discarded, reset or silently added.

- ` M` `README.md`
- `??` `SPOT_PERP_BASIS_RESEARCH_PLAN.md`
- `??` `configs/carry_experiment.yaml`
- `??` `configs/data.yaml`
- `??` `configs/economic_feasibility.yaml`
- `??` `configs/experiment.yaml`
- `??` `configs/pilot.yaml`
- `??` `docs/class_project_requirements.md`
- `??` `docs/decisions.md`
- `??` `docs/feasibility.md`
- `??` `docs/history/README_through_phase2_5.md`
- `??` `docs/history/README_through_phase2_6.md`
- `??` `docs/history/README_through_phase2_7.md`
- `??` `docs/history/decisions_through_phase2_5.md`
- `??` `docs/history/decisions_through_phase2_6.md`
- `??` `docs/history/decisions_through_phase2_7.md`
- `??` `docs/history/phase2_7_5_config_initial.yaml`
- `??` `docs/history/phase2_7_5_protocol_initial.md`
- `??` `docs/history/research_plan_v1.0.md`
- `??` `docs/phase0_readiness.md`
- `??` `docs/phase2_5_economic_feasibility.md`
- `??` `docs/phase2_5_literature_audit.md`
- `??` `docs/phase2_6_carry_predictability.md`
- `??` `docs/phase2_6_protocol.md`
- `??` `docs/phase2_7_5_collateral_feasibility.md`
- `??` `docs/phase2_7_5_decision.md`
- `??` `docs/phase2_7_5_protocol.md`
- `??` `docs/phase2_7_dataset_target_audit.md`
- `??` `docs/phase2_7_modeling_recommendations.md`
- `??` `docs/phase2_7_protocol.md`
- `??` `docs/phase2_data_dictionary.md`
- `??` `docs/phase2_data_report.md`
- `??` `docs/pilot_data_audit.md`
- `??` `docs/research_contract.md`
- `??` `docs/research_plan.md`
- `??` `docs/research_plan_v2.0.md`
- `??` `docs/reuse_audit.md`
- `??` `manifests/code_reuse.json`
- `??` `manifests/environment.json`
- `??` `manifests/five_asset_v1_acquisition.json`
- `??` `manifests/five_asset_v1_dataset.json`
- `??` `manifests/legacy_snapshot.json`
- `??` `manifests/phase0_checks.json`
- `??` `manifests/phase1_environment.json`
- `??` `manifests/phase1_review.json`
- `??` `manifests/phase1_summary.json`
- `??` `manifests/phase2_5_completion.json`
- `??` `manifests/phase2_5_measurement_audit.json`
- `??` `manifests/phase2_5_protocol.json`
- `??` `manifests/phase2_5_report.json`
- `??` `manifests/phase2_5_run.json`
- `??` `manifests/phase2_5_verification.json`
- `??` `manifests/phase2_6_analysis.json`
- `??` `manifests/phase2_6_completion.json`
- `??` `manifests/phase2_6_protocol.json`
- `??` `manifests/phase2_6_report.json`
- `??` `manifests/phase2_6_run.json`
- `??` `manifests/phase2_6_verification.json`
- `??` `manifests/phase2_7_5_availability.json`
- `??` `manifests/phase2_7_5_history.json`
- `??` `manifests/phase2_7_5_inputs.json`
- `??` `manifests/phase2_7_5_protocol.json`
- `??` `manifests/phase2_7_5_protocol_initial.json`
- `??` `manifests/phase2_7_5_run.json`
- `??` `manifests/phase2_7_5_verification.json`
- `??` `manifests/phase2_7_availability.json`
- `??` `manifests/phase2_7_history.json`
- `??` `manifests/phase2_7_labels.json`
- `??` `manifests/phase2_7_protocol.json`
- `??` `manifests/phase2_7_supplementary.json`
- `??` `manifests/phase2_7_verification.json`
- `??` `manifests/phase2_acquisition.json`
- `??` `manifests/phase2_completion.json`
- `??` `manifests/phase2_dataset.json`
- `??` `manifests/phase2_environment.json`
- `??` `manifests/phase2_sources.json`
- `??` `manifests/phase2_training_summary.json`
- `??` `manifests/phase2_verification.json`
- `??` `manifests/pilot_acquisition.json`
- `??` `manifests/plan_source.json`
- `??` `manifests/research_plan_v2_amendment.json`
- `??` `manifests/research_plan_v2_completion.json`
- `??` `manifests/reused_spot_archives.json`
- `??` `scripts/analyze_carry_pilot.py`
- `??` `scripts/analyze_collateral_scenario.py`
- `??` `scripts/analyze_target_audit.py`
- `??` `scripts/audit_economic_measurements.py`
- `??` `scripts/build_dataset.py`
- `??` `scripts/build_five_asset_dataset.py`
- `??` `scripts/check_five_asset_dataset.py`
- `??` `scripts/check_preparation.py`
- `??` `scripts/check_target_reference_data.py`
- `??` `scripts/describe_training_data.py`
- `??` `scripts/download_dataset.py`
- `??` `scripts/download_five_asset_dataset.py`
- `??` `scripts/fetch_he_replication_inputs.py`
- `??` `scripts/finalize_carry_pilot.py`
- `??` `scripts/finalize_economic_feasibility.py`
- `??` `scripts/prepare_collateral_scenario.py`
- `??` `scripts/report_five_asset_dataset.py`
- `??` `scripts/report_he_replication.py`
- `??` `scripts/run_collateral_scenario.py`
- `??` `scripts/run_economic_feasibility.py`
- `??` `scripts/run_pilot.py`
- `??` `scripts/verify_carry_pilot.py`
- `??` `scripts/verify_collateral_scenario.py`
- `??` `scripts/verify_dataset.py`
- `??` `scripts/verify_economic_feasibility.py`
- `??` `scripts/verify_five_asset_dataset.py`
- `??` `scripts/verify_he_replication.py`
- `??` `scripts/verify_target_audit.py`
- `??` `scripts/write_carry_pilot_report.py`
- `??` `scripts/write_data_report.py`
- `??` `scripts/write_economic_report.py`

## Verification and reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.test_portable
```

The portable runner retains and explicitly omits its three original archive-hash integrations. It denies reads of research data/results/experiment contents. All included executable code uses repository-relative paths; generic `/tmp` negative-test paths and content hashes in receipts are intentional. No credentials, private keys or machine-specific source/configuration roots belong in this publication.

Commit IDs are reported by Git after creation and are intentionally not self-referenced in this file. Exact commit membership can be checked with `git show --format=fuller --stat COMMIT` and `git diff-tree --no-commit-id --name-only -r COMMIT`. No push is part of this task.
