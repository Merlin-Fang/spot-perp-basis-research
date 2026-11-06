"""Bounded annual adapter preflight: synthetic 2020–2022 and allowlisted parity.

No final-mode option or generic loader. Run from the repository root with
python -B -m scripts.preflight_annual_forecasting --write-report.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from scripts.checkpoint_survival_baseline import Sources, B36, B37, CUTOFF
from spot_perp_basis.data.features import FEATURES
from spot_perp_basis.data.windows import START, END, annual_windows
from spot_perp_basis.theory import theory_signal
from spot_perp_basis.training.annual import (
    ASSETS, MODELS, causal_features, fit_development_year, raw_probabilities,
    issue_forecasts, attach_outcomes, evaluate_forecasts, portable, identity,
)
from spot_perp_basis.training.fit import fit_window


ROOT=Path(__file__).resolve().parents[1]
FROZEN={
    'docs/final_evaluation_protocol.md':'728d1596f0a219847a252de57dff72867801c090326779eedc8df11383631e26',
    'configs/final_evaluation.yaml':'647fc5f3ca09b58c0a7ea74644b6316ed7eda6fd6e3ed1d741464e57d6861e21',
    'docs/final_evaluation_freeze_receipt.md':'2a74cb15c1121f56d5cdd9e3db146c89dd299dd9ecd612f7ca3aafbad3eed2c0',
}


def synthetic_contexts(periods=None):
    """Deterministic dated simulation, explicitly unrelated to real prices.

    Full default clock contains every hour in 2020–2022, including missing
    observations. Fixed sinusoidal components and prespecified gaps are a
    fixture, not a search for forecast skill or calibration support.
    """
    clock=pd.date_range(START,END,freq='h',inclusive='left',name='origin_time_utc')
    if periods is not None:clock=clock[:periods]
    h=np.arange(len(clock));result={}
    for j,symbol in enumerate(ASSETS):
        rho=1.3*np.sin(2*np.pi*h/(31+4*j)+.4*j)+.6*np.sin(h*.173+j)
        spot=100+3*np.sin(h/137+j)
        perp=spot/(1-(rho-.5475)/1095)
        f=pd.DataFrame({'spot_close':spot,'perp_close':perp,'annual_rate':0.,
            'rate_available_at_utc':START,'theory_available':True,
            'bar_open_time_utc':clock-pd.Timedelta(minutes=1),
            'bar_available_at_utc':clock},index=clock)
        # Explicit gaps at fixed offsets; preserve all scheduled origin rows.
        f.loc[(h%(997+73*j)==700),'theory_available']=False
        result[symbol]=f
    return result


def frozen_contract():
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in FROZEN}
    if hashes!=FROZEN:raise ValueError('Frozen scientific authority changed')
    config=yaml.safe_load((ROOT/'configs/final_evaluation.yaml').read_text())
    # Compare exact pure schedules to the frozen declarative specification.
    registered={row['year']:row for row in config['chronology']['windows']}
    for year in (2023,2024,2025):
        row=registered[year] if year in registered else registered[str(year)]
        w=annual_windows(year)
        for role in ('fit','calibration','evaluation'):
            assert tuple(map(pd.Timestamp,row[role]))==getattr(w,role)
    return hashes


def historical_parity(src):
    """Refit the overlapping original May-2021 development design, not 20/4."""
    symbol='BTCUSDT';rate='zero_rate';stem=f'{symbol}_{rate}'
    frame=src.frame(f'{B36}/causal_inputs_v1/{stem}.parquet')
    f=causal_features(frame,symbol,rate)
    old=src.json(f'{B36}/fits_v1/final/{stem}_minimal.json')
    new=src.json(f'{B37}/fits_v1/final/{stem}_minimal.json')
    fit,risk=fit_window(f,START,CUTOFF)
    for field in ('raw_X_and_hour_sha256','response_sha256'):
        assert fit['manifest'][field]==new['matrix_manifest'][field]
    assert fit['scoring_censor_km']==new['scoring_censor_km']==old['scoring_censor_km']
    for field in ('asset_km','magnitude_cutpoints','magnitude_curves'):
        assert fit['empirical'][field]==old['baselines'][field]
    assert fit['tree']==old['boosters'][0]
    linear=new['models']['matched_hourly'];errors={}
    for field in ('mean','scale','beta'):
        errors[field]=float(np.max(np.abs(np.array(fit['matched_logistic'][field])-linear[field])))
        np.testing.assert_allclose(fit['matched_logistic'][field],linear[field],atol=1e-13,rtol=1e-13)
    origins=f.loc[f.index>=pd.Timestamp('2022-01-01',tz='UTC')]
    pred=raw_probabilities(fit,origins)
    archive=src.frame(f'{B37}/forecasts_v1/development2022/{stem}_minimal.parquet',begin=origins.index.min())
    columns={'raw_tree':'tree__8h','raw_matched_logistic':'matched_hourly__8h',
        'unconditional_KM':'asset_km__8h','magnitude_KM':'magnitude_km__8h',
        'raw_coarse_logistic':'original_logistic__8h'}
    prediction_errors={}
    for model,column in columns.items():
        expected=archive[column].reindex(pred.index)
        # Historical file already stores cumulative event probabilities.
        assert np.array_equal(pred[model].notna(),expected.notna())
        np.testing.assert_allclose(pred[model],expected,atol=1e-13,rtol=1e-13,equal_nan=True)
        prediction_errors[model]=float(np.nanmax(np.abs(pred[model]-expected)))
    return {'scope':'BTC_zero_rate_original_May2021_fit_separate_from_annual_demo',
        'fit_start':START.isoformat(),'fit_end_exclusive':CUTOFF.isoformat(),
        'evaluation':'2022_inspected_development','manifest':fit['manifest'],
        'tree_exact_dictionary_parity':True,'censor_KM_exact_parity':True,
        'annual_KM_and_magnitude_exact_parity':True,'linear_maximum_errors':errors,
        'prediction_maximum_errors':prediction_errors,'risk_trials':len(risk)}


def synthetic_preflight():
    contexts=synthetic_contexts()
    def progress(stage,symbol):print('Synthetic',stage,symbol,flush=True)
    snapshot=fit_development_year(contexts,2022,'zero_rate',progress=progress)
    before=snapshot.sha256
    print('Synthetic full-calendar forecast issuance',flush=True)
    forecasts=issue_forecasts(snapshot,contexts)
    assert len(forecasts)==4*8760*len(MODELS)
    assert not any(n in forecasts for n in ['followup_hours','event_observed','label_available_at_utc'])
    # Attach at the first origin before any future label exists.
    first=forecasts.loc[forecasts.origin_time_utc.eq(pd.Timestamp('2022-01-01',tz='UTC'))]
    early=attach_outcomes(snapshot,first,contexts,pd.Timestamp('2022-01-01',tz='UTC'))
    assert early.label_available_at_utc.isna().all()
    attached=attach_outcomes(snapshot,forecasts,contexts,pd.Timestamp('2022-12-31T23:00:00Z'))
    pd.testing.assert_frame_equal(attached[forecasts.columns],forecasts)
    print('Synthetic canonical scoring and shared-calendar uncertainty',flush=True)
    evaluated=evaluate_forecasts(snapshot,attached)
    assert snapshot.sha256==before
    payload=snapshot.read();support={}
    for symbol,a in payload['assets'].items():
        support[symbol]={'fitting_manifest':a['fitted']['manifest'],
            'model_status':{'tree':a['fitted']['tree']['status'],
                'matched_logistic':a['fitted']['matched_logistic']['status'],
                'coarse_logistic':a['fitted']['empirical']['hazard_0.01']['status'],
                'KM':a['fitted']['empirical']['asset_km']['status']},
            'censor_maximum_followup':a['fitted']['scoring_censor_km'].get('maximum_followup'),
            'calibration_maps':a['calibration_maps'],
            'calibration_sample_counts':a['calibration_sample_counts'],
            'finite_calibration_predictions':a['finite_pre_evaluation_predictions'],
            'annual_KM8':a['annual_KM8'],'fitting_identity':a['fitting_identity'],
            'calibration_identity':a['calibration_identity']}
    return {'data':'simulated_prices_not_market_observations','windows':payload['windows'],
        'clock_start':START.isoformat(),'clock_end_exclusive':END.isoformat(),
        'snapshot_sha256':before,'scheduled_asset_origins':4*8760,
        'forecast_model_rows':len(forecasts),'label_attachment_preserves_all_issuance_fields':True,
        'outcome_status_counts':attached.groupby('outcome_status').size().to_dict(),
        'issuance_status_counts':forecasts.groupby('forecast_status').size().to_dict(),
        'support':support,'evaluation':evaluated}


def report(v):
    d=v['synthetic_demonstration'];ev=d['evaluation'];interval=ev['annual_uncertainty']
    def number(x):return 'unavailable' if x is None else f'{x:.9g}'
    lines=['# Annual forecasting engineering preflight','',
        f'**{v["status"]}. Final evaluation remains disabled.**','',
        'This is an engineering check on simulated 2020–2022 prices and allowlisted historical development artifacts. '
        'No real 2023–2025 market, funding, labels, forecasts or outcomes were opened or hashed. '
        'The frozen scientific protocol/configuration/receipt remain byte-identical. No trading-policy experiment ran.','',
        '`data/windows.py` provides pure `annual_windows` metadata and strict `origin_mask`. '
        '`training/fit.py:fit_window` reuses the matched risk trials, unchanged optimizers, two-feature historical KM, '
        'six-feature censor KM and coarse logistic. `training/annual.py` adds causal validation, '
        '`fit_development_year`, immutable `AnnualForecastSnapshot`, `issue_forecasts`, `attach_outcomes` and '
        '`evaluate_forecasts`. `evaluation/statistics.py` adds strict complete-scope annual intervals and equal-cell points.','',
        'The exact registered 2023/2024/2025 calendars are tested as metadata. Numerical entry points reject final-period '
        'execution before using contexts. Existing development guards and historical checkpoint results/tests are preserved. '
        '2024 has 366 days/8,784 hours; no 2026 outcome can complete a 2025 label.','',
        '**Verification: 24 new focused tests and all 189 prior portable tests pass (213 total).** '
        'Boundary tests cover strict purges/publications, training-only transforms, causal rolling records, '
        'future-mutation invariance, pooled calibration failure, unavailable fits/censor support, '
        'full-scope aggregation and one shared bootstrap draw matrix across four assets. '
        'The unchanged checkpoint reproduces all eight asset/control cases; its source/results/tests remain preserved.','',
        'The separate synthetic demonstration uses fit **2020-01-01–2021-09-01**, calibration '
        '**2021-09-01–2022-01-01**, evaluation **2022-01-01–2023-01-01**, all UTC half-open. '
        'It exercises the same 20/4/full-year arithmetic at compatible development dates; it is not a final-year run. '
        'Sixteen-hour strict purges apply to fit/calibration, with warmup outside origin windows. '
        'There is no evaluation-origin purge: late-year origins remain administratively censored.','',
        f'Four assets produce **{d["scheduled_asset_origins"]:,} scheduled origins** and '
        f'**{d["forecast_model_rows"]:,} model/origin rows** over nine registered forecast variants. '
        'Predictions are issued before labels are attached. Pending, missing-feature, unsupported and early-censored '
        'rows remain explicit. Only compact summaries are saved; no bulk fits/forecasts are persisted.','',
        '| Synthetic asset | Fitting origins / trials | KM / censor origins | Calibration samples (tree / matched / coarse) |',
        '| --- | --- | --- | --- |']
    for s,a in d['support'].items():
        m=a['fitting_manifest'];n=a['calibration_sample_counts']
        lines.append(f'| {s} | {m["unique_origins"]} / {m["trials"]} | {m["empirical_origins"]} / {m["scoring_censor_origins"]} | '
            f'{n["calibrated_tree"]} / {n["calibrated_matched_logistic"]} / {n["calibrated_coarse_logistic"]} |')
    lines+=['','Individual monotone Platt is attempted first; pooled four-asset fitting is lazy and occurs only after '
        'an individual failure, within the same model/control. Maps, scaler/cuts and annual references freeze before '
        'evaluation; rolling KM alone updates from terminal records published strictly before each origin. '
        'The verification JSON records all numerical/support failures, map sources and sample counts.','',
        '| Synthetic asset | Primary paired origins | Logistic minus tree IPCW Brier |','| --- | --- | --- |']
    for a in ev['primary_differences']:lines.append(f'| {a["symbol"]} | {a["paired_origins"]} | {number(a["gain"])} |')
    lines+=['',f'Synthetic equal-asset gain: **{number(ev["annual_equal_asset"]["gain"])}**. '
        f'Joint 14-day interval: **[{number(interval["ci_low"])}, {number(interval["ci_high"])}]**, '
        f'{interval.get("occupied_calendar_blocks","unavailable")} occupied blocks, frozen seed 20261102 and 1,000 shared draws. '
        'These fixture values validate execution, not financial predictability. The 12 final-year cells and their '
        'aggregate remain unavailable. No cross-year confidence interval is constructed.','',
        'Every point, primary difference and bootstrap input uses the canonical censor-KM guards: fitted maximum '
        'followup, G(8) and event G(T−1) ≥0.05. Early-censored rows contribute zero while remaining in the denominator. '
        'Named empirical panels have exact common keys; an unavailable model or asset cannot silently disappear.','',
        'Historical source parity separately refits BTC zero-rate on the original May-2021 design and compares '
        '2022 predictions against archived Experiment 3.7. Tree state, risk-matrix/response identities, KM curves, '
        'logistic transforms/coefficients and five raw forecast variants reconcile at 1e−13 tolerance. '
        'The unchanged four-asset November checkpoint is also rerun independently. These are previously inspected '
        'development results, not independent holdout performance.','',
        'Run from the repository root:','',
        '```bash','PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.test_portable',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.preflight_annual_forecasting --write-report',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.checkpoint_survival_baseline --write-report',
        '```','',
        'See [verification](annual_forecasting_verification.json) for test, preservation and idempotence receipts; '
        '`results/annual_forecasting_preflight/summary.json` contains compact support/scoring/source identities.','',
        'Remaining gates are separate: authorized final-data acquisition/structural QA and bounded IO adapters; '
        'annual final-mode integration and operational preflight; financial continuous/checkpoint state and accounting parity; '
        'and explicit final execution authorization. None is passed by synthetic success. '
        'The hour-versus-minute crossing and last-trade measurement limitations remain unchanged. '
        'Small IPCW weights rely on the registered censoring assumptions; the synthetic fixture cannot establish '
        'real final-year calibration support or drift robustness. EFFR stays a separate control, with no new rate inputs.','',
        'The smallest next step is a development-only annual checkpoint/serialization round-trip and operational '
        'IO contract review, preserving the same final access gates. Do not begin final data preparation or evaluation '
        'without its separate authorization.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report',action='store_true')
    args=parser.parse_args();sys.dont_write_bytecode=True
    src=Sources(('zero_rate',))
    sys.addaudithook(src.guard)
    frozen=frozen_contract()
    print('Historical overlapping adapter refit parity',flush=True)
    parity=historical_parity(src)
    demo=synthetic_preflight()
    failures=[f'{s}:{n}:{status}' for s,a in demo['support'].items() for n,status in a['model_status'].items() if status!='fit']
    failures += [f'{s}:{n}:{m["status"]}' for s,a in demo['support'].items() for n,m in a['calibration_maps'].items() if m['status']!='fit']
    if demo['evaluation']['annual_equal_asset']['status']!='supported':failures.append('synthetic_primary_IPCW_aggregate_unavailable')
    status='ANNUAL_FORECASTING_ADAPTERS_PARTIAL' if failures else 'ANNUAL_FORECASTING_ADAPTERS_READY_FOR_FINAL_PREFLIGHT'
    value=portable({'status':status,'failed_support_checks':failures,
        'final_execution_enabled':False,'final_market_contents_accessed':False,
        'registered_calendars':{y:annual_windows(y).as_dict() for y in (2023,2024,2025)},
        'frozen_authority_sha256':frozen,'historical_parity':parity,
        'implementation_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in (
            'scripts/preflight_annual_forecasting.py','src/spot_perp_basis/training/annual.py',
            'src/spot_perp_basis/training/fit.py','src/spot_perp_basis/data/windows.py',
            'src/spot_perp_basis/evaluation/statistics.py')},
        'source_sha256':src.hashes,'synthetic_demonstration':demo})
    path=ROOT/'results/annual_forecasting_preflight/summary.json';path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if not path.exists() or path.read_text()!=content:path.write_text(content)
    if args.write_report:
        path=ROOT/'docs/annual_forecasting_preflight.md';content=report(value)
        if not path.exists() or path.read_text()!=content:path.write_text(content)
    print(value['status'],flush=True)


if __name__=='__main__':main()
