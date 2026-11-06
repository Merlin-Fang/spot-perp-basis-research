"""Fixed synthetic 2022 operational preflight; never a final-data runner."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import pandas as pd

from scripts.preflight_annual_forecasting import synthetic_contexts
from spot_perp_basis.training.annual import (
    ASSETS, AnnualForecastSnapshot, fit_development_year, issue_forecasts,
    attach_outcomes, identity, portable,
)
from spot_perp_basis.training.checkpoints import checkpoint_provenance, save_checkpoint, load_checkpoint
from spot_perp_basis.training.operational import (
    operational_binding, start_forecasting, forecast_chunk, save_forecasts, load_forecasts,
    save_progress, load_progress, save_resume, load_resume, assemble_forecasts,
    annual_availability, score_forecasts, _frame_identity,
)


ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/forecasting_operational_preflight'
REPORT=ROOT/'docs/forecasting_operational_preflight.md'
BEGIN=pd.Timestamp('2022-01-01',tz='UTC');END=pd.Timestamp('2023-01-01',tz='UTC')


def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):raise PermissionError('Operational preflight is offline')
    if event!='open' or not isinstance(args[0],(str,bytes)):return
    p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
    if any(p.is_relative_to(ROOT/n) for n in ('data','experiments','outputs','manifests')):
        raise PermissionError('Synthetic-only preflight cannot open research contents: '+str(p))
    if p.is_relative_to(ROOT/'results') and not p.is_relative_to(OUTPUT):
        raise PermissionError('Cannot open prior research results: '+str(p))


def selected(c,begin,end):return {s:f.loc[(f.index>=begin)&(f.index<end)] for s,f in c.items()}


def fault_snapshot(snapshot):
    """Explicit optimizer-failure injection, never a fitted research result."""
    p=snapshot.read()
    p['assets']['ADAUSDT']['fitted']['matched_logistic']['status']='unsupported_numerical_nonconvergence'
    p['assets']['DOGEUSDT']['calibration_maps']['calibrated_tree']['status']='unsupported_calibration_numerical'
    for a in p['assets'].values():
        a['fitting_identity']=identity(a['fitted']);a['calibration_identity']=identity(a['calibration_maps'])
    return AnnualForecastSnapshot(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False))


def run():
    started=time.perf_counter();c=synthetic_contexts();binding=operational_binding(ROOT)
    directory=OUTPUT/identity(binding);store=directory/'store'
    snapshot=fit_development_year(c,2022,'zero_rate',progress=lambda stage,s:print(stage,s,flush=True))
    prior=json.loads((ROOT/'docs/annual_forecasting_verification.json').read_text())
    assert snapshot.sha256==prior['synthetic_demonstration']['snapshot_sha256']
    provenance=checkpoint_provenance(snapshot,c,source_kind='synthetic',binding=binding['model_binding'])
    checkpoint=directory/'annual_2022_zero_rate.checkpoint.json'
    checkpoint_sha=save_checkpoint(snapshot,checkpoint,provenance=provenance)
    snapshot=load_checkpoint(checkpoint,expected_sha256=checkpoint_sha,expected_year=2022,
        expected_financing='zero_rate',expected_provenance=provenance)
    print('Uninterrupted full-year oracle for engineering parity',flush=True)
    reference=issue_forecasts(snapshot,c)
    state=start_forecasting(snapshot,{s:f.loc[f.index<BEGIN] for s,f in c.items()},binding=binding)
    completed=[];resumed_at=None;progress_pin=None
    boundaries=pd.date_range(BEGIN,END,freq='MS')
    for j,(a,b) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        state,f=forecast_chunk(state,selected(c,a,b))
        sha=save_forecasts(snapshot,f,store,begin=a,end=b,binding=binding)
        completed.append({'begin':a.isoformat(),'end':b.isoformat(),'sha256':sha})
        print('Published chunk',a.isoformat(),len(f),flush=True)
        if j==5:
            progress_pin=save_progress(state,store,completed)
            expected=state.history_identity;del state,completed
            state,completed=load_progress(store,progress_pin,year=2022,financing='zero_rate',binding=binding)
            assert state.history_identity==expected;resumed_at=state.next_origin.isoformat()
            print('Resumed from durable context and completed output pins',resumed_at,flush=True)
    assert state.next_origin==END and len(completed)==12
    final_progress=save_progress(state,store,completed)
    state,completed=load_progress(store,final_progress,year=2022,financing='zero_rate',binding=binding)
    pieces=[load_forecasts(snapshot,store,e['sha256'],begin=e['begin'],end=e['end'],binding=binding) for e in completed]
    actual=assemble_forecasts(snapshot,pieces,begin=BEGIN,end=END);del pieces
    pd.testing.assert_frame_equal(reference,actual,check_exact=True)
    for symbol in ASSETS:pd.testing.assert_frame_equal(state.contexts[symbol],c[symbol],check_exact=True)
    as_of=END-pd.Timedelta(hours=1)
    reference_score=score_forecasts(snapshot,attach_outcomes(snapshot,reference,c,as_of),begin=BEGIN,end=END,as_of=as_of)
    attached=attach_outcomes(snapshot,actual,state.contexts,as_of)
    annual_pin=save_forecasts(snapshot,attached,store,begin=BEGIN,end=END,binding=binding,as_of=as_of)
    reloaded=load_forecasts(snapshot,store,annual_pin,begin=BEGIN,end=END,binding=binding,as_of=as_of)
    pd.testing.assert_frame_equal(attached,reloaded,check_exact=True)
    score=score_forecasts(snapshot,reloaded,begin=BEGIN,end=END,as_of=as_of)
    assert portable(reference_score)==portable(score)
    print('Full-year forecast and scoring parity exact',flush=True)
    # Unsupported handling is an explicit fault test on a named 72h interval.
    # All 36 annual disposition cells are retained; no failed comparison is
    # replaced with a statistic from a reduced set of assets/models.
    failed=fault_snapshot(snapshot);stop=BEGIN+pd.Timedelta(hours=72)
    fault_state=start_forecasting(failed,{s:f.loc[f.index<BEGIN] for s,f in c.items()},binding=binding)
    fault_state,fault=forecast_chunk(fault_state,selected(c,BEGIN,stop))
    failure_resume=save_resume(fault_state,store)
    fault_state=load_resume(store,failure_resume,year=2022,financing='zero_rate',binding=binding)
    failed_pin=save_forecasts(failed,fault,store,begin=BEGIN,end=stop,binding=binding)
    fault=load_forecasts(failed,store,failed_pin,begin=BEGIN,end=stop,binding=binding)
    fault_score=score_forecasts(failed,attach_outcomes(failed,fault,c,stop),begin=BEGIN,end=stop,as_of=stop)
    unavailable=[r for r in fault_score['model_scores'] if r['cell_status']=='unavailable_annual_model']
    assert len(fault)==4*9*72 and len(unavailable)==3
    assert fault_score['annual_equal_asset']['status']=='unavailable_required_cell'
    print('Fault injection: three unavailable cells; all requested rows preserved',flush=True)
    elapsed=time.perf_counter()-started
    print('RESOURCE '+json.dumps({'elapsed_seconds':elapsed,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}),flush=True)
    return portable({'status':'FORECASTING_OPERATIONAL_PREFLIGHT_VERIFIED',
        'scope':'synthetic_2020_2022_zero_rate_only','binding':binding,
        'snapshot_sha256':snapshot.sha256,'matches_prior_synthetic_snapshot':True,
        'checkpoint_path':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':checkpoint_sha,
        'store_path':str(store.relative_to(ROOT)),'progress_sha256':final_progress,
        'interruption_progress_sha256':progress_pin,'resumed_at':resumed_at,'completed':completed,
        'annual_forecasts_sha256':annual_pin,'scheduled_asset_origins':4*8760,'forecast_rows':len(actual),
        'retained_context_memory_bytes':sum(int(f.memory_usage(index=True,deep=True).sum()) for f in state.contexts.values()),
        'forecast_frame_identity':_frame_identity(actual),'forecast_exact':True,
        'stored_outcomes_exact':True,'canonical_score_sha256':identity(score),'canonical_scoring_exact':True,
        'primary_scoring_asset_origins':sum(r['paired_origins'] for r in score['primary_differences']),
        'annual_availability':annual_availability(snapshot),
        'fault_injection':{'scope':'synthetic_72h_optimizer_failures','forecast_rows':len(fault),
            'annual_availability':annual_availability(failed),
            'unavailable_models':[{k:r[k] for k in ('symbol','model','annual_diagnostic','forecast_origins','ipcw_brier','cell_status')} for r in unavailable],
            'equal_asset_status':fault_score['annual_equal_asset'],'resume_sha256':failure_resume,'table_sha256':failed_pin},
        'final_execution_enabled':False,'real_post2022_contents_accessed_or_hashed':False,
        'market_downloads':False,'financial_trading_experiments':False,
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def report(v):
    return '\n'.join([
        '# Forecasting operational integration preflight','',
        '**FORECASTING_OPERATIONAL_PREFLIGHT_VERIFIED — engineering-complete for the bounded development forecasting path. '
        'Final-data access and evaluation remain disabled.**','',
        '`training/operational.py` reuses the existing annual forecaster, first-passage/KM functions, '
        'checkpoint validator and canonical scoring. It adds an observed-prefix cursor, immutable Parquet '
        'tables with versioned JSON manifests, and a durable progress record linking the cursor to every '
        'already-committed output chunk. Model/calibration/source/runtime identities and exact origins are pinned.','',
        'Full hourly causal context is retained and replayed at chunk boundaries. This preserves 90-day '
        'terminal-record history, ongoing paths and rolling-feature numerical history without implementing '
        'another label/KM engine. A temporary administrative terminal record at the current prefix end has '
        'publication equal to its last hour and cannot enter a prediction at or before that hour under the '
        'registered strict-past rule. Later chunks reconstruct it with newly observed inputs. Only supplied '
        'prefixes enter the chunk API; no future file or future outcome screen selects prediction rows.','',
        'Each table manifest binds its file SHA256, logical frame/dtypes/index identity, source/runtime, '
        'snapshot, year/control, model registry, requested interval and optional label-as-of cutoff. '
        'Load validates every asset/model/origin key and restores original UTC object/NaT representations '
        'and hourly-index frequency metadata. Missing/duplicate keys, stale/tampered history, wrong provenance '
        'and conflicts fail explicitly. Files and manifests publish atomically without clobbering; the progress '
        'manifest commits only after referenced output chunks and context are durable. Orphan blobs after an '
        'interrupted write are harmless; retry uses the same hashes and duplicate validation.','',
        'One necessary fix in `training/checkpoints.py` excludes `scoring_status` from the cross-model '
        'equality check on shared label copies. Scoring availability depends on each model, and its existing '
        'per-row check remains in force. No feature, label, model, calibration or scoring calculation changes.','',
        'The end-to-end fixture uses original synthetic 2020–2022 inputs: fitting '
        '2020-01-01–2021-09-01, calibration 2021-09-01–2022-01-01, and the full 2022 '
        'evaluation year, UTC half-open. Its fitted snapshot exactly matches the prior verified fixture. '
        'A supported model checkpoint is saved/reloaded, followed by an uninterrupted reference and twelve '
        'monthly forecast chunks. At July 1 the in-memory cursor and completed-output list are discarded '
        'and restored from the persisted progress pin. Focused tests separately exercise a fresh Python-process resume.','',
        f'**Exact parity:** {v["forecast_rows"]:,} forecast rows covering '
        f'{v["scheduled_asset_origins"]:,} asset-origins, four assets and all nine series. '
        'Uninterrupted, resumed and saved/reloaded forecasts agree exactly, including probabilities, '
        'timestamps, statuses, row ordering and metadata. The separately attached annual outcome table '
        'also round-trips exactly. Canonical point scores, paired keys/losses, equal-asset means and shared-calendar '
        f'uncertainty agree exactly on {v["primary_scoring_asset_origins"]:,} primary scoring origins. '
        'These are engineering checks on synthetic data, not independent forecasting-performance evidence.','',
        'Unsupported annual dispositions are explicit for all 36 asset/model cells. A separate named '
        '72-hour optimizer-failure injection makes ADA raw matched/calibrated matched and DOGE calibrated tree '
        'unavailable: all 2,592 requested rows remain, affected probabilities are null, diagnostics survive '
        'resume and table reload, and the complete four-asset mean is unavailable. The registered causal '
        'rolling reference remains dynamic; a frozen-fallback failure does not invent a fitted rolling curve.','',
        'The runner records reproducible pins in `results/forecasting_operational_preflight/summary.json`. '
        'Peak RSS and elapsed time are measured separately in its RESOURCE log and the '
        '[verification record](forecasting_operational_verification.json), keeping canonical outputs deterministic. '
        'Full history is practical for this fixed hourly four-asset study; repeated prefix reconstruction '
        'adds CPU and yearly scoring materializes the full forecast table. This is bounded batch processing, '
        'not a constant-memory streaming service. The harness holds reference, resumed and labeled tables '
        'simultaneously, so its peak RSS exceeds that of context retention alone. Context retention avoids '
        'unsafe pruning of long ongoing paths. Nanosecond observation clocks are tested against the unchanged '
        'annual decision-clock representation.','',
        'Reproduce from the repository root:','',
        '```bash',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -p test_annual_operational.py -v',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.test_portable',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.preflight_forecasting_operations --write-report',
        '```','',
        '**A. Forecasting engineering:** complete for the requested development-only operational path, '
        'subject to the verification receipt. No general-purpose framework, separate incremental KM engine, '
        'database, distributed queue, live service, or mid-optimizer resume is required for the registered '
        'annual batch protocol. Those features are omitted. The existing pre-2023 guards remain intact; '
        'this does not certify final-data coverage, real-year support or final-mode execution.','',
        '**B. Access/authorization tasks:** obtain explicit authorization for final-data acquisition/access '
        'and structural QA; verify four-asset spot/perpetual/funding history and EFFR reference coverage, '
        'availability clocks, source hashes, warmup and boundary handoffs. Any final-period IO activation '
        'must be narrowly authorized and checked against the frozen windows, including leap 2024; '
        'the current development guards do not automatically permit it. Final labels/performance and '
        'execution require their separately authorized stage. Financial replay remains separate from this '
        'forecasting-only path. No real 2023–2025 contents were opened, hashed or evaluated.','',
        '**C. Shortest safe path:** pin this verified code/artifacts; authorize bounded acquisition/structural '
        'QA; connect only the authorized frozen data partitions and validate IO/calendar/source boundaries '
        'without selecting on forecast outcomes; then obtain final-execution authorization and run the '
        'registered annual forecasts/storage/scoring with the existing unavailable-cell rules. Do not '
        'insert a new model search or build another forecasting framework. No such next task ran here.',''])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-report',action='store_true');args=parser.parse_args()
    sys.dont_write_bytecode=True;sys.addaudithook(guard)
    v=run();path=OUTPUT/'summary.json';text=json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if not path.exists() or path.read_text()!=text:path.write_text(text)
    if args.write_report:
        text=report(v)
        if not REPORT.exists() or REPORT.read_text()!=text:REPORT.write_text(text)
    print(v['status'],flush=True)


if __name__=='__main__':main()
