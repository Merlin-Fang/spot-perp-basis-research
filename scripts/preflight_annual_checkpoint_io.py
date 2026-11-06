"""Synthetic-only annual checkpoint and IO preflight, with final access denied.

Run: python -B -m scripts.preflight_annual_checkpoint_io --write-report
Only the new development checkpoint and compact reports may be written.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

from scripts.preflight_annual_forecasting import synthetic_contexts
from spot_perp_basis.training.annual import fit_development_year, issue_forecasts, attach_outcomes, evaluate_forecasts, portable, identity
from spot_perp_basis.training.checkpoints import (
    ASSETS, MODELS, checkpoint_provenance, implementation_binding,
    load_checkpoint, save_checkpoint, validate_contexts, validate_forecasts,
)


ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/annual_checkpoint_io_preflight'
STORE=OUTPUT/'checkpoints'
SUMMARY=OUTPUT/'summary.json'
REPORT=ROOT/'docs/annual_checkpoint_io_preflight.md'


def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):raise PermissionError('Checkpoint IO preflight is offline')
    if event!='open' or not isinstance(args[0],(str,bytes)):return
    path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
    if any(path.is_relative_to(ROOT/n) for n in ('data','experiments','outputs','manifests')):
        raise PermissionError('Synthetic preflight cannot open research contents: '+str(path))
    if path.is_relative_to(ROOT/'results') and path!=SUMMARY:
        allowed=path.is_relative_to(STORE) and len(path.relative_to(STORE).parts)==2 and len(path.parent.name)==64
        allowed &= path.name=='annual_2022_zero_rate.checkpoint.json' or path.name.startswith('.annual-') and path.suffix=='.tmp'
        if not allowed:raise PermissionError('Synthetic preflight cannot open prior results: '+str(path))


def table_digest(frame):
    hashes=pd.util.hash_pandas_object(frame,index=True,categorize=False).to_numpy(dtype='<u8')
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def run():
    contexts=synthetic_contexts();binding=implementation_binding(ROOT)
    checkpoint=STORE/identity(binding)/'annual_2022_zero_rate.checkpoint.json'
    snapshot=fit_development_year(contexts,2022,'zero_rate',progress=lambda stage,s:print('Synthetic',stage,s,flush=True))
    print('Validating inputs and saving annual model state',flush=True)
    input_contract=validate_contexts(snapshot,contexts)
    provenance=checkpoint_provenance(snapshot,contexts,source_kind='synthetic',binding=binding)
    sha=save_checkpoint(snapshot,checkpoint,provenance=provenance)
    stat_before=checkpoint.stat();loaded=load_checkpoint(checkpoint,expected_sha256=sha,
        expected_year=2022,expected_financing='zero_rate',expected_provenance=provenance)
    assert loaded.content==snapshot.content and loaded.sha256==snapshot.sha256
    print('Issuing full synthetic 2022 clock from original and loaded models',flush=True)
    original=issue_forecasts(snapshot,contexts);reproduced=issue_forecasts(loaded,contexts)
    pd.testing.assert_frame_equal(original,reproduced,check_exact=True)
    output_contract=validate_forecasts(loaded,reproduced)
    print('Validating separately attached, prefix-available outcomes',flush=True)
    as_of=pd.Timestamp('2022-12-31T23:00:00Z')
    attached=attach_outcomes(loaded,reproduced,contexts,as_of)
    attached_contract=validate_forecasts(loaded,attached,attached=True,as_of=as_of)
    pd.testing.assert_frame_equal(reproduced,attached[reproduced.columns],check_exact=True)
    # The scoring computation is reused as-is; compare complete scalar outputs,
    # not a newly selected model or economic objective.
    before=evaluate_forecasts(snapshot,attach_outcomes(snapshot,original,contexts,as_of))
    after=evaluate_forecasts(loaded,attached)
    assert portable(before)==portable(after)
    assert save_checkpoint(loaded,checkpoint,provenance=provenance)==sha
    assert (checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)==(stat_before.st_size,stat_before.st_mtime_ns)
    with tempfile.TemporaryDirectory(prefix='annual_checkpoint_copy_') as temp:
        copy=Path(temp)/checkpoint.name
        assert save_checkpoint(loaded,copy,provenance=provenance)==sha
        assert copy.read_bytes()==checkpoint.read_bytes()
    support={s:{'models':{n:a['fitted'][n]['status'] for n in ('tree','matched_logistic')},
        'calibrations':{n:c['status'] for n,c in a['calibration_maps'].items()},
        'origins':a['fitted']['manifest']['unique_origins'],'trials':a['fitted']['manifest']['trials']}
        for s,a in snapshot.read()['assets'].items()}
    return {'status':'ANNUAL_CHECKPOINT_IO_READY_FOR_BOUNDED_PREFLIGHT',
        'scope':'synthetic_2020_2022_only','financing':'zero_rate',
        'windows':snapshot.read()['windows'],'snapshot_sha256':snapshot.sha256,
        'checkpoint_path':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha,
        'checkpoint_bytes':stat_before.st_size,'provenance':provenance,
        'input_contract':input_contract,'output_contract':output_contract,'attached_contract':attached_contract,
        'model_registry':list(MODELS),'support':support,
        'parity':{'snapshot_content_exact':True,'full_forecast_frame_exact':True,
            'forecast_frame_sha256':table_digest(reproduced),'maximum_probability_error':0.,
            'canonical_scoring_identical':True,'scoring_asset_origins':sum(x['paired_origins'] for x in after['primary_differences']),
            'label_attachment_preserves_issuance':True},
        'idempotence':{'repeated_save_content_and_mtime_preserved':True,'cross_directory_bytes_identical':True},
        'scope_guards':{'final_execution_enabled':False,'real_market_content_opened_or_hashed':False,
            'market_or_EFFR_download':False,'financial_backtest':False,'model_or_rule_change':False},
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def report(v):
    return '\n'.join([
        '# Annual checkpoint serialization and IO preflight','',
        '**'+v['status']+'. Final holdout access and evaluation remain disabled.**','',
        'The new `training/checkpoints.py` wraps the existing immutable annual snapshot in canonical JSON. '
        'No existing scientific module, experiment, checkpoint or report is edited. '
        'It saves tree states/cuts, matched/coarse logistic coefficients/scalers, calibration maps, '
        'annual/magnitude/censor KM curves, fitting/calibration metadata and source identities without numerical conversion.','',
        '`checkpoint_provenance` binds only pre-evaluation fitting/calibration inputs, the frozen scientific documents, '
        'the fixed source-code allowlist and Python/NumPy/pandas versions. `save_checkpoint` uses atomic no-clobber '
        'publication; identical saves preserve file bytes and mtime. `load_checkpoint` requires a caller-pinned file '
        'SHA256, expected year/control and expected provenance. Unknown versions, duplicate JSON keys, nonfinite '
        'numbers, corruption, missing fields, changed definitions and mismatched identities fail explicitly. '
        'An embedded digest is not treated as independent provenance; the expected pin must come from a trusted run receipt.','',
        '`validate_contexts` enforces the four-asset UTC hourly causal-input schema, explicit missing hours, typed '
        'marks/rates/availability and original publication rules. `validate_forecasts` requires all requested '
        'asset × model × origin keys, including unavailable rows; it checks probabilities, raw/calibrated parent '
        'relationships, cutoffs, support/failure statuses and rolling-publication timing. Optional ex-post outcomes '
        'are checked separately against an explicit observed-prefix cutoff, with late-year censoring retained.','',
        'The existing output uses UTC Timestamp objects mixed with NaT for rolling publications and sometimes '
        'timezone-naive all-NaT columns for pending outcomes. Validation accepts these existing nullable '
        'representations; it rejects every non-null naive/string timestamp and never localizes an unknown clock.','',
        'Strict save/load requires the complete registered fitted state. A genuine unsupported fit/map raises '
        '`CheckpointUnsupportedError` with asset/model/status diagnostics. It does not substitute a different '
        'model or hide unavailable cells. The output validator separately accepts explicitly unavailable rows from '
        'the existing coordinator. A later orchestrator must translate a failed annual checkpoint into the '
        'registered unavailable annual cells and retain diagnostics; that orchestration is not implemented here.','',
        'The bounded demonstration uses the unchanged synthetic 2020–2022 fixture: fit '
        '2020-01-01–2021-09-01, calibration 2021-09-01–2022-01-01 and evaluation '
        '2022-01-01–2023-01-01, UTC half-open. All four assets and nine series are exercised under zero financing. '
        'This is not evidence about real future-year model support or performance, and it is not an EFFR annual run.','',
        f'**Exact parity:** {v["output_contract"]["rows"]:,} forecast/model rows '
        f'({v["output_contract"]["asset_origins"]:,} scheduled asset-origins), maximum probability difference **0**. '
        'Original and loaded snapshots and full forecast frames agree exactly, including nulls, statuses, metadata '
        'and row ordering. Canonical scoring also agrees exactly on '
        f'{v["parity"]["scoring_asset_origins"]:,} paired asset-origins. No refit or recalibration occurs during loading/issuance.','',
        f'Checkpoint: `{v["checkpoint_path"]}` ({v["checkpoint_bytes"]:,} bytes). SHA256 '
        f'`{v["checkpoint_sha256"]}`. Development checkpoints use source-binding version directories; prior '
        'versions remain intact. Only checkpoint state and compact summaries are retained; '
        'forecast tables stay in memory. Cross-directory serialization is byte-identical and repeated save preserves '
        'mtime. The verification record also checks a complete runner rerun.','',
        'Focused tests exercise corrupted/truncated/reformatted/oversized files; missing/conflicting/symlink '
        'targets and failed atomic publication; unsupported states; altered model/calibration/KM metadata; '
        'future/naive timestamps; incomplete scopes; parent-map mismatch; and pending/censored outcomes. '
        'See [verification](annual_checkpoint_io_verification.json) for final test counts, preservation and rerun receipts.','',
        'Reproduce from the repository root:','',
        '```bash',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -p test_annual_checkpoint_io.py -v',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.test_portable',
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m scripts.preflight_annual_checkpoint_io --write-report',
        '```','',
        'Remaining forecasting IO gates: connect an explicitly authorized, bounded data-source adapter; '
        'verify restored source identities/warmup at operational boundaries; define resumable rolling-terminal '
        'record/context persistence; preserve row populations through a versioned forecast-table storage adapter; '
        'and implement annual failure-to-unavailable reporting without fallback. The model checkpoint does not '
        'claim to persist live data history or a rolling estimator ledger. Current same-runtime JSON parity '
        'does not certify cross-version numerical compatibility.','',
        'Final data acquisition/QA (including future EFFR and four-asset coverage), final execution authorization '
        'and continuous financial replay/checkpoint engineering remain separate, unpassed gates. Existing '
        'hour-versus-minute and last-trade measurement limitations remain. No real 2023–2025 contents, new '
        'market downloads, financial backtests, staging, commits or pushes were used.','',
        'Recommended next bounded preflight: synthetic/pre-2023 chunked context and rolling-record resume parity, '
        'followed by forecast-table storage round-trips with exact key/status preservation. Do not begin '
        'holdout preparation or evaluation under this task.',''])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-report',action='store_true');args=parser.parse_args()
    sys.dont_write_bytecode=True;sys.addaudithook(guard)
    value=portable(run());text=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if not SUMMARY.exists() or SUMMARY.read_text()!=text:SUMMARY.write_text(text)
    if args.write_report:
        text=report(value)
        if not REPORT.exists() or REPORT.read_text()!=text:REPORT.write_text(text)
    print(value['status'],flush=True)


if __name__=='__main__':main()
