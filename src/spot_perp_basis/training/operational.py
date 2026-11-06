"""Bounded development forecasting: prefix replay, resume and forecast tables.

Full hourly causal history is small for this study. Replaying the existing
forecaster avoids a second first-passage/KM engine, truncation of open paths or
floating-point changes from resetting rolling-feature accumulators. Temporary
administrative records at a prefix end cannot enter its strict-past rolling KM.
"""

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow

from ..data.windows import require_training_times
from .annual import ASSETS, MODELS, CALIBRATED, AnnualForecastSnapshot, issue_forecasts, identity
from .checkpoints import (
    CheckpointContractError, _canonical, _parse, _require, _keys, _digest, _state,
    _development, _time, implementation_binding, validate_contexts, validate_forecasts,
    FROZEN, CODE,
)


FORMAT='development-forecast-operations-v1'


def operational_binding(root):
    root=Path(root)
    return {'model_binding':implementation_binding(root),'operations_sha256':hashlib.sha256(
        (root/'src/spot_perp_basis/training/operational.py').read_bytes()).hexdigest(),
        'pyarrow':pyarrow.__version__}


def annual_availability(snapshot):
    """Every asset/model annual cell, without removing unsupported cases."""
    p,w,_=_state(snapshot,supported=False);rows=[]
    for symbol in ASSETS:
        a=p['assets'][symbol];f=a['fitted'];e=f['empirical']
        status={'raw_tree':f['tree']['status'],'raw_matched_logistic':f['matched_logistic']['status'],
            'raw_coarse_logistic':e['hazard_0.01']['status'],'unconditional_KM':e['asset_km']['status'],
            'magnitude_KM':e['asset_km']['status']}
        for model in MODELS:
            if model=='rolling_magnitude_KM':value='causal_dynamic_reference'
            elif model in CALIBRATED:
                value=a['calibration_maps'][model]['status'] if status[CALIBRATED[model]]=='fit' else 'unsupported_parent_fit'
            else:value=status[model]
            rows.append({'symbol':symbol,'financing':p['financing'],'year':w.year,'model':model,
                'availability':'dynamic' if model=='rolling_magnitude_KM' else 'available' if value=='fit' else 'unavailable',
                'diagnostic':value,'snapshot_sha256':snapshot.sha256})
    return rows


def _frame_identity(frame):
    hashed=pd.util.hash_pandas_object(frame,index=True,categorize=False).to_numpy(dtype='<u8')
    return {'sha256':hashlib.sha256(hashed.tobytes()).hexdigest(),
        'columns':[[str(c),str(frame[c].dtype)] for c in frame],
        'index_dtype':str(frame.index.dtype),'index_name':frame.index.name,
        'index_frequency':frame.index.freqstr if isinstance(frame.index,pd.DatetimeIndex) else None,
        'rows':len(frame)}


@dataclass(frozen=True)
class ForecastCursor:
    snapshot: AnnualForecastSnapshot
    contexts: dict
    next_origin: pd.Timestamp
    history_identity: dict
    binding: dict


def _binding(binding):
    _keys(binding,('model_binding','operations_sha256','pyarrow'),'operational binding')
    _digest(binding['operations_sha256']);_require(binding['pyarrow']==pyarrow.__version__,'Arrow version mismatch')
    base=binding['model_binding'];_keys(base,('frozen_sha256','implementation_sha256','environment'),'model binding')
    _require(base['frozen_sha256']==FROZEN,'Scientific binding differs')
    _keys(base['implementation_sha256'],CODE,'model source bindings')
    for value in base['implementation_sha256'].values():_digest(value)
    _require(binding==operational_binding(Path(__file__).resolve().parents[3]),'Current operational source/runtime binding differs')


def _cursor(state):
    _require(isinstance(state,ForecastCursor),'Expected forecasting cursor')
    p,w,_=_state(state.snapshot,supported=False);_binding(state.binding)
    t=state.next_origin
    _require(isinstance(t,pd.Timestamp) and t.tz is not None and str(t.tz)=='UTC'
        and w.evaluation[0]<=t<=w.evaluation[1] and t.floor('h')==t,'Invalid resume cursor')
    validate_contexts(state.snapshot,state.contexts)
    actual={s:_frame_identity(f) for s,f in state.contexts.items()}
    _require(actual==state.history_identity,'Retained causal history changed')
    _require(all(len(f) and f.index.max()==t-pd.Timedelta(hours=1) for f in state.contexts.values()),'History/cursor boundary mismatch')
    return p,w


def start_forecasting(snapshot, warmup_contexts, *, binding):
    p,w,_=_state(snapshot,supported=False);validate_contexts(snapshot,warmup_contexts);_binding(binding)
    contexts={s:f.copy(deep=True) for s,f in warmup_contexts.items()}
    state=ForecastCursor(snapshot,contexts,w.evaluation[0],{s:_frame_identity(f) for s,f in contexts.items()},binding)
    _cursor(state);return state


def forecast_chunk(state, observations):
    """Append an exact consecutive observed prefix; no future context is read."""
    p,w=_cursor(state);_keys(observations,ASSETS,'chunk assets')
    first=observations[ASSETS[0]]
    _require(isinstance(first,pd.DataFrame) and len(first)>0,'Empty forecasting chunk')
    clock=first.index;require_training_times(clock)
    _require(clock.min()==state.next_origin and clock.max()<w.evaluation[1], 'Chunk skips/repeats an origin or exceeds its year')
    expected=pd.date_range(clock.min(),clock.max(),freq='h',name=clock.name)
    _require(clock.equals(expected),'Chunk must retain every scheduled hour')
    combined={}
    for symbol in ASSETS:
        new=observations[symbol];old=state.contexts[symbol]
        _require(new.index.equals(clock) and list(new.columns)==list(old.columns)
            and str(new.index.dtype)==str(old.index.dtype) and new.index.name==old.index.name
            and list(map(str,new.dtypes))==list(map(str,old.dtypes)),'Chunk schema or asset clock differs')
        combined[symbol]=pd.concat([old,new])
    validate_contexts(state.snapshot,combined)
    # Match the annual forecaster's decision-clock representation even when
    # the normalized observation index uses a different exact datetime unit.
    decision_clock=pd.date_range(*w.evaluation,freq='h',inclusive='left',name='origin_time_utc')
    decision_clock=decision_clock[(decision_clock>=clock.min())&(decision_clock<=clock.max())]
    forecasts=issue_forecasts(state.snapshot,combined,origins=decision_clock)
    validate_forecasts(state.snapshot,forecasts,origins=decision_clock)
    result=ForecastCursor(state.snapshot,combined,clock.max()+pd.Timedelta(hours=1),
        {s:_frame_identity(f) for s,f in combined.items()},state.binding)
    return result,forecasts


def _safe_path(path):
    path=Path(path)
    _require(not any(p.is_symlink() for p in (path,*path.parents)),'Artifact path traverses a symlink')
    return path


def _publish(path,content):
    """Local atomic, no-clobber publication; identical writes preserve mtime."""
    path=_safe_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.forecast-',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
        try:os.link(temp,path,follow_symlinks=False)
        except FileExistsError:_require(path.read_bytes()==content,'Conflicting immutable artifact: '+path.name)
    finally:Path(temp).unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def _put_frame(directory,frame):
    """Arrow retains binary floats; dtype metadata restores legacy UTC objects."""
    original=_frame_identity(frame);encoded=frame.copy(deep=False)
    # The only object-valued fields allowed by the existing contracts are
    # string identities or nullable UTC Timestamp columns. Arrow needs a
    # single physical datetime dtype for the latter.
    for name in frame:
        if str(frame[name].dtype)=='object' and (name.endswith('_utc')):
            encoded=encoded.copy();encoded[name]=pd.to_datetime(frame[name],utc=True).dt.as_unit('ns')
    buffer=io.BytesIO();encoded.to_parquet(buffer,engine='pyarrow',compression='zstd',index=True)
    content=buffer.getvalue();sha=hashlib.sha256(content).hexdigest()
    _publish(Path(directory)/(sha+'.parquet'),content)
    return {'file_sha256':sha,'frame_identity':original}


def _get_frame(directory,description):
    _keys(description,('file_sha256','frame_identity'),'frame description');_digest(description['file_sha256'])
    path=_safe_path(Path(directory)/(description['file_sha256']+'.parquet'))
    content=path.read_bytes();_require(hashlib.sha256(content).hexdigest()==description['file_sha256'],'Stored frame checksum mismatch')
    f=pd.read_parquet(io.BytesIO(content),engine='pyarrow');manifest=description['frame_identity']
    _keys(manifest,('sha256','columns','index_dtype','index_name','index_frequency','rows'),'frame identity');_digest(manifest['sha256'])
    _require([n for n,_ in manifest['columns']]==list(f.columns),'Stored columns differ')
    for name,dtype in manifest['columns']:
        # No executable/object deserializer is used. Only the original small
        # numeric/bool/string/UTC schemas are permitted.
        null_time=name.endswith('_utc') and f[name].isna().all() and dtype.startswith('datetime64[') and dtype.endswith(']')
        _require(dtype in ('str','object','bool','boolean','int64','int32','Int64','float64','float32','Float64')
            or dtype.startswith('datetime64[') and dtype.endswith(', UTC]') or null_time,'Unregistered stored dtype')
        f[name]=f[name].astype(dtype)
    if isinstance(f.index,pd.DatetimeIndex):
        f.index=pd.DatetimeIndex(f.index.astype(manifest['index_dtype']),freq=manifest['index_frequency'])
    f.index.name=manifest['index_name']
    _require(_frame_identity(f)==manifest,'Restored frame values/dtypes/index differ')
    return f


def _load_manifest(directory, sha, *, year, financing, binding, kind):
    _development(year,financing);_digest(sha);_binding(binding)
    path=_safe_path(Path(directory)/(sha+'.json'));content=path.read_bytes()
    _require(hashlib.sha256(content).hexdigest()==sha,'Manifest checksum mismatch')
    m=_parse(content);_require(_canonical(m)+b'\n'==content,'Noncanonical manifest')
    _require(m.get('format')==FORMAT and m.get('kind')==kind and m.get('year')==year
        and m.get('financing')==financing and m.get('binding')==binding,'Manifest version/identity/provenance differs')
    return m


def save_resume(state,directory):
    p,w=_cursor(state)
    descriptions={s:_put_frame(directory,state.contexts[s]) for s in ASSETS}
    m={'format':FORMAT,'kind':'resume','year':w.year,'financing':p['financing'],
        'binding':state.binding,'snapshot':state.snapshot.read(),'snapshot_sha256':state.snapshot.sha256,
        'next_origin':state.next_origin.isoformat(),'contexts':descriptions,
        'annual_availability':annual_availability(state.snapshot)}
    content=_canonical(m)+b'\n';sha=hashlib.sha256(content).hexdigest();_publish(Path(directory)/(sha+'.json'),content)
    return sha


def load_resume(directory,sha,*,year,financing,binding):
    m=_load_manifest(directory,sha,year=year,financing=financing,binding=binding,kind='resume')
    _keys(m,('format','kind','year','financing','binding','snapshot','snapshot_sha256','next_origin','contexts','annual_availability'),'resume manifest')
    s=AnnualForecastSnapshot(_canonical(m['snapshot']).decode());p,w,_=_state(s,supported=False)
    _require(s.sha256==m['snapshot_sha256'] and p['evaluation_year']==year and p['financing']==financing,'Resume snapshot identity differs')
    _require(m['annual_availability']==annual_availability(s),'Annual availability diagnostics changed')
    _keys(m['contexts'],ASSETS,'resume history assets')
    contexts={s:_get_frame(directory,m['contexts'][s]) for s in ASSETS}
    state=ForecastCursor(s,contexts,_time(m['next_origin'],'next origin'),
        {s:_frame_identity(f) for s,f in contexts.items()},binding)
    _cursor(state);return state


def _clock(snapshot,begin,end):
    _,w,_=_state(snapshot,supported=False)
    begin,end=pd.Timestamp(begin),pd.Timestamp(end)
    _require(begin.tz is not None and end.tz is not None and str(begin.tz)==str(end.tz)=='UTC'
        and w.evaluation[0]<=begin<end<=w.evaluation[1] and begin.floor('h')==begin and end.floor('h')==end,'Invalid forecast storage interval')
    return pd.date_range(begin,end,freq='h',inclusive='left',name='origin_time_utc')


def save_forecasts(snapshot,frame,directory,*,begin,end,binding,as_of=None):
    p,w,_=_state(snapshot,supported=False);_binding(binding);clock=_clock(snapshot,begin,end)
    validate_forecasts(snapshot,frame,origins=clock,attached=as_of is not None,as_of=as_of)
    d=_put_frame(directory,frame)
    m={'format':FORMAT,'kind':'forecasts','year':w.year,'financing':p['financing'],
        'binding':binding,'snapshot_sha256':snapshot.sha256,'model_registry':list(MODELS),
        'begin':clock.min().isoformat(),'end':pd.Timestamp(end).isoformat(),
        'as_of':pd.Timestamp(as_of).isoformat() if as_of is not None else None,
        'frame':d,'annual_availability':annual_availability(snapshot)}
    content=_canonical(m)+b'\n';sha=hashlib.sha256(content).hexdigest();_publish(Path(directory)/(sha+'.json'),content)
    return sha


def load_forecasts(snapshot,directory,sha,*,begin,end,binding,as_of=None):
    p,w,_=_state(snapshot,supported=False);clock=_clock(snapshot,begin,end)
    m=_load_manifest(directory,sha,year=w.year,financing=p['financing'],binding=binding,kind='forecasts')
    _keys(m,('format','kind','year','financing','binding','snapshot_sha256','model_registry','begin','end','as_of','frame','annual_availability'),'forecast manifest')
    _require(m['snapshot_sha256']==snapshot.sha256 and m['model_registry']==list(MODELS)
        and m['begin']==pd.Timestamp(begin).isoformat() and m['end']==pd.Timestamp(end).isoformat()
        and m['as_of']==(pd.Timestamp(as_of).isoformat() if as_of is not None else None),'Forecast request/manifest differs')
    _require(m['annual_availability']==annual_availability(snapshot),'Forecast annual cells differ')
    f=_get_frame(directory,m['frame']);validate_forecasts(snapshot,f,origins=clock,attached=as_of is not None,as_of=as_of)
    return f


def assemble_forecasts(snapshot,frames,*,begin,end,as_of=None):
    """Restore original asset/model/hour ordering; reject holes and duplicates."""
    _require(len(frames)>0,'No forecast chunks')
    f=pd.concat(frames,ignore_index=True);clock=_clock(snapshot,begin,end)
    validate_forecasts(snapshot,f,origins=clock,attached=as_of is not None,as_of=as_of)
    order=np.lexsort((pd.DatetimeIndex(f.origin_time_utc).asi8,
        f.model.map({m:j for j,m in enumerate(MODELS)}),f.symbol.map({s:j for j,s in enumerate(ASSETS)})))
    return f.iloc[order].reset_index(drop=True)


def _completed(state,directory,completed):
    """A durable resume must also identify every already-published chunk."""
    _,w=_cursor(state);end=w.evaluation[0]
    _require(isinstance(completed,list),'Expected committed chunk list')
    for entry in completed:
        _keys(entry,('begin','end','sha256'),'committed forecast chunk')
        begin,stop=_time(entry['begin'],'chunk begin'),_time(entry['end'],'chunk end')
        _require(begin==end and begin<stop<=state.next_origin,'Committed chunks have a hole/overlap')
        load_forecasts(state.snapshot,directory,entry['sha256'],begin=begin,end=stop,binding=state.binding)
        end=stop
    _require(end==state.next_origin,'Cursor advanced beyond committed forecasts')


def save_progress(state,directory,completed):
    """Publish cursor plus ordered output pins only after all blobs are durable."""
    p,w=_cursor(state);_completed(state,directory,completed)
    resume=save_resume(state,directory)
    m={'format':FORMAT,'kind':'progress','year':w.year,'financing':p['financing'],
        'binding':state.binding,'resume_sha256':resume,'completed':completed}
    content=_canonical(m)+b'\n';sha=hashlib.sha256(content).hexdigest();_publish(Path(directory)/(sha+'.json'),content)
    return sha


def load_progress(directory,sha,*,year,financing,binding):
    m=_load_manifest(directory,sha,year=year,financing=financing,binding=binding,kind='progress')
    _keys(m,('format','kind','year','financing','binding','resume_sha256','completed'),'progress manifest')
    state=load_resume(directory,m['resume_sha256'],year=year,financing=financing,binding=binding)
    _completed(state,directory,m['completed'])
    return state,m['completed']


def score_forecasts(snapshot,attached,*,begin,end,as_of):
    """Canonical scoring plus every annual cell and its failure diagnostics."""
    from .annual import evaluate_forecasts
    clock=_clock(snapshot,begin,end);validate_forecasts(snapshot,attached,origins=clock,attached=True,as_of=as_of)
    score=evaluate_forecasts(snapshot,attached)
    availability=annual_availability(snapshot)
    lookup={(r['symbol'],r['model']):r for r in availability}
    for row in score['model_scores']:
        cell=lookup[(row['symbol'],row['model'])]
        row['annual_availability']=cell['availability'];row['annual_diagnostic']=cell['diagnostic']
        row['cell_status']='unavailable_annual_model' if cell['availability']=='unavailable' else 'scored' if np.isfinite(row['ipcw_brier']) else 'unavailable_scoring_support'
    return {**score,'annual_availability':availability}
