"""Development-only annual checkpoint and frame contracts.

JSON preserves the existing snapshot without fitting, numerical conversion or
model substitution. The caller pins a file digest and expected provenance.
No market loader, final-mode option, scoring algorithm or trading policy lives
here. Complete unsupported state is diagnosed explicitly, never repaired.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

import numpy as np
import pandas as pd

from ..data.features import FEATURES
from ..data.windows import START, END, annual_windows, require_training_times
from ..models.calibration import calibrate
from ..models.survival import km_predict
from .annual import ASSETS, FINANCING, MODELS, CALIBRATED, PROTOCOL, AnnualForecastSnapshot, causal_features, identity


FORMAT='annual-development-checkpoint-v1'
RAW=('raw_tree','raw_matched_logistic','unconditional_KM','magnitude_KM','raw_coarse_logistic')
FROZEN={
    'docs/final_evaluation_protocol.md':'728d1596f0a219847a252de57dff72867801c090326779eedc8df11383631e26',
    'configs/final_evaluation.yaml':'647fc5f3ca09b58c0a7ea74644b6316ed7eda6fd6e3ed1d741464e57d6861e21',
    'docs/final_evaluation_freeze_receipt.md':'2a74cb15c1121f56d5cdd9e3db146c89dd299dd9ecd612f7ca3aafbad3eed2c0',
}
CODE=('training/annual.py','training/fit.py','training/checkpoints.py',
      'data/windows.py','data/features.py','data/labels.py','theory.py',
      'models/trees.py','models/logistic.py','models/calibration.py','models/survival.py',
      'evaluation/metrics.py','evaluation/survival_audit.py','evaluation/statistics.py')
BASE_INPUT={'spot_close','perp_close','annual_rate','rate_available_at_utc','theory_available'}
OPTIONAL_INPUT={'rho','symbol','financing','bar_open_time_utc','bar_available_at_utc',
                'funding_known','last_funding_settlement_utc','last_funding_available_at_utc'}
OUTPUT=tuple(('symbol financing evaluation_year origin_time_utc horizon_hours model '
    'raw_event_probability event_probability forecast_available_at_utc fit_cutoff_utc '
    'calibration_cutoff_utc snapshot_sha256 origin_features_available low_fee_qualified '
    'model_support_status calibration_status calibration_source forecast_status '
    'rolling_latest_record_publication_utc').split())
OUTCOMES=tuple(('followup_hours event_observed event_by_8h label_available_at_utc '
    'label_record_available_at_utc outcome_status scoring_status').split())


class CheckpointContractError(ValueError):
    """Incomplete, inconsistent, corrupted or unregistered IO state."""


class CheckpointUnsupportedError(CheckpointContractError):
    def __init__(self, failures):
        self.failures=tuple(failures)
        super().__init__('Unsupported annual checkpoint: '+', '.join(failures))


def _require(condition, message):
    if not condition:raise CheckpointContractError(message)


def _keys(value, required, label, *, optional=()):
    _require(isinstance(value,dict),label+' must be an object')
    _require(set(required)<=set(value)<=set(required)|set(optional),label+' fields missing or unregistered')


def _integer(value, label, minimum=0):
    _require(type(value) is int and value>=minimum,label+' must be an integer >= '+str(minimum))


def _number(value, label):
    _require(type(value) in (int,float) and np.isfinite(value),label+' must be finite numeric')


def _vector(value, size, label, *, positive=False):
    _require(isinstance(value,list) and len(value)==size,label+' shape mismatch')
    for x in value:_number(x,label)
    _require(not positive or all(x>0 for x in value),label+' must be positive')


def _digest(value):
    _require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'Invalid SHA256 identity')


def _time(value, label):
    _require(isinstance(value,str),label+' must be a UTC timestamp string')
    try:t=pd.Timestamp(value)
    except (TypeError,ValueError) as e:raise CheckpointContractError(label+' invalid timestamp') from e
    _require(not pd.isna(t) and t.tz is not None and str(t.tz)=='UTC',label+' requires explicit UTC')
    return t


def _canonical(value):
    try:return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    except (TypeError,ValueError,OverflowError) as e:raise CheckpointContractError('Non-JSON or nonfinite checkpoint state') from e


def _parse(content):
    def pairs(items):
        result={}
        for k,v in items:
            _require(k not in result,'Duplicate JSON key: '+k);result[k]=v
        return result
    def bad_constant(value):raise CheckpointContractError('Nonfinite JSON constant: '+value)
    try:return json.loads(content,object_pairs_hook=pairs,parse_constant=bad_constant)
    except (ValueError,TypeError,UnicodeError,RecursionError) as e:
        raise CheckpointContractError('Invalid checkpoint JSON: '+str(e)) from e


def _development(year, financing):
    _require(type(year) is int and financing in FINANCING,'Unregistered year/control identity')
    w=annual_windows(year)
    if w.fit[0]<START or w.evaluation[1]>END:
        raise CheckpointContractError('Final-period checkpoint IO is disabled; calendar metadata only')
    return w


def _km(model, reverse):
    _keys(model,('status','n'),'KM',optional=('events','maximum_followup','survival','reverse_censor'))
    _integer(model['n'],'KM origins')
    _require(model['status'] in ('fit','unsupported_training_support'),'Unknown KM status')
    if model['status']!='fit':
        _require(model['n']<20,'Inconsistent unsupported KM count');return
    _keys(model,('status','n','events','maximum_followup','survival','reverse_censor'),'fitted KM')
    _require(model['n']>=20 and model['reverse_censor'] is reverse,'KM support/role mismatch')
    _integer(model['events'],'KM events');_require(model['events']<=model['n'],'KM event count')
    _integer(model['maximum_followup'],'KM followup')
    _vector(model['survival'],model['maximum_followup']+1,'KM survival')
    s=np.array(model['survival']);_require(((s>=0)&(s<=1)).all() and (np.diff(s)<=0).all(),'Invalid KM survival')


def _tree(model, trials):
    _require(model.get('status') in ('fit','unsupported_training'),'Unknown tree status')
    for k in ('n','events'):_integer(model.get(k),'tree '+k)
    _require(model['n']==trials and model['events']<=trials,'Tree trial counts differ')
    if model['status']!='fit':
        _keys(model,('status','n','events'),'unsupported tree',optional=('depth','learning_rate'))
        _require(model.get('depth',3)==3 and model.get('learning_rate',.1)==.1,'Changed tree parameters');return
    _keys(model,('status','n','events','depth','learning_rate','maximum_rounds','cuts','intercept','trees','unweighted_training_log_loss','algorithm'),'tree')
    _require((model['depth'],model['learning_rate'],model['maximum_rounds'],model['algorithm'])==
        (3,.1,80,'local_numpy_histogram_Newton_boosting'),'Changed tree specification')
    _require(trials>=500 and min(model['events'],trials-model['events'])>=20,'Tree support counts')
    _number(model['intercept'],'tree intercept');_vector(model['unweighted_training_log_loss'],80,'boosting history')
    _require(isinstance(model['cuts'],list) and len(model['cuts'])==7,'Tree feature/hour dimensions')
    for cuts in model['cuts']:
        _require(isinstance(cuts,list) and len(cuts)<=31,'Histogram dimensions')
        _vector(cuts,len(cuts),'cuts');_require((np.diff(cuts)>0).all(),'Unordered/duplicate cuts')
    _require(isinstance(model['trees'],list) and len(model['trees'])==80,'Incomplete boosting rounds')
    def node(t,depth):
        _require(isinstance(t,dict),'Invalid tree node');_integer(t.get('n'),'node trials',200)
        if 'leaf' in t:
            _keys(t,('leaf','n','gradient_sum','hessian_sum'),'leaf')
            for k in ('leaf','gradient_sum','hessian_sum'):_number(t[k],k)
            _require(t['hessian_sum']>=0 and np.isclose(t['leaf'],-t['gradient_sum']/(t['hessian_sum']+5),atol=1e-12,rtol=1e-12),'Invalid leaf/L2 state')
        else:
            _keys(t,('feature','bin','gain','n','left','right'),'split')
            _integer(t['feature'],'split feature');_integer(t['bin'],'split bin');_number(t['gain'],'split gain')
            _require(depth<3 and t['feature']<7 and t['bin']<len(model['cuts'][t['feature']]) and t['gain']>0,'Invalid registered split')
            node(t['left'],depth+1);node(t['right'],depth+1)
            _require(t['left']['n']+t['right']['n']==t['n'],'Split trial conservation')
    for t in model['trees']:
        node(t,0);_require(t['n']==trials,'Tree root trial mismatch')


def _logistic(model, trials=None, cutoff=None):
    hourly=trials is not None
    _require(model.get('status') in ('fit','unsupported_training_support','unsupported_numerical_nonconvergence'),'Unknown logistic status')
    for k in ('n','events'):_integer(model.get(k),'logistic '+k)
    _require(model['events']<=model['n'] and (not hourly or model['n']==trials),'Logistic trial counts differ')
    if hourly:_require(model.get('alpha')==.001 and model.get('K')==16,'Changed matched logistic definition')
    else:
        _require(model.get('features')==['rho','abs_rho'],'Changed coarse feature definition')
        _require(_time(model.get('fit_cutoff_utc'),'coarse cutoff')==cutoff,'Coarse fitting cutoff mismatch')
        _digest(model.get('trial_origin_keys_sha256'))
        av=model.get('maximum_trial_available_utc')
        _require(av is None and model['n']==0 or av is not None and _time(av,'coarse trial publication')<cutoff,'Unpublished coarse trial')
    if model['status']=='unsupported_training_support':
        required=('status','n','events','alpha','K') if hourly else ('status','n','events','features','maximum_trial_available_utc','trial_origin_keys_sha256','fit_cutoff_utc')
        _keys(model,required,'unsupported logistic');return
    fields=('status','n','events','alpha','beta','iterations','maximum_gradient','objective')
    fields+=('K','mean','scale','unweighted_likelihood') if hourly else ('features_mean','features_scale','features','maximum_trial_available_utc','trial_origin_keys_sha256','fit_cutoff_utc')
    _keys(model,fields,'logistic')
    _require(model['alpha']==(.001 if hourly else .01),'Changed logistic regularization')
    if hourly:_require(model['unweighted_likelihood'] is True,'Changed likelihood weights')
    _vector(model['mean' if hourly else 'features_mean'],6 if hourly else 2,'logistic mean')
    _vector(model['scale' if hourly else 'features_scale'],6 if hourly else 2,'logistic scale',positive=True)
    _vector(model['beta'],22 if hourly else 10,'logistic coefficients')
    _integer(model['iterations'],'logistic iterations',1)
    _require(model['iterations']<=(80 if hourly else 60),'Logistic iteration metadata')
    for k in ('maximum_gradient','objective'):_number(model[k],k)
    if model['status']=='fit':
        _require(model['n']>=(500 if hourly else 100) and min(model['events'],model['n']-model['events'])>=(20 if hourly else 10),'Logistic support counts')


def _state(snapshot, *, supported):
    try:return _state_fields(snapshot,supported=supported)
    except CheckpointContractError:raise
    except (KeyError,TypeError,ValueError,IndexError,OverflowError,RecursionError) as e:
        raise CheckpointContractError('Malformed annual checkpoint state: '+str(e)) from e


def _state_fields(snapshot, *, supported):
    _require(isinstance(snapshot,AnnualForecastSnapshot),'Expected an annual snapshot')
    p=_parse(snapshot.content);_require(_canonical(p).decode()==snapshot.content,'Noncanonical snapshot encoding')
    _keys(p,('protocol','schema','execution_role','final_execution_enabled','financing','evaluation_year','windows','feature_order','information_cutoff_exclusive','assets'),'snapshot')
    w=_development(p['evaluation_year'],p['financing'])
    _require(p['protocol']==PROTOCOL and p['schema']=='annual-forecast-snapshot-v1' and p['execution_role']=='development_only' and p['final_execution_enabled'] is False,'Snapshot protocol/role mismatch')
    _require(p['windows']==w.as_dict() and p['feature_order']==list(FEATURES['minimal']),'Calendar/feature order mismatch')
    _require(_time(p['information_cutoff_exclusive'],'information cutoff')==w.evaluation[0],'Information cutoff mismatch')
    _keys(p['assets'],ASSETS,'asset registry');failures=[]
    for symbol,a in p['assets'].items():
        _keys(a,('fitted','calibration_maps','calibration_sample_counts','finite_pre_evaluation_predictions','annual_KM8','fitting_identity','calibration_identity','annual_reference_identity','normalization_and_quantiles_identity'),'asset state')
        f=a['fitted'];_keys(f,('tree','matched_logistic','scoring_censor_km','empirical','manifest'),'fitting state')
        m=f['manifest'];_keys(m,('fit_cutoff_utc','feature_columns','raw_X_and_hour_sha256','response_sha256','unique_origins','trials','maximum_trial_publication_utc','origin_interval_keys_sha256','parameters','fit_start_utc','empirical_origins','empirical_origin_keys_sha256','scoring_censor_origins'),'fitting manifest')
        _require(_time(m['fit_start_utc'],'fit start')==w.fit[0] and _time(m['fit_cutoff_utc'],'fit cutoff')==w.fit[1],'Fitting bounds mismatch')
        _require(m['feature_columns']==list(FEATURES['minimal']) and m['parameters']=={'alpha':.001,'depth':3,'learning_rate':.1,'rounds':80,'maximum_hazard_hour':16},'Fitting parameter identity mismatch')
        for k in ('unique_origins','trials','empirical_origins','scoring_censor_origins'):_integer(m[k],k)
        _require(m['unique_origins']<=m['trials']<=m['unique_origins']*16,'Hourly trial counts')
        for k in ('raw_X_and_hour_sha256','response_sha256','origin_interval_keys_sha256','empirical_origin_keys_sha256'):_digest(m[k])
        _require(m['maximum_trial_publication_utc'] in ('NaT','nan') and m['trials']==0 or m['trials']>0 and w.fit[0]<_time(m['maximum_trial_publication_utc'],'trial publication')<w.fit[1],'Unpublished hourly fitting trial')
        _tree(f['tree'],m['trials']);_logistic(f['matched_logistic'],m['trials'])
        _require(f['tree']['status']!='fit' or m['unique_origins']>=100,'Tree unique-origin support')
        _require(f['tree']['events']==f['matched_logistic']['events'],'Paired responses differ')
        e=f['empirical'];_keys(e,('asset_km','magnitude_cutpoints','magnitude_curves','hazard_0.01'),'empirical state')
        _km(e['asset_km'],False);_km(f['scoring_censor_km'],True)
        _require(e['asset_km']['n']==m['empirical_origins'] and f['scoring_censor_km']['n']==m['scoring_censor_origins'],'KM origin identity mismatch')
        cuts=e['magnitude_cutpoints'];_require(isinstance(cuts,list) and len(cuts)<=4,'Magnitude cuts')
        _vector(cuts,len(cuts),'magnitude cuts');_require(all(x>=0 for x in cuts) and (np.diff(cuts)>0).all(),'Magnitude cut order')
        _require(isinstance(e['magnitude_curves'],list) and len(e['magnitude_curves'])==len(cuts)+1,'Missing magnitude reference')
        for curve in e['magnitude_curves']:_km(curve,False)
        _logistic(e['hazard_0.01'],cutoff=w.fit[1])
        expected=float(1-km_predict(e['asset_km'],[8])[0]) if e['asset_km']['status']=='fit' else None
        _require(a['annual_KM8']==expected,'Annual KM8 reference differs')
        identities={'fitting_identity':f,'calibration_identity':a['calibration_maps'],
            'annual_reference_identity':{'empirical':e,'censor':f['scoring_censor_km']},
            'normalization_and_quantiles_identity':{'mean':f['matched_logistic'].get('mean'),'scale':f['matched_logistic'].get('scale'),'tree_cuts':f['tree'].get('cuts')}}
        for key,body in identities.items():_digest(a[key]);_require(a[key]==identity(body),key+' mismatch')
        _keys(a['finite_pre_evaluation_predictions'],RAW,'calibration forecast counts')
        calhours=int((w.calibration[1]-w.calibration[0])/pd.Timedelta(hours=1))
        for count in a['finite_pre_evaluation_predictions'].values():_integer(count,'calibration forecasts');_require(count<=calhours,'Calibration clock count')
        _keys(a['calibration_sample_counts'],CALIBRATED,'calibration sample counts');_keys(a['calibration_maps'],CALIBRATED,'calibration model registry')
        for name,parent in CALIBRATED.items():
            count=a['calibration_sample_counts'][name];_integer(count,'calibration samples')
            _require(count<=a['finite_pre_evaluation_predictions'][parent],'Calibration samples exceed predictions')
            c=a['calibration_maps'][name]
            base=('status','n','events','calibration_source','fit_start_utc','fit_end_exclusive','maximum_label_publication_utc','unique_asset_origins','training_keys_sha256')
            _require(c.get('status') in ('fit','unsupported_calibration_support','unsupported_calibration_numerical'),'Unknown calibration status')
            _keys(c,base,'Platt map',optional=('intercept','slope','objective','iterations','monotone','finite_logit_clipping_inputs'))
            for k in ('n','events','unique_asset_origins'):_integer(c[k],k)
            _require(c['events']<=c['n']==c['unique_asset_origins'],'Calibration used counts')
            _require(c['calibration_source'] in ('asset','pooled_primary_assets'),'Unknown calibration source')
            used=count if c['calibration_source']=='asset' else sum(p['assets'][s]['calibration_sample_counts'][name] for s in ASSETS)
            _require(c['n']==used,'Calibration pool/asset identity mismatch')
            _require(_time(c['fit_start_utc'],'calibration start')==w.calibration[0] and _time(c['fit_end_exclusive'],'calibration cutoff')==w.calibration[1],'Calibration bounds mismatch')
            _digest(c['training_keys_sha256'])
            _require(c['maximum_label_publication_utc'] in ('NaT','nan') and c['n']==0 or c['n']>0 and w.calibration[0]<=_time(c['maximum_label_publication_utc'],'calibration publication')<w.calibration[1],'Unpublished calibration outcome')
            if c['status']!='unsupported_calibration_support':
                _keys(c,base+('intercept','slope','objective','iterations','monotone','finite_logit_clipping_inputs'),'numerical Platt map')
                for k in ('intercept','slope','objective'):_number(c[k],k)
                _require(c['monotone'] is True and c['slope']>=0,'Nonmonotone calibration')
                _integer(c['iterations'],'Platt iterations',1);_require(c['iterations']<=100,'Platt iteration count')
                _integer(c['finite_logit_clipping_inputs'],'logit input count')
                if c['status']=='fit':
                    minimum,classes=(200,20) if c['calibration_source']=='asset' else (400,40)
                    _require(c['n']>=minimum and min(c['events'],c['n']-c['events'])>=classes,'Platt support counts')
        support={'raw_tree':f['tree']['status'],'raw_matched_logistic':f['matched_logistic']['status'],
            'raw_coarse_logistic':e['hazard_0.01']['status'],'unconditional_KM':e['asset_km']['status'],
            'magnitude_KM':e['asset_km']['status'],'rolling_magnitude_KM':e['asset_km']['status'],
            'scoring_censor_KM':f['scoring_censor_km']['status'],**{n:c['status'] for n,c in a['calibration_maps'].items()}}
        failures.extend(symbol+':'+n+':'+s for n,s in support.items() if s!='fit')
    if supported and failures:raise CheckpointUnsupportedError(failures)
    return p,w,failures


def validate_contexts(snapshot, contexts):
    """Causal hourly-input schema, without a filesystem data loader."""
    p,w,_=_state(snapshot,supported=False);_keys(contexts,ASSETS,'input asset scope')
    for symbol,f in contexts.items():
        _require(isinstance(f,pd.DataFrame) and f.columns.is_unique,'Expected unique input columns')
        require_training_times(f.index)
        _require(isinstance(f.index,pd.DatetimeIndex) and str(f.index.tz)=='UTC','Input index must be UTC')
        _require(BASE_INPUT<=set(f.columns)<=BASE_INPUT|OPTIONAL_INPUT,'Incomplete/unregistered causal input schema')
        for name,value in (('symbol',symbol),('financing',p['financing'])):
            if name in f:_require(f[name].notna().all() and f[name].eq(value).all(),'Missing/mixed input identity: '+name)
        for group in ({'bar_open_time_utc','bar_available_at_utc'},{'funding_known','last_funding_settlement_utc','last_funding_available_at_utc'}):
            _require(not (set(f.columns)&group) or group<=set(f.columns),'Incomplete publication fields')
        for name in ('theory_available','funding_known'):
            if name in f:_require(pd.api.types.is_bool_dtype(f[name]) and not f[name].isna().any(),'Boolean input required: '+name)
        for name in ('spot_close','perp_close','annual_rate','rho'):
            if name in f:
                _require(pd.api.types.is_numeric_dtype(f[name]) and not pd.api.types.is_bool_dtype(f[name]),'Numeric input required: '+name)
                values=f[name].to_numpy(float);_require(not np.isinf(values).any(),'Infinite input: '+name)
                if name.endswith('_close'):_require((values[np.isfinite(values)]>0).all(),'Nonpositive observed mark')
        for name in ('rate_available_at_utc','bar_open_time_utc','bar_available_at_utc','last_funding_settlement_utc','last_funding_available_at_utc'):
            if name in f:_require(isinstance(f[name].dtype,pd.DatetimeTZDtype) and str(f[name].dt.tz)=='UTC','UTC publication dtype required: '+name)
        active=f.theory_available
        _require(f.loc[active,['spot_close','perp_close','annual_rate','rate_available_at_utc']].notna().all().all(),'Declared theory availability lacks required inputs')
        causal_features(f,symbol,p['financing'])
    return {'status':'valid','assets':list(ASSETS),'evaluation_year':w.year}


def implementation_binding(root):
    """Hash only a fixed allowlist of scientific source/configuration files."""
    root=Path(root);frozen={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in FROZEN}
    _require(frozen==FROZEN,'Frozen authority changed')
    return {'frozen_sha256':frozen,'implementation_sha256':{n:hashlib.sha256((root/'src/spot_perp_basis'/n).read_bytes()).hexdigest() for n in CODE},
        'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__}}


def checkpoint_provenance(snapshot, contexts, *, source_kind, binding):
    """Bind fitting/calibration inputs only; never evaluation outcomes."""
    p,w,_=_state(snapshot,supported=False);validate_contexts(snapshot,contexts)
    _require(source_kind in ('synthetic','pre2023_development'),'Unregistered source kind')
    inputs={}
    for symbol,f in contexts.items():
        c=f.loc[f.index<w.evaluation[0],sorted(f.columns)]
        _require(len(c)>0,'Missing fitting/calibration input provenance')
        hashed=pd.util.hash_pandas_object(c,index=True,categorize=False).to_numpy(dtype='<u8')
        inputs[symbol]={'sha256':hashlib.sha256(hashed.tobytes()).hexdigest(),'rows':len(c),
            'start_utc':c.index.min().isoformat(),'end_exclusive_utc':(c.index.max()+pd.Timedelta(hours=1)).isoformat(),
            'columns_and_dtypes':[[n,str(c[n].dtype)] for n in c]}
    provenance={'source_kind':source_kind,'inputs':inputs,'binding':binding}
    _provenance(provenance,w)
    return provenance


def _provenance(value,w):
    _keys(value,('source_kind','inputs','binding'),'provenance')
    _require(value['source_kind'] in ('synthetic','pre2023_development'),'Source kind')
    b=value['binding'];_keys(b,('frozen_sha256','implementation_sha256','environment'),'implementation binding')
    _require(b['frozen_sha256']==FROZEN,'Scientific provenance mismatch');_keys(b['implementation_sha256'],CODE,'implementation source identities')
    for h in b['implementation_sha256'].values():_digest(h)
    _keys(b['environment'],('python','numpy','pandas'),'environment')
    for v in b['environment'].values():_require(isinstance(v,str) and bool(v),'Missing runtime version')
    _require(b['environment']=={'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__},'Checkpoint runtime environment differs')
    _keys(value['inputs'],ASSETS,'source provenance assets')
    for d in value['inputs'].values():
        _keys(d,('sha256','rows','start_utc','end_exclusive_utc','columns_and_dtypes'),'input identity');_digest(d['sha256']);_integer(d['rows'],'input rows',1)
        start,end=_time(d['start_utc'],'source start'),_time(d['end_exclusive_utc'],'source end')
        _require(START<=start<end<=w.evaluation[0] and (end-start)/pd.Timedelta(hours=1)==d['rows'],'Input provenance bounds/count mismatch')
        pairs=d['columns_and_dtypes'];_require(isinstance(pairs,list) and all(isinstance(x,list) and len(x)==2 and all(isinstance(v,str) for v in x) for x in pairs),'Input dtype manifest')
        names=[x[0] for x in pairs];_require(len(names)==len(set(names)) and BASE_INPUT<=set(names)<=BASE_INPUT|OPTIONAL_INPUT,'Input column manifest')


def _path(path,year,financing):
    _development(year,financing);path=Path(path)
    _require(path.name==f'annual_{year}_{financing}.checkpoint.json','Checkpoint filename must match year/control')
    _require(not any(p.is_symlink() for p in (path,*path.parents)),'Checkpoint path may not traverse symlinks')
    return path


def _read(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        s=os.fstat(f.fileno());_require(stat.S_ISREG(s.st_mode) and s.st_size<=16*1024*1024,'Expected bounded regular checkpoint JSON')
        return f.read()


def _utc_column(series, label, *, nullable=False):
    """Accept existing UTC Timestamp/NaT object columns without coercing times.

    Annual output concatenation creates object dtype for rolling publications
    and timezone-naive all-NaT columns for pending outcomes. There is no clock
    to reinterpret in a wholly missing nullable column. Non-null naive times
    and string times are rejected, rather than silently localized.
    """
    if isinstance(series.dtype,pd.DatetimeTZDtype):
        _require(str(series.dt.tz)=='UTC','UTC timezone required: '+label)
    elif nullable:
        known=series.dropna()
        _require(all(isinstance(t,pd.Timestamp) and t.tz is not None and str(t.tz)=='UTC' for t in known),'UTC Timestamp objects required: '+label)
    else:raise CheckpointContractError('UTC timestamp dtype required: '+label)
    _require(nullable or not series.isna().any(),'Missing timestamp: '+label)
    return pd.to_datetime(series,utc=True)


def checkpoint_bytes(snapshot, provenance):
    p,w,_=_state(snapshot,supported=True);_provenance(provenance,w)
    return _canonical({'format':FORMAT,'snapshot':p,'snapshot_sha256':snapshot.sha256,
        'model_registry':list(MODELS),'provenance':provenance})+b'\n'


def save_checkpoint(snapshot,path,*,provenance):
    """Atomic no-clobber save; identical bytes preserve content and mtime."""
    content=checkpoint_bytes(snapshot,provenance);p=snapshot.read()
    path=_path(path,p['evaluation_year'],p['financing']);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.annual-',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
        try:os.link(temp,path,follow_symlinks=False)
        except FileExistsError:
            _require(_read(path)==content,'Existing checkpoint conflicts; use a new version/directory')
    finally:Path(temp).unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def load_checkpoint(path,*,expected_sha256,expected_year,expected_financing,expected_provenance):
    """Verify trusted pins before returning the original immutable snapshot."""
    path=_path(path,expected_year,expected_financing);_digest(expected_sha256)
    w=_development(expected_year,expected_financing);_provenance(expected_provenance,w)
    content=_read(path);_require(hashlib.sha256(content).hexdigest()==expected_sha256,'Checkpoint checksum mismatch')
    record=_parse(content);_keys(record,('format','snapshot','snapshot_sha256','model_registry','provenance'),'checkpoint envelope')
    _require(_canonical(record)+b'\n'==content,'Noncanonical checkpoint encoding')
    _require(record['format']==FORMAT and record['model_registry']==list(MODELS),'Checkpoint format/model registry mismatch')
    snapshot=AnnualForecastSnapshot(_canonical(record['snapshot']).decode());p,w,_=_state(snapshot,supported=True)
    _require(record['snapshot_sha256']==snapshot.sha256,'Snapshot checksum mismatch')
    _require(p['evaluation_year']==expected_year and p['financing']==expected_financing,'Requested checkpoint identity differs')
    _provenance(record['provenance'],w);_require(record['provenance']==expected_provenance,'Checkpoint provenance mismatch')
    return snapshot


def validate_forecasts(snapshot, forecasts, *, origins=None, attached=False, as_of=None):
    """Validate the entire requested four-asset/nine-model output population."""
    p,w,_=_state(snapshot,supported=False)
    clock=pd.date_range(*w.evaluation,freq='h',inclusive='left',name='origin_time_utc') if origins is None else pd.DatetimeIndex(origins,name='origin_time_utc')
    require_training_times(clock)
    _require(len(clock)>0 and str(clock.tz)=='UTC' and clock.floor('h').equals(clock) and clock.min()>=w.evaluation[0] and clock.max()<w.evaluation[1],'Invalid requested forecast clock')
    _require(isinstance(forecasts,pd.DataFrame) and forecasts.columns.is_unique,'Forecast table schema')
    _require(set(forecasts)==set(OUTPUT+(OUTCOMES if attached else ())), 'Incomplete/unregistered forecast columns')
    nonnull=set(OUTPUT)-{'raw_event_probability','event_probability','rolling_latest_record_publication_utc'}
    _require(forecasts[list(nonnull)].notna().all().all(),'Missing forecast identity/status metadata')
    if attached:_require(forecasts[['event_observed','outcome_status','scoring_status']].notna().all().all(),'Missing outcome status metadata')
    keys=['symbol','model','origin_time_utc']
    _require(not forecasts.duplicated(keys).any(),'Duplicate forecast keys')
    expected=pd.MultiIndex.from_product([ASSETS,MODELS,clock],names=keys)
    observed=pd.MultiIndex.from_frame(forecasts[keys])
    _require(len(observed)==len(expected) and observed.difference(expected).empty and expected.difference(observed).empty,'Dropped/extra asset, model or origin rows')
    d=forecasts
    for name,value in [('financing',p['financing']),('evaluation_year',w.year),('horizon_hours',8),('snapshot_sha256',snapshot.sha256),('fit_cutoff_utc',w.fit[1]),('calibration_cutoff_utc',w.calibration[1])]:
        _require(d[name].eq(value).all(),'Forecast identity mismatch: '+name)
    for name in ('evaluation_year','horizon_hours'):
        _require(pd.api.types.is_integer_dtype(d[name]) and not pd.api.types.is_bool_dtype(d[name]),'Integer identity field: '+name)
    for name in ('origin_time_utc','forecast_available_at_utc','fit_cutoff_utc','calibration_cutoff_utc','rolling_latest_record_publication_utc'):
        _utc_column(d[name],name,nullable=name=='rolling_latest_record_publication_utc')
    _require(d.forecast_available_at_utc.eq(d.origin_time_utc).all(),'Forecast availability mismatch')
    for name in ('origin_features_available','low_fee_qualified'):
        _require(pd.api.types.is_bool_dtype(d[name]) and not d[name].isna().any(),'Boolean forecast field: '+name)
        _require(d.groupby(['symbol','origin_time_utc'])[name].nunique().eq(1).all(),'Inconsistent origin flags')
    for name in ('event_probability','raw_event_probability'):
        _require(pd.api.types.is_numeric_dtype(d[name]) and not pd.api.types.is_bool_dtype(d[name]),'Numeric forecast required')
        x=d[name].to_numpy(float);_require(not np.isinf(x).any() and ((x[np.isfinite(x)]>=0)&(x[np.isfinite(x)]<=1)).all(),'Invalid probability')
    for (symbol,model),g in d.groupby(['symbol','model'],sort=False):
        a=p['assets'][symbol];f=a['fitted'];values=g.event_probability.to_numpy(float);available=g.origin_features_available.to_numpy(bool)
        parent=CALIBRATED.get(model,model)
        base={'raw_tree':f['tree']['status'],'raw_matched_logistic':f['matched_logistic']['status'],
            'raw_coarse_logistic':f['empirical']['hazard_0.01']['status'],'unconditional_KM':f['empirical']['asset_km']['status'],'magnitude_KM':f['empirical']['asset_km']['status']}
        if model in CALIBRATED:
            c=a['calibration_maps'][model]
            status=c['status'] if base[parent]=='fit' else 'unsupported_parent_fit'
            _require(g.calibration_status.eq(c['status']).all() and g.calibration_source.eq(c['calibration_source']).all(),'Calibration identity/status differs')
            reference=d.loc[d.symbol.eq(symbol)&d.model.eq(parent)].set_index('origin_time_utc').event_probability.reindex(pd.DatetimeIndex(g.origin_time_utc)).to_numpy(float)
            _require(np.array_equal(reference,g.raw_event_probability.to_numpy(float),equal_nan=True),'Wrong raw parent probabilities')
            with np.errstate(invalid='ignore'):mapped=calibrate(c,reference)
            _require(np.array_equal(mapped,values,equal_nan=True),'Calibration application differs')
        else:
            status=np.where(np.isfinite(values),'fit','unsupported_reference') if model=='rolling_magnitude_KM' else base[model]
            _require(g.calibration_status.eq('not_applicable').all() and g.calibration_source.eq('not_applicable').all(),'Unregistered calibration metadata')
            _require(np.array_equal(values,g.raw_event_probability.to_numpy(float),equal_nan=True),'Raw probability orientation differs')
        _require(np.array_equal(g.model_support_status.to_numpy(),np.broadcast_to(status,len(g))),'Model support/status differs')
        target=np.where(np.broadcast_to(status,len(g))!='fit','unsupported_model_or_map',np.where(~available,'unavailable_origin_features',np.where(np.isfinite(values),'issued','unavailable_prediction')))
        _require(np.array_equal(target,g.forecast_status),'Invalid forecast failure/issuance status')
        _require(not np.isfinite(values[~available]).any(),'Forecast issued without causal features')
        _require(not np.isfinite(values[np.broadcast_to(status,len(g))!='fit']).any(),'Unsupported model has finite output')
        publication=_utc_column(g.rolling_latest_record_publication_utc,'rolling publication',nullable=True)
        _require((publication.isna() | publication.lt(g.origin_time_utc)).all() if model=='rolling_magnitude_KM' else publication.isna().all(),'Unpublished/wrong rolling record')
    if attached:_outcomes(d,pd.Timestamp(as_of) if as_of is not None else None)
    return {'status':'valid','rows':len(d),'asset_origins':len(clock)*len(ASSETS),'models':len(MODELS),'attached':attached}


def _outcomes(d,as_of):
    _require(as_of is not None,'Outcome validation needs an explicit observed-prefix cutoff')
    require_training_times(pd.DatetimeIndex([as_of]))
    for name in ('label_available_at_utc','label_record_available_at_utc'):
        publication=_utc_column(d[name],name,nullable=True)
        _require((publication.isna()|publication.le(as_of)).all(),'Future outcome publication')
        _require((publication.isna()|publication.ge(d.origin_time_utc)).all(),'Outcome publication precedes origin')
    _require(pd.api.types.is_bool_dtype(d.event_observed) and not d.event_observed.isna().any(),'Boolean event indicator')
    for name in ('followup_hours','event_by_8h'):_require(pd.api.types.is_numeric_dtype(d[name]),'Numeric outcome required')
    t=d.followup_hours.to_numpy(float);y=d.event_by_8h.to_numpy(float)
    _require(not np.isinf(t).any() and (t[np.isfinite(t)]>=0).all() and (t[np.isfinite(t)]%1==0).all(),'Invalid hourly followup')
    _require(not np.isinf(y).any() and np.isin(y[np.isfinite(y)],[0,1]).all(),'Invalid event-by-horizon label')
    _require(not (d.event_observed.to_numpy()&(~np.isfinite(t)|(t<1))).any(),'Event without observed future followup')
    expected=np.where(d.event_observed.to_numpy()&(t<=8),1.,np.where(t>=8,0.,np.nan))
    _require(np.array_equal(expected,y,equal_nan=True),'Event/followup/8h outcome mismatch')
    _require(d.event_by_8h.notna().eq(d.label_available_at_utc.notna()).all(),'Known label publication missing')
    known=d.event_by_8h.notna();finite=d.followup_hours.notna();observed=d.origin_time_utc.le(as_of)
    _require(finite.eq(d.label_record_available_at_utc.notna()).all(),'Terminal-record publication missing')
    future=d.origin_time_utc+pd.to_timedelta(d.followup_hours,unit='h')
    record=_utc_column(d.label_record_available_at_utc,'terminal publication',nullable=True)
    _require((~finite | (record.ge(future)&record.le(future+pd.Timedelta(hours=1)))).all(),'Terminal-record clock mismatch')
    _require((~d.event_observed | record.eq(future)).all(),'Event publication clock mismatch')
    horizon=d.origin_time_utc+pd.Timedelta(hours=8)
    published=_utc_column(d.label_available_at_utc,'label publication',nullable=True)
    due=future.where(d.event_observed & d.followup_hours.le(8),horizon)
    _require((~known | published.eq(due)).all(),'8h label publication clock mismatch')
    target=np.select([~observed,known,finite],['pending_observation','known_8h','early_censored'],default='unavailable_origin')
    _require(np.array_equal(target,d.outcome_status),'Outcome status mismatch')
    _require(not (finite&~observed).any(),'Future origin outcome attached')
    scoring=np.where(d.forecast_status.ne('issued'),'unavailable_prediction',np.where(finite,'eligible_pending_censor_KM_check','unavailable_outcome'))
    _require(np.array_equal(scoring,d.scoring_status),'Scoring availability status mismatch')
    # Label copies are common across models; scoring availability also depends
    # on each model's issued/unsupported status and is checked above per row.
    for name in (n for n in OUTCOMES if n!='scoring_status'):
        _require(d.groupby(['symbol','origin_time_utc'])[name].nunique(dropna=False).le(1).all(),'Different outcome copies across models')
