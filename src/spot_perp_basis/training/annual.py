"""Frozen annual forecast orchestration on supplied development frames only.

No data loader, final-mode switch, model search or financial policy lives here.
Calendar specifications may describe final years; numerical operations retain
the existing pre-2023 guards. Snapshots contain immutable canonical JSON.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from ..data.features import FEATURES, features
from ..data.labels import labels
from ..data.windows import START, END, annual_windows, require_training_times
from ..models import logistic, trees
from ..models.calibration import calibrate
from ..models.survival import km_predict, rolling_curves
from ..theory import theory_signal
from ..evaluation.survival_audit import matched_ipcw_scores, probability_audit
from ..evaluation.statistics import joint_calendar_interval, complete_matrix_mean
from .fit import fit_window, calibration_sample, fit_calibrators


ASSETS=('BTCUSDT','ETHUSDT','DOGEUSDT','ADAUSDT')
FINANCING=('zero_rate','nyfed_effr_lag7d')
MODELS=('raw_tree','raw_matched_logistic','unconditional_KM','magnitude_KM',
        'rolling_magnitude_KM','raw_coarse_logistic','calibrated_tree',
        'calibrated_matched_logistic','calibrated_coarse_logistic')
CALIBRATED={'calibrated_tree':'raw_tree','calibrated_matched_logistic':'raw_matched_logistic',
            'calibrated_coarse_logistic':'raw_coarse_logistic'}
PROTOCOL='final-survival-evaluation-v1.0.0'


def portable(value):
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [portable(v) for v in value]
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(pd.Timestamp,np.datetime64)):return pd.Timestamp(value).isoformat()
    return value


def identity(value):
    return hashlib.sha256(json.dumps(portable(value),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def causal_features(frame, symbol, financing):
    """Validate availability and build the unchanged minimal causal features."""
    if symbol not in ASSETS or financing not in FINANCING:raise ValueError('Unregistered asset/control')
    require_training_times(frame.index)
    f=frame.copy();f.index=f.index.tz_convert('UTC').rename('origin_time_utc')
    if not len(f):raise ValueError('Require an explicit observed hourly clock')
    if not f.index.equals(pd.date_range(f.index.min(),f.index.max(),freq='h',name=f.index.name)) or not f.index.floor('h').equals(f.index):
        raise ValueError('Retain missing scheduled hours explicitly')
    for name,value in [('symbol',symbol),('financing',financing)]:
        if name in f and not f[name].eq(value).all():raise ValueError('Mixed asset/control context')
        f[name]=value
    for name in ['rate_available_at_utc','bar_available_at_utc','last_funding_available_at_utc']:
        if name in f:
            av=pd.to_datetime(f[name],utc=True)
            if (av.notna() & av.gt(f.index)).any():raise ValueError('Future input publication: '+name)
    if 'rate_available_at_utc' not in f:raise ValueError('Explicit financing availability required')
    if 'bar_open_time_utc' in f:
        bt=pd.to_datetime(f.bar_open_time_utc,utc=True)
        if (bt.notna() & (bt+pd.Timedelta(minutes=1)).ne(f.index)).any():
            raise ValueError('Primary prices must be immediately completed minute closes')
        if 'bar_available_at_utc' in f and not pd.to_datetime(f.bar_available_at_utc,utc=True).equals(bt+pd.Timedelta(minutes=1)):
            raise ValueError('Minute availability must be bar open plus one minute')
    if 'funding_known' in f:
        known=f.funding_known.eq(True)
        age=(f.index-pd.DatetimeIndex(pd.to_datetime(f.last_funding_settlement_utc,utc=True))).total_seconds()/3600
        if (known & ((age<0)|(age>12)|f.last_funding_available_at_utc.isna())).any():
            raise ValueError('Unknown/stale funding declared available')
        expected=pd.to_datetime(f.last_funding_settlement_utc,utc=True)+pd.Timedelta(minutes=1)
        if (known & pd.to_datetime(f.last_funding_available_at_utc,utc=True).ne(expected)).any():
            raise ValueError('Funding publication must retain settlement plus one minute')
    valid=f.theory_available.eq(True)&np.isfinite(f[['spot_close','perp_close','annual_rate']]).all(axis=1)&f.rate_available_at_utc.notna()
    rho=pd.Series(np.nan,index=f.index)
    if valid.any():rho.loc[valid]=theory_signal(f.loc[valid,'spot_close'],f.loc[valid,'perp_close'],f.loc[valid,'annual_rate'])[0]
    if 'rho' in f:
        if not np.allclose(f.rho,rho,equal_nan=True,atol=1e-12,rtol=1e-12):raise ValueError('Inconsistent theoretical rho')
    else:f['rho']=rho
    f['theory_available']=valid
    f['abs_rho']=f.rho.abs();f['rho_change1']=f.rho.diff()
    f['minimal_eligible']=valid&f.rho.notna()
    return features(f,cohorts=('minimal',))


def raw_probabilities(fitted, context):
    """Reuse the registered estimators; all returned values are event probabilities."""
    require_training_times(context.index)
    if len(context) and context.index.min()<pd.Timestamp(fitted['manifest']['fit_cutoff_utc']):
        raise ValueError('Prediction origin precedes fitting cutoff')
    x=context[list(FEATURES['minimal'])].to_numpy()
    s,q=trees.survival(fitted['tree'],x,80)
    p,lq=logistic.predict(fitted['matched_logistic'],x)
    probability_audit(q,s,1-s);probability_audit(lq,np.cumprod(1-lq,axis=1),p)
    e=fitted['empirical'];own=e['asset_km']
    result=pd.DataFrame({'raw_tree':1-s[:,7],'raw_matched_logistic':p[:,7],
                         'unconditional_KM':1-km_predict(own,[8])[0]},index=context.index)
    bins=np.searchsorted(e['magnitude_cutpoints'],context.abs_rho.to_numpy())
    magnitude=np.full(len(context),np.nan)
    for j,curve in enumerate(e['magnitude_curves']):magnitude[bins==j]=1-km_predict(curve,[8])[0]
    result['magnitude_KM']=magnitude
    coarse,_=logistic.hazard_predict(e['hazard_0.01'],context[['rho','abs_rho']].to_numpy())
    result['raw_coarse_logistic']=1-coarse[:,3]
    result.loc[~context.minimal_eligible36.eq(True),:]=np.nan
    return result


@dataclass(frozen=True)
class AnnualForecastSnapshot:
    content: str

    @property
    def sha256(self):return hashlib.sha256(self.content.encode()).hexdigest()

    def read(self):return json.loads(self.content)


def fit_development_year(contexts, evaluation_year, financing, *, progress=None):
    """Fit full 20/4 development windows; registered final years are disabled."""
    w=annual_windows(evaluation_year)
    if w.evaluation[1]>END or w.fit[0]<START:
        raise PermissionError('Final-period numerical execution is disabled; calendar metadata only')
    if set(contexts)!=set(ASSETS) or financing not in FINANCING:
        raise ValueError('Require the frozen four-asset scope and one financing control')
    prepared={};fitted={};raw={};samples={name:{} for name in CALIBRATED.values()}
    for symbol in ASSETS:
        frame=contexts[symbol];require_training_times(frame.index)
        # No evaluation values enter even validation/preprocessing of the fit snapshot.
        f=causal_features(frame.loc[frame.index<w.calibration[1]],symbol,financing)
        prepared[symbol]=f
        fitted[symbol],_=fit_window(f,*w.fit)
        c=f.loc[(f.index>=w.calibration[0])&(f.index<w.calibration[1])]
        raw[symbol]=raw_probabilities(fitted[symbol],c)
        label=labels(f,*w.calibration)
        for name in samples:
            samples[name][symbol]=calibration_sample(raw[symbol][name],c,label,symbol,*w.calibration)
        if progress:progress('fit_and_calibration_samples',symbol)
    maps={name:fit_calibrators(sample,*w.calibration) for name,sample in samples.items()}
    assets={}
    for symbol in ASSETS:
        asset_maps={name:maps[parent][symbol] for name,parent in CALIBRATED.items()}
        counts={name:len(samples[parent][symbol]) for name,parent in CALIBRATED.items()}
        assets[symbol]={'fitted':fitted[symbol],'calibration_maps':asset_maps,
            'calibration_sample_counts':counts,'finite_pre_evaluation_predictions':{n:int(raw[symbol][n].notna().sum()) for n in raw[symbol]},
            'annual_KM8':float(1-km_predict(fitted[symbol]['empirical']['asset_km'],[8])[0]),
            'fitting_identity':identity(fitted[symbol]),'calibration_identity':identity(asset_maps),
            'annual_reference_identity':identity({'empirical':fitted[symbol]['empirical'],
                                                'censor':fitted[symbol]['scoring_censor_km']}),
            'normalization_and_quantiles_identity':identity({'mean':fitted[symbol]['matched_logistic'].get('mean'),
                'scale':fitted[symbol]['matched_logistic'].get('scale'),'tree_cuts':fitted[symbol]['tree'].get('cuts')})}
    payload={'protocol':PROTOCOL,'schema':'annual-forecast-snapshot-v1','execution_role':'development_only',
        'final_execution_enabled':False,'financing':financing,'evaluation_year':evaluation_year,
        'windows':w.as_dict(),'feature_order':list(FEATURES['minimal']),
        'information_cutoff_exclusive':w.evaluation[0].isoformat(),'assets':assets}
    return AnnualForecastSnapshot(json.dumps(portable(payload),sort_keys=True,separators=(',',':'),allow_nan=False))


def _snapshot(snapshot):
    p=snapshot.read();w=annual_windows(p['evaluation_year'])
    if p['protocol']!=PROTOCOL or p['windows']!=portable(w.as_dict()) or p['feature_order']!=list(FEATURES['minimal']):
        raise ValueError('Snapshot design/window mismatch')
    if p['final_execution_enabled'] or w.evaluation[1]>END:
        raise PermissionError('Final forecast execution remains disabled')
    if set(p['assets'])!=set(ASSETS) or p['financing'] not in FINANCING:
        raise ValueError('Snapshot scope mismatch')
    for a in p['assets'].values():
        f=a['fitted'];m=f['manifest']
        if pd.Timestamp(m['fit_start_utc'])!=w.fit[0] or pd.Timestamp(m['fit_cutoff_utc'])!=w.fit[1]:
            raise ValueError('Snapshot fit bounds mismatch')
        if identity(f)!=a['fitting_identity'] or identity(a['calibration_maps'])!=a['calibration_identity']:
            raise ValueError('Snapshot model/map identity mismatch')
        if any(pd.Timestamp(c['fit_end_exclusive'])!=w.calibration[1] for c in a['calibration_maps'].values()):
            raise ValueError('Snapshot calibration cutoff mismatch')
    return p,w


def issue_forecasts(snapshot, contexts, *, origins=None):
    """Issue every scheduled model/origin row, independently of future labels.

    Rolling KM is the registered terminal-record estimator. Its ledger contains
    publication times and is filtered strictly before each origin by the reused
    source implementation. Raw forecasts never consume that ledger.
    """
    p,w=_snapshot(snapshot)
    if set(contexts)!=set(ASSETS):raise ValueError('Missing requested context; never drop an asset')
    clock=pd.date_range(*w.evaluation,freq='h',inclusive='left',name='origin_time_utc') if origins is None else pd.DatetimeIndex(origins,name='origin_time_utc')
    require_training_times(clock)
    if len(clock) and (clock.min()<w.evaluation[0] or clock.max()>=w.evaluation[1] or not clock.floor('h').equals(clock)):
        raise ValueError('Forecast clock outside the annual hourly evaluation interval')
    output=[]
    for symbol in ASSETS:
        f=causal_features(contexts[symbol],symbol,p['financing'])
        c=f.reindex(clock);allowed=c.minimal_eligible36.eq(True)
        a=p['assets'][symbol];fit=a['fitted'];r=raw_probabilities(fit,c)
        terminal=labels(f,f.index.min(),w.evaluation[1])
        v,detail=rolling_curves(f,clock,fit['empirical'],terminal,horizons=(8,))
        r['rolling_magnitude_KM']=np.where(allowed,v[:,0],np.nan)
        # Missing predictions remain NaN; the source logit/sigmoid may warn on
        # those intentional NaNs before its final availability mask.
        with np.errstate(invalid='ignore'):
            for name,parent in CALIBRATED.items():r[name]=calibrate(a['calibration_maps'][name],r[parent])
        statuses={'raw_tree':fit['tree']['status'],'raw_matched_logistic':fit['matched_logistic']['status'],
            'raw_coarse_logistic':fit['empirical']['hazard_0.01']['status'],
            **{n:fit['empirical']['asset_km']['status'] for n in ['unconditional_KM','magnitude_KM','rolling_magnitude_KM']},
            **{n:(a['calibration_maps'][n]['status'] if fit['tree' if parent=='raw_tree' else 'matched_logistic']['status']=='fit' else 'unsupported_parent_fit') for n,parent in CALIBRATED.items() if parent!='raw_coarse_logistic'}}
        statuses['calibrated_coarse_logistic']=a['calibration_maps']['calibrated_coarse_logistic']['status'] if statuses['raw_coarse_logistic']=='fit' else 'unsupported_parent_fit'
        for model in MODELS:
            parent=CALIBRATED.get(model,model)
            values=r[model].to_numpy(float)
            status=np.where(np.isfinite(values),'fit','unsupported_reference') if model=='rolling_magnitude_KM' else statuses[model]
            if np.isinf(values).any() or ((values[np.isfinite(values)]<0)|(values[np.isfinite(values)]>1)).any():
                raise ValueError('Invalid forecast probability')
            output.append(pd.DataFrame({'symbol':symbol,'financing':p['financing'],'evaluation_year':w.year,
                'origin_time_utc':clock,'horizon_hours':8,'model':model,
                'raw_event_probability':r[parent].to_numpy(),'event_probability':values,
                'forecast_available_at_utc':clock,'fit_cutoff_utc':w.fit[1],
                'calibration_cutoff_utc':w.calibration[1],'snapshot_sha256':snapshot.sha256,
                'origin_features_available':allowed.to_numpy(),'low_fee_qualified':c.low_fee_margin.gt(0).to_numpy(),
                'model_support_status':status,
                'calibration_status':a['calibration_maps'][model]['status'] if model in CALIBRATED else 'not_applicable',
                'calibration_source':a['calibration_maps'][model].get('calibration_source','unavailable') if model in CALIBRATED else 'not_applicable',
                'forecast_status':np.where(status!='fit','unsupported_model_or_map',
                    np.where(~allowed,'unavailable_origin_features',np.where(np.isfinite(values),'issued','unavailable_prediction'))),
                'rolling_latest_record_publication_utc':detail.maximum_used_record_publication_utc.reindex(clock).to_numpy() if model=='rolling_magnitude_KM' else pd.NaT}))
    return pd.concat(output,ignore_index=True)


def attach_outcomes(snapshot, forecasts, contexts, as_of):
    """Attach only outcomes from the observed prefix; never edit issuance rows."""
    p,w=_snapshot(snapshot);as_of=pd.Timestamp(as_of)
    require_training_times(pd.DatetimeIndex([as_of]))
    _forecast_identity(snapshot,forecasts,p,w)
    if any(n in forecasts for n in ['followup_hours','event_observed','label_available_at_utc']):
        raise ValueError('Outcomes already attached')
    output=[]
    for symbol in ASSETS:
        f=causal_features(contexts[symbol].loc[contexts[symbol].index<=as_of],symbol,p['financing'])
        end=min(w.evaluation[1],as_of+pd.Timedelta(nanoseconds=1))
        if end<=w.evaluation[0]:
            label=pd.DataFrame(columns=['followup_hours','event_observed','event_by_8h','available_8h_utc','record_available_at_utc'])
            label.index=pd.DatetimeIndex([],tz='UTC',name='origin_time_utc')
        else:label=labels(f,w.evaluation[0],end)
        d=forecasts.loc[forecasts.symbol.eq(symbol)].copy()
        ix=pd.DatetimeIndex(d.origin_time_utc);lab=label.reindex(ix)
        d['followup_hours']=lab.followup_hours.to_numpy()
        d['event_observed']=lab.event_observed.eq(True).to_numpy()
        d['event_by_8h']=lab.event_by_8h.to_numpy()
        d['label_available_at_utc']=lab.available_8h_utc.to_numpy()
        d['label_record_available_at_utc']=lab.record_available_at_utc.to_numpy()
        known=lab.event_by_8h.notna().to_numpy();finite=lab.followup_hours.notna().to_numpy()
        observed=ix<=as_of
        d['outcome_status']=np.select([~observed,known,finite],['pending_observation','known_8h','early_censored'],default='unavailable_origin')
        d['scoring_status']=np.where(d.forecast_status.ne('issued'),'unavailable_prediction',np.where(finite,'eligible_pending_censor_KM_check','unavailable_outcome'))
        publication=pd.to_datetime(d.label_available_at_utc,utc=True)
        if (publication.notna() & publication.gt(as_of)).any():raise ValueError('Unpublished outcome attachment')
        output.append(d)
    return pd.concat(output,ignore_index=True)


def _forecast_identity(snapshot, forecasts, p, w):
    """Keep asset/control/time/model keys bound to the immutable annual state."""
    if (not forecasts.snapshot_sha256.eq(snapshot.sha256).all()
            or not forecasts.financing.eq(p['financing']).all()
            or not forecasts.horizon_hours.eq(8).all()
            or not forecasts.symbol.isin(ASSETS).all()
            or not forecasts.model.isin(MODELS).all()):
        raise ValueError('Forecast/snapshot information identity mismatch')
    clock=pd.DatetimeIndex(forecasts.origin_time_utc)
    if (clock.tz is None or clock.hasnans or (clock<w.evaluation[0]).any()
            or (clock>=w.evaluation[1]).any() or not clock.floor('h').equals(clock)):
        raise ValueError('Forecast origins outside the annual hourly clock')
    if forecasts.duplicated(['symbol','financing','origin_time_utc','horizon_hours','model']).any():
        raise ValueError('Duplicate forecast keys')
    if (not forecasts.fit_cutoff_utc.eq(w.fit[1]).all()
            or not forecasts.calibration_cutoff_utc.eq(w.calibration[1]).all()
            or not forecasts.forecast_available_at_utc.eq(forecasts.origin_time_utc).all()):
        raise ValueError('Forecast publication/cutoff mismatch')


def evaluate_forecasts(snapshot, attached):
    """Canonical per-model and named paired panels; no comparator disappears."""
    p,w=_snapshot(snapshot);rows=[];differences=[];gain_parts=[];panels=[]
    _forecast_identity(snapshot,attached,p,w)
    for symbol in ASSETS:
        d=attached.loc[attached.symbol.eq(symbol)]
        if d.duplicated(['origin_time_utc','model','horizon_hours']).any():raise ValueError('Duplicate forecast keys')
        wide=d.pivot(index='origin_time_utc',columns='model',values='event_probability').reindex(columns=MODELS)
        original=d.drop_duplicates('origin_time_utc').set_index('origin_time_utc').reindex(wide.index)
        qualified=original.low_fee_qualified.eq(True)
        lab=original[['followup_hours','event_observed']].loc[qualified]
        censor=p['assets'][symbol]['fitted']['scoring_censor_km']
        for model in MODELS:
            score,_,_,_,_=matched_ipcw_scores(wide.loc[qualified,[model]],lab,censor)
            rows.append({'symbol':symbol,'financing':p['financing'],'year':w.year,'model':model,
                'requested_origins':len(wide),'finite_predictions':int(wide[model].notna().sum()),**score[model]})
        for name,models in [('primary',['raw_tree','raw_matched_logistic']),
            *[(f'primary_plus_{m}',['raw_tree','raw_matched_logistic',m]) for m in MODELS if m not in ['raw_tree','raw_matched_logistic']]]:
            score,loss,_,_,_=matched_ipcw_scores(wide.loc[qualified,models],lab,censor)
            panels.append({'symbol':symbol,'panel':name,'origins':len(loss),
                'origin_keys_sha256':hashlib.sha256(loss.index.as_unit('ns').asi8.tobytes()).hexdigest(),
                'scores':score})
            if name=='primary':
                gain=loss.raw_matched_logistic-loss.raw_tree
                point=float(gain.mean()) if len(gain) and gain.notna().all() else np.nan
                differences.append({'symbol':symbol,'financing':p['financing'],'year':w.year,'gain':point,'paired_origins':len(gain)})
                if np.isfinite(point):gain_parts.append(pd.DataFrame({'symbol':symbol,'origin_time_utc':gain.index,'gain':gain.to_numpy()}))
    records=pd.concat(gain_parts,ignore_index=True) if gain_parts else pd.DataFrame(columns=['symbol','origin_time_utc','gain'])
    return {'model_scores':rows,'matched_panels':panels,'primary_differences':differences,
        'annual_uncertainty':joint_calendar_interval(records,w.year,ASSETS),
        'annual_equal_asset':complete_matrix_mean(pd.DataFrame(differences),ASSETS,[w.year],p['financing']),
        'final_12_cell_aggregate':complete_matrix_mean(pd.DataFrame(differences),ASSETS,[2023,2024,2025],p['financing'])}
