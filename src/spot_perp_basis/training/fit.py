"""Registered risk sets, paired model fitting, forecast issuance and calibration.

Inputs and windows are explicit. No experiment registry, model search, annual
holdout schedule or research-directory IO occurs here. Empirical controls keep
their original two-feature training cohort, distinct from the six-feature fit."""

import hashlib

import numpy as np
import pandas as pd

from ..data.features import FEATURES
from ..data.windows import START, END, require_training_times
from ..models import logistic, trees
from ..models.calibration import platt_fit
from ..models.logistic import logistic_fit, hazard_predict
from ..models.survival import km_fit, km_predict


# Observed hourly likelihood trials

def hourly_risk_set(f,label,cohort,cutoff, *, maximum_hazard_hour=16):
    c=f.reindex(label.index);allowed=c[cohort+'_eligible36']&label.survival_origin_eligible&~label.boundary_window_purged&(label.index<cutoff);rows=[]
    for k in range(1,maximum_hazard_hour+1):
        av=pd.Series(label.index+pd.Timedelta(hours=k),index=label.index);m=allowed&label.followup_hours.ge(k)&av.lt(cutoff)
        event=label.event_observed&label.followup_hours.eq(k)
        if m.any():rows.append(pd.DataFrame({'origin_time_utc':label.index[m],'hazard_hour':k,'response':event.loc[m].astype(int).to_numpy(),'available_at_utc':av.loc[m].to_numpy()}))
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['origin_time_utc','hazard_hour','response','available_at_utc'])


# Paired model fitting and provenance

def array_hash(values):
    """Original 3.7 portable little-endian float64 matrix fingerprint."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<f8').tobytes()).hexdigest()


def fit_hourly(f, label, cohort, cutoff, *, alpha=0.001, depth=3,
               learning_rate=0.1, rounds=80):
    """Coordinate the matched 3.6/3.7 fits; retain their distinct support gates.

    The label builder supplies the registered 16h purge flag. This function
    checks it rather than inventing a new split or borrowing future context.
    """
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tz is None or not START < cutoff <= END:
        raise ValueError('Development-only fit cutoff required')
    require_training_times(f.index)
    require_training_times(label.index)
    expected_purge = label.index + pd.Timedelta(hours=16) >= cutoff
    if not np.array_equal(label.boundary_window_purged, expected_purge):
        raise ValueError('Labels must use the fitting prefix and 16h purge')
    if not label.label_prefix_end_exclusive.eq(cutoff).all():
        raise ValueError('Label prefix differs from the fit cutoff')
    risk = hourly_risk_set(f, label, cohort, cutoff)
    names = FEATURES[cohort]
    rows = f.reindex(pd.DatetimeIndex(risk.origin_time_utc))
    X = rows[list(names)].to_numpy()
    y = risk.response.to_numpy(float)
    bins = risk.hazard_hour.to_numpy(int) - 1
    raw = np.c_[X, risk.hazard_hour.to_numpy()]
    origins = int(risk.origin_time_utc.nunique())
    if len(risk) and not risk.available_at_utc.lt(cutoff).all():
        raise ValueError('Unpublished risk trial')
    if origins < 100:
        tree = {'status': 'unsupported_training', 'n': len(y),
                'events': int(y.sum()), 'depth': depth, 'learning_rate': learning_rate}
    else:
        tree = trees.train(raw, y, depth, learning_rate, max_rounds=rounds)
    linear = logistic.fit(X, bins, y, alpha, 16)
    scoring = (f.reindex(label.index)[cohort + '_eligible36']
               & ~label.boundary_window_purged & label.survival_origin_eligible)
    censor = km_fit(label.loc[scoring, 'followup_hours'],
                    label.loc[scoring, 'event_observed'], reverse=True)
    manifest = {
        'fit_cutoff_utc': str(cutoff), 'feature_columns': list(names),
        'raw_X_and_hour_sha256': array_hash(raw), 'response_sha256': array_hash(y),
        'unique_origins': origins, 'trials': len(risk),
        'maximum_trial_publication_utc': str(risk.available_at_utc.max()),
        'origin_interval_keys_sha256': hashlib.sha256(
            risk[['origin_time_utc', 'hazard_hour']].to_csv(index=False).encode()).hexdigest(),
        'parameters': {'alpha': alpha, 'depth': depth, 'learning_rate': learning_rate,
                       'rounds': rounds, 'maximum_hazard_hour': 16},
    }
    return {'tree': tree, 'matched_logistic': linear, 'scoring_censor_km': censor,
            'manifest': manifest}, risk


# Issue raw forecasts from a frozen fit

def predict_hourly(fitted, features, cohort):
    """Issue both raw curves on causal feature eligibility, independent of labels."""
    require_training_times(features.index)
    cutoff = pd.Timestamp(fitted['manifest']['fit_cutoff_utc'])
    if len(features) and features.index.min() < cutoff:
        raise ValueError('Prediction origins precede the frozen fit cutoff')
    names = fitted['manifest']['feature_columns']
    X = features[names].to_numpy()
    allowed = features[cohort + '_eligible36'].eq(True).to_numpy()
    survival, tree_hazard = trees.survival(fitted['tree'], X)
    probability, linear_hazard = logistic.predict(fitted['matched_logistic'], X)
    out = pd.DataFrame(index=features.index)
    for h in (1, 2, 4, 8, 16):
        out[f'tree__{h}h'] = np.where(allowed, 1-survival[:, h-1], np.nan)
        out[f'matched_logistic__{h}h'] = np.where(allowed, probability[:, h-1], np.nan)
    out['feature_eligible'] = allowed
    out['fit_cutoff_utc'] = cutoff
    return out, {'tree': np.where(allowed[:, None], tree_hazard, np.nan),
                 'matched_logistic': np.where(allowed[:, None], linear_hazard, np.nan)}


# Past published calibration samples and asset/pool fallback

def calibration_sample(prediction, context, label, symbol, begin, end):
    """Select exactly the 3.6 known-outcome calibration population."""
    begin, end = pd.Timestamp(begin), pd.Timestamp(end)
    if not START <= begin < end <= END:
        raise ValueError('Calibration must stay within the development window')
    p = prediction.loc[(prediction.index >= begin) & (prediction.index < end)]
    c = context.reindex(p.index)
    labels = label.reindex(p.index)
    if not labels.label_prefix_end_exclusive.eq(end).all():
        raise ValueError('Calibration label prefix mismatch')
    if not (labels.boundary_window_purged.to_numpy()
            == (labels.index + pd.Timedelta(hours=16) >= end)).all():
        raise ValueError('Calibration labels require the registered purge')
    good = (p.notna() & c.low_fee_margin.gt(0) & ~labels.boundary_window_purged
            & labels.event_by_8h.notna() & labels.available_8h_utc.lt(end))
    return pd.DataFrame({
        'symbol': symbol, 'origin_time_utc': p.index[good],
        'prediction': p.loc[good].to_numpy(),
        'actual': labels.loc[good, 'event_by_8h'].to_numpy(),
        'label_available_utc': labels.loc[good, 'available_8h_utc'].to_numpy(),
    })


def fit_calibrators(samples, begin, end):
    """One model/control/cohort at a time, in caller-supplied asset order.

    Pooling is the original four-asset fallback, never across models or rates.
    The caller supplies the asset scope; this routine does not choose assets.
    """
    begin, end = pd.Timestamp(begin), pd.Timestamp(end)
    if not START <= begin < end <= END:
        raise ValueError('Development-only calibration window required')
    for symbol, sample in samples.items():
        if not sample.symbol.eq(symbol).all():
            raise ValueError('Calibration sample symbol mismatch')
        if (not sample.origin_time_utc.ge(begin).all()
                or not sample.origin_time_utc.lt(end).all()
                or not sample.label_available_utc.lt(end).all()
                or sample.duplicated(['symbol', 'origin_time_utc']).any()):
            raise ValueError('Duplicate, out-of-window or unpublished calibration sample')
    pooled = pd.concat(samples.values(), ignore_index=True)
    pool = platt_fit(pooled.prediction, pooled.actual, 400, 40)
    maps = {}
    for symbol, sample in samples.items():
        fitted = platt_fit(sample.prediction, sample.actual)
        source = 'asset'
        if fitted['status'] != 'fit':
            fitted = pool.copy()
            source = 'pooled_primary_assets'
        used = sample if source == 'asset' else pooled
        keys = '\n'.join(used.symbol.astype(str) + '|' + used.origin_time_utc.astype(str))
        fitted.update(calibration_source=source, fit_start_utc=str(begin),
                      fit_end_exclusive=str(end),
                      maximum_label_publication_utc=str(used.label_available_utc.max()),
                      unique_asset_origins=len(used),
                      training_keys_sha256=hashlib.sha256(keys.encode()).hexdigest())
        maps[symbol] = fitted
    return maps


# Original two-feature empirical/coarse controls

EMPIRICAL_FEATURES = {'minimal': ('rho', 'abs_rho'), 'rich': ('rho', 'abs_rho', 'rho_change1', 'spot_logreturn24_bps', 'perp_logreturn24_bps', 'spot_rv24_bps', 'basis_change_rv24_bps', 'known_funding_bps', 'mean_known_funding24_bps', 'log1p_spot_quote_volume24', 'log1p_perp_quote_volume24', 'nominal_next_funding_hours')}


def empirical_features(cohort):return list(EMPIRICAL_FEATURES['minimal'] if cohort=='minimal' else EMPIRICAL_FEATURES['rich'])


def empirical_eligible(c,cohort,survival=False):
    ok=c[cohort+'_eligible'].eq(True)&np.isfinite(c[empirical_features(cohort)]).all(axis=1)
    return ok&c.rho.ne(0) if survival else ok


def digest_index(index):return hashlib.sha256(pd.DatetimeIndex(index).as_unit('ns').asi8.tobytes()).hexdigest()


def survival_training(c,label,cohort):
    d=c.reindex(label.index);return empirical_eligible(d,cohort,True)&label.survival_origin_eligible&label.followup_hours.notna()


def fit_survival(c,label,cutoff,cohort,alphas,pooled, *, score_grid_hours=(1,2,4,8,16,32,64,128), minimum_magnitude_bin=100):
    d=c.reindex(label.index);base=survival_training(c,label,'minimal')&(label.index<cutoff)
    own=km_fit(label.loc[base,'followup_hours'],label.loc[base,'event_observed'])
    censor_mask=survival_training(c,label,cohort)&(label.index<cutoff)
    censor=km_fit(label.loc[censor_mask,'followup_hours'],label.loc[censor_mask,'event_observed'],reverse=True)
    cuts=np.unique(np.quantile(d.loc[base,'abs_rho'],[.2,.4,.6,.8])).tolist() if base.any() else []
    bins=np.searchsorted(cuts,d.abs_rho.to_numpy());curves=[]
    for j in range(len(cuts)+1):
        selected=base&(bins==j)
        curves.append(km_fit(label.loc[selected,'followup_hours'],label.loc[selected,'event_observed']) if selected.sum()>=minimum_magnitude_bin else own)
    out={'fit_cutoff_utc':str(cutoff),'asset_km':own,'pooled_km':pooled,'magnitude_cutpoints':cuts,'magnitude_curves':curves,
        'censor_km':censor,'cohort':cohort,'training_origins':int(censor_mask.sum()),'training_keys_sha256':digest_index(label.index[censor_mask]),
        'last_training_record_available_utc':str(label.loc[censor_mask,'record_available_at_utc'].max())}
    # Origin-only volatility scale, with a training-only single calibration.
    vol=d.basis_change_rv24_bps.to_numpy()*1095/10000
    good=base.to_numpy(bool)&np.isfinite(vol)&(vol>0)
    if good.sum()>=20:
        floor=float(np.quantile(vol[good],.1));scale=np.maximum(vol[good],floor)**2/(24*d.abs_rho.to_numpy()[good]**2+1e-24)
        curve=km_fit(label.loc[good,'followup_hours'],label.loc[good,'event_observed']);target=1-float(km_predict(curve,np.array([8]))[0])
        lo=0.;hi=1.
        while np.mean(1-np.exp(-hi*8*scale))<target and hi<1e8:hi*=2
        for _ in range(60):
            mid=(lo+hi)/2
            if np.mean(1-np.exp(-mid*8*scale))<target:lo=mid
            else:hi=mid
        out['volatility_heuristic']={'status':'fit','positive_rv_floor':floor,'multiplier':(lo+hi)/2,'n':int(good.sum()),'training_KM8_convergence':target}
    else:out['volatility_heuristic']={'status':'unsupported_training_support','n':int(good.sum())}
    x=[];bin_index=[];response=[];available=[];origin_rows=[];prev=0
    for j,h in enumerate(score_grid_hours):
        event=label.event_observed&label.followup_hours.gt(prev)&label.followup_hours.le(h)
        survivor=label.followup_hours.ge(h)&~(label.event_observed&label.followup_hours.le(h))
        av=pd.Series(label.index+pd.Timedelta(hours=h),index=label.index);av.loc[event]=label.loc[event,'event_time_utc']
        good=censor_mask&(event|survivor)&av.lt(cutoff)
        if good.any():
            x.append(d.loc[good,empirical_features(cohort)].to_numpy());bin_index.extend([j]*int(good.sum()));response.extend(event.loc[good].astype(int));available.extend(av.loc[good]);origin_rows.extend(label.index[good])
        prev=h
    X=np.vstack(x) if x else np.empty((0,len(empirical_features(cohort))))
    for a in alphas:
        model=logistic_fit(X,bin_index,response,a, score_grid_hours=score_grid_hours)
        model.update(features=empirical_features(cohort),maximum_trial_available_utc=str(max(available)) if available else None,
            trial_origin_keys_sha256=digest_index(origin_rows),fit_cutoff_utc=str(cutoff))
        if available:assert max(available)<cutoff
        out[f'hazard_{a:g}']=model
    return out


def predict_survival(c,origins,fit,cohort,alphas, *, score_grid_hours=(1,2,4,8,16,32,64,128)):
    d=c.reindex(origins);allowed=empirical_eligible(d,cohort,True);grid=np.array(score_grid_hours);J=len(grid)
    result={};support={};hazards={}
    for name in ['asset_km','pooled_km']:
        curve=fit[name];result[name]=np.broadcast_to(km_predict(curve,grid),(len(d),J)).copy();support[name]=curve['status']=='fit'
    bins=np.searchsorted(fit['magnitude_cutpoints'],d.abs_rho.to_numpy());v=np.full((len(d),J),np.nan)
    for j,curve in enumerate(fit['magnitude_curves']):v[bins==j]=km_predict(curve,grid)
    result['magnitude_km']=v;support['magnitude_km']=fit['asset_km']['status']=='fit'
    if cohort=='rich':
        m=fit['volatility_heuristic'];v=np.full((len(d),J),np.nan)
        if m['status']=='fit':
            vol=d.basis_change_rv24_bps.to_numpy()*1095/10000;scale=np.maximum(vol,m['positive_rv_floor'])**2/(24*d.abs_rho.to_numpy()**2+1e-24)
            v=np.exp(-m['multiplier']*scale[:,None]*grid[None,:])
        result['volatility_heuristic']=v;support['volatility_heuristic']=m['status']=='fit'
    for a in alphas:
        name=f'hazard_{a:g}';m=fit[name];surv,hazard=hazard_predict(m,d[empirical_features(cohort)], score_grid_hours=score_grid_hours)
        result[name]=surv;hazards[name]=np.where(allowed.to_numpy()[:,None],hazard,np.nan);support[name]=m['status']=='fit'
    result={name:np.where(allowed.to_numpy()[:,None],values,np.nan) for name,values in result.items()}
    return result,support,allowed,hazards
