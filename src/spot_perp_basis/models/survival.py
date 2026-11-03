"""Historical KM, reverse-censor KM and causal rolling magnitude survival.

Rolling curves use only terminal records published strictly before the origin."""

import numpy as np
import pandas as pd


# KM and the discrete reverse-censor tie rule

def km_fit(duration,event,reverse=False, *, minimum_curve_origins=20):
    t=np.asarray(duration,float);e=np.asarray(event,bool);ok=np.isfinite(t)&(t>=0);t=t[ok].astype(int);e=e[ok]
    if len(t)<minimum_curve_origins:return {'status':'unsupported_training_support','n':len(t)}
    if reverse:t=np.where(e,t,t+1)
    n=np.bincount(t);risk=np.cumsum(n[::-1])[::-1]
    deaths=np.bincount(t,weights=(~e if reverse else e).astype(float),minlength=len(n))
    denominator=risk.astype(float)
    if reverse:denominator-=np.bincount(t,weights=e.astype(float),minlength=len(n))
    hazard=np.divide(deaths,denominator,out=np.zeros(len(n),float),where=denominator>0)
    surv=np.cumprod(1-hazard)
    return {'status':'fit','n':len(t),'events':int(e.sum()),'maximum_followup':int(t.max()),'survival':surv.tolist(),'reverse_censor':reverse}


def km_predict(model,h):
    h=np.asarray(h,int)
    if model['status']!='fit':return np.full(h.shape,np.nan)
    s=np.asarray(model['survival']);return s[np.clip(h,0,len(s)-1)]


# Published terminal-record rolling KM and frozen fallback

def rolling_curves(old,origins,fit,p, *, horizons=(1,2,4,8,16), rolling_days=90, rolling_minimum_local=100, rolling_minimum_asset=200):
    t=p.index.as_unit('ns').asi8;av=pd.DatetimeIndex(p.record_available_at_utc).as_unit('ns').asi8;duration=p.followup_hours.to_numpy();event=p.event_observed.to_numpy(bool)
    valid=p.survival_origin_eligible.to_numpy(bool)&np.isfinite(duration);cuts=np.array(fit['magnitude_cutpoints']);bins=np.searchsorted(cuts,old.abs_rho.reindex(p.index).to_numpy());current=old.abs_rho.reindex(origins).to_numpy();currentbin=np.searchsorted(cuts,current)
    result=np.full((len(origins),len(horizons)),np.nan);details=[];lookback=rolling_days*86400*10**9
    for j,origin in enumerate(origins.as_unit('ns').asi8):
        m=valid&(t>origin-lookback)&(t<origin)&(av<origin);local=m&(bins==currentbin[j]);n=int(local.sum());nasset=int(m.sum())
        if n>=rolling_minimum_local:curve=km_fit(duration[local],event[local]);source='rolling_magnitude'
        elif nasset>=rolling_minimum_asset:curve=km_fit(duration[m],event[m]);source='rolling_asset_fallback'
        else:curve=fit['magnitude_curves'][currentbin[j]];source='frozen_magnitude_fallback'
        result[j]=1-km_predict(curve,np.array(horizons));details.append({'origin_time_utc':origins[j],'local_terminal_records':n,'asset_terminal_records':nasset,'source':source,'maximum_used_record_publication_utc':pd.Timestamp(int(av[m].max()),tz='UTC') if m.any() else pd.NaT})
    result[~np.isfinite(current)]=np.nan
    return result,pd.DataFrame(details).set_index('origin_time_utc')
