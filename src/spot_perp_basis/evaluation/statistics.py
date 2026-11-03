"""Registered block uncertainty on explicit statistical or cash inputs.

Occupied statistical blocks and all-calendar financial blocks are distinct;
weighted cash paths are perturbations, not executable strategy replays."""

import numpy as np
import pandas as pd


# Registered block resampling; statistical and financial clocks stay distinct

def block_mean_interval(values_by_year, repetitions=1000, block=8, seed=0):
    """Stratified moving-block percentile interval for a paired mean loss difference."""
    rng = np.random.default_rng(seed)
    values = [np.asarray(x, dtype=float) for x in values_by_year if len(x)]
    if not values:
        return np.nan, np.nan
    draws = np.zeros(repetitions)
    n = sum(map(len, values))
    for array in values:
        width = min(block, len(array))
        starts = rng.integers(0, len(array)-width+1, size=(repetitions, int(np.ceil(len(array)/width))))
        positions = (starts[..., None]+np.arange(width)).reshape(repetitions, -1)[:, :len(array)]
        draws += array[positions].sum(axis=1)/n
    return tuple(float(x) for x in np.quantile(draws, [.025, .975]))


def block_interval(records,year, *, seed=20261102, bootstrap_replicates=1000):
    d=records.dropna(subset=['gain']);assets=sorted(d.symbol.unique());anchor=pd.Timestamp(f'{year}-01-01', tz='UTC');allbins=((pd.DatetimeIndex(d.origin_time_utc)-anchor).total_seconds()//(14*86400)).astype(int);occupied=sorted(set(allbins));lookup={v:j for j,v in enumerate(occupied)};s=np.zeros((len(assets),len(occupied)));n=np.zeros_like(s)
    for j,symbol in enumerate(assets):
        sub=d.loc[d.symbol.eq(symbol)];bins=((pd.DatetimeIndex(sub.origin_time_utc)-anchor).total_seconds()//(14*86400)).astype(int);b=np.array([lookup[x] for x in bins]);s[j]=np.bincount(b,weights=sub.gain,minlength=len(occupied));n[j]=np.bincount(b,minlength=len(occupied))
    total=n.sum(axis=1);point=np.mean(s.sum(axis=1)/total) if len(assets) else np.nan;row={'paired_origins':len(d),'assets':len(assets),'occupied_calendar_blocks':len(occupied),'macro_gain':point,'observation_weighted_gain':d.gain.mean(),'ci_low':np.nan,'ci_high':np.nan,'leave_one_block_out_min':np.nan,'leave_one_block_out_max':np.nan,'interval_status':'sparse_support'}
    if len(d)<30 or len(occupied)<8 or not len(assets) or (np.sum(n>0,axis=1)<8).any():return row
    rng=np.random.default_rng(seed);w=rng.multinomial(len(occupied),np.ones(len(occupied))/len(occupied),size=bootstrap_replicates);den=w@n.T;good=(den>0).all(axis=1)
    if good.mean()<.8:return row
    draws=np.mean((w[good]@s.T)/den[good],axis=1);loo=[]
    for j in range(len(occupied)):
        denom=total-n[:,j]
        if (denom>0).all():loo.append(float(np.mean((s.sum(axis=1)-s[:,j])/denom)))
    row.update(ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),leave_one_block_out_min=min(loo),leave_one_block_out_max=max(loo),interval_status='conditional_shared_calendar_development_diagnostic');return row


def block_weights(index,repetitions=None, *, anchor=pd.Timestamp("2022-01-01", tz="UTC"), block_days=14, seed=20261102, bootstrap_replicates=1000):
    blocks=((index-anchor).days//block_days).to_numpy();B=int(blocks.max())+1
    rng=np.random.default_rng(seed);R=bootstrap_replicates if repetitions is None else repetitions
    counts=np.array([np.bincount(x,minlength=B) for x in rng.integers(0,B,(R,B))]);return blocks,counts[:,blocks]


def weighted_risk(p,weights,seed, *, capital_usdt=3000000):
    p=np.asarray(p,float);w=np.asarray(weights,float);total=w.sum(axis=1);cash=w*p;equity=seed+cash.cumsum(axis=1);peaks=np.maximum.accumulate(np.c_[np.full(len(w),seed),equity],axis=1)[:,1:]
    return {'net_pnl':cash.sum(axis=1),'daily_downside_deviation_C0':np.sqrt((w*np.minimum(p/capital_usdt,0)**2).sum(axis=1)/total),'maximum_daily_drawdown_usdt':(peaks-equity).max(axis=1)}
