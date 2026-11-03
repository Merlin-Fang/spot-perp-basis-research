"""Deterministic synthetic inputs; these are not historical market observations."""

import numpy as np
import pandas as pd

from spot_perp_basis.data.market import Market


def context(n=900):
    i = np.arange(n)
    clock = pd.date_range('2020-01-01', periods=n, freq='h', tz='UTC', name='origin_time_utc')
    rho = 2.7*np.sin(i*.49) + 1.3*np.cos(i*.131)
    rho[80] = 0
    rho[[j for j in [45, 46, 300, 555] if j < n]] = np.nan
    f = pd.DataFrame(index=clock)
    f['rho'] = rho
    f['abs_rho'] = np.abs(rho)
    f['rho_change1'] = f.rho.diff()
    f['theory_available'] = np.isfinite(rho)
    f['minimal_eligible'] = f.theory_available
    f['rich_eligible'] = f.theory_available & (i >= 24)
    f['spot_close'] = 100 + i*.01 + np.sin(i*.3)
    f['perp_close'] = f.spot_close*(1+np.nan_to_num(rho)/1095)
    f['annual_rate'] = .01 + np.sin(i*.02)*.001
    # Synthetic causal context inputs used by the original rich transformation.
    for j, name in enumerate(['spot_logreturn24_bps', 'perp_logreturn24_bps',
                             'spot_rv24_bps', 'basis_change_rv24_bps',
                             'known_funding_bps', 'mean_known_funding24_bps',
                             'log1p_spot_quote_volume24', 'log1p_perp_quote_volume24',
                             'nominal_next_funding_hours']):
        f[name] = 1 + (j+1)*.1 + np.cos(i*.1+j)*.2
    return f


def numerical_trials():
    rng = np.random.default_rng(7301)
    x = rng.normal(size=(2400, 6))
    x[:, 3] = 4.0  # Constant feature exercises the scale/cut conventions.
    hour = np.tile(np.arange(16), 150)
    eta = -.9 + .6*x[:, 0] - .2*x[:, 1] + .45*(x[:, 2] > 0) + hour*.025
    y = (rng.uniform(size=len(x)) < 1/(1+np.exp(-eta))).astype(float)
    return x, hour, y


def context_inputs():
    """Separate supplied source frames for the old IO-coupled context builder."""
    from spot_perp_basis.theory import theory_signal
    c = context(120)
    old = c[['spot_close','perp_close','annual_rate']].copy()
    old['rate_available_at_utc'] = old.index-pd.Timedelta(hours=1)
    old['signal_available'] = c.theory_available
    old['rho'] = np.nan
    good = old.signal_available
    old.loc[good,'rho'] = theory_signal(old.loc[good,'spot_close'],old.loc[good,'perp_close'],old.loc[good,'annual_rate'])[0]
    causal = c[['spot_logreturn24_bps','perp_logreturn24_bps','spot_rv24_bps',
                'basis_change_rv24_bps','mean_known_funding24_bps']].copy()
    causal['last_settled_funding_rate'] = .0001
    causal['spot_quote_volume24'] = 10000.
    causal['perp_quote_volume24'] = 12000.
    causal['last_funding_settlement_utc'] = causal.index.floor('8h')-pd.Timedelta(hours=8)
    causal['last_funding_available_at_utc'] = causal.last_funding_settlement_utc+pd.Timedelta(minutes=1)
    causal['funding_known'] = True
    causal['context_price_window_complete'] = np.arange(len(causal))>=24
    hourly = old[['spot_close','perp_close']].copy()
    for venue in ['spot','perp']:
        for field in ['open','high','low']:hourly[f'{venue}_{field}']=hourly[venue+'_close']
    hourly['last_funding_interval_hours']=8.
    hourly['paired_valid']=True
    minute = pd.DataFrame({'paired_valid':True,'active_sample':True},index=old.index)
    return old,hourly,causal,minute,old.spot_close.shift(1)


def cash_case():
    clock = pd.date_range('2020-01-02', periods=360, freq='min', tz='UTC', name='decision_time_utc')
    i = np.arange(len(clock))
    s = 100 + np.sin(i*.09)*2 + i*.003
    p = s*(1+.004*np.sin(i*.14))
    f = pd.DataFrame({'bar_open_time_utc': clock-pd.Timedelta(minutes=1)}, index=clock)
    for venue, v in [('spot', s), ('perp', p)]:
        f[venue+'_open'] = v-.01
        f[venue+'_close'] = v
        f[venue+'_volume'] = 10.
        f[venue+'_trades'] = 4
    f.loc[clock[52:61], ['spot_volume', 'perp_trades']] = 0
    # A missing interval must remain missing, including in financing/exposure.
    f = f.drop(clock[141:145])
    market = Market(f)
    origins = clock[[0, 40, 50, 90, 130, 170, 220, 300, 359]]
    obs = pd.DataFrame({'rho': [2, 1, -2, 0, 3, -3, 0, 2, -2],
                        'lambda': [1.0004]*9, 'annual_rate': [.01]*9,
                        'signal_available': [True]*9,
                        'rate_available_at_utc': [clock[0]-pd.Timedelta(days=1)]*9}, index=origins)
    et = clock[[30, 61, 91, 131, 201, 301]] + pd.Timedelta(milliseconds=5)
    row = np.searchsorted(market.times, et.as_unit('ns').asi8, side='right')-1
    events = pd.DataFrame({'settlement_time_utc': et,
                           'funding_rate': [.0001, -.0002, .0003, .0001, -.0001, .0002],
                           'perp_price_proxy': market.frame.perp_close.to_numpy()[row],
                           'perp_proxy_age_minutes': (et.as_unit('ns').asi8-market.times[row])/60e9})
    rates = pd.DataFrame({'observation_time_utc': [pd.Timestamp('2020-01-01', tz='UTC')],
                          'available_at_utc': [pd.Timestamp('2020-01-01', tz='UTC')],
                          'annual_rate': [.01]})
    return market, obs, events, rates
