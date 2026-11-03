"""Causal context and registered survival/carry predictor schemas.

No future outcomes, fill completeness or outcome availability select features."""

import numpy as np
import pandas as pd

from ..theory import compute_rho, cost_bound, theory_signal
from .funding import rates_asof
from .market import prefix
from .windows import MINUTE_NS, ns_index, require_training_times


# Supplied causal source context

def build_context(symbol,rate,old,h,causal):
    require_training_times(old.index);hh=h.reindex(old.index)
    c=pd.DataFrame(index=old.index.rename('origin_time_utc'));c['symbol']=symbol;c['financing']=rate
    for x in ['spot_close','perp_close','annual_rate','rate_available_at_utc']:c[x]=old[x]
    c['theory_available']=old.signal_available.eq(True)
    c['rho']=compute_rho(c.spot_close.to_numpy(),c.perp_close.to_numpy(),c.annual_rate.to_numpy(),c.theory_available)
    assert np.allclose(c.rho,old.rho,atol=1e-12,rtol=1e-12,equal_nan=True)
    c['abs_rho']=c.rho.abs();c['rho_change1']=c.rho.diff()
    need=['spot_logreturn24_bps','perp_logreturn24_bps','spot_rv24_bps','basis_change_rv24_bps','last_settled_funding_rate',
        'mean_known_funding24_bps','spot_quote_volume24','perp_quote_volume24','last_funding_settlement_utc','last_funding_available_at_utc','funding_known','context_price_window_complete']
    for name in need:c[name]=causal[name].reindex(c.index)
    c['known_funding_bps']=c.last_settled_funding_rate*1e4
    for venue in ('spot','perp'):c[f'log1p_{venue}_quote_volume24']=np.log1p(c[f'{venue}_quote_volume24'])
    interval=hh.last_funding_interval_hours
    elapsed=(c.index-pd.DatetimeIndex(c.last_funding_settlement_utc)).total_seconds()/3600
    c['nominal_next_funding_hours']=np.where(c.funding_known.eq(True)&interval.gt(0),interval-(elapsed%interval),np.nan)
    c['minimal_eligible']=c.theory_available&np.isfinite(c[['rho', 'abs_rho']]).all(axis=1)
    c['rich_eligible']=c.minimal_eligible&c.context_price_window_complete.eq(True)&np.isfinite(c[['rho', 'abs_rho', 'rho_change1', 'spot_logreturn24_bps', 'perp_logreturn24_bps', 'spot_rv24_bps', 'basis_change_rv24_bps', 'known_funding_bps', 'mean_known_funding24_bps', 'log1p_spot_quote_volume24', 'log1p_perp_quote_volume24', 'nominal_next_funding_hours']]).all(axis=1)
    for col in ['rate_available_at_utc','last_funding_available_at_utc']:
        known=c[col].notna();assert (c.loc[known,col]<=c.index[known]).all()
    return c


# Registered six-feature and rich survival cohorts

FEATURES = {'minimal': ('rho', 'abs_rho', 'proposed_direction', 'low_fee_margin', 'rho_change1', 'rho_minus_mean24'), 'rich': ('rho', 'abs_rho', 'proposed_direction', 'low_fee_margin', 'rho_change1', 'rho_minus_mean24', 'spot_logreturn24_bps', 'perp_logreturn24_bps', 'spot_rv24_bps', 'basis_change_rv24_bps', 'known_funding_bps', 'mean_known_funding24_bps', 'log1p_spot_quote_volume24', 'log1p_perp_quote_volume24', 'nominal_next_funding_hours', 'log_spot_price', 'log_perp_price', 'log_basis_bps', 'basis_change1_bps', 'annual_rate', 'annual_rate_change24', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'past24_theory_valid_fraction')}


def features(old, *, cohorts=("minimal", "rich")):
    f=old.copy();f['proposed_direction']=-np.sign(f.rho);f['low_fee_margin']=f.abs_rho-cost_bound(2.25,.18)
    mean=f.rho.where(f.theory_available).rolling('24h',min_periods=12,closed='right').mean();f['rho_minus_mean24']=f.rho-mean
    f['log_spot_price']=np.log(f.spot_close);f['log_perp_price']=np.log(f.perp_close);f['log_basis_bps']=1e4*(f.log_perp_price-f.log_spot_price);f['basis_change1_bps']=f.log_basis_bps.diff()
    f['annual_rate_change24']=f.annual_rate-f.annual_rate.shift(24)
    for name,value,period in [('hour',f.index.hour,24),('weekday',f.index.weekday,7)]:
        f[name+'_sin']=np.sin(2*np.pi*value/period);f[name+'_cos']=np.cos(2*np.pi*value/period)
    f['past24_theory_valid_fraction']=f.theory_available.astype(float).rolling('24h',min_periods=1,closed='right').mean()
    for cohort in cohorts:
        ok=f.theory_available&f.rho.ne(0)&np.isfinite(f[list(FEATURES[cohort])]).all(axis=1)
        if cohort=='rich':ok&=f.rich_eligible
        f[cohort+'_eligible36']=ok
    return f


# Theory computed from completed minute endpoints

def hourly_theory_observations(market, rates, start, end):
    require_training_times(market.frame.index)
    clock = pd.date_range(start, end, freq="h", inclusive="left")
    require_training_times(clock)
    source = market.frame[["spot_close", "perp_close"]].copy()
    source["active_sample"] = market.eligible
    frame = source.reindex(clock)
    frame.index.name = "decision_time_utc"
    r, available = rates_asof(clock, rates)
    frame["annual_rate"], frame["rate_available_at_utc"] = r, available
    frame["rho"], frame["lambda"] = np.nan, np.nan
    eligible = frame.active_sample.eq(True) & frame.annual_rate.notna()
    rho, hedge = theory_signal(frame.loc[eligible, "spot_close"], frame.loc[eligible, "perp_close"],
                               frame.loc[eligible, "annual_rate"])
    frame.loc[eligible, "rho"], frame.loc[eligible, "lambda"] = rho, hedge
    frame["signal_available"] = eligible
    return frame


# Earlier carry predictors; separate registered schemas

def origin_grid(cfg):
    grid = pd.date_range(cfg["origin_start"], cfg["train_end"], freq=cfg["origin_frequency"], inclusive="left")
    return grid[grid < pd.Timestamp(cfg["train_end"])]


def input_features(market, events, origins, cfg):
    """Compute only backward information. Future fill data never enters inputs."""
    available = ns_index(events.available_at_utc)
    settled = ns_index(events.settlement_time_utc)
    rates = events.funding_rate.to_numpy()*1e4
    if np.any(np.diff(available) <= 0) or np.any(available < settled):
        raise ValueError("Funding availability is not ordered and causal")
    b = market.basis("close")*1e4
    spot = market.frame.spot_close.to_numpy()
    perp = market.frame.perp_close.to_numpy()
    continuous = np.r_[False, np.diff(market.times) == MINUTE_NS]
    changes = {}
    for venue, prices in (("spot", spot), ("perp", perp)):
        delta = np.r_[np.nan, np.diff(np.log(prices))]
        delta[~continuous] = np.nan
        changes[venue] = delta
    records = []
    for number, origin in enumerate(origins):
        t = origin.value
        pos = int(np.searchsorted(market.times, t))
        exact = pos < len(market.times) and market.times[pos] == t
        row = {"decision_time_utc": origin, "year": origin.year, "origin_number": number,
               "primary_sparse": number % cfg["primary_nonoverlap_stride_days"] == 0,
               "origin_observed": exact, "origin_active": bool(exact and market.eligible[pos]),
               "price_input_available_at_utc": origin if exact else pd.NaT}
        end = int(np.searchsorted(available, t, side="right"))
        row["funding_events_known"] = end
        row["funding_last_bps"] = float(rates[end-1]) if end else np.nan
        row["funding_expanding_mean_bps"] = float(np.mean(rates[:end])) if end else np.nan
        row["funding_last_available_at_utc"] = pd.Timestamp(int(available[end-1]), tz="UTC") if end else pd.NaT
        row["funding_age_hours"] = (t-settled[end-1])/MINUTE_NS/60 if end else np.nan
        for days in cfg["funding_rolling_days"]:
            left = int(np.searchsorted(available, t-days*24*60*MINUTE_NS, side="right"))
            values = rates[left:end]
            enough = len(values) >= days*cfg["minimum_funding_events_per_day"]
            row[f"funding_mean_{days}d_bps"] = float(np.mean(values)) if enough else np.nan
            row[f"funding_n_{days}d"] = len(values)
        if exact:
            left = int(np.searchsorted(market.times, t-24*60*MINUTE_NS, side="right"))
            past24 = np.searchsorted(market.times, t-24*60*MINUTE_NS)
            past1 = np.searchsorted(market.times, t-60*MINUTE_NS)
            row.update(basis_bps=float(b[pos]), spot_close=float(spot[pos]), perp_close=float(perp[pos]),
                       basis_mean_24h_bps=float(np.mean(b[left:pos+1])),
                       basis_std_24h_bps=float(np.std(b[left:pos+1])),
                       basis_change_1h_bps=float(b[pos]-b[past1]) if past1 <= pos and market.times[past1] == t-60*MINUTE_NS else np.nan,
                       past_24h_coverage=len(b[left:pos+1])/1440)
            for venue, prices in (("spot", spot), ("perp", perp)):
                r = changes[venue][left:pos+1]
                enough = np.isfinite(r).sum() >= .99*1440
                row[f"{venue}_realized_vol_24h_bps"] = float(np.sqrt(np.nansum(r*r))*1e4) if enough else np.nan
                row[f"{venue}_return_24h_bps"] = (float(np.log(prices[pos]/prices[past24])*1e4)
                    if market.times[past24] == t-24*60*MINUTE_NS else np.nan)
                row[f"{venue}_volume_24h"] = float(market.frame[f"{venue}_volume"].iloc[left:pos+1].sum())
                row[f"{venue}_trades_24h"] = float(market.frame[f"{venue}_trades"].iloc[left:pos+1].sum())
            row["past_24h_inactive_fraction"] = float(np.mean(~market.eligible[left:pos+1]))
        row["funding_features_ready"] = bool(end and all(np.isfinite(row[f"funding_mean_{d}d_bps"])
                                                    for d in cfg["funding_rolling_days"]))
        records.append(row)
    return pd.DataFrame(records)


def historical_features(market, events, origins, cfg):
    """Vectorized backward windows, ending at completed origin, never future masks."""
    origins = pd.DatetimeIndex(origins).as_unit("ns")
    t, mt = ns_index(origins), market.times
    p = np.searchsorted(mt, t)
    bounded = np.minimum(p, len(mt)-1)
    exact = (p < len(mt)) & (mt[bounded] == t)
    p = bounded
    frame = market.frame
    b = market.basis("close")*1e4
    result = pd.DataFrame({"decision_time_utc": origins, "year": origins.year,
        "origin_observed": exact, "origin_active": exact & market.eligible[p],
        "basis_bps": np.where(exact, b[p], np.nan),
        "spot_close": np.where(exact, frame.spot_close.to_numpy()[p], np.nan),
        "perp_close": np.where(exact, frame.perp_close.to_numpy()[p], np.nan)})
    result["price_input_available_at_utc"] = pd.to_datetime(np.where(exact, t, np.iinfo('int64').min), utc=True)
    available = ns_index(events.available_at_utc)
    settled = ns_index(events.settlement_time_utc)
    if np.any(np.diff(available) <= 0) or np.any(available < settled):
        raise ValueError("Noncausal funding availability")
    rates = events.funding_rate.to_numpy()*1e4
    e = np.searchsorted(available, t, side="right")
    last = np.maximum(0, e-1)
    result["funding_last_bps"] = np.where(e > 0, rates[last], np.nan)
    result["funding_last_available_at_utc"] = pd.to_datetime(np.where(e > 0, available[last], np.iinfo('int64').min), utc=True)
    result["funding_age_hours"] = np.where(e > 0, (t-settled[last])/MINUTE_NS/60, np.nan)
    for days in cfg["funding_lookbacks_days"]:
        a = np.searchsorted(available, t-days*1440*MINUTE_NS, side="right")
        n = e-a
        enough = n >= days*3
        for suffix, values in (("mean", rates), ("positive_fraction", rates > 0)):
            total = prefix(values)
            result[f"funding_{suffix}_{days}d" + ("_bps" if suffix == "mean" else "")] = np.where(
                enough, (total[e]-total[a])/np.maximum(1, n), np.nan)
        result[f"funding_n_{days}d"] = n
    continuous = np.r_[False, np.diff(mt) == MINUTE_NS]
    for hours in cfg["implemented_audit_lookbacks_hours"]:
        a = np.searchsorted(mt, t-hours*60*MINUTE_NS, side="right")
        z = np.searchsorted(mt, t-hours*60*MINUTE_NS)
        zb = np.minimum(z, len(mt)-1)
        n = p+1-a
        coverage = np.where(exact, n/(hours*60), 0.)
        result[f"past_{hours}h_coverage"] = coverage
        for suffix, values in (("mean", b), ("second_moment", b*b), ("inactive_fraction", ~market.eligible)):
            total = prefix(values)
            result[f"basis_{suffix}_{hours}h" if suffix != "inactive_fraction" else f"past_{hours}h_inactive_fraction"] = np.where(
                exact & (n > 0), (total[p+1]-total[a])/np.maximum(1, n), np.nan)
        result[f"basis_std_{hours}h_bps"] = np.sqrt(np.maximum(0, result.pop(f"basis_second_moment_{hours}h")-
                                                               result[f"basis_mean_{hours}h"]**2))
        result.rename(columns={f"basis_mean_{hours}h": f"basis_mean_{hours}h_bps"}, inplace=True)
        for venue in ("spot", "perp"):
            prices = frame[f"{venue}_close"].to_numpy()
            delta = np.r_[0., np.diff(np.log(prices))]
            delta[~continuous] = 0.
            rv = prefix(delta*delta)
            result[f"{venue}_rv_{hours}h"] = np.where(exact & (coverage >= .99), np.sqrt(rv[p+1]-rv[a]), np.nan)
            result[f"{venue}_return_{hours}h"] = np.where(exact & (mt[zb] == t-hours*60*MINUTE_NS),
                                                           np.log(prices[p]/prices[zb]), np.nan)
            for field in ("quote_volume", "trades", "taker_buy_quote_volume"):
                total = prefix(frame[f"{venue}_{field}"].to_numpy())
                result[f"{venue}_{field}_{hours}h"] = np.where(exact, total[p+1]-total[a], np.nan)
            totalq = result[f"{venue}_quote_volume_{hours}h"]
            result[f"{venue}_taker_imbalance_{hours}h"] = (2*result.pop(f"{venue}_taker_buy_quote_volume_{hours}h")/totalq-1).where(totalq > 0)
    return result


def matured_summaries(labels, origins, cfg):
    """A label joins a backward statistic only after its explicit availability."""
    source = labels.loc[labels.label_status == "observed"].copy()
    source = source.sort_values("label_available_at_utc")
    output = pd.DataFrame({"decision_time_utc": pd.DatetimeIndex(origins).as_unit("ns")})
    window = f"{cfg['historical_label_window_days']}D"
    minimum = cfg["minimum_matured_label_history"]
    for target in ("basis_pnl_bps", "funding_cash_bps", "collateral_ratio"):
        series = pd.Series(source[target].to_numpy(), index=pd.DatetimeIndex(source.label_available_at_utc).as_unit("ns"))
        stats = pd.DataFrame({"matured_at_utc": series.index})
        stats[target+"_last"] = series.ffill().to_numpy()
        stats[target+"_mean30"] = series.rolling(window, min_periods=minimum).mean().to_numpy()
        if target == "collateral_ratio":
            for quantile in cfg["quantiles"]:
                stats[f"risk_q{int(100*quantile)}_30d"] = series.rolling(window, min_periods=minimum).quantile(quantile).to_numpy()
        joined = pd.merge_asof(output[["decision_time_utc"]].sort_values("decision_time_utc"), stats,
                               left_on="decision_time_utc", right_on="matured_at_utc", direction="backward")
        for column in joined.columns.difference(["decision_time_utc"]):
            if column == "matured_at_utc":
                output[target+"_history_available_at_utc"] = joined[column].to_numpy()
            else:
                output[column] = joined[column].to_numpy()
    return output
