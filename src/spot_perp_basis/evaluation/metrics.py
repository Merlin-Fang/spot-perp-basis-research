"""Forecast/IPCW, observable calibration and financial performance metrics.

Cash and position ledgers are supplied by the caller, not constructed here."""

import numpy as np
import pandas as pd

from ..backtest.accounting import COMPONENTS
from ..data.windows import MINUTE_NS
from ..models.baselines import fit_mean_reversion
from ..models.calibration import platt_fit
from ..models.survival import km_predict


# Forecast support, IPCW and observable calibration diagnostics

def auc_binary(y,score):
    y=np.asarray(y,int);n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return np.nan
    ranks=pd.Series(score).rank(method='average').to_numpy()
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def survival_metrics(survival,event,duration,h,censor, *, minimum_censor_survival=0.05):
    s=np.asarray(survival,float);event=np.asarray(event,bool);duration=np.asarray(duration,float)
    eligible=np.isfinite(s)&np.isfinite(duration);s=s[eligible];event=event[eligible];duration=duration[eligible]
    occurred=event&(duration<=h);alive=(duration>=h)&~occurred;known=occurred|alive
    truth=alive.astype(float);brier=float(np.mean((s[known]-truth[known])**2)) if known.any() else np.nan
    with np.errstate(divide='ignore',invalid='ignore'):
        logs=np.where(truth[known]>0,-np.log(s[known]),-np.log1p(-s[known]))
    loss=float(np.mean(logs)) if known.any() else np.nan
    ipcw=np.nan;weights=np.zeros(len(s));status='unsupported_censor_distribution'
    if len(s) and censor['status']=='fit':
        gh=float(km_predict(censor,np.array([h]))[0]);ge=km_predict(censor,np.maximum(duration.astype(int)-1,0))
        if h<=censor['maximum_followup'] and gh>=minimum_censor_survival and (not occurred.any() or ge[occurred].min()>=minimum_censor_survival):
            weights[occurred]=1/ge[occurred];weights[alive]=1/gh
            ipcw=float(np.sum(weights*(s-truth)**2)/len(s));status='training_censor_KM_assumption_required'
    return {'forecast_origins':len(s),'observable_outcomes':int(known.sum()),'censored_before_h':int((~known).sum()),
        'brier_observable':brier,'log_loss_observable':loss,'ipcw_brier':ipcw,'ipcw_status':status,
        'convergence_auc_observable':auc_binary(occurred[known],1-s[known]) if known.any() else np.nan,
        'mean_predicted_convergence':float(np.mean(1-s)) if len(s) else np.nan,'observed_convergence_fraction_known':float(occurred[known].mean()) if known.any() else np.nan},known,weights


def metrics(p,label,h,censor,diagnostic_calibration=True):
    p=np.asarray(p,float);event=label.event_observed.to_numpy(bool);duration=label.followup_hours.to_numpy(float);good=np.isfinite(p)&np.isfinite(duration);d=label.loc[good];pp=p[good]
    result,known,weights=survival_metrics(1-pp,d.event_observed,d.followup_hours,h,censor);truth=(d.event_observed.to_numpy(bool)&d.followup_hours.le(h).to_numpy()).astype(float)
    result['mean_predicted_probability']=float(pp.mean()) if len(pp) else np.nan;result['observed_frequency_known']=float(truth[known].mean()) if known.any() else np.nan
    bins=np.minimum((pp*10).astype(int),9);ece=0.;records=[]
    for j in range(10):
        m=known&(bins==j);n=int(m.sum());pred=float(pp[m].mean()) if n else np.nan;actual=float(truth[m].mean()) if n else np.nan
        if n:ece+=n*abs(pred-actual)
        records.append({'probability_bin':j,'forecast_count':int((bins==j).sum()),'known_count':n,'mean_predicted_known':pred,'observed_frequency_known':actual})
    result['ece_observable']=ece/known.sum() if known.any() else np.nan
    if diagnostic_calibration:
        fit=platt_fit(pp[known],truth[known],minimum=30,minimum_class=5,monotone=False);result['diagnostic_calibration_intercept']=fit.get('intercept',np.nan);result['diagnostic_calibration_slope']=fit.get('slope',np.nan)
    return result,records,known,weights


# Full-calendar financial risk and denominator support

def ratio(a,b,positive=True):
    if not np.isfinite(a) or not np.isfinite(b) or (b<=1e-12 if positive else abs(b)<=1e-12):return np.nan
    return float(a/b)


def drawdown(pnl,initial_equity):
    p=np.asarray(pnl,float);equity=initial_equity+np.cumsum(p);peak=np.maximum.accumulate(np.r_[initial_equity,equity])[1:];dollar=peak-equity
    return float(dollar.max()) if len(p) else np.nan,float(np.max(np.divide(dollar,peak,out=np.full(len(p),np.nan),where=peak>0))) if len(p) else np.nan


def risk_metrics(pnl,initial_equity,days=None, *, capital_usdt=3000000, tail_probability=0.05, minimum_daily_tail_sample=200, minimum_tail_observations=10, minimum_negative_tail_days=10):
    p=np.asarray(pnl,float);N=len(p);C0=capital_usdt;days=N if days is None else days
    if not N or not np.isfinite(p).all():raise ValueError('No missing/invented daily cash risk observations')
    r=p/C0;mean=float(r.mean());sd=float(r.std(ddof=1)) if N>1 else np.nan;down=float(np.sqrt(np.mean(np.minimum(r,0)**2)));dd,dfrac=drawdown(p,initial_equity);tail_n=int(np.ceil(tail_probability*N));tail=np.sort(p)[:tail_n];negative=int((p<0).sum());tail_supported=N>=minimum_daily_tail_sample and tail_n>=minimum_tail_observations and negative>=minimum_negative_tail_days
    worst7=float(pd.Series(p).rolling(7,min_periods=7).sum().min()) if N>=7 else np.nan
    result={'calendar_days':float(days),'daily_observations':N,'net_pnl':float(p.sum()),'fixed_C0_return':float(p.sum()/C0),'simple_calendar_annualized_C0_return':float(p.sum()/C0*365/days),
        'daily_marked_pnl_volatility_usdt':float(p.std(ddof=1)) if N>1 else np.nan,'daily_C0_volatility':sd,'daily_downside_deviation_C0':down,'daily_downside_deviation_usdt':down*C0,
        'maximum_daily_drawdown_usdt':dd,'maximum_daily_drawdown_fraction_source_equity':dfrac,'maximum_daily_drawdown_C0':dd/C0,'worst_daily_pnl':float(p.min()),'worst_observed_rolling7day_pnl':worst7,'daily_pnl_lower_5pct_quantile':float(np.quantile(p,.05)),
        'descriptive_5pct_tail_mean_pnl':float(tail.mean()),'expected_shortfall_5pct_loss_usdt':float(-tail.mean()) if tail_supported else np.nan,'tail_statistic_supported':bool(tail_supported),'tail_observation_count':tail_n,'negative_calendar_days':negative,'negative_day_fraction':negative/N,
        'sharpe_like_daily':ratio(np.sqrt(365)*mean,sd),'sortino_like_daily':ratio(np.sqrt(365)*mean,down),'return_to_max_drawdown':ratio(p.sum(),dd),'calmar_like_simple_calendar':ratio(p.sum()/C0*365/days,dd/C0),'initial_marked_equity_usdt':float(initial_equity)}
    return result


def classify(net_delta,downside_delta,exposure_delta):
    if abs(net_delta)<=1e-6 and abs(downside_delta)<=1e-12 and abs(exposure_delta)<=1e-6:return 'unchanged'
    if net_delta>1e-6 and downside_delta< -1e-12:return 'A_return_and_downside_improvement'
    if net_delta>1e-6 and downside_delta>1e-12:return 'B_return_gain_with_downside_increase'
    if net_delta< -1e-6 and (downside_delta< -1e-12 or exposure_delta< -1e-6):return 'C_profit_sacrifice_with_risk_or_exposure_reduction'
    if net_delta<=1e-6 and downside_delta>=-1e-12 and exposure_delta>=-1e-6:return 'D_no_return_downside_or_exposure_improvement'
    return 'risk_return_boundary_tradeoff'


# He calendar and active-hour summaries

def active_hour_metrics(returns, active, years, complete=True):
    """Section 4.4 scaling, caller-defined denominator; no idle dilution."""
    values = np.asarray(returns, float)[np.asarray(active, bool)]
    na = len(values)/years if years > 0 else np.nan
    result = {"active_hour_bins": len(values), "active_hours_per_365d_year": na,
              "paper_style_notional_annual_return": np.nan,
              "paper_style_notional_annual_volatility": np.nan,
              "paper_style_notional_sharpe": np.nan}
    if not complete or not len(values) or not np.isfinite(values).all():
        return result
    mu, sd = values.mean(), values.std(ddof=1) if len(values) > 1 else np.nan
    result.update(paper_style_notional_annual_return=mu*na,
                  paper_style_notional_annual_volatility=sd*np.sqrt(na),
                  paper_style_notional_sharpe=mu/sd*np.sqrt(na) if sd > 0 else np.nan)
    return result


def summarize(records, path, market, *, start, end, capital, notional):
    """Calendar PnL (including open MTM), daily capital and separate active hours."""
    cols = COMPONENTS+["price_convergence_pnl", "gross_pnl", "transaction_costs", "net_pnl"]
    path = path.loc[(path.index >= pd.Timestamp(start)) & (path.index < pd.Timestamp(end))]
    daily = path[cols].resample("1D").sum(min_count=1)
    expected = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end)-pd.Timedelta(days=1), freq="D")
    daily = daily.reindex(expected)
    if daily.net_pnl.isna().any():
        raise ValueError("Missing daily valuation; cannot invent an equity curve")
    daily.index.name = "valuation_date_utc"
    daily["equity"] = capital+daily.net_pnl.cumsum()
    daily["capital_return"] = daily.net_pnl/capital
    hourly = path[cols].resample("1h", closed="right", label="right").sum()
    hourly["active"] = path.active.resample("1h", closed="right", label="right").max().eq(True)
    # An entire missing hour still carries exposure if a position spans it.
    # Infer occupied bins from actual holding intervals, never from zero PnL.
    exposure = hourly.active.to_numpy(copy=True)
    for record in records.itertuples():
        first = hourly.index.searchsorted(pd.Timestamp(record.entry_utc), side="right")
        terminal = pd.Timestamp(record.valuation_utc)+pd.Timedelta(hours=1)
        last = hourly.index.searchsorted(terminal, side="left")
        exposure[first:last] = True
    hourly["active"] = exposure
    # Initiation/closing charges in a bin count as strategy activity as well.
    hourly["active"] |= hourly.transaction_costs.gt(0)
    hourly = hourly.loc[(hourly.index > pd.Timestamp(start)) & (hourly.index < pd.Timestamp(end))]
    known = pd.DatetimeIndex(market.frame.index)
    hourly["complete_endpoints"] = hourly.index.isin(known) & (hourly.index-pd.Timedelta(hours=1)).isin(known)
    hourly["notional_return"] = hourly.net_pnl/notional
    rows = []
    ex = pd.to_datetime(records.exit_utc, utc=True) if len(records) else pd.Series([], dtype="datetime64[ns, UTC]")
    en = pd.to_datetime(records.entry_utc, utc=True) if len(records) else ex
    for year in (2020, 2021, 2022):
        left = max(pd.Timestamp(start), pd.Timestamp(f"{year}-01-01", tz="UTC"))
        right = min(pd.Timestamp(end), pd.Timestamp(f"{year+1}-01-01", tz="UTC"))
        days = (right-left).total_seconds()/86400
        if days <= 0:
            continue
        d = daily.loc[daily.index.year == year]
        h = hourly.loc[hourly.index.year == year]
        completed = records.loc[ex.dt.year.eq(year)] if len(records) else records
        pp = path.loc[path.index.year == year]
        prior = capital+path.loc[path.index < left, "net_pnl"].sum()
        equity = np.r_[prior, pp.equity.to_numpy()]
        drawdown = equity/np.maximum.accumulate(equity)-1
        mean, sd = d.capital_return.mean(), d.capital_return.std(ddof=1)
        complete = bool(h.loc[h.active, "complete_endpoints"].all())
        row = {"year": year, "calendar_days": days,
            "completed_trade_count": len(completed),
            "entry_count": int(en.dt.year.eq(year).sum()),
            "average_holding_hours": completed.duration_hours.mean() if len(completed) else np.nan,
            "median_holding_hours": completed.duration_hours.median() if len(completed) else np.nan,
            "open_positions_at_year_end": int(((en < right) & (ex.isna() | (ex >= right))).sum()),
            "notional_usdt": notional, "reporting_capital_usdt": capital,
            **{c: float(d[c].sum()) for c in cols},
            "capital_gross_return": float(d.gross_pnl.sum()/capital),
            "capital_net_return": float(d.net_pnl.sum()/capital),
            "capital_calendar_annual_return": float(d.net_pnl.sum()/capital*365/days),
            "capital_calendar_annual_volatility": float(sd*np.sqrt(365)),
            "capital_calendar_sharpe": float(mean/sd*np.sqrt(365)) if sd > 0 else np.nan,
            "observed_minute_max_drawdown": float(drawdown.min()),
            "paper_style_hourly_endpoints_complete": complete,
            "missing_active_hour_endpoints": int((h.active & ~h.complete_endpoints).sum()),
            **active_hour_metrics(h.notional_return, h.active, days/365, complete)}
        rows.append(row)
    return pd.DataFrame(rows), daily, hourly


# Earlier equity-return convention

def return_metrics(equity, initial_capital, days):
    """Full-calendar daily returns, with initial cash before the first day."""
    x = np.asarray(equity, dtype=float)
    previous = np.r_[initial_capital, x[:-1]]
    returns = np.divide(x-previous, previous, out=np.full_like(x, np.nan), where=previous > 0)
    vol = np.std(returns, ddof=1) if len(x) > 1 else np.nan
    sharpe = float(np.mean(returns)/vol*np.sqrt(365)) if np.isfinite(vol) and vol > 0 else None
    net = float(x[-1]-initial_capital)
    return {"total_net_pnl": net, "simple_annual_return": net/initial_capital*365/days,
            "cagr": float((x[-1]/initial_capital)**(365/days)-1) if x[-1] > 0 else None,
            "daily_sharpe": sharpe}


# Earlier chronological basis pilot

def benchmark(targets, cutoff, end):
    """Single predeclared chronological calibration/evaluation, entirely training-era."""
    cutoff, end = pd.Timestamp(cutoff), pd.Timestamp(end)
    eligible = targets.dropna(subset=["basis_bps", "target_bps"])
    calibration = eligible.loc[(eligible.index < cutoff) & eligible.target_end_utc.lt(cutoff)]
    evaluation = eligible.loc[(eligible.index >= cutoff) & eligible.target_end_utc.lt(end)].copy()
    if evaluation.empty:
        raise ValueError("Empty pilot evaluation")
    model = fit_mean_reversion(calibration)
    evaluation["prediction_no_change_bps"] = 0.0
    evaluation["prediction_mean_reversion_bps"] = -model["kappa"] * (evaluation.basis_bps - model["mean_bps"])
    rows = []
    for name, column in (("no_change", "prediction_no_change_bps"), ("mean_reversion", "prediction_mean_reversion_bps")):
        error = evaluation.target_bps - evaluation[column]
        zero_mse = float(np.mean(evaluation.target_bps ** 2))
        mse = float(np.mean(error ** 2))
        rows.append({"model": name, "evaluation_samples": len(evaluation), "MSE_bps2": mse,
                     "MAE_bps": float(np.mean(np.abs(error))), "R2_OOS_vs_no_change": 1 - mse / zero_mse if zero_mse else None})
    return model, pd.DataFrame(rows), evaluation


def cost_comparison(target, scenarios):
    absolute = target.dropna().abs()
    rows = []
    for scenario in scenarios:
        cost = 2 * scenario["spot_fee_bps"] + 2 * scenario["perp_fee_bps"] + 4 * scenario["slippage_bps_per_fill"]
        rows.append({**scenario, "round_trip_bps": cost, "samples": len(absolute),
                     "share_abs_basis_move_exceeds_cost": float(absolute.gt(cost).mean()),
                     "share_short_basis_favorable_move_exceeds_cost": float((-target.dropna()).gt(cost).mean())})
    return pd.DataFrame(rows)
