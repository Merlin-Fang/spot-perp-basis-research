"""He cash ledgers: fixed entry lambda, actual funding, financing and four-leg fees.

Reference and array-loop accountants are independent verified implementations.
Observed exposure preserves gaps and terminal unknown time."""

import numpy as np
import pandas as pd

from ..data.funding import rates_asof
from ..data.windows import START, END, MINUTE_NS, YEAR_NS, ns_index, stamp, require_training_times


# Signed funding cash and fill charges

def funding_payment(quantity, rate, price, perpetual_side):
    """Side +1 is long (pays positive funding); -1 is short (receives)."""
    if perpetual_side not in (-1, 1):
        raise ValueError("Perpetual side must be +1 or -1")
    return -perpetual_side * quantity * rate * price


def fill_cost(quantity, spot, perp, scenario):
    """One two-leg execution: fees and slippage separately, in USDT."""
    fees = quantity * (spot * scenario["spot_fee_bps"] + perp * scenario["perp_fee_bps"]) / 1e4
    slip = quantity * (spot + perp) * scenario["slippage_bps"] / 1e4
    return fees, slip


# Original reference He accountant

COMPONENTS = ["spot_pnl", "perp_pnl", "funding_pnl", "financing_cost", "entry_cost", "exit_cost"]


def selected_funding(events, entry, exit):
    require_training_times(events.settlement_time_utc)
    return events.loc[(events.settlement_time_utc > pd.Timestamp(entry)) &
                      (events.settlement_time_utc <= pd.Timestamp(exit))].copy()


def account_positions(market, positions, events, rates, *, notional, capital,
                      spot_fee_bps, perp_fee_bps):
    """Minute observed MTM, actual-event cash, fixed entry hedge, four fees.

    The output increments telescope across trades and calendar years; missing
    minutes are counted rather than fabricated. Open trades have no exit fee.
    """
    require_training_times(market.frame.index)
    require_training_times(events.settlement_time_utc)
    if notional <= 0 or capital <= 0:
        raise ValueError("Positive notional and reporting capital required")
    times, frame = market.times, market.frame
    n = len(times)
    inc = {key: np.zeros(n) for key in COMPONENTS}
    active = np.zeros(n, bool)
    records, ledgers, previous_exit = [], [], -1
    previous_open = False
    s = frame.spot_close.to_numpy(float)
    f = frame.perp_close.to_numpy(float)
    cs, cf = spot_fee_bps/1e4, perp_fee_bps/1e4
    for position in positions:
        en, ex = position["entry_row"], position["exit_row"]
        en_ns = int(market.opens[en])
        ex_ns = int(market.opens[ex]) if ex is not None else int(times[-1])
        if previous_open or en_ns < previous_exit or ex_ns <= en_ns:
            raise ValueError("Overlapping, duplicate, or reversed positions")
        if en_ns < START.value or ex_ns >= END.value:
            raise ValueError("Trade outside 2020–2022")
        if pd.Timestamp(position["entry_signal_utc"]).value+MINUTE_NS > en_ns:
            raise ValueError("Entry is not later than the completed signal")
        previous_exit, previous_open = ex_ns, ex is None
        left = int(np.searchsorted(times, en_ns))
        last = int(np.searchsorted(times, ex_ns))
        if last >= n:
            raise ValueError("No terminal observed valuation")
        right = last+1
        vt = np.minimum(times[left:right], ex_ns)
        sp, fp = s[left:right].copy(), f[left:right].copy()
        s0, f0 = float(frame.spot_open.iloc[en]), float(frame.perp_open.iloc[en])
        sx = float(frame.spot_open.iloc[ex]) if ex is not None else sp[-1]
        fx = float(frame.perp_open.iloc[ex]) if ex is not None else fp[-1]
        if times[left] == en_ns:
            sp[0], fp[0] = s0, f0
        sp[-1], fp[-1] = sx, fx
        d = position["perpetual_side"]
        if d not in (-1, 1):
            raise ValueError("Invalid perpetual side")
        q = notional/f0
        qs = q*position["hedge_lambda"]
        spot_path = -d*qs*(sp-s0)
        perp_path = d*q*(fp-f0)
        inc["spot_pnl"][left:right] += np.diff(spot_path, prepend=0.)
        inc["perp_pnl"][left:right] += np.diff(perp_path, prepend=0.)
        previous_times = np.r_[en_ns, vt[:-1]]
        previous_spot = np.r_[s0, sp[:-1]]
        rr, _ = rates_asof(pd.to_datetime(previous_times, utc=True), rates)
        if not np.isfinite(rr).all():
            raise ValueError("Missing financing rate on an occupied interval")
        finance = -d*qs*previous_spot*rr*((vt-previous_times).astype(float)/YEAR_NS)
        inc["financing_cost"][left:right] += finance
        entry_s, entry_f = qs*s0*cs, q*f0*cf
        exit_s, exit_f = (qs*sx*cs, q*fx*cf) if ex is not None else (0., 0.)
        inc["entry_cost"][left] += entry_s+entry_f
        inc["exit_cost"][last] += exit_s+exit_f
        active[left:right] |= (vt > en_ns) & (times[left:right] <= ex_ns)
        held = selected_funding(events, pd.Timestamp(en_ns, tz="UTC"), pd.Timestamp(ex_ns, tz="UTC"))
        cash = funding_payment(q, held.funding_rate.to_numpy(float),
                               held.perp_price_proxy.to_numpy(float), d)
        if not np.isfinite(cash).all():
            raise ValueError("A held settlement has unknown cash notional")
        if "perp_proxy_age_minutes" in held and (held.perp_proxy_age_minutes > 1).any():
            raise ValueError("A held funding notional is stale")
        ix = np.searchsorted(times, ns_index(held.settlement_time_utc))
        np.add.at(inc["funding_pnl"], ix, cash)
        if len(held):
            held["trade_number"], held["perpetual_side"] = position["trade_number"], d
            held["perp_quantity"], held["funding_pnl"] = q, cash
            held["cash_booked_at_utc"] = pd.to_datetime(times[ix], utc=True)
            ledgers.append(held)
        observed = np.sum((times[left:right] > en_ns) & (times[left:right] <= ex_ns))
        missing = max(0, int((ex_ns-en_ns)//MINUTE_NS)-int(observed))
        spot_pnl, perp_pnl = float(spot_path[-1]), float(perp_path[-1])
        fund, financing = float(cash.sum()), float(finance.sum())
        gross = spot_pnl+perp_pnl+fund
        record = {key: value for key, value in position.items() if key not in ("entry_row", "exit_row")}
        record.update(status="closed" if ex is not None else "open_censored",
            direction="long_spot_short_perp" if d == -1 else "short_spot_long_perp",
            entry_utc=stamp(en_ns), exit_utc=stamp(ex_ns) if ex is not None else None,
            valuation_utc=stamp(ex_ns), duration_hours=(ex_ns-en_ns)/(60*MINUTE_NS),
            entry_wait_minutes=(en_ns-pd.Timestamp(position["entry_signal_utc"]).value)/MINUTE_NS,
            entry_spot=s0, entry_perp=f0, exit_or_mark_spot=sx, exit_or_mark_perp=fx,
            perp_quantity=q, spot_quantity=qs, entry_perp_notional_usdt=notional,
            reporting_capital_usdt=capital, spot_pnl=spot_pnl, perp_pnl=perp_pnl,
            price_convergence_pnl=spot_pnl+perp_pnl, funding_pnl=fund, financing_cost=financing,
            entry_spot_fee=entry_s, entry_perp_fee=entry_f, exit_spot_fee=exit_s, exit_perp_fee=exit_f,
            entry_cost=entry_s+entry_f, exit_cost=exit_s+exit_f,
            transaction_costs=entry_s+entry_f+exit_s+exit_f,
            gross_pnl=gross, net_pnl=gross-financing-entry_s-entry_f-exit_s-exit_f,
            funding_events=len(held), missing_held_minutes=missing,
            inactive_held_minutes=int((~market.eligible[left:right]).sum()),
            fill_count=4 if ex is not None else 2)
        records.append(record)
    path = pd.DataFrame(inc, index=frame.index)
    path["active"] = active
    path["price_convergence_pnl"] = path.spot_pnl+path.perp_pnl
    path["gross_pnl"] = path.price_convergence_pnl+path.funding_pnl
    path["transaction_costs"] = path.entry_cost+path.exit_cost
    path["net_pnl"] = path.gross_pnl-path.financing_cost-path.transaction_costs
    path["equity"] = capital+path.net_pnl.cumsum()
    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(
        columns=[*events.columns, "trade_number", "perpetual_side", "perp_quantity", "funding_pnl", "cash_booked_at_utc"])
    if len(ledger) and ledger.settlement_time_utc.duplicated().any():
        raise ValueError("A funding event was booked to more than one position")
    return pd.DataFrame(records), path, ledger


# Verified array-loop accountant

def fast_account(market,positions,events,rate,fees, *, perpetual_notional_usdt=1000000, reporting_capital_usdt=3000000):
    """Same observed-minute ledger as frozen accountant, with array-only loops."""
    require_training_times(market.frame.index);require_training_times(events.settlement_time_utc)
    t=market.times;n=len(t);s=market.frame.spot_close.to_numpy();f=market.frame.perp_close.to_numpy()
    so=market.frame.spot_open.to_numpy();fo=market.frame.perp_open.to_numpy()
    inc={k:np.zeros(n) for k in ['spot_pnl','perp_pnl','funding_pnl','financing_cost','entry_cost','exit_cost']};active=np.zeros(n,bool)
    r_av=ns_index(rate.available_at_utc);r_values=rate.annual_rate.to_numpy()
    prev_t=np.r_[t[0],t[:-1]];prev_r_idx=np.searchsorted(r_av,prev_t,side='right')-1
    previous_rates=np.where(prev_r_idx>=0,r_values[np.maximum(prev_r_idx,0)],np.nan)
    event_t=ns_index(events.settlement_time_utc);event_rate=events.funding_rate.to_numpy();marks=events.perp_price_proxy.to_numpy();age=events.perp_proxy_age_minutes.to_numpy()
    records=[];cash_rows=[];last_exit=-1;open_seen=False
    for p in positions:
        eni=p['entry_row'];exi=p['exit_row'];en=int(market.opens[eni]);ex=int(market.opens[exi]) if exi is not None else int(t[-1])
        if open_seen or en<last_exit or ex<=en or ex>=END.value:raise ValueError('Invalid overlapping/out-of-window position')
        last_exit=ex;open_seen=exi is None
        left=int(np.searchsorted(t,en));last=int(np.searchsorted(t,ex));right=last+1
        vt=np.minimum(t[left:right],ex);sp=s[left:right].copy();fp=f[left:right].copy();s0=float(so[eni]);f0=float(fo[eni])
        sx=float(so[exi]) if exi is not None else float(sp[-1]);fx=float(fo[exi]) if exi is not None else float(fp[-1])
        if t[left]==en:sp[0]=s0;fp[0]=f0
        sp[-1]=sx;fp[-1]=fx;d=p['perpetual_side'];q=perpetual_notional_usdt/f0;qs=q*p['hedge_lambda']
        sv=-d*qs*(sp-s0);fv=d*q*(fp-f0)
        inc['spot_pnl'][left:right]+=np.diff(sv,prepend=0.);inc['perp_pnl'][left:right]+=np.diff(fv,prepend=0.)
        rr=previous_rates[left:right].copy();ri=np.searchsorted(r_av,en,side='right')-1
        if ri<0:raise ValueError('Missing entry financing')
        rr[0]=r_values[ri]
        if not np.isfinite(rr).all():raise ValueError('Missing held financing')
        prev=np.r_[en,vt[:-1]];ps=np.r_[s0,sp[:-1]];finance=-d*qs*ps*rr*((vt-prev).astype(float)/YEAR_NS)
        inc['financing_cost'][left:right]+=finance
        entry_s=qs*s0*fees['spot_fee_bps']/1e4;entry_f=q*f0*fees['perp_fee_bps']/1e4
        exit_s=qs*sx*fees['spot_fee_bps']/1e4 if exi is not None else 0.;exit_f=q*fx*fees['perp_fee_bps']/1e4 if exi is not None else 0.
        inc['entry_cost'][left]+=entry_s+entry_f;inc['exit_cost'][last]+=exit_s+exit_f
        active[left:right]|=(vt>en)&(t[left:right]<=ex)
        lo=np.searchsorted(event_t,en,side='right');hi=np.searchsorted(event_t,ex,side='right');ei=np.arange(lo,hi)
        cash=-d*q*event_rate[lo:hi]*marks[lo:hi]
        if not np.isfinite(cash).all() or np.any(age[lo:hi]>1):raise ValueError('Unknown/stale held funding notional')
        ix=np.searchsorted(t,event_t[lo:hi]);np.add.at(inc['funding_pnl'],ix,cash)
        for j,c in zip(ei,cash):cash_rows.append({'event_row':int(j),'trade_number':p['trade_number'],'settlement_time_utc':events.settlement_time_utc.iloc[j],
            'funding_rate':float(event_rate[j]),'perp_price_proxy':float(marks[j]),'perp_proxy_age_minutes':float(age[j]),'funding_pnl':float(c),
            'cash_booked_at_utc':pd.Timestamp(t[np.searchsorted(t,event_t[j])],tz='UTC')})
        price=float(sv[-1]+fv[-1]);fund=float(cash.sum());fin=float(finance.sum());cost=entry_s+entry_f+exit_s+exit_f
        observed=int(((t[left:right]>en)&(t[left:right]<=ex)).sum());missing=max(0,int((ex-en)//MINUTE_NS)-observed)
        rec={k:v for k,v in p.items() if k not in ('entry_row','exit_row')}
        rec.update(status='closed' if exi is not None else 'open_censored',direction='long_spot_short_perp' if d==-1 else 'short_spot_long_perp',
            entry_utc=stamp(en),exit_utc=stamp(ex) if exi is not None else None,valuation_utc=stamp(ex),duration_hours=(ex-en)/(60*MINUTE_NS),
            entry_spot=s0,entry_perp=f0,exit_or_mark_spot=sx,exit_or_mark_perp=fx,perp_quantity=q,spot_quantity=qs,
            spot_pnl=float(sv[-1]),perp_pnl=float(fv[-1]),price_convergence_pnl=price,funding_pnl=fund,financing_cost=fin,
            entry_cost=entry_s+entry_f,exit_cost=exit_s+exit_f,entry_spot_fee=entry_s,entry_perp_fee=entry_f,exit_spot_fee=exit_s,exit_perp_fee=exit_f,
            transaction_costs=cost,gross_pnl=price+fund,net_pnl=price+fund-fin-cost,funding_events=len(ei),missing_held_minutes=missing,
            inactive_held_minutes=int((~market.eligible[left:right]).sum()),peak_observed_gross_leg_notional_usdt=float(np.max(np.r_[qs*sp+q*fp,qs*s0+q*f0])),
            observed_price_mae_usdt=float(max(0.,-np.min(np.r_[0.,sv+fv]))),entry_wait_minutes=(en-pd.Timestamp(p['entry_signal_utc']).value)/MINUTE_NS,
            reporting_capital_usdt=reporting_capital_usdt,entry_perp_notional_usdt=perpetual_notional_usdt)
        records.append(rec)
    rec=pd.DataFrame(records)
    if not len(rec):rec=pd.DataFrame(columns=['entry_utc','exit_utc','entry_signal_utc','valuation_utc','duration_hours','status','perpetual_side','direction','net_pnl','price_convergence_pnl','spot_pnl','perp_pnl','funding_pnl','financing_cost','entry_cost','exit_cost','transaction_costs','funding_events','peak_observed_gross_leg_notional_usdt','observed_price_mae_usdt'])
    path=pd.DataFrame(inc,index=market.frame.index);path['active']=active
    path['price_convergence_pnl']=path.spot_pnl+path.perp_pnl;path['transaction_costs']=path.entry_cost+path.exit_cost
    path['net_pnl']=path.price_convergence_pnl+path.funding_pnl-path.financing_cost-path.transaction_costs
    path['equity']=reporting_capital_usdt+path.net_pnl.cumsum()
    cash=pd.DataFrame(cash_rows,columns=['event_row','trade_number','settlement_time_utc','funding_rate','perp_price_proxy','perp_proxy_age_minutes','funding_pnl','cash_booked_at_utc'])
    if len(cash) and cash.event_row.duplicated().any():raise ValueError('Double funding ownership')
    for k in ['spot_pnl','perp_pnl','funding_pnl','financing_cost','entry_cost','exit_cost','transaction_costs','net_pnl']:
        if not np.isclose(rec[k].sum(),path[k].sum(),atol=1e-6,rtol=1e-9):raise ValueError('Trade/calendar mismatch: '+k)
    return rec,path,cash


# Left-mark exposure and explicit unknown intervals

HOUR = 3600 * 10**9


def exposure_panel(rec,market,start,end):
    """Left-observed marks across gaps; no extrapolation beyond final valuation."""
    clock=pd.date_range(start,end,freq='h',inclusive='left',name='hour_utc');N=len(clock);start_ns=start.value
    additive=['occupied_seconds','gross_exposure_usdt_hours','strict_minute_gross_usdt_hours','strict_active_minute_gross_usdt_hours','valued_held_seconds','strict_minute_seconds','gap_interval_seconds','missing_scheduled_seconds','inactive_left_mark_seconds','unvalued_terminal_held_seconds','turnover_usdt']
    a={k:np.zeros(N,float) for k in additive};peaks={k:np.zeros(N,float) for k in ['peak_spot_exposure_usdt','peak_perp_exposure_usdt','peak_gross_exposure_usdt']};observed=np.zeros(N,int)
    def add_intervals(left,right,weights):
        left=np.maximum(np.asarray(left,np.int64),start.value);right=np.minimum(np.asarray(right,np.int64),end.value);good=right>left
        if not good.any():return
        left=left[good];right=right[good];weights={k:np.broadcast_to(np.asarray(v,float),good.shape)[good] for k,v in weights.items()};ib=(left-start_ns)//HOUR;ie=(right-1-start_ns)//HOUR;simple=ib==ie;dt=(right-left)/1e9
        for key,value in weights.items():np.add.at(a[key],ib[simple],dt[simple]*value[simple])
        for j in np.flatnonzero(~simple):
            t=int(left[j]);terminal=int(right[j])
            while t<terminal:
                b=int((t-start_ns)//HOUR);u=min(terminal,start_ns+(b+1)*HOUR);seconds=(u-t)/1e9
                for key,value in weights.items():a[key][b]+=seconds*value[j]
                t=u
    times=market.times;spot=market.frame.spot_close.to_numpy();perp=market.frame.perp_close.to_numpy();eligible=market.eligible
    for row in rec.itertuples():
        en=pd.Timestamp(row.entry_utc).value;valuation=pd.Timestamp(row.valuation_utc).value;terminal=pd.Timestamp(row.exit_utc).value if row.status=='closed' else end.value
        add_intervals([en],[terminal],{'occupied_seconds':[1.]})
        if terminal>valuation:add_intervals([valuation],[terminal],{'unvalued_terminal_held_seconds':[1.]})
        left=np.searchsorted(times,en);last=np.searchsorted(times,valuation);vt=np.minimum(times[left:last+1],valuation);sp=spot[left:last+1].copy();fp=perp[left:last+1].copy();active=eligible[left:last+1].copy()
        if not len(vt):continue
        if vt[0]==en:sp[0]=row.entry_spot;fp[0]=row.entry_perp;active[0]=True
        sp[-1]=row.exit_or_mark_spot;fp[-1]=row.exit_or_mark_perp
        if row.status=='closed':active[-1]=True
        previous=np.r_[en,vt[:-1]];ps=np.r_[row.entry_spot,sp[:-1]];pf=np.r_[row.entry_perp,fp[:-1]];pa=np.r_[True,active[:-1]];dt=(vt-previous)/1e9;gross=row.spot_quantity*ps+row.perp_quantity*pf;strict=dt<=60.000001;strict_active=strict&pa&active
        add_intervals(previous,vt,{'gross_exposure_usdt_hours':gross/3600,'strict_minute_gross_usdt_hours':gross*strict/3600,'strict_active_minute_gross_usdt_hours':gross*strict_active/3600,'valued_held_seconds':np.ones(len(dt)),'strict_minute_seconds':strict.astype(float),'gap_interval_seconds':(~strict).astype(float),'missing_scheduled_seconds':np.divide(np.maximum(dt-60,0),dt,out=np.zeros(len(dt)),where=dt>0),'inactive_left_mark_seconds':(~pa).astype(float)})
        mt=np.r_[en,vt];ss=np.r_[row.entry_spot,sp]*row.spot_quantity;ff=np.r_[row.entry_perp,fp]*row.perp_quantity;ix=(mt-start_ns)//HOUR;valid=(ix>=0)&(ix<N)
        for name,value in [('peak_spot_exposure_usdt',ss),('peak_perp_exposure_usdt',ff),('peak_gross_exposure_usdt',ss+ff)]:np.maximum.at(peaks[name],ix[valid],value[valid])
        np.add.at(observed,ix[valid],1)
        for stamp,notional in [(en,row.spot_quantity*row.entry_spot+row.perp_quantity*row.entry_perp)]+([(valuation,row.spot_quantity*row.exit_or_mark_spot+row.perp_quantity*row.exit_or_mark_perp)] if row.status=='closed' else []):
            k=int((stamp-start_ns)//HOUR)
            if 0<=k<N:a['turnover_usdt'][k]+=notional
    result=pd.DataFrame(a,index=clock)
    for key,values in peaks.items():result[key]=np.where((observed==0)&(result.occupied_seconds>0),np.nan,values)
    result['observed_mark_records']=observed;result['observed_mean_gross_exposure_usdt']=np.divide(result.gross_exposure_usdt_hours,result.valued_held_seconds/3600,out=np.full(N,np.nan),where=result.valued_held_seconds>0)
    daily=result[additive].resample('D').sum()
    for key in peaks:daily[key]=result[key].resample('D').max()
    daily['observed_mark_records']=result.observed_mark_records.resample('D').sum();daily.index.name='valuation_date_utc'
    return result,daily
