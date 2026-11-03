"""Future outcomes: first passage, complete-window basis and carry exposure.

These objects are outcomes/diagnostics, never predictors. Censoring, origin
direction and publication timestamps retain their registered conventions."""

import numpy as np
import pandas as pd

from ..theory import entry_side, exit_crossed
from ..backtest.accounting import fill_cost
from .market import MINUTE, RangeMax, prefix
from .windows import MINUTE_NS, YEAR_NS, ns_index


# Complete-minute basis outcomes

def horizon_targets(frame, horizon=15, column="basis_bps"):
    """Exact-time labels, rejecting every window containing any missing minute."""
    expected = pd.date_range(frame.index[0], frame.index[-1], freq="min").as_unit("ns")
    if not frame.index.equals(expected):
        raise ValueError("Target construction requires an explicit, regular minute grid")
    basis = frame[column]
    complete = basis.notna().rolling(horizon + 1, min_periods=horizon + 1).sum().shift(-horizon).eq(horizon + 1)
    target = (basis.shift(-horizon) - basis).where(complete)
    return pd.DataFrame({"basis_bps": basis, "target_bps": target,
                         "target_end_utc": frame.index + horizon * MINUTE}, index=frame.index)


# Origin-fixed first passage and gap censoring

def first_passage(rho,valid,clock,direction=None):
    """Observed-clock first crossing plus gap-censored primary follow-up."""
    rho=np.asarray(rho,float);valid=np.asarray(valid,bool)&np.isfinite(rho);n=len(rho);index=np.arange(n)
    d=np.array([entry_side(v,0.) for v in rho]) if direction is None else np.asarray(direction,int)
    def future(mask):
        a=np.minimum.accumulate(np.where(mask,index,n)[::-1])[::-1]
        return np.r_[a[1:],n]
    negative=future(valid&np.array([exit_crossed(-1,v) if np.isfinite(v) else False for v in rho]))
    positive=future(valid&np.array([exit_crossed(1,v) if np.isfinite(v) else False for v in rho]))
    event_index=np.where(d==-1,negative,np.where(d==1,positive,n));gap=future(~valid)
    event=(event_index<n)&(event_index<gap);follow=np.where(event,event_index,np.minimum(gap-1,n-1))
    available=np.where(event,event_index,np.where(gap<n,gap,n-1))
    eligible=valid&(d!=0);duration=(follow-index).astype(float);duration[~eligible]=np.nan
    out=pd.DataFrame(index=pd.DatetimeIndex(clock))
    out['direction']=d;out['survival_origin_eligible']=eligible;out['event_observed']=event&eligible;out['followup_hours']=duration
    out['followup_until_utc']=pd.DatetimeIndex(clock).take(follow).where(eligible)
    out['event_time_utc']=pd.DatetimeIndex(clock).take(np.minimum(event_index,n-1)).where(event&eligible)
    out['record_available_at_utc']=pd.DatetimeIndex(clock).take(available).where(eligible)
    out['censor_reason']=np.select([~eligible,event,(gap<n)],['unsupported_origin_or_zero_direction','observed_crossing','missing_or_inactive_hour'],default='administrative_prefix_end')
    out['observed_clock_event_utc']=pd.DatetimeIndex(clock).take(np.minimum(event_index,n-1)).where((event_index<n)&eligible)
    out['gap_before_observed_clock_event']=(gap<event_index)&eligible
    out['observed_clock_followup_hours']=np.where(event_index<n,event_index-index,n-1-index).astype(float)
    out.loc[~eligible,'observed_clock_followup_hours']=np.nan
    return out


def labels(old,begin,end, *, horizons=(1,2,4,8,16), maximum_hazard_hour=16):
    old=old.loc[old.index<end];p=first_passage(old.rho,old.theory_available,old.index);p['rho_origin']=old.rho
    for h in horizons:
        event=p.event_observed&p.followup_hours.le(h);alive=p.followup_hours.ge(h)&~event;p[f'event_by_{h}h']=np.where(event,1.,np.where(alive,0.,np.nan))
        av=pd.Series(pd.NaT,index=p.index,dtype='datetime64[ns, UTC]');av.loc[event]=p.loc[event,'event_time_utc'];av.loc[alive]=p.index[alive]+pd.Timedelta(hours=h)
        p[f'available_{h}h_utc']=av
    p['boundary_window_purged']=p.index+pd.Timedelta(hours=maximum_hazard_hour)>=end;p['label_prefix_end_exclusive']=end
    return p.loc[p.index>=begin]


# Earlier fixed-horizon carry/path outcomes

class LabelEngine:
    def __init__(self, market, events, cfg):
        self.market, self.cfg = market, cfg
        self.et = ns_index(events.settlement_time_utc)
        if np.any(np.diff(self.et) <= 0):
            raise ValueError("Unordered/duplicate funding")
        if np.any(self.et < pd.Timestamp(cfg["start"]).value) or np.any(self.et >= pd.Timestamp(cfg["end"]).value):
            raise ValueError("Funding outside training")
        if market.times[0] < pd.Timestamp(cfg["start"]).value or market.times[-1] >= pd.Timestamp(cfg["end"]).value:
            raise ValueError("Prices outside training")
        amounts = events.funding_rate.to_numpy()*events.perp_price_proxy.to_numpy()
        self.fund = prefix(np.nan_to_num(amounts, nan=0.))
        self.bad_fund = prefix(~np.isfinite(amounts))
        self.rates = prefix(events.funding_rate.to_numpy())
        self.global_at_price = self.fund[np.searchsorted(self.et, market.times, side="right")]
        self.perp = market.frame.perp_close.to_numpy()
        self.rmq = {float(m): RangeMax((1+m)*self.perp-self.global_at_price) for m in cfg["maintenance_fractions"]}
        self.perp_max = RangeMax(self.perp)
        self.inactive = prefix(~market.eligible)
        delta = np.r_[0., np.diff(np.log(self.perp))]
        delta[np.r_[True, np.diff(market.times) != MINUTE_NS]] = 0.
        self.variance = prefix(delta*delta)

    def outcome(self, origin, delay=None):
        origin = pd.Timestamp(origin)
        t = origin.value
        cfg, market = self.cfg, self.market
        H = cfg["primary_horizon_hours"]
        delay = cfg["entry_delay_minutes"] if delay is None else delay
        if delay < 1 or H <= 0:
            raise ValueError("Require delayed positive horizon")
        row = {"decision_time_utc": origin, "year": origin.year, "delay_minutes": delay,
               "label_status": "unavailable_origin", "risk_valid": False}
        if t < pd.Timestamp(cfg["start"]).value or t >= pd.Timestamp(cfg["end"]).value:
            raise ValueError("Origin outside training")
        p = np.searchsorted(market.times, t)
        if p == len(market.times) or market.times[p] != t or not market.eligible[p]:
            return row
        due = t+delay*MINUTE_NS
        lag = cfg["label_publication_lag_minutes"]*MINUTE_NS
        end = pd.Timestamp(cfg["end"]).value
        if due+H*60*MINUTE_NS+lag >= end:
            return {**row, "label_status": "censored_training_boundary"}
        en = market.fill_after(due)
        wait = cfg["maximum_fill_wait_minutes"]*MINUTE_NS
        if en is None or market.opens[en]-due > wait:
            return {**row, "label_status": "unfilled_entry"}
        entry = market.opens[en]
        due_exit = entry+H*60*MINUTE_NS
        if due_exit+lag >= end:
            return {**row, "label_status": "censored_training_boundary"}
        ex = market.fill_after(due_exit)
        if ex is None or market.opens[ex]-due_exit > wait:
            return {**row, "label_status": "unfilled_exit"}
        exit_time = market.opens[ex]
        if exit_time+lag >= end:
            return {**row, "label_status": "censored_training_boundary"}
        s0, f0 = market.frame.spot_open.iat[en], market.frame.perp_open.iat[en]
        sx, fx = market.frame.spot_open.iat[ex], market.frame.perp_open.iat[ex]
        N = cfg["spot_notional"]
        q = N/s0
        a, b = np.searchsorted(self.et, [entry, exit_time], side="right")
        left = np.searchsorted(market.times, entry, side="right")
        right = np.searchsorted(market.times, exit_time)
        missing = int((exit_time-entry)//MINUTE_NS-1-(right-left))
        funding = q*(self.fund[b]-self.fund[a])
        basis = q*(sx-s0+f0-fx)
        en_cost = sum(fill_cost(q, s0, f0, cfg["costs"]))
        ex_cost = sum(fill_cost(q, sx, fx, cfg["costs"]))
        fund_complete = b-a == H//8 and self.bad_fund[b] == self.bad_fund[a]
        valid = missing == 0 and fund_complete
        row.update(label_status="observed", entry_utc=pd.Timestamp(entry, tz="UTC"),
            exit_utc=pd.Timestamp(exit_time, tz="UTC"), label_available_at_utc=pd.Timestamp(exit_time+lag, tz="UTC"),
            entry_wait_minutes=float((entry-due)/MINUTE_NS), exit_wait_minutes=float((exit_time-due_exit)/MINUTE_NS),
            duration_hours=float((exit_time-entry)/MINUTE_NS/60), quantity=q,
            spot_entry_price=s0, perp_entry_price=f0, spot_exit_price=sx, perp_exit_price=fx,
            basis_pnl_usdt=basis, basis_pnl_bps=basis/N*1e4,
            funding_cash_usdt=funding if fund_complete else np.nan,
            funding_cash_bps=funding/N*1e4 if fund_complete else np.nan,
            funding_rate_sum_bps=(self.rates[b]-self.rates[a])*1e4,
            funding_events=int(b-a), funding_complete=bool(fund_complete),
            missing_minutes=missing, inactive_minutes=int(self.inactive[right]-self.inactive[left]),
            risk_valid=bool(valid), total_cost_usdt=en_cost+ex_cost,
            economic_net_bps=((basis+funding-en_cost-ex_cost-N*cfg["economic_financing_hurdle_apr"]*(exit_time-entry)/YEAR_NS)/N*1e4
                              if fund_complete else np.nan),
            future_perp_rv=np.sqrt(max(0., self.variance[right]-self.variance[left])),
            future_max_up_fraction=max(0., max(f0, fx, self.perp_max.query(left, right))/f0-1))
        for m in cfg["maintenance_fractions"]:
            interior = q*(self.rmq[float(m)].query(left, right)-f0+self.fund[a])+en_cost
            reserve = max(0., m*q*f0+en_cost, interior,
                          q*((1+m)*fx-f0)-funding+en_cost+ex_cost)
            key = "collateral_ratio" if m == cfg["primary_maintenance_fraction"] else "collateral_ratio_2pct"
            row["observed_"+key] = reserve/N
            row[key] = reserve/N if valid else np.nan
        R = row["collateral_ratio"]
        row["total_required_capital_usdt"] = N*(1+R)
        row["extra_cash_beyond_reserve_usdt"] = max(0., N*R-(cfg["committed_capital"]-N)) if valid else np.nan
        row["feasible_at_committed_capital"] = bool(valid and N*(1+R) <= cfg["committed_capital"])
        return row
