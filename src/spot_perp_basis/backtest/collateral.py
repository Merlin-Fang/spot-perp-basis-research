"""Earlier closed-treasury collateral mechanics and chronological episode replay.

Retained for unique public functionality and substantive regression tests;
these reserve policies are not part of the survival entry-veto study."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.market import RangeMax, prefix
from ..data.windows import MINUTE_NS, ns_index
from .accounting import fill_cost


# Closed-treasury collateral and reserve conventions

@dataclass
class Wallet:
    cash: float
    treasury: float
    quantity: float
    entry_price: float
    transfer_cost: float = 0.
    transfer_net: float = 0.

    def equity(self, mark):
        return self.cash+self.quantity*(self.entry_price-mark)

    def command(self, target_equity, reference, cash_floor=0.):
        # A positive unrealized gain is not withdrawable USDT cash.
        return max(target_equity-self.equity(reference), cash_floor-self.cash)

    def transfer(self, command, fee_bps=0.):
        amount = max(float(command), -self.cash)
        fee = abs(amount)*fee_bps/1e4
        if amount > self.treasury-fee+1e-8 or self.treasury-amount-fee < -1e-8:
            return False, 0., 0.
        before = self.cash+self.treasury
        self.cash += amount
        self.treasury -= amount+fee
        self.transfer_net += amount
        self.transfer_cost += fee
        if self.cash < -1e-7 or self.treasury < -1e-7:
            raise AssertionError("Cash borrowing forbidden")
        if not np.isclose(self.cash+self.treasury, before-fee, atol=1e-7):
            raise AssertionError("Transfer creates/destroys cash")
        return True, amount, fee


def required_equity(quantity, reference, marks, funding_since, maintenance, charges=0.):
    """Independent direct formula; used by synthetic and source reconstruction tests."""
    marks = np.asarray(marks, float)
    funding_since = np.asarray(funding_since, float)
    if not np.isfinite(marks).all() or not np.isfinite(funding_since).all():
        return np.nan
    return max(0., float(np.max(maintenance*quantity*marks+quantity*(marks-reference)
                                -funding_since+np.asarray(charges))))


def position_requirement(quantity, entry_price, reference, marks, funding_since, maintenance, charges=0.):
    margin = required_equity(quantity, reference, marks, funding_since, maintenance, charges)
    if not np.isfinite(margin):
        return {"required_equity": np.nan, "required_cash": np.nan}
    cash_flow = max(0., float(np.max(np.asarray(charges)-np.asarray(funding_since))))
    unrealized = quantity*(entry_price-reference)
    cash = max(0., margin-unrealized, cash_flow)
    return {"required_equity": cash+unrealized, "required_cash": cash,
            "margin_equity": margin, "cash_flow_buffer": cash_flow}


class CollateralTimeline:
    """Independent perpetual-minute and funding-event grid, with sample mark prices."""

    def __init__(self, perp, events, marks, cfg, price="hybrid_close"):
        start, end = pd.Timestamp(cfg["start"]).value, pd.Timestamp(cfg["end"]).value
        self.minute_times = ns_index(perp.index)
        self.event_times = ns_index(events.settlement_time_utc)
        if (np.any(np.diff(self.minute_times) <= 0) or np.any(np.diff(self.event_times) <= 0)
                or self.minute_times.min() < start or self.minute_times.max() >= end
                or self.event_times.min() < start or self.event_times.max() >= end):
            raise ValueError("Ordered training-only prices/events required")
        self.perp, self.events, self.cfg, self.price = perp, events, cfg, price
        self.mark_times = ns_index(marks.observation_time_utc)
        self.mark_available = ns_index(marks.feature_available_at_utc)
        self.mark_values = marks.close.to_numpy()
        self.sample_start = pd.Timestamp(cfg["sample_mark_month"]+"-01", tz="UTC").value
        self.sample_end = (pd.Timestamp(cfg["sample_mark_month"]+"-01", tz="UTC")+pd.offsets.MonthBegin()).value
        self.use_marks = price != "trade_close"
        self.high_mode = price == "hybrid_high"
        close = perp.close.to_numpy().copy()
        high = perp.high.to_numpy().copy()
        sample = (self.minute_times > self.sample_start) & (self.minute_times <= self.sample_end) & self.use_marks
        idx = np.searchsorted(self.mark_times, self.minute_times[sample])
        good = (idx < len(marks))
        bounded = np.minimum(idx, len(marks)-1)
        good &= self.mark_times[bounded] == self.minute_times[sample]
        close[sample] = np.where(good, self.mark_values[bounded], np.nan)
        high[sample] = np.where(good, marks.high.to_numpy()[bounded], np.nan)
        valid = perp.valid.to_numpy(bool)
        close[~valid & ~sample] = np.nan; high[~valid & ~sample] = np.nan
        self.minute_close, self.minute_high = close, high
        notionals = events.perp_price_proxy.to_numpy().copy()
        em = (self.event_times > self.sample_start) & (self.event_times <= self.sample_end) & self.use_marks
        j = np.searchsorted(self.mark_times, self.event_times[em], side="right")-1
        bounded = np.maximum(0, j)
        ok = (j >= 0) & (self.event_times[em]-self.mark_times[bounded] <= 2*MINUTE_NS)
        notionals[em] = np.where(ok, self.mark_values[bounded], np.nan)
        self.event_per_coin = notionals*events.funding_rate.to_numpy()
        self.event_prefix = prefix(np.nan_to_num(self.event_per_coin, nan=0.))
        self.bad_event_prefix = prefix(~np.isfinite(self.event_per_coin))
        self.times = np.union1d(self.minute_times, self.event_times)
        last = np.searchsorted(self.minute_times, self.times, side="right")-1
        bounded = np.maximum(0, last)
        self.close = np.where(last >= 0, close[bounded], np.nan)
        self.high = self.close.copy()
        exact = np.isin(self.times, self.minute_times)
        self.high[exact] = high[np.searchsorted(self.minute_times, self.times[exact])]
        self.is_minute = exact
        self.minute_count = prefix(exact)
        self.inactive = np.zeros(len(self.times), bool)
        inactive = (perp.volume.to_numpy() <= 0) | (perp.trades.to_numpy() <= 0)
        self.inactive[exact] = inactive[np.searchsorted(self.minute_times, self.times[exact])]
        self.inactive_trade = self.inactive.copy()
        self.mark_source = (self.times > self.sample_start) & (self.times <= self.sample_end) & self.use_marks
        self.funding = self.event_prefix[np.searchsorted(self.event_times, self.times, side="right")]
        self.cash_unknown = np.isin(self.times, self.event_times[~np.isfinite(self.event_per_coin)])
        self.risk_funding = self.funding.copy()
        if self.high_mode:
            a = np.searchsorted(self.event_times, self.times[exact]-MINUTE_NS, side="right")
            b = np.searchsorted(self.event_times, self.times[exact], side="right")
            neg = prefix(np.minimum(np.nan_to_num(self.event_per_coin), 0.))
            self.risk_funding[exact] = self.event_prefix[a]+neg[b]-neg[a]
        self.risk_price = self.high if self.high_mode else self.close
        self.bad_price = prefix(~np.isfinite(self.risk_price))
        self.inactive_prefix = prefix(self.inactive)
        self.rmq = {m: RangeMax(np.nan_to_num((1+m)*self.risk_price-self.risk_funding, nan=-np.inf)) for m in [.01, .02]}
        self.cash_rmq = RangeMax(-self.funding)
        raw = perp.close.to_numpy()
        changes = np.r_[np.nan, np.diff(np.log(raw))]
        continuous = np.r_[False, np.diff(self.minute_times) == MINUTE_NS]
        changes[~continuous] = np.nan
        self.rv_prefix = prefix(np.nan_to_num(changes*changes, nan=0.))
        self.rv_count = prefix(np.isfinite(changes))

    def funding_at(self, t):
        return float(self.event_prefix[np.searchsorted(self.event_times, t, side="right")])

    def actual_mark(self, t):
        p = np.searchsorted(self.minute_times, t, side="right")-1
        if p < 0 or t-self.minute_times[p] > MINUTE_NS:
            return np.nan
        return float(self.minute_close[p])

    def known(self, t):
        p = np.searchsorted(self.minute_times, t)
        if p >= len(self.minute_times) or self.minute_times[p] != t:
            return np.nan, False, pd.NaT
        if self.use_marks and self.sample_start < t <= self.sample_end:
            j = np.searchsorted(self.mark_available, t, side="right")-1
            if j < 0 or t-self.mark_available[j] > 2*MINUTE_NS:
                return np.nan, False, pd.NaT
            return float(self.mark_values[j]), True, pd.Timestamp(int(self.mark_available[j]), tz="UTC")
        if not self.perp.valid.iat[p]:
            return np.nan, False, pd.NaT
        active = bool(self.perp.volume.iat[p] > 0 and self.perp.trades.iat[p] > 0)
        return float(self.perp.close.iat[p]), active, pd.Timestamp(int(t), tz="UTC")

    def rv(self, t):
        a, b = np.searchsorted(self.minute_times, [t-1440*MINUTE_NS, t], side="right")
        count = self.rv_count[b]-self.rv_count[a]
        return float(np.sqrt(self.rv_prefix[b]-self.rv_prefix[a])) if count >= .99*1440 else np.nan

    def bounds(self, t, until):
        return np.searchsorted(self.times, [t, until], side="right")

    def risk(self, t, until, maintenance=.01, quantity=1., reference=None, exit_mark=None, exit_cost=0., inactive_unknown=False,
             position_entry_price=None):
        if t >= until or until+MINUTE_NS >= pd.Timestamp(self.cfg["end"]).value:
            return {"valid": False, "status": "boundary_censored", "required_equity": np.nan}
        P = self.known(t)[0] if reference is None else reference
        a, b = self.bounds(t, until)
        ea, eb = np.searchsorted(self.event_times, [t, until], side="right")
        bad = self.bad_price[b]-self.bad_price[a]
        n = self.inactive_prefix[b]-self.inactive_prefix[a]
        # The explicit minute grid includes invalid bars as NaN; never fill them.
        expected = (until//MINUTE_NS-t//MINUTE_NS)
        observed = int(self.minute_count[b]-self.minute_count[a])
        bad += max(0, int(expected)-observed)
        valid = np.isfinite(P) and np.isfinite(self.actual_mark(t)) and bad == 0 and self.bad_event_prefix[eb] == self.bad_event_prefix[ea]
        valid &= not (inactive_unknown and n > 0)
        if not valid:
            return {"valid": False, "status": "unknown_path", "required_equity": np.nan,
                    "missing_prices": int(bad), "inactive_minutes": int(n)}
        G = self.funding_at(t)
        value = max(0., quantity*(maintenance*self.actual_mark(t)+self.actual_mark(t)-P),
                    quantity*(self.rmq[maintenance].query(a, b)-P+G))
        if exit_mark is not None:
            value = max(value, quantity*((1+maintenance)*exit_mark-P-self.funding_at(until)+G)+exit_cost)
        cash_flow = max(0., quantity*(self.cash_rmq.query(a, b)+G), exit_cost-quantity*(self.funding_at(until)-G))
        unrealized = quantity*((P if position_entry_price is None else position_entry_price)-P)
        cash = max(0., value-unrealized, cash_flow)
        state = cash+unrealized
        return {"valid": True, "status": "observed", "required_equity": float(state),
                "margin_required_equity": float(value), "cash_flow_buffer": float(cash_flow),
                "required_wallet_cash": float(cash), "risk_ratio": float(state/(quantity*P)),
                "margin_risk_ratio": float(value/(quantity*P)), "missing_prices": 0, "inactive_minutes": int(n)}


def historical_daily_labels(timeline, cfg, maintenance, inactive_unknown=False):
    rows = []
    days = pd.date_range(pd.Timestamp(cfg["start"])+pd.Timedelta(hours=9), cfg["end"], freq="D", inclusive="left")
    for H in cfg["historical_label_horizons_hours"]:
        for day in days:
            until = day+pd.Timedelta(hours=H)
            available = until+pd.Timedelta(minutes=cfg["label_publication_lag_minutes"])
            split_end = min(pd.Timestamp(cfg["end"]), pd.Timestamp(f"{day.year+1}-01-01", tz="UTC"))
            r = {"decision_time_utc": day, "year": day.year, "planning_hours": H,
                 "label_available_at_utc": available, "risk_ratio": np.nan, "status": "boundary_censored"}
            if available < split_end:
                result = timeline.risk(day.value, until.value, maintenance, inactive_unknown=inactive_unknown)
                r.update(status=result["status"], risk_ratio=result.get("risk_ratio", np.nan))
            rows.append(r)
    return pd.DataFrame(rows)


def reserve_forecast(t, H, timeline, history, cfg, maintenance, policy):
    P, active, available = timeline.known(t)
    rv = timeline.rv(t)
    output = {"reference_price": P, "reference_active": active, "price_available_at_utc": available,
              "rv24": rv, "planning_hours": H, "matured_labels": 0, "last_label_available_at_utc": pd.NaT,
              "fallback": False, "fraction": cfg["fixed_equity_fraction"]}
    if policy == "rolling95":
        now = pd.Timestamp(int(t), tz="UTC")
        matured = history.loc[(history.planning_hours == H) & history.risk_ratio.notna()
            & (history.label_available_at_utc <= now)
            & (history.label_available_at_utc > now-pd.Timedelta(days=cfg["rolling_window_days"]))]
        output["matured_labels"] = len(matured)
        output["last_label_available_at_utc"] = matured.label_available_at_utc.max()
        if len(matured) >= cfg["rolling_minimum_matured_days"]:
            output["fraction"] = max(maintenance+cfg["adaptive_buffer_fraction"],
                float(matured.risk_ratio.quantile(cfg["rolling_quantile"]))+cfg["adaptive_buffer_fraction"])
        else:
            output["fallback"] = True
    elif policy == "volatility3":
        if np.isfinite(rv):
            output["fraction"] = maintenance+cfg["adaptive_buffer_fraction"]+cfg["volatility_sigma_multiple"]*rv*np.sqrt(H/24)
        else:
            output["fallback"] = True
    if output["fallback"]:
        output["fraction"] = cfg["warmup_fallback_fraction"]
    return output


# Chronological collateral episode simulation

def episodes(market, cfg, symbol):
    result = []
    for quarter in pd.period_range("2020Q1", "2022Q4", freq="Q"):
        first = quarter.start_time.tz_localize("UTC")
        after = (quarter+1).start_time.tz_localize("UTC")
        origin = first+pd.Timedelta(hours=9)
        close_signal = after-pd.Timedelta(days=1)+pd.Timedelta(hours=23, minutes=50)
        r = {"episode_id": f"{symbol}_{quarter}", "symbol": symbol, "year": first.year,
             "quarter": str(quarter), "calendar_start_utc": first, "calendar_end_utc": after,
             "entry_decision_utc": origin, "exit_signal_utc": close_signal, "status": "unavailable_entry"}
        p = np.searchsorted(market.times, origin.value)
        if p >= len(market.times) or market.times[p] != origin.value or not market.eligible[p]:
            result.append(r); continue
        en = market.fill_after(origin.value+cfg["entry_exit_delay_minutes"]*MINUTE_NS)
        ex = market.fill_after(close_signal.value+cfg["entry_exit_delay_minutes"]*MINUTE_NS)
        wait = cfg["maximum_fill_wait_minutes"]*MINUTE_NS
        if en is None or market.opens[en]-origin.value-MINUTE_NS > wait:
            result.append(r); continue
        if ex is None or market.opens[ex]-close_signal.value-MINUTE_NS > wait:
            result.append({**r, "status": "unavailable_exit"}); continue
        entry, exit_time = int(market.opens[en]), int(market.opens[ex])
        if exit_time+MINUTE_NS >= pd.Timestamp(cfg["end"]).value:
            result.append({**r, "status": "boundary_censored"}); continue
        s0, f0 = market.frame.spot_open.iat[en], market.frame.perp_open.iat[en]
        sx, fx = market.frame.spot_open.iat[ex], market.frame.perp_open.iat[ex]
        q = cfg["spot_notional"]/s0
        r.update(status="observed", entry_utc=pd.Timestamp(entry, tz="UTC"), exit_utc=pd.Timestamp(exit_time, tz="UTC"),
            spot_entry=s0, perp_entry=f0, spot_exit=sx, perp_exit=fx, quantity=q,
            seed_spot_decision=market.frame.spot_close.iat[p], seed_perp_decision=market.frame.perp_close.iat[p],
            entry_cost=sum(fill_cost(q, s0, f0, cfg["costs"])), exit_cost=sum(fill_cost(q, sx, fx, cfg["costs"])),
            price_pnl=q*(sx-s0+f0-fx))
        result.append(r)
    return result


def decision_times(ep):
    return pd.date_range(ep["calendar_start_utc"]+pd.Timedelta(hours=9), ep["exit_utc"], freq="D")


def planning_hours(day, scenario):
    if scenario["monday_outage"] and day.dayofweek == 6:
        return 48
    return scenario["planning_hours"]


def segment_statistics(timeline, ep, a, b, cash, treasury, maintenance):
    if b <= a:
        return {"hours": 0., "cash_dollar_hours": 0., "equity_dollar_hours": 0., "treasury_dollar_hours": 0.,
                "unused_dollar_hours": 0., "peak_cash": cash, "peak_equity": np.nan, "peak_treasury": treasury}
    left, right = timeline.bounds(a, b)
    inner = timeline.times[left:right]
    stamps = np.unique(np.r_[a, inner, b])
    # Values on [stamp_i, stamp_i+1); account cash responds once to each event.
    marks = np.array([timeline.actual_mark(a)])
    loc = np.searchsorted(timeline.times, stamps[1:], side="right")-1
    marks = np.r_[marks, timeline.close[np.maximum(0, loc)]]
    funds = timeline.event_prefix[np.searchsorted(timeline.event_times, stamps, side="right")]
    c = cash+ep["quantity"]*(funds-timeline.funding_at(a))
    e = c+ep["quantity"]*(ep["perp_entry"]-marks)
    hours = np.diff(stamps)/MINUTE_NS/60
    unused = np.maximum(0., e-maintenance*ep["quantity"]*marks)
    return {"hours": float(hours.sum()), "cash_dollar_hours": float(np.nansum(c[:-1]*hours)),
            "equity_dollar_hours": float(np.nansum(e[:-1]*hours)),
            "treasury_dollar_hours": float(treasury*hours.sum()),
            "unused_dollar_hours": float(np.nansum(unused[:-1]*hours)),
            "peak_cash": float(np.nanmax(c)), "peak_equity": float(np.nanmax(e)), "peak_treasury": treasury}


def simulate_episode(timeline, ep, history, cfg, scenario, policy, forecast_cache=None):
    """A failed/unknown episode stops transfers and never receives terminal credit."""
    forecast_cache = {} if forecast_cache is None else forecast_cache
    ident = {"episode_id": ep["episode_id"], "symbol": ep["symbol"], "year": ep["year"],
             "quarter": ep["quarter"], "scenario": scenario["name"], "policy": policy}
    N, pool = cfg["spot_notional"], scenario["pool"]*cfg["spot_notional"]
    result = {**ident, "status": ep["status"], "initial_liquid_pool": pool, "initial_total_commitment": N+pool,
              "genuinely_released_cash": 0., "funded_terminal_pnl": np.nan,
              "failure_time_utc": pd.NaT, "max_cash_conservation_error": 0.}
    if ep["status"] != "observed":
        return result, [], [], []
    q, F0 = ep["quantity"], ep["perp_entry"]
    en, ex = ep["entry_utc"].value, ep["exit_utc"].value
    maintenance = scenario["maintenance"]
    result.update(counterfactual_price_pnl=ep["price_pnl"],
        counterfactual_funding=q*(timeline.funding_at(ex)-timeline.funding_at(en)),
        counterfactual_trading_cost=ep["entry_cost"]+ep["exit_cost"],
        calendar_days=(ep["calendar_end_utc"]-ep["calendar_start_utc"])/pd.Timedelta(days=1),
        first_entry_utc=ep["entry_utc"], scheduled_exit_utc=ep["exit_utc"])
    result["counterfactual_trading_pnl"] = result["counterfactual_price_pnl"]+result["counterfactual_funding"]-result["counterfactual_trading_cost"]
    def forecast(day):
        H = planning_hours(day, scenario)
        key = (timeline.price, maintenance, scenario["inactive_unknown"], day.value, H, policy)
        if key not in forecast_cache:
            forecast_cache[key] = reserve_forecast(day.value, H, timeline, history, cfg, maintenance, policy)
        return dict(forecast_cache[key])
    first = forecast(ep["entry_decision_utc"])
    initial_reference = first["reference_price"]
    seed = 2*N if policy == "generous" else first["fraction"]*N*initial_reference/ep["seed_spot_decision"]
    if not np.isfinite(seed) or seed > pool+1e-7 or seed < ep["entry_cost"]:
        result.update(status="infeasible_entry", failure_time_utc=ep["entry_utc"], transfer_count=0, transfer_turnover=0.,
                      transfer_cost=0., active_hours=0., mean_posted_cash=np.nan, mean_wallet_equity=np.nan,
                      mean_treasury=np.nan, peak_posted_cash=np.nan, peak_wallet_equity=np.nan, peak_treasury=np.nan,
                      committed_dollar_days=(N+pool)*result["calendar_days"])
        return result, [], [], []
    wallet = Wallet(seed-ep["entry_cost"], pool-seed, q, F0)
    now, status = en, "active"
    decisions, transfers, segments = [], [], []
    all_days = [d for d in decision_times(ep) if d.value > en]
    due = {int(d.value+scenario["transfer_delay_minutes"]*MINUTE_NS): d for d in all_days
           if not (scenario["monday_outage"] and d.dayofweek == 0) and d.value+scenario["transfer_delay_minutes"]*MINUTE_NS < ex}
    controls = sorted(set([d.value for d in all_days]+list(due)+[ex]))
    days_by_time = {d.value: d for d in all_days}
    pending = {}
    result.update(transfer_count=0, transfer_turnover=0., transfer_cost=0., skipped_outage_commands=0,
                  inactive_decisions=0, max_shortfall_usdt=0., shadow_shortfall_minutes=0., shadow_longest_spell_minutes=0.,
                  shadow_complete=True, initial_seed_cash=seed, first_breach_before_transfer=False)

    def conservation():
        expected = pool+q*(timeline.funding_at(now)-timeline.funding_at(en))-ep["entry_cost"]-wallet.transfer_cost
        error = abs(wallet.cash+wallet.treasury-expected)
        result["max_cash_conservation_error"] = max(result["max_cash_conservation_error"], error)
        if error > 2e-6:
            raise AssertionError("Funding/cash transfer conservation failed")

    def fail_window(t0, stop):
        # Only a labeled unrescued shadow of the failing operational interval.
        nonlocal status
        future_due = sorted(t for t in due if t >= stop)
        limit = min(ex, future_due[0] if future_due else ex)
        if policy == "generous":
            future_days = [d.value for d in all_days if d.value >= stop]
            limit = min(ex, future_days[0] if future_days else ex)
        a, b = timeline.bounds(t0, limit)
        points = timeline.times[a:b]
        e = wallet.cash+q*(timeline.risk_funding[a:b]-timeline.funding_at(t0))+q*(F0-timeline.risk_price[a:b])
        deficit = maintenance*q*timeline.risk_price[a:b]-e
        bad = ~np.isfinite(deficit) | timeline.cash_unknown[a:b]
        if scenario["inactive_unknown"]:
            bad |= timeline.inactive[a:b]
        result["shadow_complete"] = not bool(bad.any())
        if bad.any():
            result.update(max_shortfall_usdt=np.nan, shadow_shortfall_minutes=np.nan, shadow_longest_spell_minutes=np.nan)
            return
        positive = deficit > 1e-7
        spans = np.diff(np.r_[points, limit])/MINUTE_NS
        result["max_shortfall_usdt"] = max(0., float(deficit.max(initial=0.)))
        result["shadow_shortfall_minutes"] = float(np.sum(spans[positive]))
        longest, run = 0., 0.
        for flag, dt in zip(positive, spans):
            run = run+dt if flag else 0.
            longest = max(longest, run)
        result["shadow_longest_spell_minutes"] = float(longest)

    def check_post_transfer():
        nonlocal status
        mark = timeline.actual_mark(now)
        if not np.isfinite(mark):
            status = "unknown_path"
            result["failure_time_utc"] = pd.Timestamp(now, tz="UTC")
        elif wallet.equity(mark) < maintenance*q*mark-1e-7:
            status = "margin_shortfall"
            result["failure_time_utc"] = pd.Timestamp(now, tz="UTC")
            deficit_now = maintenance*q*mark-wallet.equity(mark)
            fail_window(now, now+1)
            result["max_shortfall_usdt"] = max(deficit_now, result["max_shortfall_usdt"])

    if wallet.equity(F0) < maintenance*q*F0:
        status = "margin_shortfall"
        result.update(failure_time_utc=ep["entry_utc"], first_breach_before_transfer=True)
    for point in controls:
        if status != "active":
            break
        a, b = timeline.bounds(now, point)
        times = timeline.times[a:b]
        current_cash = wallet.cash+q*(timeline.funding[a:b]-timeline.funding_at(now))
        equity = wallet.cash+q*(timeline.risk_funding[a:b]-timeline.funding_at(now))+q*(F0-timeline.risk_price[a:b])
        deficit = maintenance*q*timeline.risk_price[a:b]-equity
        unknown = ~np.isfinite(deficit) | timeline.cash_unknown[a:b]
        if scenario["inactive_unknown"]:
            unknown |= timeline.inactive[a:b]
        breach = deficit > 1e-7
        cash_bad = current_cash < -1e-7
        bad = np.flatnonzero(unknown | breach | cash_bad)
        until = int(times[bad[0]]) if len(bad) else point
        segments.append({**ident, "start_ns": now, "end_ns": until, "initial_cash": wallet.cash,
                         "treasury": wallet.treasury})
        if len(bad):
            i = bad[0]
            status = "unknown_path" if unknown[i] else "margin_shortfall" if breach[i] else "cash_shortfall"
            result.update(failure_time_utc=pd.Timestamp(until, tz="UTC"), first_breach_before_transfer=bool(breach[i]))
            if status == "margin_shortfall":
                fail_window(now, until)
            else:
                result.update(max_shortfall_usdt=np.nan, shadow_shortfall_minutes=np.nan, shadow_longest_spell_minutes=np.nan)
            wallet.cash += q*(timeline.funding_at(until)-timeline.funding_at(now))
            now = until
            conservation()
            break
        wallet.cash += q*(timeline.funding_at(point)-timeline.funding_at(now))
        now = point
        conservation()
        # A previously fixed delayed command executes BEFORE today's new command.
        if point in pending:
            command, original_day = pending.pop(point)
            ok, amount, fee = wallet.transfer(command, scenario["transfer_fee_bps"])
            transfers.append({**ident, "decision_time_utc": original_day, "execution_time_utc": pd.Timestamp(point, tz="UTC"),
                              "requested_usdt": command, "executed_usdt": amount, "fee_usdt": fee, "feasible": ok})
            if not ok:
                status = "liquidity_infeasible"
                result["failure_time_utc"] = pd.Timestamp(point, tz="UTC")
                break
            result["transfer_count"] += int(abs(amount) > 1e-7)
            result["transfer_turnover"] += abs(amount)
            conservation()
            check_post_transfer()
            if status != "active":
                break
        if point == ex:
            wallet.cash -= ep["exit_cost"]
            mark = max(timeline.actual_mark(ex), ep["perp_exit"])
            if not np.isfinite(mark):
                status = "unknown_path"
            elif wallet.cash < -1e-7 or wallet.equity(mark) < maintenance*q*mark-1e-7:
                status = "exit_cash_or_margin_shortfall"
                result["max_shortfall_usdt"] = max(0., maintenance*q*mark-wallet.equity(mark), -wallet.cash)
                result["failure_time_utc"] = ep["exit_utc"]
            else:
                wallet.cash += q*(F0-ep["perp_exit"])
                wealth = wallet.cash+wallet.treasury+q*ep["spot_exit"]
                expected = N+pool+result["counterfactual_trading_pnl"]-wallet.transfer_cost
                if not np.isclose(wealth, expected, atol=2e-6):
                    raise AssertionError("Closing wealth fails hedge PnL identity")
                status = "completed"
                result.update(funded_terminal_pnl=wealth-(N+pool), terminal_wealth=wealth,
                              terminal_cash_conservation_error=abs(wealth-expected))
            break
        if point in days_by_time:
            day = days_by_time[point]
            f = forecast(day)
            P = f["reference_price"]
            Eref = wallet.equity(P)
            target = f["fraction"]*q*P
            H = f["planning_hours"]
            if point+(H*60+scenario["transfer_delay_minutes"])*MINUTE_NS >= ex:
                # Known-price estimate; actual closing costs are paid only at exit.
                target += q*P*(cfg["costs"]["spot_fee_bps"]+cfg["costs"]["perp_fee_bps"]+2*cfg["costs"]["slippage_bps"])/1e4
            row = {**ident, **f, "decision_time_utc": day, "cash_before": wallet.cash, "treasury_before": wallet.treasury,
                "unrealized_ref": q*(F0-P), "equity_ref_before": Eref, "target_equity": target,
                "cash_after": wallet.cash, "treasury_after": wallet.treasury, "equity_ref_after": Eref,
                "action": "no_transfer", "net_transfer_requested": 0.}
            if policy != "generous":
                if not np.isfinite(P) or not f["reference_active"]:
                    row["action"] = "inactive_or_unknown_hold"
                    result["inactive_decisions"] += 1
                elif scenario["monday_outage"] and day.dayofweek == 0:
                    row["action"] = "outage_hold"
                    result["skipped_outage_commands"] += 1
                else:
                    floor = cfg["wallet_cash_floor_fraction"]*q*P
                    command = wallet.command(target, P, cash_floor=floor)
                    row["cash_floor_binding"] = bool(floor > target-q*(F0-P))
                    execution = point+scenario["transfer_delay_minutes"]*MINUTE_NS
                    row["net_transfer_requested"] = command
                    if execution >= ex:
                        row["action"] = "after_exit_not_sent"
                    elif execution > point:
                        pending[execution] = (command, day)
                        row["action"] = "queued"
                    else:
                        ok, amount, fee = wallet.transfer(command, scenario["transfer_fee_bps"])
                        transfers.append({**ident, "decision_time_utc": day, "execution_time_utc": day,
                                          "requested_usdt": command, "executed_usdt": amount, "fee_usdt": fee, "feasible": ok})
                        row["action"] = "transferred" if ok else "liquidity_infeasible"
                        if not ok:
                            status = "liquidity_infeasible"
                            result["failure_time_utc"] = day
                        else:
                            result["transfer_count"] += int(abs(amount) > 1e-7)
                            result["transfer_turnover"] += abs(amount)
                            conservation()
                            check_post_transfer()
                        row.update(cash_after=wallet.cash, treasury_after=wallet.treasury, equity_ref_after=wallet.equity(P))
            decisions.append(row)
    result.update(status=status, transfer_cost=wallet.transfer_cost,
                  committed_dollar_days=(N+pool)*result["calendar_days"],
                  opportunity_cost=(N+pool)*cfg["opportunity_hurdle_apr"]*result["calendar_days"]/365,
                  alternative_use_earnings=0.)
    stats = [segment_statistics(timeline, ep, s["start_ns"], s["end_ns"], s["initial_cash"], s["treasury"], maintenance) for s in segments]
    hours = sum(s["hours"] for s in stats)
    result["active_hours"] = hours
    for field, label in [("cash", "posted_cash"), ("equity", "wallet_equity"), ("treasury", "treasury"), ("unused", "unused_equity")]:
        result["mean_"+label] = sum(s[field+"_dollar_hours"] for s in stats)/hours if hours else np.nan
    for field, label in [("cash", "posted_cash"), ("equity", "wallet_equity"), ("treasury", "treasury")]:
        result["peak_"+label] = max((s["peak_"+field] for s in stats), default=np.nan)
    result["full_calendar_posted_cash_lower_bound"] = sum(s["cash_dollar_hours"] for s in stats)/(result["calendar_days"]*24)
    result["funded_roc"] = result["funded_terminal_pnl"]/(N+pool)
    return result, decisions, transfers, segments
