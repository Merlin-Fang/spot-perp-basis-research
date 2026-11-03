"""Earlier equal-base carry/threshold simulation and fixed-horizon outcomes.

This retained exploratory family has distinct sizing and accounting conventions
from He. It is required by public carry/target tests and existing research CLIs."""

import numpy as np
import pandas as pd

from ..data.windows import MINUTE_NS, YEAR_NS, ns_index, stamp
from ..evaluation.metrics import return_metrics
from .accounting import fill_cost


# Earlier equal-base carry accounting; not the He hedge

def threshold_trades(market, *, measurement="close", spacing=1, delay=1,
                     entry=.005, exit=-.001):
    """Signals at completed timestamps; orders wait for eligible later opens."""
    if delay < 1 or spacing not in (1, 60):
        raise ValueError("A full next-bar delay and prespecified clock are required")
    b = market.basis(measurement)
    observed = market.eligible & ((market.times // MINUTE_NS) % spacing == 0)
    entries = np.flatnonzero(observed & (b >= entry - 1e-12))
    exits = np.flatnonzero(observed & (b <= exit + 1e-12))
    trades, cursor = [], -1
    while True:
        k = np.searchsorted(market.times[entries], cursor, side="right")
        if k == len(entries):
            break
        signal = int(entries[k])
        fill = market.fill_after(market.times[signal] + delay * MINUTE_NS)
        if fill is None:
            break
        entry_time = market.opens[fill]
        j = np.searchsorted(market.times[exits], entry_time, side="right")
        exit_signal = None if j == len(exits) else int(exits[j])
        exit_fill = None if exit_signal is None else market.fill_after(
            market.times[exit_signal] + delay * MINUTE_NS)
        trades.append({"entry_signal_row": signal, "entry_row": fill,
                       "exit_signal_row": exit_signal, "exit_row": exit_fill,
                       "entry_signal_basis": float(b[signal]),
                       "exit_signal_basis": None if exit_signal is None else float(b[exit_signal])})
        if exit_fill is None:
            break
        cursor = market.opens[exit_fill]
    qualifying = pd.DatetimeIndex(pd.to_datetime(market.times[entries], utc=True))
    opportunities = {str(y): int((qualifying.year == y).sum()) for y in (2020, 2021, 2022)}
    opportunities["all"] = len(entries)
    opportunities["observations"] = int(observed.sum())
    opportunities["episodes"] = (int(1 + np.sum(np.diff(market.times[entries]) > spacing*MINUTE_NS))
                                   if len(entries) else 0)
    return trades, opportunities


def carry_trades(market, hold="annual", delay=1):
    if delay < 1 or hold not in ("annual", "90d", "continuous"):
        raise ValueError("Unknown carry schedule or noncausal delay")
    trades = []
    years = [2020] if hold == "continuous" else [2020, 2021, 2022]
    for year in years:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC").value
        signal = int(np.searchsorted(market.times, start))
        if signal == len(market.times):
            continue
        fill = market.fill_after(market.times[signal] + delay * MINUTE_NS)
        if fill is None:
            continue
        exit_time = (None if hold == "continuous" else
                     pd.Timestamp(f"{year}-12-31 23:50", tz="UTC").value if hold == "annual"
                     else market.opens[fill] + 90 * 24 * 60 * MINUTE_NS)
        exit_fill = None if exit_time is None else market.fill_after(exit_time + delay * MINUTE_NS)
        trades.append({"entry_signal_row": signal, "entry_row": fill,
                       "exit_signal_row": None, "scheduled_exit_signal_ns": exit_time,
                       "exit_row": exit_fill, "entry_signal_basis": None, "exit_signal_basis": None})
        if exit_fill is None:
            break
    return trades, {"all": len(trades), "observations": len(years), "episodes": len(trades),
                    **{str(y): int(any(pd.Timestamp(market.opens[t["entry_row"]], tz="UTC").year == y
                                                   for t in trades)) for y in (2020, 2021, 2022)}}


def build_accounting(market, trades, events, start, end, notional=1e6):
    """Build friction-independent cumulative cash/MTM paths once per rule.

    Funding uses events, never the repeated last-settled context on bar rows.
    A missing funding-price proxy is fatal only if an actual held position needs it.
    """
    n = len(market.times)
    columns = ("spot", "perp", "funding", "funding_spot_proxy", "spot_turnover",
               "perp_turnover", "financing_dollar_years", "realized_spot")
    increments = {key: np.zeros(n) for key in columns}
    active_notional, active_perp = np.zeros(n), np.zeros(n)
    event_times = ns_index(events.settlement_time_utc)
    if (np.any(np.diff(event_times) <= 0) or np.any(event_times < pd.Timestamp(start).value)
            or np.any(event_times >= pd.Timestamp(end).value)):
        raise ValueError("Funding must be unique, ordered, and within training")
    event_cash = events.funding_rate.to_numpy() * events.perp_price_proxy.to_numpy()
    spot_event_cash = events.funding_rate.to_numpy() * events.spot_price_proxy.to_numpy()
    records, previous_exit = [], -1
    for number, trade in enumerate(trades):
        en, ex = trade["entry_row"], trade["exit_row"]
        entry_time = int(market.opens[en])
        exit_time = int(market.opens[ex]) if ex is not None else int(market.times[-1])
        if entry_time <= previous_exit or exit_time < entry_time:
            raise ValueError("Overlapping or reversed positions")
        if not (pd.Timestamp(start).value <= entry_time <= exit_time < pd.Timestamp(end).value):
            raise ValueError("Trade outside training")
        previous_exit = exit_time
        left = int(np.searchsorted(market.times, entry_time))
        right = int(np.searchsorted(market.times, exit_time, side="right"))
        if right <= left:
            raise ValueError("Position has no observable valuation")
        st = market.frame.spot_close.to_numpy()[left:right].copy()
        ft = market.frame.perp_close.to_numpy()[left:right].copy()
        s0, f0 = float(market.frame.spot_open.iloc[en]), float(market.frame.perp_open.iloc[en])
        q = notional / s0
        if market.times[left] == entry_time:
            st[0], ft[0] = s0, f0
        if ex is not None:
            st[-1] = market.frame.spot_open.iloc[ex]
            ft[-1] = market.frame.perp_open.iloc[ex]
        times = market.times[left:right]
        e0 = np.searchsorted(event_times, entry_time, side="right")
        e1 = np.searchsorted(event_times, exit_time, side="right")
        if not (np.isfinite(event_cash[e0:e1]).all() and np.isfinite(spot_event_cash[e0:e1]).all()):
            raise ValueError("Missing funding notional for a held settlement")
        positions = np.searchsorted(event_times[e0:e1], times, side="right")
        fund = q * np.r_[0., np.cumsum(event_cash[e0:e1])][positions]
        spot_fund = q * np.r_[0., np.cumsum(spot_event_cash[e0:e1])][positions]
        paths = {"spot": q * (st - s0), "perp": q * (f0 - ft), "funding": fund,
                 "funding_spot_proxy": spot_fund,
                 "financing_dollar_years": ((times - entry_time).astype(float) / YEAR_NS) * notional}
        for key, values in paths.items():
            increments[key][left:right] += np.diff(np.r_[0., values])
        increments["spot_turnover"][left] += q * s0
        increments["perp_turnover"][left] += q * f0
        if ex is not None:
            increments["spot_turnover"][right-1] += q * st[-1]
            increments["perp_turnover"][right-1] += q * ft[-1]
            increments["realized_spot"][right-1] += paths["spot"][-1]
        active_right = right - int(ex is not None and times[-1] == exit_time)
        active_notional[left:active_right] = notional
        active_perp[left:active_right] = q * ft[:active_right-left]
        es = trade["entry_signal_row"]
        xs = trade.get("exit_signal_row")
        missing = max(0, int((exit_time-entry_time)//MINUTE_NS) - int(np.sum(times > entry_time)))
        records.append({"trade_number": number + 1, "status": "closed" if ex is not None else "open",
                        "entry_signal_utc": stamp(market.times[es]), "entry_utc": stamp(entry_time),
                        "exit_signal_utc": stamp(market.times[xs]) if xs is not None else
                            stamp(trade["scheduled_exit_signal_ns"]) if trade.get("scheduled_exit_signal_ns") else None,
                        "exit_utc": None if ex is None else stamp(exit_time), "valuation_utc": stamp(exit_time),
                        "entry_signal_basis": trade["entry_signal_basis"],
                        "exit_signal_basis": trade["exit_signal_basis"],
                        "entry_fill_basis": (f0-s0)/s0, "exit_or_mark_basis": (ft[-1]-st[-1])/st[-1],
                        "spot_entry_price": s0, "perp_entry_price": f0,
                        "spot_exit_or_mark_price": float(st[-1]), "perp_exit_or_mark_price": float(ft[-1]),
                        "quantity": q, "duration_hours": (exit_time-entry_time)/MINUTE_NS/60,
                        "funding_events": int(e1-e0), "missing_paired_minutes_held": missing,
                        "spot_pnl": float(paths["spot"][-1]), "perp_pnl": float(paths["perp"][-1]),
                        "funding_perp_proxy": float(fund[-1]), "funding_spot_proxy": float(spot_fund[-1]),
                        "spot_turnover": q*(s0+(st[-1] if ex is not None else 0)),
                        "perp_turnover": q*(f0+(ft[-1] if ex is not None else 0)),
                        "financing_dollar_years": float(paths["financing_dollar_years"][-1]),
                        "fill_count": 4 if ex is not None else 2,
                        "entry_wait_minutes": (entry_time-market.times[es])/MINUTE_NS,
                        "left": left, "right": right})
    paths = {key: np.cumsum(value) for key, value in increments.items()}
    paths.update(active_spot_notional=active_notional, active_perp_notional=active_perp)
    return paths, records


def evaluate(market, paths, base_records, scenario, start, end, *, capital=3e6,
             financing_apr=0., funding_on=True, funding_proxy="perp", maintenance=.01):
    """Apply cash debits, produce trade and daily records, and diagnose margin."""
    funding_key = "funding" if funding_proxy == "perp" else "funding_spot_proxy"
    funding = paths[funding_key] if funding_on else np.zeros(len(market.times))
    fees = (paths["spot_turnover"]*scenario["spot_fee_bps"]
            + paths["perp_turnover"]*scenario["perp_fee_bps"])/1e4
    slippage = (paths["spot_turnover"]+paths["perp_turnover"])*scenario["slippage_bps"]/1e4
    financing = paths["financing_dollar_years"]*financing_apr
    gross = paths["spot"]+paths["perp"]
    net = gross+funding-fees-slippage-financing
    equity = capital+net
    wallet = (capital-paths["active_spot_notional"]+paths["realized_spot"]+paths["perp"]
              + funding-fees-slippage-financing)
    cash_needed = np.maximum(maintenance*paths["active_perp_notional"]-wallet, 0)
    active = paths["active_spot_notional"] > 0
    peaks = np.maximum.accumulate(np.r_[capital, equity])[1:]
    dd = equity/peaks-1
    records, realized = [], 0.
    for original in base_records:
        r = dict(original)
        left, right = r.pop("left"), r.pop("right")
        r["gross_price_pnl"] = r["spot_pnl"]+r["perp_pnl"]
        r["funding_pnl"] = r[f"funding_{funding_proxy}_proxy"] if funding_on else 0.
        r["fees"] = (r["spot_turnover"]*scenario["spot_fee_bps"]+r["perp_turnover"]*scenario["perp_fee_bps"])/1e4
        r["slippage"] = (r["spot_turnover"]+r["perp_turnover"])*scenario["slippage_bps"]/1e4
        r["financing_hurdle"] = r["financing_dollar_years"]*financing_apr
        r["net_pnl"] = r["gross_price_pnl"]+r["funding_pnl"]-r["fees"]-r["slippage"]-r["financing_hurdle"]
        r["maximum_adverse_excursion_usdt"] = float(min(0., np.min(net[left:right]-realized)))
        r["gross_price_mae_usdt"] = float(min(0., np.min(gross[left:right] -
                                                    (gross[left-1] if left else 0))))
        r["margin_proxy_breach"] = bool(np.any(cash_needed[left:right] > 0))
        r["additional_cash_required_proxy"] = float(np.max(cash_needed[left:right]))
        r["terminal_liquidation_cost_estimate"] = (r["quantity"]*(
            r["spot_exit_or_mark_price"]*scenario["spot_fee_bps"]+
            r["perp_exit_or_mark_price"]*scenario["perp_fee_bps"]+
            (r["spot_exit_or_mark_price"]+r["perp_exit_or_mark_price"])*scenario["slippage_bps"])/1e4
            if r["status"] == "open" else 0.)
        r["break_even_uniform_bps_per_fill"] = ((r["gross_price_pnl"]+r["funding_pnl"]-r["financing_hurdle"])
                                                /(r["spot_turnover"]+r["perp_turnover"])*1e4)
        records.append(r)
        if r["status"] == "closed":
            realized += r["net_pnl"]
    frame = pd.DataFrame({"spot_pnl": paths["spot"], "perp_pnl": paths["perp"], "funding_pnl": funding,
                          "fees": fees, "slippage": slippage, "financing_hurdle": financing,
                          "net_pnl": net, "equity": equity, "wallet_proxy": wallet,
                          "additional_cash_required_proxy": cash_needed,
                          "valuation_time_utc": pd.to_datetime(market.times, utc=True)},
                         index=pd.to_datetime(market.times, utc=True))
    # Select the last observable valuation per calendar day, retaining idle days.
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(),
                         freq="D", inclusive="left")
    daily = frame.groupby(frame.index.floor("D")).last().reindex(days)
    if daily.equity.isna().any():
        raise ValueError("Missing entire daily valuation; do not silently forward fill")
    previous = np.r_[capital, daily.equity.to_numpy()[:-1]]
    daily["daily_pnl"] = daily.equity.to_numpy()-previous
    daily["daily_return"] = daily.daily_pnl.to_numpy()/previous
    daily["valuation_age_minutes"] = ((daily.index+pd.Timedelta(days=1))-pd.DatetimeIndex(daily.valuation_time_utc)).total_seconds()/60
    trade_frame = pd.DataFrame(records)
    closed = [r for r in records if r["status"] == "closed"]
    duration = [r["duration_hours"] for r in closed]
    total_days = (pd.Timestamp(end)-pd.Timestamp(start)).total_seconds()/86400
    turnover = float(paths["spot_turnover"][-1]+paths["perp_turnover"][-1])
    summary = {**return_metrics(daily.equity, capital, total_days), "initial_capital": capital,
               "calendar_days": total_days, "completed_trades": len(closed),
               "open_trades": len(records)-len(closed), "entered_positions": len(records),
               "mean_duration_hours_closed": float(np.mean(duration)) if duration else None,
               "median_duration_hours_closed": float(np.median(duration)) if duration else None,
               "mean_gross_price_pnl_closed": float(np.mean([r["gross_price_pnl"] for r in closed])) if closed else None,
               "mean_net_pnl_closed": float(np.mean([r["net_pnl"] for r in closed])) if closed else None,
               "win_rate_closed": float(np.mean([r["net_pnl"] > 0 for r in closed])) if closed else None,
               "spot_pnl": float(paths["spot"][-1]), "perp_pnl": float(paths["perp"][-1]),
               "gross_price_pnl": float(gross[-1]), "funding_pnl": float(funding[-1]),
               "fees": float(fees[-1]), "slippage": float(slippage[-1]),
               "financing_hurdle": float(financing[-1]), "maximum_drawdown_fraction": float(np.min(dd)),
               "worst_trade_mae_usdt": min([r["maximum_adverse_excursion_usdt"] for r in records], default=0.),
               "active_time_fraction": sum(r["duration_hours"] for r in records)/(total_days*24),
               "observed_active_minutes": int(active.sum()),
               "missing_paired_minutes_held": sum(r["missing_paired_minutes_held"] for r in records),
               "break_even_uniform_bps_per_fill": float((gross[-1]+funding[-1]-financing[-1])/turnover*1e4) if turnover else None,
               "turnover_usdt": turnover, "margin_proxy_breach": bool(np.any(cash_needed > 0)),
               "additional_cash_required_proxy": float(np.max(cash_needed)),
               "minimum_active_wallet_proxy": float(np.min(wallet[active])) if active.any() else None,
               "funding_events_booked": sum(r["funding_events"] for r in records) if funding_on else 0,
               "terminal_liquidation_cost_estimate": sum(r["terminal_liquidation_cost_estimate"] for r in records),
               "maximum_daily_valuation_age_minutes": float(daily.valuation_age_minutes.max())}
    # Calendar PnL and returns include positions that cross year boundaries.
    yearly = []
    for year, group in daily.groupby(daily.index.year):
        before = float(group.equity.iloc[0]-group.daily_pnl.iloc[0])
        row = {"year": int(year), **return_metrics(group.equity, before, len(group))}
        row["return_on_initial_capital"] = float(group.daily_pnl.sum()/capital)
        for key in ("spot_pnl", "perp_pnl", "funding_pnl", "fees", "slippage", "financing_hurdle"):
            prev = float(daily[key].loc[daily.index < group.index[0]].iloc[-1]) if (daily.index < group.index[0]).any() else 0.
            row[key] = float(group[key].iloc[-1]-prev)
        year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        year_end = pd.Timestamp(f"{year+1}-01-01", tz="UTC")
        completed = [r for r in closed if year_start <= pd.Timestamp(r["exit_utc"]) < year_end]
        active_hours = sum(max(0., (min(pd.Timestamp(r["valuation_utc"]), year_end)
                                   - max(pd.Timestamp(r["entry_utc"]), year_start)).total_seconds()/3600)
                           for r in records)
        row.update(completed_trades_exit_year=len(completed),
                   entered_positions=sum(year_start <= pd.Timestamp(r["entry_utc"]) < year_end for r in records),
                   mean_net_pnl_closed_exit_year=float(np.mean([r["net_pnl"] for r in completed])) if completed else None,
                   win_rate_closed_exit_year=float(np.mean([r["net_pnl"] > 0 for r in completed])) if completed else None,
                   active_time_fraction=active_hours/(len(group)*24))
        observed_year = (market.times >= year_start.value) & (market.times < year_end.value)
        row["margin_proxy_breach"] = bool(np.any(cash_needed[observed_year] > 0))
        row["additional_cash_required_proxy"] = float(np.max(cash_needed[observed_year]))
        yearly.append(row)
    return trade_frame, daily, summary, yearly


# Isolated fixed-horizon carry labels

def fixed_outcome(market, events, origin, horizon_hours, delay_minutes, cfg):
    """One isolated fixed-horizon example; no capital sharing with other labels."""
    origin = pd.Timestamp(origin)
    if delay_minutes < 1 or horizon_hours <= 0:
        raise ValueError("Require a later opening and positive fixed horizon")
    t, end = origin.value, pd.Timestamp(cfg["train_end"]).value
    lag = cfg["label_availability_lag_minutes"]*MINUTE_NS
    maximum_wait = cfg["maximum_fill_wait_minutes"]*MINUTE_NS
    row = {"decision_time_utc": origin, "year": origin.year, "horizon_hours": horizon_hours,
           "delay_minutes": delay_minutes, "label_status": "unfilled_entry", "risk_complete": False}
    pos = int(np.searchsorted(market.times, t))
    if pos == len(market.times) or market.times[pos] != t or not market.eligible[pos]:
        return {**row, "label_status": "unavailable_origin"}, [], []
    due = t+delay_minutes*MINUTE_NS
    if due+horizon_hours*60*MINUTE_NS+lag >= end:
        return {**row, "label_status": "censored_training_boundary"}, [], []
    en = market.fill_after(due)
    if en is None or market.opens[en]-due > maximum_wait:
        return row, [], []
    entry = int(market.opens[en])
    due_exit = entry+horizon_hours*60*MINUTE_NS
    row.update(entry_utc=pd.Timestamp(entry, tz="UTC"), scheduled_exit_utc=pd.Timestamp(due_exit, tz="UTC"))
    if due_exit+lag >= end:
        return {**row, "label_status": "censored_training_boundary"}, [], []
    ex = market.fill_after(due_exit)
    if ex is None or market.opens[ex]-due_exit > maximum_wait:
        return {**row, "label_status": "unfilled_exit"}, [], []
    exit_time = int(market.opens[ex])
    if exit_time+lag >= end:
        return {**row, "label_status": "censored_training_boundary"}, [], []
    N = float(cfg["spot_notional"])
    C = float(cfg["initial_capital"])
    s0, f0 = float(market.frame.spot_open.iloc[en]), float(market.frame.perp_open.iloc[en])
    sx, fx = float(market.frame.spot_open.iloc[ex]), float(market.frame.perp_open.iloc[ex])
    q = N/s0
    left = int(np.searchsorted(market.times, entry, side="right"))
    right = int(np.searchsorted(market.times, exit_time))
    times = np.r_[entry, market.times[left:right], exit_time]
    s = np.r_[s0, market.frame.spot_close.to_numpy()[left:right], sx]
    f = np.r_[f0, market.frame.perp_close.to_numpy()[left:right], fx]
    expected = (exit_time-entry)//MINUTE_NS+1
    missing = int(expected-len(times))
    et = ns_index(events.settlement_time_utc)
    if np.any(np.diff(et) <= 0):
        raise ValueError("Funding events must be unique and ordered")
    a, b = np.searchsorted(et, [entry, exit_time], side="right")
    held = events.iloc[a:b]
    if not np.isfinite(held.perp_price_proxy.to_numpy()).all():
        return {**row, "label_status": "missing_funding_notional"}, [], []
    amounts = q*held.perp_price_proxy.to_numpy()*held.funding_rate.to_numpy()
    funding = np.r_[0., np.cumsum(amounts)][np.searchsorted(et[a:b], times, side="right")]
    spot_pnl, perp_pnl = q*(s-s0), q*(f0-f)
    gross = spot_pnl+perp_pnl
    elapsed_years = (times-entry).astype(float)/YEAR_NS
    row.update(label_status="observed", exit_utc=pd.Timestamp(exit_time, tz="UTC"),
               label_available_at_utc=pd.Timestamp(exit_time+lag, tz="UTC"),
               entry_wait_minutes=(entry-due)/MINUTE_NS, exit_wait_minutes=(exit_time-due_exit)/MINUTE_NS,
               duration_hours=(exit_time-entry)/MINUTE_NS/60, quantity=q,
               spot_entry_price=s0, perp_entry_price=f0, spot_exit_price=sx, perp_exit_price=fx,
               spot_pnl=float(spot_pnl[-1]), perp_pnl=float(perp_pnl[-1]), gross_price_pnl=float(gross[-1]),
               funding_cash_proxy=float(funding[-1]), future_funding_rate_sum_bps=float(held.funding_rate.sum()*1e4),
               funding_event_count=len(held), funding_event_count_expected=horizon_hours//8,
               funding_complete=len(held) == horizon_hours//8,
               first_funding_utc=held.settlement_time_utc.iloc[0] if len(held) else pd.NaT,
               last_funding_utc=held.settlement_time_utc.iloc[-1] if len(held) else pd.NaT,
               missing_paired_minutes=missing, risk_complete=missing == 0,
               inactive_paired_minutes_held=int(np.sum(~market.eligible[left:right])),
               maximum_path_gap_minutes=float(np.max(np.diff(times))/MINUTE_NS),
               entry_basis_bps=(f0-s0)/s0*1e4, exit_basis_bps=(fx-sx)/sx*1e4,
               gross_price_mae=float(min(0., np.min(gross))))
    scenarios = []
    for name, costs in cfg["cost_scenarios"].items():
        fee_entry, slip_entry = fill_cost(q, s0, f0, costs)
        fee_exit, slip_exit = fill_cost(q, sx, fx, costs)
        fees = np.full(len(times), fee_entry)
        slip = np.full(len(times), slip_entry)
        fees[-1] += fee_exit
        slip[-1] += slip_exit
        finance = N*float(costs["financing_apr"])*elapsed_years
        net = gross+funding-fees-slip-finance
        # Conservative pre-unwind cash: charge exit friction before releasing spot sale cash.
        short_cash = perp_pnl+funding-fees-slip-finance
        reserve_needed = max(0., float(np.max(cfg["maintenance_fraction"]*q*f-short_cash)))
        total_needed = N+reserve_needed
        margin_breach = total_needed > C+1e-7
        risk_known = row["risk_complete"] and row["funding_complete"]
        result = {**row, "scenario": name, "fees": float(fees[-1]), "slippage": float(slip[-1]),
                  "financing_hurdle": float(finance[-1]), "net_pnl": float(net[-1]),
                  "net_return_bps": float(net[-1]/C*1e4), "net_spot_notional_bps": float(net[-1]/N*1e4),
                  "net_mae_usdt": float(min(0., np.min(net))),
                  "net_mae_capital_bps": float(min(0., np.min(net))/C*1e4),
                  "minimum_wallet_proxy": float(np.min(C-N+short_cash)),
                  "required_initial_cash": total_needed, "required_capital_multiple": total_needed/N,
                  "observed_margin_breach": margin_breach,
                  "feasible_complete": bool(risk_known and not margin_breach),
                  "risk_status": "observed_breach" if margin_breach else "complete" if risk_known else "unknown",
                  "required_cash_maintenance_2pct": N+max(0., float(np.max(cfg["maintenance_sensitivity"]*q*f-short_cash)))}
        for multiple in cfg["capital_multiple_sensitivities"]:
            result[f"margin_breach_at_{multiple:g}x"] = bool(total_needed > multiple*N+1e-7)
        scenarios.append(result)
    ledger = [{"settlement_time_utc": event.settlement_time_utc, "available_at_utc": event.available_at_utc,
               "funding_rate": event.funding_rate, "perp_price_proxy": event.perp_price_proxy,
               "quantity": q, "funding_cash_proxy": float(amount)}
              for event, amount in zip(held.itertuples(index=False), amounts)]
    return row, scenarios, ledger
