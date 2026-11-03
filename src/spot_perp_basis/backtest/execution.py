"""Original He delayed fills, pending exits and one-position chronological replay.

The original unfiltered and veto-capable implementations retain their verified
registered interfaces. Neither implements annual restart/checkpoint behavior."""

import numpy as np
import pandas as pd

from ..data.windows import MINUTE_NS, ns_index, stamp, require_training_times
from ..theory import cost_bound, entry_side, exit_crossed


# Original unfiltered He occupancy and delayed fills

def convergence_positions(market, observations, *, spot_fee_bps, perp_fee_bps, delay_minutes=1):
    """One position per asset. Pending orders execute at later active opens.

    A crossing can close and reverse at the same execution timestamp, closing
    first. Orders are never canceled using future price or rate information.
    """
    require_training_times(market.frame.index)
    require_training_times(observations.index)
    if delay_minutes < 1:
        raise ValueError("Require a full minute after the completed signal")
    bound = cost_bound(spot_fee_bps, perp_fee_bps)
    positions, current, blocked_until = [], None, -1
    for time, row in observations.iterrows():
        t = time.value
        if t <= blocked_until or not row.signal_available:
            continue
        if current is not None:
            if t <= market.opens[current["entry_row"]] or not exit_crossed(current["perpetual_side"], row.rho):
                continue
            fill = market.fill_after(t+delay_minutes*MINUTE_NS)
            current.update(exit_signal_utc=stamp(t), exit_signal_rho=float(row.rho), exit_row=fill)
            if fill is None:
                break  # retain open, exit pending, not a fabricated completion
            blocked_until = int(market.opens[fill])
            current = None
            # New opposite entry at this signal closes first at the same fill.
        else:
            fill = None
        side = entry_side(row.rho, bound)
        if not side:
            continue
        if fill is None:
            fill = market.fill_after(t+delay_minutes*MINUTE_NS)
        if fill is None:
            continue
        current = {"trade_number": len(positions)+1, "entry_signal_utc": stamp(t),
            "entry_signal_rho": float(row.rho), "entry_rate_annual": float(row.annual_rate),
            "entry_rate_available_at_utc": row.rate_available_at_utc,
            "hedge_lambda": float(row["lambda"]), "perpetual_side": side,
            "entry_row": fill, "exit_row": None, "exit_signal_utc": None,
            "exit_signal_rho": np.nan, "entry_bound_annual_decimal": bound}
        positions.append(current)
    return positions


# Chronological entry-veto replay

def replay(market,obs,fees,gate=None):
    """Only gate(t,side) can reveal Oracle values; it receives no fill or exit."""
    require_training_times(obs.index);require_training_times(market.frame.index)
    bound=cost_bound(**fees);positions=[];current=None;blocked=-1;decisions=[]
    tt=ns_index(obs.index);rho=obs.rho.to_numpy();valid=obs.signal_available.to_numpy(bool)
    lam=obs['lambda'].to_numpy();r=obs.annual_rate.to_numpy();av=obs.rate_available_at_utc.to_numpy()
    for i,t in enumerate(tt):
        if t<=blocked or not valid[i]:continue
        fill=None
        if current is not None:
            if t<=market.opens[current['entry_row']]:continue
            crossing=rho[i]<=0 if current['perpetual_side']==-1 else rho[i]>=0
            if not crossing:continue
            fill=market.fill_after(int(t)+MINUTE_NS)
            current.update(exit_signal_utc=stamp(t),exit_signal_rho=float(rho[i]),exit_row=fill)
            if fill is None:break
            blocked=int(market.opens[fill]);current=None
        side=-1 if rho[i]>bound else 1 if rho[i]<-bound else 0
        if not side:continue
        accepted=True if gate is None else bool(gate(i,side))
        decisions.append({'origin_time_utc':obs.index[i],'perpetual_side':side,'accepted':accepted,'origin_row':i})
        if not accepted:continue
        if fill is None:fill=market.fill_after(int(t)+MINUTE_NS)
        if fill is None:continue
        current={'trade_number':len(positions)+1,'entry_signal_utc':stamp(t),'entry_signal_rho':float(rho[i]),
            'entry_rate_annual':float(r[i]),'entry_rate_available_at_utc':av[i],'hedge_lambda':float(lam[i]),'perpetual_side':side,
            'entry_row':fill,'exit_row':None,'exit_signal_utc':None,'exit_signal_rho':np.nan,'entry_bound_annual_decimal':bound}
        positions.append(current)
    return positions,pd.DataFrame(decisions,columns=['origin_time_utc','perpetual_side','accepted','origin_row'])
