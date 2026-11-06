"""Read-only observation adapters for the frozen annual forecasting contract.

No rho/features, labels, risk sets, fitted state or forecasts are constructed.
Final input permission does not change any existing numerical access guard.
"""

import numpy as np
import pandas as pd

from .final_sources import ASSETS, START, END
from .funding import rates_asof, prepare_effr, join_funding
from .windows import annual_windows, origin_mask


FINANCING = ("zero_rate", "nyfed_effr_lag7d")
HISTORY_START = pd.Timestamp("2020-12-31", tz="UTC")
RAW = ["symbol", "bar_open_time_utc"] + [v+"_"+c for v in ("spot", "perp")
       for c in ("close", "volume", "trades", "observed", "valid")]
FUNDING = ["funding_known", "last_funding_settlement_utc", "last_funding_available_at_utc",
           "last_settled_funding_rate", "last_funding_interval_hours", "funding_age_minutes", "funding_stale"]
REFERENCE = ["effr_annual_rate", "effr_observation_time_utc", "effr_available_at_utc", "zero_rate_annual_rate"]


def utc_clock(index):
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC"
            or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing
            or index.unit != "ns"):
        raise ValueError("Require unique ordered UTC-nanosecond decision timestamps")
    if len(index) and (index.min() < HISTORY_START or index.max() >= END):
        raise ValueError("Input outside required historical warmup/final clock")


def observation_masks(frame, symbol):
    """Preserve the grid and separately identify missing, invalid and inactivity."""
    utc_clock(frame.index)
    if symbol not in ASSETS or not frame.columns.is_unique or not set(RAW) <= set(frame):
        raise ValueError("Wrong asset or incomplete observation schema")
    if not frame.symbol.eq(symbol).all():
        raise ValueError("Mixed asset observations")
    opens = pd.DatetimeIndex(frame.bar_open_time_utc)
    if str(opens.tz) != "UTC" or not (opens + pd.Timedelta(minutes=1)).equals(frame.index):
        raise ValueError("Require immediately completed minute, not current opening bar")
    masks = pd.DataFrame(index=frame.index)
    for v in ("spot", "perp"):
        for c in ("observed", "valid"):
            x = frame[v+"_"+c]
            if not pd.api.types.is_bool_dtype(x) or x.isna().any():
                raise ValueError("Explicit boolean observation/validity masks required")
            masks[v+"_"+c] = x
        observed, valid = frame[v+"_observed"], frame[v+"_valid"]
        if (valid & ~observed).any():
            raise ValueError("Missing observation declared valid")
        for c in ("close", "volume", "trades"):
            x = frame[v+"_"+c]
            if not pd.api.types.is_numeric_dtype(x) or pd.api.types.is_bool_dtype(x):
                raise ValueError("Numeric price/activity input required")
            if c == "close": good = np.isfinite(x) & x.gt(0)
            else: good = np.isfinite(x) & x.ge(0)
            if c == "trades": good &= x.mod(1).eq(0)
            if (valid & ~good).any():
                raise ValueError("Invalid value declared a valid market observation")
        active = valid & frame[v+"_volume"].gt(0) & frame[v+"_trades"].gt(0)
        masks[v+"_active"] = active
        masks[v+"_state"] = np.select([~observed, ~valid, ~active],
                                      ["missing", "invalid", "inactive"], default="active")
    masks["pair_valid"] = masks.spot_valid & masks.perp_valid
    masks["pair_active"] = masks.spot_active & masks.perp_active
    for old, new in (("paired_valid", "pair_valid"), ("pair_valid", "pair_valid"),
                     ("active_sample", "pair_active"), ("pair_active", "pair_active")):
        if old in frame and not frame[old].equals(masks[new].rename(old)):
            raise ValueError("Stored validity/activity disagrees with registered definition")
    return masks


def reference_history(raw):
    """Reuse the existing guarded percent/100 and seven-day historical treatment."""
    rates = prepare_effr(raw)
    for c in ("observation_time_utc", "available_at_utc"):
        rates[c] = pd.DatetimeIndex(rates[c]).as_unit("ns")
    if rates.observation_time_utc.duplicated().any():
        raise ValueError("Duplicate historical EFFR dates/revisions")
    return rates


def adapt_hourly(frame, symbol, financing, *, historical_rates=None):
    """Return the existing annual base-input schema plus separate validity masks.

    Input prices are copied at their original scale; invalid/missing prices are
    masked, valid inactive marks stay present. No future status selects rows.
    Rho and the registered six features remain the existing builder's job, under
    later numerical authorization. Nothing here issues a forecasting origin.
    """
    if financing not in FINANCING:
        raise ValueError("Unregistered financing control")
    masks = observation_masks(frame, symbol)
    if len(frame) and not frame.index.equals(pd.date_range(frame.index.min(), frame.index.max(), freq="h").as_unit("ns")):
        raise ValueError("Retain every scheduled hour, including unavailable observations")
    if not frame.index.floor("h").equals(frame.index):
        raise ValueError("Annual inputs require exact hourly origins")
    if not set(FUNDING) <= set(frame):
        raise ValueError("Missing funding availability schema")
    for c in ("last_funding_settlement_utc", "last_funding_available_at_utc"):
        if not isinstance(frame[c].dtype, pd.DatetimeTZDtype) or str(frame[c].dt.tz) != "UTC":
            raise ValueError("UTC funding timestamps required")
    known = frame.funding_known
    if not pd.api.types.is_bool_dtype(known) or known.isna().any():
        raise ValueError("Explicit funding-known flags required")
    settlement, available = frame.last_funding_settlement_utc, frame.last_funding_available_at_utc
    age = (frame.index - pd.DatetimeIndex(settlement)).total_seconds()/3600
    if (available.notna() & (available.gt(frame.index) | available.ne(settlement+pd.Timedelta(minutes=1)))).any():
        raise ValueError("Funding metadata contains future/nonregistered publication")
    if (known & (~np.isfinite(age) | (age<0) | (age>12) | available.isna()
                 | available.gt(frame.index) | available.ne(settlement+pd.Timedelta(minutes=1)))).any():
        raise ValueError("Noncausal/stale funding declared known")
    out = pd.DataFrame(index=frame.index.rename("origin_time_utc"))
    out["symbol"], out["financing"] = symbol, financing
    for v in ("spot", "perp"):
        out[v+"_close"] = frame[v+"_close"].where(masks[v+"_valid"]).to_numpy(dtype=float)
    out["bar_open_time_utc"] = pd.DatetimeIndex(frame.bar_open_time_utc).as_unit("ns")
    out["bar_available_at_utc"] = out.index
    if financing == "zero_rate":
        out["annual_rate"] = 0.0
        out["rate_available_at_utc"] = out.index
    elif set(REFERENCE) <= set(frame):
        av, effective = frame.effr_available_at_utc, frame.effr_observation_time_utc
        if (av.isna().any() or av.gt(frame.index).any()
                or not av.eq(effective+pd.Timedelta(days=7)).all()):
            raise ValueError("Noncausal/incomplete lagged EFFR input")
        if not frame.zero_rate_annual_rate.eq(0).all():
            raise ValueError("Wrong constant-zero financing input")
        out["annual_rate"] = frame.effr_annual_rate.to_numpy(dtype=float)
        out["rate_available_at_utc"] = pd.DatetimeIndex(av).as_unit("ns")
    else:
        if historical_rates is None or (frame.index >= START).any():
            raise ValueError("Verified historical EFFR or canonical final reference required")
        values, av = rates_asof(frame.index, historical_rates)
        out["annual_rate"] = values
        out["rate_available_at_utc"] = pd.to_datetime(av, utc=True).as_unit("ns")
    for c in ("funding_known", "last_funding_settlement_utc", "last_funding_available_at_utc"):
        out[c] = frame[c].to_numpy() if c == "funding_known" else pd.DatetimeIndex(frame[c]).as_unit("ns")
    out["theory_available"] = masks.pair_active.to_numpy() & np.isfinite(out.annual_rate) & out.rate_available_at_utc.notna()
    # Auxiliary source input, deliberately outside the registered model schema.
    masks["last_settled_funding_rate"] = frame.last_settled_funding_rate
    return out, masks


def check_funding_clock(frame, events):
    """Reuse the source join, validating statuses and exact timestamps only."""
    joined = join_funding(frame[["symbol"]], events, maximum_age_hours=12)
    for c in ("funding_known", "last_funding_settlement_utc", "last_funding_available_at_utc"):
        if not joined[c].equals(frame[c]):
            raise ValueError("Canonical funding clock/status differs from actual events: "+c)


def annual_plan(year):
    w = annual_windows(year)
    if year not in (2023, 2024, 2025):
        raise ValueError("Only registered final annual plans")
    result = w.as_dict()
    result["warmup"] = [(w.fit[0]-pd.Timedelta(hours=24)).isoformat(), w.fit[0].isoformat()]
    result["fit_calibration_origin_rule"] = "begin<=t and t+16h<end"
    result["trial_or_label_publication_rule"] = "publication<segment_end"
    result["input_publication_rule"] = "availability<=origin"
    result["rolling_terminal_publication_rule"] = "publication<origin"
    result["evaluation_origin_purge_hours"] = 0
    result["schedule_counts"] = {}
    for name in ("fit", "calibration", "evaluation"):
        begin, end = getattr(w, name)
        clock = pd.date_range(begin, end, freq="h", inclusive="left").as_unit("ns")
        result["schedule_counts"][name] = {"clock": len(clock), "origin_bound_permitted": int(origin_mask(clock, begin, end, purge=name!="evaluation").sum())}
    return result
