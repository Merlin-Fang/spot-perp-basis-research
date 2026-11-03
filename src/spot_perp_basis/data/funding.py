"""Actual funding events and causal funding/financing/reference availability.

Predictor publication lag is separate from cash ownership at (entry, exit].
The EFFR control is financing, not perpetual funding or original Aave history."""

import numpy as np
import pandas as pd

from .windows import START, END, MINUTE_NS, ns_index, require_training_times


# Settlements published to predictors

def prepare_funding(events, lag_minutes=1):
    events = events.sort_values("settlement_time_utc").reset_index(drop=True).copy()
    if events.settlement_time_utc.duplicated().any():
        raise ValueError("Duplicate funding settlements across archives")
    if lag_minutes < 0:
        raise ValueError("Negative publication lag is forbidden")
    events["available_at_utc"] = events.settlement_time_utc + pd.Timedelta(minutes=lag_minutes)
    events["observed_gap_hours"] = events.settlement_time_utc.diff().dt.total_seconds() / 3600
    # Interval metadata describes a settled event, never a promised future schedule.
    events["cadence_gap_flag"] = events.observed_gap_hours.gt(events.funding_interval_hours + 1 / 3600)
    return events


def join_funding(paired, events, maximum_age_hours=12):
    if not paired.index.is_monotonic_increasing or paired.index.has_duplicates:
        raise ValueError("Paired decision timestamps must be sorted and unique")
    columns = ["settlement_time_utc", "available_at_utc", "funding_rate", "funding_interval_hours"]
    result = pd.merge_asof(paired.reset_index(), events[columns].sort_values("available_at_utc"),
                           left_on="decision_time_utc", right_on="available_at_utc", direction="backward",
                           allow_exact_matches=True).set_index("decision_time_utc")
    result = result.rename(columns={"settlement_time_utc": "last_funding_settlement_utc",
                                   "available_at_utc": "last_funding_available_at_utc",
                                   "funding_rate": "last_settled_funding_rate",
                                   "funding_interval_hours": "last_funding_interval_hours"})
    result["funding_age_minutes"] = (result.index - result.last_funding_settlement_utc).dt.total_seconds() / 60
    result["funding_stale"] = result.funding_age_minutes.gt(maximum_age_hours * 60)
    result["funding_known"] = result.last_funding_settlement_utc.notna() & ~result.funding_stale
    result.loc[~result.funding_known, "last_settled_funding_rate"] = np.nan
    if (result.last_funding_available_at_utc.dropna() > result.index[result.last_funding_available_at_utc.notna()]).any():
        raise ValueError("Future funding leaked into decision-time context")
    return result


# Launch quarantine and observed cadence

def funding_history(events, symbol, calendar, lag_minutes=1):
    if not events.symbol.eq(symbol).all():
        raise ValueError("Cross-symbol funding contamination")
    times = pd.DatetimeIndex(events.settlement_time_utc)
    if len(times) and (times.min()<START or times.max()>=END):
        raise ValueError("Out-of-window funding")
    result = prepare_funding(events, lag_minutes)
    public = pd.Timestamp(calendar.get("public_open_utc") or START)
    observed = pd.Timestamp(calendar["first_observed_bar_utc"])
    result["quarantine_reason"] = np.where(result.settlement_time_utc<public,"before_public_open",
        np.where(result.settlement_time_utc<observed,"before_first_observed_bar",""))
    result["quarantined"] = result.quarantine_reason.ne("")
    result["eligible_for_causal_use"] = ~result.quarantined & result.available_at_utc.lt(END)
    result["eligible_for_accounting"] = ~result.quarantined
    result["interval_change_flag"] = result.funding_interval_hours.ne(result.funding_interval_hours.shift()) & result.index.to_series().gt(0)
    result["interval_mismatch_flag"] = (result.observed_gap_hours-result.funding_interval_hours).abs().gt(1/3600)
    result["cross_month_event"] = result.source_month.ne(result.source_month.shift()) & result.index.to_series().gt(0)
    return result


# Separate lagged EFFR financing control

def prepare_effr(raw, lag_days=7):
    """Official percent quotes -> annual decimal proxy; explicit availability.

    The ACT/365 treatment is our sensitivity convention, not a conversion
    into an actual stablecoin loan. No APY conversion or incentive yield.
    """
    if lag_days < 7:
        raise ValueError("The registered conservative lag is at least seven days")
    frame = pd.DataFrame(raw["refRates"])
    times = pd.to_datetime(frame.effectiveDate, utc=True)
    require_training_times(pd.DatetimeIndex(times).sort_values())
    if not frame.type.eq("EFFR").all():
        raise ValueError("Unexpected rate instrument")
    values = pd.to_numeric(frame.percentRate, errors="raise").to_numpy(float)/100
    if not np.isfinite(values).all():
        raise ValueError("Invalid EFFR quotes")
    result = pd.DataFrame({"observation_time_utc": times,
        "available_at_utc": times+pd.Timedelta(days=lag_days), "annual_rate": values,
        "source": "NYFed_EFFR_annual_percent_ACT365_proxy"})
    return result.sort_values("available_at_utc").reset_index(drop=True)


def rates_asof(times, rates):
    """Backward only; no fill before the first known rate, no future context."""
    requested = pd.DatetimeIndex(times)
    require_training_times(requested, allow_duplicates=True)
    av = pd.DatetimeIndex(rates.available_at_utc)
    if av.has_duplicates or not av.is_monotonic_increasing or av.tz is None:
        raise ValueError("Rate availability must be ordered, unique and timezone aware")
    require_training_times(pd.DatetimeIndex(rates.observation_time_utc))
    if np.any(ns_index(rates.observation_time_utc) > ns_index(av)):
        raise ValueError("Rate cannot be available before observation")
    rows = np.searchsorted(ns_index(av), ns_index(requested), side="right")-1
    found = rows >= 0
    result = np.full(len(requested), np.nan)
    available = pd.Series(pd.NaT, index=np.arange(len(requested)), dtype="datetime64[ns, UTC]")
    result[found] = rates.annual_rate.to_numpy(float)[rows[found]]
    available.loc[found] = av[rows[found]]
    return result, available.to_numpy()


# Causal reference-price availability

def reference_asof(frame, queries, *, feature, max_age_minutes=2):
    """Never forward join or silently substitute a last-trade price."""
    times = ns_index(frame.feature_available_at_utc if feature else frame.observation_time_utc)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Reference history must be unique and ordered")
    queries = ns_index(queries)
    rows = np.searchsorted(times, queries, side="right") - 1
    values = np.full(len(queries), np.nan)
    ages = np.full(len(queries), np.nan)
    good = rows >= 0
    ages[good] = (queries[good]-times[rows[good]])/MINUTE_NS
    good &= ages <= max_age_minutes
    values[good] = frame.close.to_numpy()[rows[good]]
    return values, ages
