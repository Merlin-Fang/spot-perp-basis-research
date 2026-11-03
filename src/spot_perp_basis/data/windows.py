"""UTC units, development access guards and publication-purged chronological windows."""

import numpy as np
import pandas as pd


# Development access bounds

START = pd.Timestamp("2020-01-01", tz="UTC")


END = pd.Timestamp("2023-01-01", tz="UTC")


def require_training_times(times, *, allow_duplicates=False):
    idx = pd.DatetimeIndex(times)
    if idx.tz is None or idx.hasnans or (idx.has_duplicates and not allow_duplicates) or not idx.is_monotonic_increasing:
        raise ValueError("Require unique ordered UTC-aware timestamps")
    if len(idx) and (idx.min() < START or idx.max() >= END):
        raise ValueError("Only 2020–2022 observations are permitted")


# Time units and conversion

MINUTE_NS = 60_000_000_000


YEAR_NS = 365 * 24 * 60 * MINUTE_NS


def ns_index(values):
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def stamp(value):
    return str(pd.Timestamp(int(value), tz="UTC"))


# Frame boundary validation

def bounded(frame):
    if frame.index.tz is None or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Require unique ordered UTC observation times")
    if len(frame) and (frame.index.min()<START or frame.index.max()>=END):
        raise ValueError("Observation outside 2020–2022")


# Select partitions before IO

def training_partitions(partitions, train_start, train_end):
    """Filter paths by catalog month BEFORE reading any values from disk."""
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    selected = []
    for row in partitions:
        month_start = pd.Timestamp(row["month"] + "-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthBegin()
        if month_start < end and month_end > start:
            selected.append(row)
    return sorted(selected, key=lambda row: (row["symbol"], row["month"]))


def require_training_period(start, end, splits):
    left, right = map(pd.Timestamp, splits["train"])
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if not (left <= start < end <= right):
        raise ValueError("Descriptive price analysis is restricted to the training interval")
    return start, end


# Publication-purged pilot folds

def purged_fold(frame, fold, embargo_hours):
    start, end = pd.Timestamp(fold["eval_start"]), pd.Timestamp(fold["eval_end"])
    origin = pd.to_datetime(frame.decision_time_utc, utc=True)
    available = pd.to_datetime(frame.label_available_at_utc, utc=True)
    fit = frame.loc[(origin >= pd.Timestamp(fold["fit_start"])) & (origin < start)
                    & (available < start-pd.Timedelta(hours=embargo_hours))]
    evaluation = frame.loc[(origin >= start) & (origin < end) & (available < end)]
    return fit, evaluation
