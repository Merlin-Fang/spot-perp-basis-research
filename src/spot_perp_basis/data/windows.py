"""UTC units, development access guards and publication-purged chronological windows."""

import numpy as np
import pandas as pd
from dataclasses import dataclass


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


@dataclass(frozen=True)
class AnnualWindows:
    """Pure calendar metadata; constructing it grants no observation access."""
    year: int
    fit: tuple
    calibration: tuple
    evaluation: tuple

    def __post_init__(self):
        start = pd.Timestamp(f'{self.year}-01-01', tz='UTC')
        expected = ((start-pd.DateOffset(months=24), start-pd.DateOffset(months=4)),
                    (start-pd.DateOffset(months=4), start),
                    (start, start+pd.DateOffset(years=1)))
        actual = (self.fit, self.calibration, self.evaluation)
        if actual != expected:
            raise ValueError('Require exact UTC 20/4/full-calendar-year half-open windows')
        if any(t.tz is None or str(t.tz) != 'UTC' for w in actual for t in w):
            raise ValueError('Require UTC calendar boundaries')

    def as_dict(self):
        return {'year':self.year, **{n:[t.isoformat() for t in getattr(self,n)]
                for n in ('fit','calibration','evaluation')},
                'role':'registered_calendar_only' if self.year in (2023,2024,2025) else 'development_demonstration'}


def annual_windows(year):
    """Registered schedules, plus compatible earlier dates for explicit demos."""
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError('Integer calendar year required')
    start = pd.Timestamp(f'{year}-01-01', tz='UTC')
    return AnnualWindows(year, (start-pd.DateOffset(months=24),start-pd.DateOffset(months=4)),
                         (start-pd.DateOffset(months=4),start), (start,start+pd.DateOffset(years=1)))


def origin_mask(index, begin, end, *, purge=False):
    """Pure half-open selection; optional strict t+16h<C fitting purge."""
    idx = pd.DatetimeIndex(index)
    begin, end = pd.Timestamp(begin), pd.Timestamp(end)
    if idx.tz is None or begin.tz is None or end.tz is None or not begin < end:
        raise ValueError('Timezone-aware ordered interval required')
    mask = (idx>=begin)&(idx<end)
    if purge: mask &= idx+pd.Timedelta(hours=16)<end
    return mask
