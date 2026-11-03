"""Market-bar validation, exact pairing and launch-aware observation panels.

Completed minute endpoints and trailing-hour aggregates remain distinct."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .funding import join_funding
from .windows import MINUTE_NS, ns_index, bounded


# Exact-minute spot/perpetual alignment

MINUTE = pd.Timedelta(minutes=1)


def align_markets(spot, perpetual):
    """Inner join valid paired observations; keep a separate explicit missing grid."""
    columns = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
               "taker_buy_volume", "taker_buy_quote_volume"]
    joined = spot.loc[spot.valid, columns].add_prefix("spot_").join(
        perpetual.loc[perpetual.valid, columns].add_prefix("perp_"), how="inner", validate="one_to_one")
    joined.insert(0, "bar_open_time_utc", joined.index)
    joined.index = (joined.index + MINUTE).rename("decision_time_utc")
    joined["basis_bps"] = 10_000 * (np.log(joined.perp_close) - np.log(joined.spot_close))
    spot_ohlc4 = joined[["spot_open", "spot_high", "spot_low", "spot_close"]].mean(axis=1)
    perp_ohlc4 = joined[["perp_open", "perp_high", "perp_low", "perp_close"]].mean(axis=1)
    joined["basis_ohlc4_bps"] = 10_000 * (np.log(perp_ohlc4) - np.log(spot_ohlc4))
    grid = (spot.index.union(perpetual.index) + MINUTE).rename("decision_time_utc")
    return joined, joined.reindex(grid)


# OHLCV quality and paired bars

PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
                 "taker_buy_volume", "taker_buy_quote_volume"]


def intervals(index, mask):
    positions = np.flatnonzero(np.asarray(mask))
    if not len(positions):
        return []
    groups = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
    return [{"start_utc": str(index[g[0]]), "end_exclusive_utc": str(index[g[-1]] + MINUTE),
             "minutes": len(g)} for g in groups]


def audit_market(frame, audit, previous=None):
    """Audit only structural anomalies and counts, including cross-file boundaries."""
    result = dict(audit)
    for name in ("raw_first_row", "raw_last_row"):
        result.pop(name, None)  # Never export validation/test prices into integrity tables.
    price_mean = frame.quote_volume / frame.volume.where(frame.volume.gt(0))
    frame = frame.copy()
    frame["quote_price_flag"] = frame.valid & ((price_mean < frame.low - 1e-4) | (price_mean > frame.high + 1e-4))
    frame["zero_activity_inconsistent"] = frame.valid & ((frame.volume.eq(0) & frame.quote_volume.ne(0)) |
                                         (frame.trades.eq(0) & frame.volume.gt(0)))
    comparable = frame.valid & frame.valid.shift(1, fill_value=False)
    unchanged = comparable & frame.close.eq(frame.close.shift(1))
    returns = np.log(frame.close.where(frame.valid)).diff()
    if previous is not None and previous["valid"] and frame.valid.iloc[0]:
        unchanged.iloc[0] = frame.close.iloc[0] == previous["close"]
        returns.iloc[0] = np.log(frame.close.iloc[0] / previous["close"])
    frame["unchanged_close"] = unchanged
    frame["large_return_flag"] = returns.abs().gt(.10)
    # These are integrity flags, not return distributions or held-out performance.
    result.update(unchanged_close_minutes=int(unchanged.sum()),
                  quote_price_flags=int(frame.quote_price_flag.sum()),
                  zero_activity_inconsistent_minutes=int(frame.zero_activity_inconsistent.sum()),
                  large_return_flags=int(frame.large_return_flag.sum()))
    return frame, result


def pair_bars(spot, perp):
    """Join completed, valid observations only; no basis or future labels calculated."""
    selected = PRICE_COLUMNS + ["unchanged_close", "quote_price_flag", "large_return_flag"]
    paired = spot.loc[spot.valid, selected].add_prefix("spot_").join(
        perp.loc[perp.valid, selected].add_prefix("perp_"), how="inner", validate="one_to_one")
    paired.insert(0, "bar_open_time_utc", paired.index)
    paired.index = (paired.index + MINUTE).rename("decision_time_utc")
    paired["spot_zero_volume"] = paired.spot_volume.eq(0)
    paired["perp_zero_volume"] = paired.perp_volume.eq(0)
    paired["contiguous_previous_minute"] = paired.index.to_series().diff().eq(MINUTE).to_numpy()
    return paired


# Completed-bar market and backward marks

@dataclass
class Market:
    frame: pd.DataFrame

    def __post_init__(self):
        self.times = ns_index(self.frame.index)
        self.opens = ns_index(self.frame.bar_open_time_utc)
        if (len(self.times) == 0 or np.any(np.diff(self.times) <= 0)
                or np.any(self.times != self.opens + MINUTE_NS)):
            raise ValueError("Require unique ordered completed minute bars")
        self.eligible = np.ones(len(self.times), dtype=bool)
        for venue in ("spot", "perp"):
            for column in ("volume", "trades"):
                self.eligible &= self.frame[f"{venue}_{column}"].to_numpy() > 0
        self.fill_rows = np.flatnonzero(self.eligible)
        self.fill_times = self.opens[self.fill_rows]

    def fill_after(self, time):
        pos = np.searchsorted(self.fill_times, time)
        return None if pos == len(self.fill_rows) else int(self.fill_rows[pos])

    def basis(self, measurement):
        if measurement == "close":
            spot, perp = self.frame.spot_close.to_numpy(), self.frame.perp_close.to_numpy()
        elif measurement == "OHLC4":
            spot = self.frame[[f"spot_{c}" for c in ("open", "high", "low", "close")]].mean(axis=1).to_numpy()
            perp = self.frame[[f"perp_{c}" for c in ("open", "high", "low", "close")]].mean(axis=1).to_numpy()
        else:
            raise ValueError("Unknown measurement")
        return (perp - spot) / spot


def asof_prices(events, price_times, prices):
    """No future interpolation; return an explicit age with each price proxy."""
    t = ns_index(events)
    pt = ns_index(price_times)
    if np.any(np.diff(pt) <= 0):
        raise ValueError("Price context must be unique and ordered")
    rows = np.searchsorted(pt, t, side="right") - 1
    valid = rows >= 0
    result = np.full(len(t), np.nan)
    age = np.full(len(t), np.nan)
    result[valid] = np.asarray(prices)[rows[valid]]
    age[valid] = (t[valid] - pt[rows[valid]]) / MINUTE_NS
    return result, age


# Launch-aware minute/hour observation panels

def market_quality(bars, previous, cfg):
    bars, audit = audit_market(bars, {}, previous)
    implied = bars.quote_volume/bars.volume.where(bars.volume.gt(0))
    tolerance = np.maximum(bars.close.abs()*cfg["quote_price_relative_tolerance"],cfg["quote_price_absolute_floor"])
    bars["quote_price_flag"] = bars.valid & ((implied<bars.low-tolerance)|(implied>bars.high+tolerance))
    return bars


def minute_panel(spot, perp, events, symbol, cfg, previous=None):
    calendar = cfg["assets"][symbol]
    grid = spot.index.union(perp.index)
    # Filter before creating any completed-bar observation at/after 2023.
    grid = grid[(grid>=pd.Timestamp(cfg["start"])) & (grid+MINUTE<pd.Timestamp(cfg["end"]))]
    spot,perp = spot.reindex(grid),perp.reindex(grid)
    frame = pd.DataFrame(index=(grid+MINUTE).rename("decision_time_utc"))
    frame["symbol"] = symbol
    frame["bar_open_time_utc"] = grid
    frame["bar_available_at_utc"] = frame.index
    earliest = frame.index+cfg["execution_delay_minutes"]*MINUTE
    frame["earliest_execution_time_utc"] = earliest.where(earliest<pd.Timestamp(cfg["end"]))
    public = pd.Timestamp(calendar.get("public_open_utc") or cfg["start"])
    frame["structural_prelisting"] = grid<public
    for venue,bars in (("spot",spot),("perp",perp)):
        fields = PRICE_COLUMNS+["observed","valid","bad_price","bad_activity","bad_close_time",
            "unchanged_close","quote_price_flag","large_return_flag","zero_activity_inconsistent"]
        for field in fields: frame[f"{venue}_{field}"] = bars[field].to_numpy()
        frame[f"{venue}_close_time_utc"] = pd.to_datetime(bars.close_time,unit=bars.attrs.get("timestamp_unit","ms"),utc=True).to_numpy()
        listed = np.ones(len(frame),dtype=bool) if venue=="spot" else ~frame.structural_prelisting
        frame[f"{venue}_missing"] = listed & ~frame[f"{venue}_observed"]
        frame[f"{venue}_inactive"] = frame[f"{venue}_valid"] & (frame[f"{venue}_volume"].eq(0)|frame[f"{venue}_trades"].eq(0))
    # Use the existing exact-minute pairing, preserving an explicit outer grid.
    paired = pair_bars(spot,perp)
    frame["paired_valid"] = frame.index.isin(paired.index) & ~frame.structural_prelisting
    frame["active_sample"] = frame.paired_valid & ~frame.spot_inactive & ~frame.perp_inactive
    frame["pair_status"] = np.select([frame.structural_prelisting,~frame.paired_valid,~frame.active_sample],
        ["structural_prelisting","missing_or_invalid","inactive"],default="active")
    frame["contiguous_previous_minute"] = frame.paired_valid & frame.paired_valid.shift(1,fill_value=False)
    if previous is not None and len(frame):
        frame.iloc[0,frame.columns.get_loc("contiguous_previous_minute")] = bool(
            frame.paired_valid.iloc[0] and previous["paired_valid"] and frame.index[0]-previous.name==MINUTE)
    frame["basis_bps"] = (np.log(frame.perp_close)-np.log(frame.spot_close)).mul(1e4).where(frame.paired_valid)
    s4 = frame[[f"spot_{c}" for c in ("open","high","low","close")]].mean(axis=1)
    f4 = frame[[f"perp_{c}" for c in ("open","high","low","close")]].mean(axis=1)
    frame["basis_ohlc4_bps"] = (np.log(f4)-np.log(s4)).mul(1e4).where(frame.paired_valid)
    frame["premium_perp_denominator_bps"] = ((frame.perp_close-frame.spot_close)/frame.perp_close*1e4).where(frame.paired_valid)
    frame["cross_market_dislocation_flag"] = frame.basis_bps.abs().gt(cfg["cross_market_log_basis_threshold_bps"])
    eligible = events.loc[events.eligible_for_causal_use]
    if not eligible.symbol.eq(symbol).all(): raise ValueError("Cross-symbol funding")
    frame = join_funding(frame,eligible,cfg["funding_maximum_age_hours"])
    hashes = eligible.set_index("settlement_time_utc").source_sha256
    frame["last_funding_source_sha256"] = frame.last_funding_settlement_utc.map(hashes)
    bounded(frame)
    return frame


def hourly_panel(minute, symbol, cfg):
    bounded(minute)
    calendar = cfg["assets"][symbol]
    first = max(pd.Timestamp(cfg["start"]),pd.Timestamp(calendar.get("public_open_utc") or cfg["start"]))+MINUTE
    clock = pd.date_range(max(first.ceil("h"),minute.index.min().ceil("h")),
        min(pd.Timestamp(cfg["end"])-MINUTE,minute.index.max()),freq="h")
    hourly = minute.reindex(clock).copy()
    hourly.index.name = "decision_time_utc"
    hourly["symbol"] = symbol
    hourly["hour_start_utc"] = clock-pd.Timedelta(hours=1)
    hourly["hour_end_utc"] = clock
    hourly["pair_complete_hour"] = minute.paired_valid.rolling(60,min_periods=1).sum().eq(60).reindex(clock)
    hourly["active_minutes_in_hour"] = minute.active_sample.rolling(60,min_periods=1).sum().reindex(clock).astype("int64")
    for venue in ("spot","perp"):
        valid = minute[f"{venue}_valid"]
        complete = valid.rolling(60,min_periods=1).sum().eq(60)
        hourly[f"{venue}_hour_observed_minutes"] = minute[f"{venue}_observed"].rolling(60,min_periods=1).sum().reindex(clock).astype("int64")
        hourly[f"{venue}_hour_valid_minutes"] = valid.rolling(60,min_periods=1).sum().reindex(clock).astype("int64")
        hourly[f"{venue}_hour_complete"] = complete.reindex(clock)
        for field in PRICE_COLUMNS:
            values = minute[f"{venue}_{field}"].where(valid)
            if field=="open": agg=values.shift(59)
            elif field=="close": agg=values
            elif field=="high": agg=values.rolling(60,min_periods=60).max()
            elif field=="low": agg=values.rolling(60,min_periods=60).min()
            else: agg=values.rolling(60,min_periods=60).sum()
            hourly[f"{venue}_hour_{field}"] = agg.where(complete).reindex(clock)
    bounded(hourly)
    return hourly


# Observed path summary helpers shared by predictors and outcomes

class RangeMax:
    """O(n) storage, bounded 256-value edge queries and O(1) interior RMQ."""

    def __init__(self, values, block=256):
        self.values = np.asarray(values, float)
        self.block = block
        n = len(self.values)
        padded = np.full(((n+block-1)//block)*block, -np.inf)
        padded[:n] = self.values
        matrix = padded.reshape(-1, block)
        self.prefix = np.maximum.accumulate(matrix, axis=1).ravel()[:n]
        self.suffix = np.maximum.accumulate(matrix[:, ::-1], axis=1)[:, ::-1].ravel()[:n]
        self.levels = [matrix.max(axis=1)]
        width = 2
        while width <= len(matrix):
            previous = self.levels[-1]
            half = width//2
            self.levels.append(np.maximum(previous[:-half], previous[half:]))
            width *= 2

    def query(self, left, right):
        if left >= right:
            return -np.inf
        if left < 0 or right > len(self.values):
            raise IndexError("RMQ outside price history")
        a, b = left//self.block, (right-1)//self.block
        if a == b:
            return float(self.values[left:right].max())
        result = max(self.suffix[left], self.prefix[right-1])
        length = int(b-a-1)
        if length:
            level = length.bit_length()-1
            width = 1 << level
            result = max(result, self.levels[level][a+1], self.levels[level][b-width])
        return float(result)


def prefix(values):
    return np.r_[0., np.cumsum(np.asarray(values, float))]
