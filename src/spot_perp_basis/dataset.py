"""Minimal compatibility for immutable archived imports; no implementations."""
from .data.market import PRICE_COLUMNS, intervals, audit_market, pair_bars
from .data.ingest import parse_klines
from .data.windows import MINUTE_NS, training_partitions, require_training_period
