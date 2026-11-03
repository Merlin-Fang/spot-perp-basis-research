"""Minimal compatibility for immutable archived imports; no implementations."""
from .data.ingest import COLUMNS, parse_klines
from .data.market import MINUTE, align_markets
from .data.labels import horizon_targets
from .models.baselines import fit_mean_reversion
from .evaluation.metrics import benchmark, cost_comparison
