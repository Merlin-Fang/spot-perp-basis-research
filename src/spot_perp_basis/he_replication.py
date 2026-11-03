"""Minimal compatibility for immutable archived imports; no implementations."""
from .theory import ANNUAL_PERIODS, KAPPA, GAMMA, IOTA, theory_signal, parameter_signal, cost_bound, entry_side, exit_crossed
from .data.windows import START, END, require_training_times
from .data.funding import prepare_effr, rates_asof
from .data.features import hourly_theory_observations as hourly_observations
from .backtest.execution import convergence_positions
from .backtest.accounting import COMPONENTS, selected_funding, account_positions
from .evaluation.metrics import active_hour_metrics, summarize
