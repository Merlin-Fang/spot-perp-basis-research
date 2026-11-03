"""Earlier funding and basis mean-reversion numerical controls.

Retained for public carry/pilot tests and workflows; not new survival models."""

import numpy as np
import pandas as pd


# Earlier funding controls

def funding_forecasts(fit, evaluation, minimum_rows=100):
    """Frozen baselines and two tiny OLS specifications; no hyperparameter search."""
    if len(fit) < minimum_rows:
        raise ValueError("Insufficient prespecified calibration data")
    multiplier = evaluation.horizon_hours.to_numpy()/8
    predictions = {"zero": np.zeros(len(evaluation)),
                   "persistence": evaluation.funding_last_bps.to_numpy()*multiplier,
                   "expanding_mean": evaluation.funding_expanding_mean_bps.to_numpy()*multiplier}
    for days in (1, 7, 30):
        predictions[f"mean_{days}d"] = evaluation[f"funding_mean_{days}d_bps"].to_numpy()*multiplier
    coefficients = []
    for name, columns in (("linear_last", ["funding_last_bps"]),
                          ("linear_last_and_7d", ["funding_last_bps", "funding_mean_7d_bps"])):
        x = np.column_stack([np.ones(len(fit)), fit[columns].to_numpy()])
        y = fit.future_funding_rate_sum_bps.to_numpy()
        coefficient, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank < x.shape[1]:
            raise ValueError("Rank-deficient predeclared pilot regression")
        predictions[name] = np.column_stack([np.ones(len(evaluation)), evaluation[columns].to_numpy()]) @ coefficient
        coefficients.append({"model": name, "fit_rows": len(fit), "rank": int(rank),
                             "terms": ["intercept", *columns], "coefficients": coefficient.tolist(),
                             "last_fit_label_available_at": str(fit.label_available_at_utc.max())})
    return predictions, coefficients


# Earlier basis mean-reversion control

def fit_mean_reversion(calibration):
    if len(calibration) < 2:
        raise ValueError("Insufficient calibration labels")
    mean = float(calibration.basis_bps.mean())
    deviation = calibration.basis_bps.to_numpy() - mean
    denominator = float(deviation @ deviation)
    kappa = max(0.0, -float(deviation @ calibration.target_bps.to_numpy()) / denominator) if denominator else 0.0
    return {"mean_bps": mean, "kappa": kappa, "fit_samples": len(calibration)}
