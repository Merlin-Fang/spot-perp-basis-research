"""Probability and same-key IPCW checks; no fitting, IO or probability repairs."""

import numpy as np
import pandas as pd

from .metrics import survival_metrics


def probability_audit(hazard, survival, probability):
    """Validate coherent complete curves, retaining entirely unavailable rows."""
    arrays = [np.asarray(v, float) for v in (hazard, survival, probability)]
    if any(a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Require matching two-dimensional hazard/survival/probability")
    joined = np.concatenate(arrays, axis=1)
    finite = np.isfinite(joined).all(axis=1)
    missing = np.isnan(joined).all(axis=1)
    if not (finite | missing).all():
        raise ValueError("Partial or infinite probability curve")
    if ((joined[finite] < 0) | (joined[finite] > 1)).any():
        raise ValueError("Probability outside [0,1]; no clipping is permitted")
    q, s, p = [a[finite] for a in arrays]
    if (np.diff(s, axis=1) > 1e-12).any() or (np.diff(p, axis=1) < -1e-12).any():
        raise ValueError("Nonmonotone cumulative curve")
    expected = np.cumprod(1-q, axis=1)
    if not np.allclose(s, expected, atol=1e-12, rtol=1e-12):
        raise ValueError("Survival is not the ordered product of conditional survivals")
    if not np.allclose(p, 1-s, atol=1e-12, rtol=1e-12):
        raise ValueError("Event/survival orientation mismatch")
    return {"rows": len(joined), "finite_rows": int(finite.sum()),
            "unavailable_rows": int(missing.sum()), "horizons": arrays[0].shape[1],
            "maximum_product_error": float(np.max(np.abs(s-expected))) if len(s) else None}


def matched_ipcw_scores(predictions, label, censor, horizon=8):
    """One named intersection and canonical support contract for all losses.

    Columns are required models, including globally unsupported all-null models.
    Returned losses contain early-censored zero contributions in the denominator.
    Unsupported censor support produces null losses, never clipped weights.
    """
    if not predictions.index.equals(label.index) or predictions.index.has_duplicates:
        raise ValueError("Prediction and label keys must match exactly and be unique")
    p = predictions.to_numpy(float)
    if np.isinf(p).any() or ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
        raise ValueError("Invalid event probability; no repairs")
    valid = np.isfinite(p).all(axis=1) & np.isfinite(label.followup_hours.to_numpy(float))
    if (label.followup_hours.loc[valid] < 0).any():
        raise ValueError("Negative followup")
    d = label.loc[valid]
    scores, losses = {}, pd.DataFrame(index=d.index)
    weights = None
    known = np.zeros(len(d), bool)
    truth = (d.event_observed & d.followup_hours.le(horizon)).to_numpy(float)
    for model in predictions:
        metric, mask, w = survival_metrics(1-predictions.loc[valid, model],
            d.event_observed, d.followup_hours, horizon, censor)
        supported = np.isfinite(metric['ipcw_brier'])
        scores[model] = metric
        if weights is not None and (not np.array_equal(w, weights) or not np.array_equal(mask, known)):
            raise ValueError("Censor support differs across the matched panel")
        known, weights = mask, w
        losses[model] = w*(predictions.loc[valid, model].to_numpy()-truth)**2 if supported else np.nan
        if supported and not np.isclose(losses[model].mean(), metric['ipcw_brier'], atol=1e-14):
            raise ValueError("Point score and per-origin losses disagree")
    return scores, losses, valid, known, weights
