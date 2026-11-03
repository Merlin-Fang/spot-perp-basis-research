"""Existing strict entry-veto policies consuming supplied information.

These gates do not train models, load future labels or change He exits/sizing."""

import hashlib
import numpy as np
import pandas as pd


# Existing entry vetoes; no model fitting or target loading

def sign_gate(direction,value,channel):
    d=np.asarray(direction);v=np.asarray(value,float)
    if channel not in ('basis','funding'):raise ValueError('Unknown information channel')
    gain=(d if channel=='basis' else -d)*v
    usable=np.isfinite(v)
    return (~usable)|(gain>0),usable,gain


def verdict(symbol,rate,clock,q, *, seed=20261102):
    values=[int.from_bytes(hashlib.sha256(f'{seed}|{symbol}|{rate}|{t.isoformat()}'.encode()).digest()[:8],'big')/2**64 for t in clock]
    return np.array(values)<q


def probability_gate(predictions, reference):
    """Historical per-hour missing-prediction pass-through; no support fallback.

    A missing global reference is a caller support error, not a new policy.
    """
    p = pd.Series(predictions)
    return (p > reference).where(p.notna(), True).to_numpy(bool)
