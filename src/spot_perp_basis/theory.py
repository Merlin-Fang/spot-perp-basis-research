"""He annual-decimal rho, fixed-entry hedge and strict/inclusive directional conventions."""

import numpy as np
import pandas as pd

ANNUAL_PERIODS = 365 * 3
KAPPA = float(ANNUAL_PERIODS)
GAMMA = 0.0005 * ANNUAL_PERIODS
IOTA = 0.0001 * ANNUAL_PERIODS


def theory_signal(spot, perp, annual_rate):
    """Eq. 17 and Eq. 8, in annual decimal units, not percentage points.

    Missing rates remain missing. The r=iota equilibrium interval has no
    unique lambda; reject it rather than selecting an undocumented midpoint.
    """
    s, f, r = np.broadcast_arrays(np.asarray(spot, float), np.asarray(perp, float),
                                  np.asarray(annual_rate, float))
    if np.any(~np.isfinite(s) | ~np.isfinite(f) | (s <= 0) | (f <= 0)):
        raise ValueError("Prices must be finite and positive")
    if np.any(np.isinf(r)) or np.any(np.isfinite(r) & (r == IOTA)):
        raise ValueError("Invalid rate or r=iota interval equilibrium")
    denominator = KAPPA + np.sign(IOTA-r)*GAMMA-r
    if np.any(denominator <= 0):
        raise ValueError("Nonpositive fundamental-price denominator")
    return KAPPA*(f-s)/f + np.sign(IOTA-r)*GAMMA-r, KAPPA/denominator


def cost_bound(spot_fee_bps, perp_fee_bps):
    if min(spot_fee_bps, perp_fee_bps) < 0:
        raise ValueError("Negative fees are not an audited scenario")
    return KAPPA * 2 * (spot_fee_bps+perp_fee_bps)/1e4


def entry_side(rho, bound):
    """Perpetual side: +1 long, -1 short, 0 no entry. Strict inequalities."""
    if not np.isfinite(rho):
        return 0
    return -1 if rho > bound else 1 if rho < -bound else 0


def exit_crossed(side, rho):
    if side not in (-1, 1):
        raise ValueError("Invalid perpetual side")
    return bool(np.isfinite(rho) and (rho <= 0 if side == -1 else rho >= 0))


def parameter_signal(spot,perp,annual_rate,iota):
    """Eq8/17 only for a unique equilibrium; zero iota is not forced through sign(0)."""
    s,f,r=np.broadcast_arrays(np.asarray(spot,float),np.asarray(perp,float),np.asarray(annual_rate,float))
    if not np.isfinite(iota) or iota<0:raise ValueError('Unsupported mechanism parameter')
    if np.any(~np.isfinite(s)|~np.isfinite(f)|(s<=0)|(f<=0)|np.isinf(r)):raise ValueError('Invalid prices/rates')
    if np.any(np.isfinite(r)&(r==iota)):raise ValueError('Unsupported interval equilibrium r=iota')
    denominator=KAPPA+np.sign(iota-r)*GAMMA-r
    if np.any(denominator<=0):raise ValueError('Nonpositive denominator')
    return KAPPA*(f-s)/f+np.sign(iota-r)*GAMMA-r,KAPPA/denominator


def compute_rho(s,f,r,eligible):
    out=np.full(len(s),np.nan);ok=np.asarray(eligible,bool)&np.isfinite(s)&np.isfinite(f)&np.isfinite(r)&(np.asarray(s)>0)&(np.asarray(f)>0)
    if ok.any():out[ok]=theory_signal(np.asarray(s)[ok],np.asarray(f)[ok],np.asarray(r)[ok])[0]
    return out
