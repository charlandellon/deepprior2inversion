"""Wood/Gassmann in SI, with a bounded baseline calibration of dry bulk modulus."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PetroParameters:
    K_o: float
    K_w: float
    rho_o: float
    rho_w: float
    K_s: float
    rho_s: float
    G_dry: float
    phi: float

    def __post_init__(self):
        values = list(vars(self).values())
        if not np.all(np.isfinite(values)):
            raise ValueError('petrophysical parameters must be finite SI values')
        if min(self.K_o, self.K_w, self.rho_o, self.rho_w, self.K_s, self.rho_s) <= 0:
            raise ValueError('bulk moduli and densities must be positive')
        if self.G_dry < 0 or not 0 < self.phi < 1:
            raise ValueError('require G_dry >= 0 and 0 < phi < 1')
        if max(self.K_o, self.K_w) >= self.K_s:
            raise ValueError('fluid moduli must be smaller than the mineral modulus')


def wood(so, params):
    so = np.asarray(so, dtype=np.float64)
    if not np.isfinite(so).all() or np.any((so < 0) | (so > 1)):
        raise ValueError('So must be finite and between zero and one')
    return (1/(so/params.K_o + (1-so)/params.K_w),
            so*params.rho_o + (1-so)*params.rho_w)


def gassmann(so, k_dry, params):
    """Return Vp, Ksat and rho_sat; density is not a seismic-solver input in V1."""
    k_dry = np.asarray(k_dry, dtype=np.float64)
    if not np.isfinite(k_dry).all() or np.any((k_dry < 0) | (k_dry >= params.K_s)):
        raise ValueError('require finite 0 <= K_dry < K_s')
    kf, rhof = wood(so, params)
    denominator = params.phi/kf + (1-params.phi)/params.K_s - k_dry/params.K_s**2
    if np.any(denominator <= 0):
        raise ValueError('nonpositive Gassmann denominator')
    ksat = k_dry + (1-k_dry/params.K_s)**2/denominator
    rho = (1-params.phi)*params.rho_s + params.phi*rhof
    vp = np.sqrt((ksat+4*params.G_dry/3)/rho)
    if not np.isfinite(vp).all() or np.any(vp <= 0):
        raise ValueError('nonphysical Gassmann velocity')
    return vp, ksat, rho


def calibrate_k_dry(vp_baseline, so_baseline, params, tolerance_m_s=1e-6):
    """Vectorized bisection; reject unattainable velocities rather than clipping."""
    target = np.asarray(vp_baseline, dtype=np.float64)
    if not np.isfinite(target).all() or np.any(target <= 0):
        raise ValueError('baseline velocity must be finite and positive')
    if not np.isfinite(tolerance_m_s) or tolerance_m_s <= 0:
        raise ValueError('calibration tolerance must be positive')
    low = np.zeros_like(target)
    high = np.full_like(target, params.K_s*(1-1e-10))
    vlow = gassmann(so_baseline, low, params)[0]
    vhigh = gassmann(so_baseline, high, params)[0]
    if np.any((target < vlow) | (target > vhigh)):
        raise ValueError('baseline cannot be matched within physical K_dry bounds; adjust petro config')
    for _ in range(80):
        middle = (low+high)/2
        estimate = gassmann(so_baseline, middle, params)[0]
        low = np.where(estimate < target, middle, low)
        high = np.where(estimate < target, high, middle)
    result = (low+high)/2
    if np.max(np.abs(gassmann(so_baseline, result, params)[0]-target)) > tolerance_m_s:
        raise ValueError('baseline calibration did not reach requested tolerance')
    return result
