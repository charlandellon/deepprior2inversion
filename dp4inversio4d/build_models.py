"""Build calibrated, locally changing three-survey velocity truth."""

from dataclasses import dataclass

import numpy as np

from .rock_physics import PetroParameters, calibrate_k_dry, gassmann


@dataclass
class TruthModels:
    v_true: np.ndarray
    k_dry: np.ndarray
    calibration_error_m_s: dict


def build_velocity_truth(velocity_baseline, so, masks, petro_config, fixed_k_dry=None):
    baseline = np.asarray(velocity_baseline, dtype=np.float64)
    so = np.asarray(so, dtype=np.float64)
    if baseline.ndim != 2 or so.shape != (3,)+baseline.shape:
        raise ValueError('expected baseline (z,x) and So (3,z,x)')
    if not np.isfinite(baseline).all() or np.any(baseline <= 0):
        raise ValueError('baseline velocities must be finite and positive')
    velocity = np.repeat(baseline[None], 3, axis=0)
    k_dry = np.full(baseline.shape, np.nan)
    occupied = np.zeros(baseline.shape, dtype=bool)
    errors = {}
    for name, mask in masks.items():
        if mask.dtype != bool or mask.shape != baseline.shape or not mask.any():
            raise ValueError('each reservoir needs a nonempty boolean mask matching baseline')
        if np.any(occupied & mask):
            raise ValueError('reservoirs must not overlap')
        occupied |= mask
        params = PetroParameters(**petro_config[name])
        if fixed_k_dry is None:
            calibrated = calibrate_k_dry(baseline[mask], so[0, mask], params)
        else:
            calibrated = np.full(int(mask.sum()), float(fixed_k_dry[name]))
        k_dry[mask] = calibrated
        for k in range(3):
            velocity[k, mask] = gassmann(so[k, mask], calibrated, params)[0]
        # In replacement mode the reference is the prescribed petroelastic t0,
        # not the original (possibly gas-bearing) low velocity.
        reference = baseline[mask] if fixed_k_dry is None else gassmann(so[0, mask], calibrated, params)[0]
        errors[name] = float(np.max(np.abs(velocity[0, mask]-reference)))
    return TruthModels(velocity, k_dry, errors)
