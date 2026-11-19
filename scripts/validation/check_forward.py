"""Check whether the conventional custom forward matches a fresh solver.

Uses the reduced Marmousi (399x112, 5 Hz) and one physical shot.
This script is diagnostic only; it does not modify the application.
"""

import json
from pathlib import Path

import numpy as np
import torch
from devito import initialize_function
from examples.seismic import Model
from examples.seismic.acoustic import AcousticWaveSolver
from scipy.ndimage import gaussian_filter

from dp4inversion.devito_layer import ForwardModeling

from common import INITIAL_PATH, TRUE_PATH, build_case, fresh_forward, load_vp


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def custom_forward(base_model, geometry, vp, dt, solver_vp):
    # Build the solver with the same dt as the reference (true model), then let
    # the custom operator update vp in place.
    initialize_function(base_model.vp, solver_vp, base_model.nbl)
    solver = AcousticWaveSolver(base_model, geometry, kernel='OT2',
                                space_order=base_model.space_order)
    src = geometry.src
    dm = torch.from_numpy(1.0 / vp**2).unsqueeze(0).unsqueeze(0)
    d = ForwardModeling.apply(dm, src, base_model, geometry, solver,
                              torch.device('cpu'), False, 0, dt)
    return torch.as_tensor(d).detach().cpu().numpy().astype(np.float64)


def main():
    vp_true, dx, dy = load_vp(TRUE_PATH)
    vp0, _, _ = load_vp(INITIAL_PATH)
    base_model, geometry, dt, nt = build_case(vp_true, dx, dy)
    d_obs = fresh_forward(base_model, geometry, vp_true, dt)

    p = gaussian_filter(np.random.default_rng(0).standard_normal(vp0.shape), 2.0)
    p /= np.abs(p).max()
    vp_pert = vp0 + 0.01 * p

    fresh0 = fresh_forward(base_model, geometry, vp0, dt)
    custom0 = custom_forward(base_model, geometry, vp0, dt, vp_true)
    fresh1 = fresh_forward(base_model, geometry, vp_pert, dt)
    custom1 = custom_forward(base_model, geometry, vp_pert, dt, vp_true)

    result = dict(
        model_shape=list(vp0.shape), nt=int(nt), dt_ms=dt,
        base=dict(fresh_vs_observed=relative(fresh0, d_obs),
                  custom_vs_observed=relative(custom0, d_obs),
                  custom_vs_fresh=relative(custom0, fresh0)),
        perturbed=dict(fresh_vs_base=relative(fresh1, fresh0),
                       custom_vs_base=relative(custom1, custom0),
                       custom_vs_fresh=relative(custom1, fresh1)),
    )
    out = Path('outputs/validation')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'forward_consistency.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
