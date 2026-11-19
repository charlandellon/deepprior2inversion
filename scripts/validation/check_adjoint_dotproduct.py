"""Dot-product test and per-point finite differences for the acoustic adjoint.

Diagnostic only.

1. Dot-product test:  <J dm, r>  vs  <dm, J^T r>  with the Born operator (J)
   and the Gradient operator (J^T).  This checks the operator pair itself and
   does not involve finite differences of the nonlinear objective.
2. Per-point finite differences of the nonlinear objective compared with the
   corresponding entry of the adjoint gradient.
"""

import json
from pathlib import Path

import numpy as np
from devito import Function, initialize_function
from examples.seismic.acoustic import AcousticWaveSolver
from scipy.ndimage import gaussian_filter

from common import (INITIAL_PATH, TRUE_PATH, build_case, direct_adjoint,
                    fresh_forward, load_vp)


def smooth_direction(shape, seed):
    p = gaussian_filter(np.random.default_rng(seed).standard_normal(shape), 2.0)
    return p / np.abs(p).max()


def objective_vp(vp, base_model, geometry, dt, d_obs):
    pred = fresh_forward(base_model, geometry, vp, dt)
    return 0.5 * float(np.sum((pred - d_obs) ** 2))


def main():
    vp_true, dx, dy = load_vp(TRUE_PATH)
    vp0, _, _ = load_vp(INITIAL_PATH)
    base_model, geometry, dt, nt = build_case(vp_true, dx, dy)

    d_obs = fresh_forward(base_model, geometry, vp_true, dt)
    d_syn = fresh_forward(base_model, geometry, vp0, dt)
    residual = d_syn - d_obs
    m0 = 1.0 / vp0 ** 2

    # ---- solver at vp0, forward wavefield and adjoint --------------------
    initialize_function(base_model.vp, vp0, base_model.nbl)
    solver = AcousticWaveSolver(base_model, geometry, kernel='OT2',
                                space_order=base_model.space_order)
    _, u0, _ = solver.forward(save=True, dt=dt)

    rec_adj = geometry.rec
    rec_adj.data[:] = residual
    grad = Function(name='grad_dot', grid=base_model.grid)
    solver.jacobian_adjoint(rec=rec_adj, u=u0, grad=grad, dt=dt)
    nb = base_model.nbl
    g = np.array(grad.data[nb:-nb, nb:-nb], dtype=np.float64)

    report = dict(
        nt=int(nt), dt_ms=float(dt),
        dot_product=[], per_point=[],
    )

    # ---- 1. dot-product test for several directions / data vectors -----
    for seed in range(3):
        p = smooth_direction(vp0.shape, seed)
        dm_func = Function(name='dm', grid=base_model.grid, space_order=0)
        initialize_function(dm_func, p, base_model.nbl)
        rec_born = geometry.new_rec(name=f'rec_born_{seed}')
        solver.jacobian(dm_func, rec=rec_born, dt=dt)
        d_born = np.array(rec_born.data, dtype=np.float64)

        rng = np.random.default_rng(100 + seed)
        r = rng.standard_normal(d_born.shape)

        lhs = float(np.sum(d_born * r))

        # adjoint of the same r
        rec_r = geometry.new_rec(name=f'rec_r_{seed}')
        rec_r.data[:] = r
        grad_r = Function(name=f'grad_r_{seed}', grid=base_model.grid)
        solver.jacobian_adjoint(rec=rec_r, u=u0, grad=grad_r, dt=dt)
        gt_r = np.array(grad_r.data[nb:-nb, nb:-nb], dtype=np.float64)
        rhs = float(np.sum(p * gt_r))

        report['dot_product'].append(dict(
            seed=seed, lhs_born=lhs, rhs_adjoint=rhs,
            relative_error=abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)))

    # also with the physical residual to stay close to the real use case
    dm_func = Function(name='dm', grid=base_model.grid, space_order=0)
    p0 = smooth_direction(vp0.shape, 0)
    initialize_function(dm_func, p0, base_model.nbl)
    rec_born = geometry.new_rec(name='rec_born_phys')
    solver.jacobian(dm_func, rec=rec_born, dt=dt)
    lhs = float(np.sum(np.array(rec_born.data) * residual))
    rhs = float(np.sum(p0 * g))
    report['dot_product'].append(dict(
        seed='physical_residual', lhs_born=lhs, rhs_adjoint=rhs,
        relative_error=abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)))

    # ---- 2. per-point finite differences --------------------------------
    shape = vp0.shape
    points = [
        ('shallow_left', 40, 10),
        ('shallow_center', shape[0] // 2, 10),
        ('shallow_right', shape[0] - 41, 10),
        ('mid_center', shape[0] // 2, shape[1] // 2),
        ('deep_center', shape[0] // 2, shape[1] - 11),
        ('deep_left', 40, shape[1] - 11),
        ('deep_right', shape[0] - 41, shape[1] - 11),
        ('near_receiver', shape[0] // 2, 3),
    ]
    eps = 1e-5
    for name, ix, iy in points:
        m_high = m0.copy()
        m_low = m0.copy()
        m_high[ix, iy] += eps
        m_low[ix, iy] -= eps
        vp_high = np.sqrt(1.0 / m_high)
        vp_low = np.sqrt(1.0 / m_low)
        fd = (objective_vp(vp_high, base_model, geometry, dt, d_obs)
              - objective_vp(vp_low, base_model, geometry, dt, d_obs)) / (2 * eps)
        report['per_point'].append(dict(
            name=name, ix=int(ix), iy=int(iy), fd=fd,
            adjoint=float(g[ix, iy]),
            relative_error=abs(fd - g[ix, iy])
            / max(abs(fd), abs(g[ix, iy]), 1e-30)))

    out = Path('outputs/validation')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'adjoint_dotproduct.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
