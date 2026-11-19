"""Measure the near-surface/near-source tail of the adjoint mismatch.

Diagnostic only.  Builds a mask that excludes the first `depth` rows and
repeats the dot-product test, plus a vertical profile of pointwise FD vs
adjoint along the source column.
"""

import json

import numpy as np
from devito import Function, initialize_function
from examples.seismic.acoustic import AcousticWaveSolver
from scipy.ndimage import gaussian_filter

from common import INITIAL_PATH, TRUE_PATH, build_case, fresh_forward, load_vp


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

    initialize_function(base_model.vp, vp0, base_model.nbl)
    solver = AcousticWaveSolver(base_model, geometry, kernel='OT2',
                                space_order=base_model.space_order)
    _, u0, _ = solver.forward(save=True, dt=dt)
    rec = geometry.rec
    rec.data[:] = residual
    grad = Function(name='grad_ns', grid=base_model.grid)
    solver.jacobian_adjoint(rec=rec, u=u0, grad=grad, dt=dt)
    nb = base_model.nbl
    g = np.array(grad.data[nb:-nb, nb:-nb], dtype=np.float64)

    p = smooth_direction(vp0.shape, 0)

    report = dict(masked_dot_product=[], vertical_profile=[], source_column=199)

    for depth in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24):
        mask = np.ones_like(p)
        mask[:, :depth] = 0.0
        pm = p * mask
        dm_func = Function(name=f'dm_mask_{depth}', grid=base_model.grid,
                           space_order=0)
        initialize_function(dm_func, pm, base_model.nbl)
        rec_born = geometry.new_rec(name=f'rec_mask_{depth}')
        solver.jacobian(dm_func, rec=rec_born, dt=dt)
        lhs = float(np.sum(np.array(rec_born.data) * residual))
        rhs = float(np.sum(pm * g))
        report['masked_dot_product'].append(dict(
            excluded_top_rows=depth, lhs_born=lhs, rhs_adjoint=rhs,
            relative_error=abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)))

    eps = 1e-5

    def point_fd(ix, iy):
        hi = m0.copy()
        lo = m0.copy()
        hi[ix, iy] += eps
        lo[ix, iy] -= eps
        return (objective_vp(np.sqrt(1.0 / hi), base_model, geometry, dt, d_obs)
                - objective_vp(np.sqrt(1.0 / lo), base_model, geometry, dt, d_obs)
                ) / (2 * eps)

    for iy in (0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 50, 80, 110):
        fd = point_fd(199, iy)
        report['vertical_profile'].append(dict(
            ix=199, iy=iy, fd=fd, adjoint=float(g[199, iy]),
            relative_error=abs(fd - g[199, iy])
            / max(abs(fd), abs(g[199, iy]), 1e-30)))

    with open('outputs/validation/adjoint_nearsurface.json', 'w',
              encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
