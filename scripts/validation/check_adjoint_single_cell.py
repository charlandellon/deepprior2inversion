"""Separate a real boundary adjoint error from an artifact of initialize_function.

initialize_function replicates the model edges into the padding.  Perturbing
row 0 of the interior therefore also changes 50 padding cells, while the
adjoint entry g[0, :] represents a single cell.  Here we perturb a single
padded cell instead.
"""

import json

import numpy as np
from devito import initialize_function
from examples.seismic.acoustic import AcousticWaveSolver

from common import INITIAL_PATH, TRUE_PATH, build_case, fresh_forward, load_vp


def main():
    vp_true, dx, dy = load_vp(TRUE_PATH)
    vp0, _, _ = load_vp(INITIAL_PATH)
    base_model, geometry, dt, nt = build_case(vp_true, dx, dy)
    d_obs = fresh_forward(base_model, geometry, vp_true, dt)
    nb = base_model.nbl

    initialize_function(base_model.vp, vp0, nb)
    vp_pad = np.array(base_model.vp.data, dtype=np.float64)
    m_pad = 1.0 / vp_pad ** 2

    # adjoint at vp0
    solver = AcousticWaveSolver(base_model, geometry, kernel='OT2',
                                space_order=base_model.space_order)
    _, u0, _ = solver.forward(save=True, dt=dt)
    d_syn = fresh_forward(base_model, geometry, vp0, dt)
    rec = geometry.rec
    rec.data[:] = d_syn - d_obs
    from devito import Function
    grad = Function(name='grad_single', grid=base_model.grid)
    solver.jacobian_adjoint(rec=rec, u=u0, grad=grad, dt=dt)
    g = np.array(grad.data[nb:-nb, nb:-nb], dtype=np.float64)

    def objective(m_padded):
        base_model.vp.data[:] = np.sqrt(1.0 / m_padded)
        s = AcousticWaveSolver(base_model, geometry, kernel='OT2',
                               space_order=base_model.space_order)
        pred = np.array(s.forward(save=False, dt=dt)[0].data, dtype=np.float64)
        return 0.5 * float(np.sum((pred - d_obs) ** 2))

    eps = 1e-5
    report = dict(note='single padded cell perturbation', points=[])
    for iy in (0, 1, 2, 3, 5):
        hi = m_pad.copy()
        lo = m_pad.copy()
        hi[nb + 199, nb + iy] += eps
        lo[nb + 199, nb + iy] -= eps
        fd = (objective(hi) - objective(lo)) / (2 * eps)
        report['points'].append(dict(
            ix=199, iy=iy, fd_single_cell=fd, adjoint=float(g[199, iy]),
            relative_error=abs(fd - g[199, iy])
            / max(abs(fd), abs(g[199, iy]), 1e-30)))

    # Also perturb the full replicated column (boundary + padding above it)
    m_full_hi = m_pad.copy()
    m_full_lo = m_pad.copy()
    m_full_hi[nb + 199, 0:nb + 1] += eps     # whole top padding column
    m_full_lo[nb + 199, 0:nb + 1] -= eps
    fd_col = (objective(m_full_hi) - objective(m_full_lo)) / (2 * eps)
    report['replicated_column'] = dict(
        ix=199, iy=0, fd=fd_col, adjoint=float(g[199, 0]),
        relative_error=abs(fd_col - g[199, 0])
        / max(abs(fd_col), abs(g[199, 0]), 1e-30))

    with open('outputs/validation/adjoint_single_cell.json', 'w',
              encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
