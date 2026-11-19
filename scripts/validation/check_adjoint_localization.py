"""Locate where the directional derivative and the adjoint disagree."""

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
    grad = Function(name='grad_loc', grid=base_model.grid)
    solver.jacobian_adjoint(rec=rec, u=u0, grad=grad, dt=dt)
    nb = base_model.nbl
    g = np.array(grad.data[nb:-nb, nb:-nb], dtype=np.float64)

    p = smooth_direction(vp0.shape, 0)
    peak = np.unravel_index(np.argmax(np.abs(p)), p.shape)
    gmax = np.unravel_index(np.argmax(np.abs(g)), g.shape)

    # directional
    eps_m = 1e-5
    m_high = m0 + eps_m * p
    m_low = m0 - eps_m * p
    dir_fd = (objective_vp(np.sqrt(1.0 / m_high), base_model, geometry, dt, d_obs)
              - objective_vp(np.sqrt(1.0 / m_low), base_model, geometry, dt, d_obs)
              ) / (2 * eps_m)
    dir_adj = float(np.sum(g * p))

    def point_fd(ix, iy, eps=1e-5):
        hi = m0.copy()
        lo = m0.copy()
        hi[ix, iy] += eps
        lo[ix, iy] -= eps
        return (objective_vp(np.sqrt(1.0 / hi), base_model, geometry, dt, d_obs)
                - objective_vp(np.sqrt(1.0 / lo), base_model, geometry, dt, d_obs)
                ) / (2 * eps)

    p_peak_fd = point_fd(*peak)
    g_peak_fd = point_fd(*gmax)

    g_top = np.argsort(-np.abs(g).ravel())[:10]
    top = []
    for flat in g_top:
        ix, iy = np.unravel_index(flat, g.shape)
        top.append(dict(ix=int(ix), iy=int(iy), adjoint=float(g[ix, iy]),
                        p=float(p[ix, iy])))

    # how much of the directional derivative the top-k points explain
    flat_p = p.ravel()
    flat_g = g.ravel()
    contrib = flat_p * flat_g
    order = np.argsort(-np.abs(contrib))[:20]
    explained = float(np.sum(contrib[order]))

    report = dict(
        direction=dict(peak_index=[int(peak[0]), int(peak[1])],
                       peak_value=float(p[peak]),
                       directional_fd=dir_fd,
                       directional_adjoint=dir_adj,
                       relative_error=abs(dir_fd - dir_adj)
                       / max(abs(dir_fd), abs(dir_adj), 1e-30)),
        adjoint_max=dict(index=[int(gmax[0]), int(gmax[1])],
                         value=float(g[gmax]),
                         point_fd_at_that_index=g_peak_fd,
                         point_fd_at_direction_peak=p_peak_fd),
        grad_stats=dict(min=float(g.min()), max=float(g.max()),
                        mean=float(g.mean()),
                        l1=float(np.abs(g).sum()),
                        l2=float(np.linalg.norm(g))),
        top_abs_gradient=top,
        top20_contribution_to_directional=explained,
        directional_adjoint_total=dir_adj,
    )
    with open('outputs/validation/adjoint_localization.json', 'w',
              encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
