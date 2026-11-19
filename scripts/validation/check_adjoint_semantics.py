"""Diagnose what jacobian_adjoint returns and why the old test failed.

Diagnostic only.  No application code is changed.

The old check compared

    sum(g * p)              against  dJ/dvp
    sum(-0.5*vp**3 * g * p) against  dJ/dm

but Devito's GradientOperator accumulates ``Inc(grad, -u * v.dt2)``, i.e. it
returns the gradient with respect to the squared slowness ``m = 1/vp**2``.
Here we measure that directly and quantify each mistake.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from common import (INITIAL_PATH, TRUE_PATH, build_case, direct_adjoint,
                    fresh_forward, load_vp)


def direction(shape, seed):
    p = gaussian_filter(np.random.default_rng(seed).standard_normal(shape), 2.0)
    return p / np.abs(p).max()


def objective_vp(vp, base_model, geometry, dt, d_obs):
    pred = fresh_forward(base_model, geometry, vp, dt)
    return 0.5 * float(np.sum((pred - d_obs) ** 2))


def fd_m(vp, base_model, geometry, dt, d_obs, p, eps):
    """Central difference in m = 1/vp**2 along p."""
    vp_high = np.sqrt(1.0 / (1.0 / vp ** 2 + eps * p))
    vp_low = np.sqrt(1.0 / (1.0 / vp ** 2 - eps * p))
    high = objective_vp(vp_high, base_model, geometry, dt, d_obs)
    low = objective_vp(vp_low, base_model, geometry, dt, d_obs)
    return (high - low) / (2 * eps)


def fd_vp(vp, base_model, geometry, dt, d_obs, p, eps):
    high = objective_vp(vp + eps * p, base_model, geometry, dt, d_obs)
    low = objective_vp(vp - eps * p, base_model, geometry, dt, d_obs)
    return (high - low) / (2 * eps)


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b)
                 / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def fit_scale(fd, adj):
    fd = np.asarray(fd, dtype=np.float64)
    adj = np.asarray(adj, dtype=np.float64)
    alpha = float(np.dot(fd, adj)) / max(float(np.dot(adj, adj)), 1e-30)
    resid = float(np.linalg.norm(alpha * adj - fd)
                  / max(np.linalg.norm(fd), 1e-30))
    return alpha, resid


def main():
    vp_true, dx, dy = load_vp(TRUE_PATH)
    vp0, _, _ = load_vp(INITIAL_PATH)
    base_model, geometry, dt, nt = build_case(vp_true, dx, dy)

    d_obs = fresh_forward(base_model, geometry, vp_true, dt)
    d_syn = fresh_forward(base_model, geometry, vp0, dt)
    residual = d_syn - d_obs               # Devito convention (inversion_utils)

    g = direct_adjoint(base_model, geometry, vp0, residual, dt,
                       checkpointing=False)
    g_neg = direct_adjoint(base_model, geometry, vp0, -residual, dt,
                           checkpointing=False)

    report = dict(
        devito_gradient_definition='dJ/dm with m = 1/vp**2',
        nt=int(nt), dt_ms=float(dt),
        adjoint_is_linear_in_rec=dict(
            max_abs_g_plus_gneg=float(np.abs(g + g_neg).max()),
            relative=float(np.linalg.norm(g + g_neg)
                           / max(np.linalg.norm(g), 1e-30)),
        ),
        dm_test=[], vp_test=[], old_test=[],
    )

    fd_dm_vec, adj_dm_vec = [], []
    fd_vp_vec, adj_vp_vec = [], []
    fd_old_vec, adj_old_vec = [], []

    for seed in range(3):
        p = direction(vp0.shape, seed)

        # --- correct dm test: g is already dJ/dm -------------------------
        for eps in (1e-5, 1e-6, 1e-7):
            fd = fd_m(vp0, base_model, geometry, dt, d_obs, p, eps)
            adj = float(np.sum(g * p))
            report['dm_test'].append(dict(
                seed=seed, eps=eps, fd=fd, adjoint=adj,
                relative_error=rel(fd, adj)))
            if eps == 1e-5:
                fd_dm_vec.append(fd)
                adj_dm_vec.append(adj)

        # --- correct vp test: dJ/dvp = dJ/dm * dm/dvp = -2/vp**3 dJ/dm ---
        eps = 1e-3
        fd = fd_vp(vp0, base_model, geometry, dt, d_obs, p, eps)
        adj_vp = float(np.sum(g * (-2.0 / vp0 ** 3) * p))
        report['vp_test'].append(dict(
            seed=seed, eps=eps, fd=fd, adjoint=adj_vp,
            relative_error=rel(fd, adj_vp)))
        fd_vp_vec.append(fd)
        adj_vp_vec.append(adj_vp)

        # --- what the old check did --------------------------------------
        adj_old_vp = float(np.sum(g * p))                     # no chain rule
        p_dm = direction(vp0.shape, seed + 10)
        fd_dm = fd_m(vp0, base_model, geometry, dt, d_obs, p_dm, 1e-5)
        g_old_dm = -0.5 * vp0 ** 3 * g                        # spurious chain
        adj_old_dm = float(np.sum(g_old_dm * p_dm))
        report['old_test'].append(dict(
            seed=seed,
            vp_fd=fd, vp_adjoint_without_chain_rule=adj_old_vp,
            vp_relative_error=rel(fd, adj_old_vp),
            dm_fd=fd_dm, dm_adjoint_with_spurious_chain=adj_old_dm,
            dm_relative_error=rel(fd_dm, adj_old_dm),
            dm_adjoint_if_raw_g=float(np.sum(g * p_dm)),
            dm_relative_error_if_raw_g=rel(fd_dm, float(np.sum(g * p_dm))),
        ))
        fd_old_vec.append(fd_dm)
        adj_old_vec.append(adj_old_dm)

    # --- is the mismatch a shape error or only a scaling? ----------------
    for name, fd_vec, adj_vec in (
            ('dm_correct', fd_dm_vec, adj_dm_vec),
            ('vp_correct', fd_vp_vec, adj_vp_vec),
            ('dm_old_test', fd_old_vec, adj_old_vec)):
        alpha, resid = fit_scale(fd_vec, adj_vec)
        report.setdefault('multi_direction', {})[name] = dict(
            cosine_similarity=cosine(fd_vec, adj_vec),
            best_scale_alpha=alpha,
            residual_after_scalar_fit=resid)

    out = Path('outputs/validation')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'adjoint_semantics.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
