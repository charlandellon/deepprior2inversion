"""Validate the acoustic adjoint on the reduced Marmousi.

Devito's ``jacobian_adjoint`` returns dJ/dm with m = 1/vp**2, because
``GradientOperator`` accumulates ``Inc(grad, -u * v.dt2)``.  Therefore:

* dm test: compare ``g`` directly with centered FD in m;
* vp test: compare ``-2/vp**3 * g`` with centered FD in vp.

Directions are masked to zero on the outer two rows/columns.  ``initialize_function``
replicates the model edge into the nbl padding, so perturbing a boundary cell also
perturbs the padding and the FD would no longer represent a single interior cell.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from devito import initialize_function
from examples.seismic.acoustic import AcousticWaveSolver
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

from dp4inversion.devito_layer import (ForwardModeling,
                                       ForwardModelingConvencional)

from common import (INITIAL_PATH, TRUE_PATH, build_case, direct_adjoint,
                    fresh_forward, load_vp)


def objective(vp, base_model, geometry, dt, d_obs):
    pred = fresh_forward(base_model, geometry, vp, dt)
    return 0.5 * float(np.sum((pred - d_obs) ** 2))


def masked_direction(shape, seed, border=2):
    """Smooth random direction that vanishes on the model boundary."""
    p = gaussian_filter(np.random.default_rng(seed).standard_normal(shape), 2.0)
    p = p / np.abs(p).max()
    p[:border, :] = 0.0
    p[-border:, :] = 0.0
    p[:, :border] = 0.0
    p[:, -border:] = 0.0
    return p


def fd_dm(vp, base_model, geometry, dt, d_obs, p, eps):
    vp_high = np.sqrt(1.0 / (1.0 / vp ** 2 + eps * p))
    vp_low = np.sqrt(1.0 / (1.0 / vp ** 2 - eps * p))
    high = objective(vp_high, base_model, geometry, dt, d_obs)
    low = objective(vp_low, base_model, geometry, dt, d_obs)
    return (high - low) / (2 * eps)


def fd_vp(vp, base_model, geometry, dt, d_obs, p, eps):
    high = objective(vp + eps * p, base_model, geometry, dt, d_obs)
    low = objective(vp - eps * p, base_model, geometry, dt, d_obs)
    return (high - low) / (2 * eps)


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def save_image(field, dx, dy, path, title):
    extent = [0, field.shape[0] * dx * 1e-3,
              0, field.shape[1] * dy * 1e-3]
    limit = max(float(np.quantile(np.abs(field), 0.99)), 1e-12)
    fig = plt.figure(figsize=(15, 6))
    image = plt.imshow(field.T, cmap='seismic', vmin=-limit, vmax=limit,
                       extent=extent)
    plt.xlabel('Length (km)')
    plt.ylabel('Depth (km)')
    plt.title(title)
    cax = make_axes_locatable(plt.gca()).append_axes('right', size='3%',
                                                     pad=0.05)
    plt.colorbar(image, cax=cax)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def autograd_gradient(layer, dm0, obs):
    x = dm0.clone().requires_grad_(True)
    pred = layer(x)
    loss = 0.5 * (pred - obs).pow(2).sum()
    return torch.autograd.grad(loss, x)[0][0, 0].detach().numpy()


def main():
    vp_true, dx, dy = load_vp(TRUE_PATH)
    vp0, _, _ = load_vp(INITIAL_PATH)
    base_model, geometry, dt, nt = build_case(vp_true, dx, dy)

    d_obs = fresh_forward(base_model, geometry, vp_true, dt)
    d_syn = fresh_forward(base_model, geometry, vp0, dt)
    residual = d_syn - d_obs          # Devito convention

    g = direct_adjoint(base_model, geometry, vp0, residual, dt,
                       checkpointing=False)
    g_ck = direct_adjoint(base_model, geometry, vp0, residual, dt,
                          checkpointing=True)

    report = dict(
        model_shape=list(vp0.shape), nt=int(nt), dt_ms=float(dt),
        convention='jacobian_adjoint returns dJ/dm, m = 1/vp**2',
        checkpoint_relative_difference=float(
            np.linalg.norm(g_ck - g) / max(np.linalg.norm(g), 1e-30)),
        dm_vs_adjoint=[], vp_vs_chain_rule=[],
        custom_autograd={}, conventional_autograd={},
    )

    for seed in range(3):
        p_dm = masked_direction(vp0.shape, seed)
        for eps in (1e-5, 1e-6):
            fd = fd_dm(vp0, base_model, geometry, dt, d_obs, p_dm, eps)
            adj = float(np.sum(g * p_dm))
            report['dm_vs_adjoint'].append(dict(
                seed=seed, eps=eps, fd=fd, adjoint=adj,
                relative_error=rel(fd, adj)))

        p_vp = masked_direction(vp0.shape, seed + 10)
        eps = 1e-3
        fd = fd_vp(vp0, base_model, geometry, dt, d_obs, p_vp, eps)
        adj = float(np.sum((-2.0 / vp0 ** 3) * g * p_vp))
        report['vp_vs_chain_rule'].append(dict(
            seed=seed, eps=eps, fd=fd, adjoint=adj,
            relative_error=rel(fd, adj)))

    # ---- autograd wrappers ---------------------------------------------
    initialize_function(base_model.vp, vp0, base_model.nbl)
    solver = AcousticWaveSolver(base_model, geometry, kernel='OT2',
                                space_order=base_model.space_order)
    src = geometry.src
    dm0 = torch.from_numpy(1.0 / vp0 ** 2).unsqueeze(0).unsqueeze(0)
    obs = torch.from_numpy(d_obs)

    custom = autograd_gradient(
        lambda x: ForwardModeling.apply(
            x, src, base_model, geometry, solver, torch.device('cpu'),
            False, 0, dt), dm0, obs)
    conventional = autograd_gradient(
        lambda x: ForwardModelingConvencional.apply(
            x, src, base_model, geometry, solver, torch.device('cpu'), dt),
        dm0, obs)

    report['custom_autograd'] = dict(
        relative_difference_vs_adjoint=float(
            np.linalg.norm(custom - g) / max(np.linalg.norm(g), 1e-30)))
    report['conventional_autograd'] = dict(
        relative_difference_vs_adjoint=float(
            np.linalg.norm(conventional - g)
            / max(np.linalg.norm(g), 1e-30)))

    out = Path('outputs/validation')
    out.mkdir(parents=True, exist_ok=True)
    save_image(g, dx, dy, out / 'gradient_dm.png',
               'Adjoint gradient dJ/dm')
    save_image(-2.0 / vp0 ** 3 * g, dx, dy, out / 'gradient_vp.png',
               'Adjoint gradient dJ/dvp (chain rule)')
    save_image(custom, dx, dy, out / 'gradient_custom.png',
               'Custom autograd gradient (dm input)')

    (out / 'gradient_validation.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')

    print(json.dumps(report, indent=2))
    dm_errors = [d['relative_error'] for d in report['dm_vs_adjoint']]
    vp_errors = [d['relative_error'] for d in report['vp_vs_chain_rule']]
    print()
    print(f"max dm error : {max(dm_errors):.3e}")
    print(f"max vp error : {max(vp_errors):.3e}")
    print(f"checkpoint   : {report['checkpoint_relative_difference']:.3e}")
    print(f"custom dm    : "
          f"{report['custom_autograd']['relative_difference_vs_adjoint']:.3e}")
    print(f"conventional : "
          f"{report['conventional_autograd']['relative_difference_vs_adjoint']:.3e}")


if __name__ == '__main__':
    main()
