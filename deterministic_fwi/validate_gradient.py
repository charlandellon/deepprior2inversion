"""Finite-difference validation of the conventional nonlinear operator.

Builds a small acoustic model, compares the legacy and isolated operators, and
checks autograd directional derivatives against centered finite differences.
"""

import json

import numpy as np
import torch
from examples.seismic import (AcquisitionGeometry, Model, RickerSource,
                              TimeAxis)
from examples.seismic.acoustic import AcousticWaveSolver
from scipy.ndimage import gaussian_filter

from dp4inversion.devito_layer import ForwardModeling as LegacyForward
from deterministic_fwi.devito_layer import ForwardModeling as IsolatedForward


def build_case():
    nx, nz = 120, 80
    spacing = (20.0, 20.0)
    x = np.linspace(2.0, 3.5, nx, dtype=np.float64)[:, None]
    z = np.linspace(0.0, 1.0, nz, dtype=np.float64)[None, :]
    vp = x + 0.3 * np.sin(2.0 * np.pi * z)
    model = Model(space_order=8, vp=vp, origin=(0.0, 0.0),
                  shape=(nx, nz), spacing=spacing, nbl=20,
                  bcs='damp', dtype=np.float64)
    time = TimeAxis(start=0.0, stop=500.0, step=model.critical_dt)
    rec_x = np.linspace(100.0, nx * spacing[0] - 100.0, 20)
    rec = np.column_stack([rec_x, np.full(20, 2.0 * spacing[1])])
    src = RickerSource(name='src', grid=model.grid, f0=0.01,
                       time_range=time, npoint=1)
    src.coordinates.data[:, 0] = nx * spacing[0] / 2.0
    src.coordinates.data[:, 1] = 2.0 * spacing[1]
    geometry = AcquisitionGeometry(model, rec, src.coordinates.data,
                                   t0=0.0, tn=time.stop, src_type='Ricker',
                                   f0=0.01)
    solver = AcousticWaveSolver(model, geometry, kernel='OT2', space_order=8)
    dm = torch.from_numpy(1.0 / vp**2).unsqueeze(0).unsqueeze(0)
    return model, solver, src, geometry, dm, time.num


class Wrapper(torch.nn.Module):
    def __init__(self, op_cls, model, src, geometry, solver, device,
                 checkpointing=False, num_checkpoints=0):
        super().__init__()
        self.op = op_cls()
        self.model = model
        self.src = src
        self.geometry = geometry
        self.solver = solver
        self.device = device
        self.checkpointing = bool(checkpointing)
        self.num_checkpoints = int(num_checkpoints)

    def forward(self, x):
        return self.op.apply(x, self.src, self.model, self.geometry,
                             self.solver, self.device, self.checkpointing,
                             self.num_checkpoints)


def objective(layer, dm, obs):
    return 0.5 * (layer(dm) - obs).pow(2).sum()


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    model, solver, src, geometry, dm, nt = build_case()
    obs = torch.randn(nt, 20, dtype=torch.float64) * 1e-4
    device = torch.device('cpu')

    results = {}
    gradients = {}
    for name, cls, kwargs in (
        ('legacy_full', LegacyForward, dict(checkpointing=False)),
        ('legacy_checkpoint', LegacyForward,
         dict(checkpointing=True, num_checkpoints=10)),
        ('isolated_full', IsolatedForward, dict(checkpointing=False)),
        ('isolated_checkpoint', IsolatedForward,
         dict(checkpointing=True, num_checkpoints=10)),
    ):
        layer = Wrapper(cls, model, src, geometry, solver, device, **kwargs)
        x = dm.clone().requires_grad_(True)
        value = objective(layer, x, obs)
        grad = torch.autograd.grad(
            value, x, grad_outputs=torch.ones_like(value))[0]
        gradients[name] = grad[0, 0].detach().cpu().numpy()
        results[name] = float(value.detach())

    directions = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        p = gaussian_filter(rng.standard_normal(dm.shape[-2:]), 3.0)
        p /= np.abs(p).max()
        directions.append(p)

    layer = Wrapper(LegacyForward, model, src, geometry, solver, device,
                    checkpointing=False)
    fd_results = []
    eps = 1e-5
    base = None
    for k, direction in enumerate(directions):
        p = torch.from_numpy(direction).unsqueeze(0).unsqueeze(0)
        plus = objective(layer, dm + eps * p, obs)
        minus = objective(layer, dm - eps * p, obs)
        fd = float((plus - minus) / (2 * eps))
        analytic = float((torch.from_numpy(gradients['legacy_full']) * p[0, 0]).sum())
        fd_results.append(dict(
            direction=k, finite_difference=fd, autograd=analytic,
            relative_error=abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)))
        base = float(objective(layer, dm, obs))

    comparison = {}
    reference = gradients['legacy_full']
    for name, grad in gradients.items():
        comparison[name] = float(
            np.linalg.norm(grad - reference) / max(np.linalg.norm(reference), 1e-30))

    print(json.dumps(dict(
        nt=nt, objective=base, finite_difference=fd_results,
        gradient_comparison=comparison, eps=eps), indent=2))


if __name__ == '__main__':
    main()
