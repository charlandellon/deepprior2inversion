"""Minimal Devito/PyTorch operator for deterministic FWI.

Copied from the validated nonlinear forward operator, keeping only the
pathways used here: full wavefield storage or checkpointed recomputation.
"""

from typing import Tuple

import numpy as np
import torch
from devito import Function, initialize_function
from examples.seismic import (AcquisitionGeometry, Model, WaveletSource)
from examples.seismic.acoustic import AcousticWaveSolver


class ForwardModeling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, src: WaveletSource, model: Model,
                geometry: AcquisitionGeometry, solver: AcousticWaveSolver,
                device: torch.device, checkpointing: bool = False,
                num_checkpoints: int = 0, dt: float = None) -> torch.Tensor:
        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device
        ctx.src = src
        ctx.checkpointing = bool(checkpointing)
        ctx.num_checkpoints = int(num_checkpoints)
        ctx.dt = float(dt) if dt is not None else float(solver.dt)

        # Update the existing Function in place.  Replacing model.vp with a new
        # Function would leave the memoized gradient operator referencing the
        # old m = 1/vp**2 and invalidate the adjoint.
        initialize_function(model.vp,
                            input[0, 0, ...].detach().cpu().numpy() ** (-0.5),
                            model.nbl)
        d_nonlin, ctx.u0 = solver.forward(
            src=src, model=ctx.model, dt=ctx.dt,
            save=not ctx.checkpointing)[:2]
        return torch.from_numpy(np.array(d_nonlin.data)).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        grad_output = grad_output.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = grad_output[:]
        if ctx.checkpointing and ctx.num_checkpoints > 0:
            g = _revolver_gradient(ctx, rec)
        elif ctx.checkpointing:
            g = ctx.solver.jacobian_adjoint(
                rec, ctx.u0, src=ctx.src, model=ctx.model,
                dt=ctx.dt, checkpointing=True)[0].data
        else:
            g = ctx.solver.jacobian_adjoint(
                rec, model=ctx.model, u=ctx.u0, dt=ctx.dt)[0].data
        nb = ctx.model.nbl
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)
        return (g.view(1, 1, g.shape[0], g.shape[1]), None, None, None,
                None, None, None, None, None)


def _revolver_gradient(ctx, rec):
    from devito import TimeFunction
    from devito.checkpointing import CheckpointOperator, DevitoCheckpoint
    from pyrevolve import Revolver

    solver, model = ctx.solver, ctx.model
    grad = Function(name='grad', grid=model.grid)
    v = TimeFunction(name='v', grid=model.grid, time_order=2,
                     space_order=solver.space_order)
    u = TimeFunction(name='u', grid=model.grid, time_order=2,
                     space_order=solver.space_order)
    kwargs = dict(model.physical_params())
    dt = getattr(ctx, 'dt', None) or getattr(solver, 'dt', None) or model.critical_dt
    checkpoint = DevitoCheckpoint([u])
    wrap_fw = CheckpointOperator(solver.op_fwd(save=False), src=ctx.src,
                                 u=u, dt=dt, **kwargs)
    wrap_rev = CheckpointOperator(solver.op_grad(save=False), u=u, v=v,
                                  rec=rec, dt=dt, grad=grad, **kwargs)
    nsteps = int(rec.data.shape[0] - 2)
    revolver = Revolver(checkpoint, wrap_fw, wrap_rev,
                        int(ctx.num_checkpoints), nsteps)
    revolver.apply_forward()
    revolver.apply_reverse()
    return grad.data


class ForwardModelingLayer(torch.nn.Module):
    def __init__(self, model, src, geometry, solver, device,
                 checkpointing=False, num_checkpoints=0, dt=None):
        super(ForwardModelingLayer, self).__init__()
        self.op = ForwardModeling()
        self.model = model
        self.src = src
        self.geometry = geometry
        self.solver = solver
        self.device = device
        self.checkpointing = bool(checkpointing)
        self.num_checkpoints = int(num_checkpoints)
        self.dt = float(dt) if dt is not None else float(self.solver.dt)

    def forward(self, x):
        return self.op.apply(x, self.src, self.model, self.geometry,
                             self.solver, self.device, self.checkpointing,
                             self.num_checkpoints, self.dt)
