"""Devito's linearized Born scattering forward operator as a Pytorch "layer".

This implementation is based on the original implementation of Devito4PyTorch at
https://github.com/slimgroup/Devito4PyTorch.

Typical usage example:

forward_born = ForwardBorn()
y_pred = forward_born.apply(x, src, model, geometry, solver, device)

# Some objective function.
obj = objective_function(y_obs, y_pred)

# The following computes the gradient of the objictive function with respect to
# the seismic image `x`, which uses `ForwardBorn.backward`` method.
obj.backward ()
"""

from examples.seismic import Model, AcquisitionGeometry, WaveletSource, plot_velocity
from examples.seismic.acoustic import AcousticWaveSolver
from devito.builtins import initialize_function
from devito import Function, gaussian_smooth
import numpy as np
import torch
from typing import Tuple
from scipy.ndimage import gaussian_filter


class ForwardBorn(torch.autograd.Function):
    """PyTorch wrapper for Devito's linearized Born scattering forward operator.

    This module allows to have access to the automatic differentiation utilities
    of PyTorch while exploiting Devito’s highly optimized migration and
    demigration operators.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, src: WaveletSource, model: Model,
                geometry: AcquisitionGeometry, solver: AcousticWaveSolver,
                device: torch.device) -> torch.Tensor:
        """Linearized Born scattering forward operator.

        Args:
            x: A torch.Tensor containing the input seismic image.
            src: A WaveletSource object containing the source signature.
            model: A Model object containing the finite-difference grid and
                wave-equation coefficients.
            geometry: An AcquisitionGeometry object containing the receiver
                and source positions.
            solver: An AcousticWaveSolver object that contains the
                highly-optimized finite-difference solver.
            device: A torch.device object that specifies the device that
                seismic image is on.

        Returns:
            A torch.Tensor containing the output linearized seismic data.
        """
        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device
        ctx.src = src

        # Move seismic image to cpu, cast to numpy array, and include
        # finite-difference related boundary padding.
        x = x.cpu().detach()
        x = torch.nn.ReplicationPad2d((ctx.model.nbl))(x).numpy()

        # Apply Devito's linearized forward modeling operator to the image.
        d_lin = ctx.solver.born(x[0, 0, :, :], src=src)[0].data

        return torch.from_numpy(np.array(d_lin)).to(ctx.device)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[torch.Tensor]:
        """Adjoint of the linearized Born scattering forward operator.

        Args:
           dy: A torch.Tensor containing the data residual.

        Returns:
            A torch.Tensor containing gradient with respect to the input
                seismic image.
        """
        # Move data residual image to cpu, cast to numpy array.
        dy = dy.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = dy[:]

        # Apply the adjoint of the linearized modeling operator.
        u0 = ctx.solver.forward(src=ctx.src, save=True)[1]
        dx = ctx.solver.gradient(rec, u0)[0].data

        # Undo the padding and cast to torch.Tensor.
        nb = ctx.model.nbl
        dx = torch.from_numpy(np.array(dx[nb:-nb, nb:-nb])).to(ctx.device)
        dx = dx.view(1, 1, dx.shape[0], dx.shape[1])

        return dx, None, None, None, None, None


class ForwardModeling(torch.autograd.Function):
    """
    Wrapping forward-modeling operator
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, src: WaveletSource, model: Model,
                geometry: AcquisitionGeometry, solver: AcousticWaveSolver,
                device: torch.device) -> torch.Tensor:

        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device
        ctx.src = src

        # input = input.cpu().detach()
        # input = torch.nn.ReplicationPad2d((ctx.model.nbl))(input).numpy()
        # ctx.model.update('vp', input[0, 0, ...])

        # Prepare input
        # input = input[0, 0, ...].detach().cpu().numpy()

        # print(f'minimum: {input.min()} \n maximum: {input.max()}')

        # input = (input - input.min()) * (4.8 - 1.5) / (input.max() - input.min()) + 1.5
        vp = Function(name='vp', grid=ctx.model.grid,
                    space_order=ctx.model.space_order)
        initialize_function(vp, input[0, 0, ...].detach().cpu().numpy()**(-0.5), ctx.model.nbl)
        ctx.model.vp = vp

        # ctx.solver = AcousticWaveSolver(ctx.model, ctx.geometry, space_order=ctx.model.space_order)

        # Nonlinear forward modeling
        d_nonlin, ctx.u0 = ctx.solver.forward(src=src, vp=ctx.model.vp, model=ctx.model, save=True)[:2]

        # d_nonlin = np.array(d_nonlin.data)
        d_nonlin = torch.from_numpy(np.array(d_nonlin.data)).to(ctx.device)
        # predicted_norm = (d_nonlin.detach()**2).mean().sqrt().item()
        # d_nonlin /= predicted_norm

        return d_nonlin

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:

        grad_output = grad_output.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = grad_output[:]

        g = ctx.solver.jacobian_adjoint(rec, model=ctx.model, vp=ctx.model.vp, u=ctx.u0)[0].data

        # Remove padding
        nb = ctx.model.nbl

        # g = np.array(g[nb:-nb, nb:-nb])
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)

        # Normalização do gradiente por camada
        # mean = g.mean(dim=-2, keepdim=True)
        # std = g.std(dim=-2, keepdim=True)
        # epsilon = 1e-8  # Valor pequeno para evitar divisão por zero
        # g = (g - mean) / (std + epsilon)

        # g_norm = (g.detach()**2).mean().sqrt()
        # g /= g.detach().abs().max()
        # g = torch.clip(g, min=-0.98, max=0.98)
        # g = torch.clip(g, min=-torch.quantile(g.detach().abs(), 0.99), max=torch.quantile(g.detach().abs(), 0.99))
        # print(f'min_grad: {g.min()} max_grad: {g.max()}')


        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None, None

import torch.nn.functional as F

def laplacian_2d(m):
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], device=m.device, dtype=m.dtype).unsqueeze(0).unsqueeze(0)
    # m = m.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    lap = F.conv2d(m, kernel, padding=1)
    return lap

class ForwardModelingConvencional(torch.autograd.Function):
    """
    Wrapping forward-modeling operator
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, src: WaveletSource, model: Model,
                geometry: AcquisitionGeometry, solver: AcousticWaveSolver,
                device: torch.device) -> torch.Tensor:

        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device
        ctx.src = src


        vp = Function(name='vp', grid=ctx.model.grid,
                    space_order=ctx.model.space_order)
        vp_tensor = input.pow(-0.5)
        vp_numpy = gaussian_filter(vp_tensor[0, 0, ...].cpu().numpy(), sigma=0.5)
        # vp_numpy = vp_tensor[0, 0, ...].cpu().numpy()

        initialize_function(vp, vp_numpy, ctx.model.nbl)
        ctx.model.vp = vp

        # ctx.solver = AcousticWaveSolver(ctx.model, ctx.geometry, space_order=ctx.model.space_order)

        # Nonlinear forward modeling
        d_nonlin, ctx.u0 = ctx.solver.forward(src=src, vp=ctx.model.vp, model=ctx.model, save=True)[:2]

        # d_nonlin = np.array(d_nonlin.data)
        d_nonlin = torch.from_numpy(np.array(d_nonlin.data)).to(ctx.device)

        return d_nonlin

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:

        grad_output = grad_output.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = grad_output[:]

        g = ctx.solver.jacobian_adjoint(rec, model=ctx.model, vp=ctx.model.vp, u=ctx.u0)[0].data

        # Remove padding
        nb = ctx.model.nbl

        # g = np.array(g[nb:-nb, nb:-nb])
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)

        # Normalização do gradiente por camada
        # mean = g.mean(dim=-2, keepdim=True)
        # std = g.std(dim=-2, keepdim=True)
        # epsilon = 1e-8  # Valor pequeno para evitar divisão por zero
        # g = (g - mean) / (std + epsilon)

        # g_norm = (g.detach()**2).mean().sqrt()
        # g /= g.detach().abs().max()
        # g = torch.clip(g, min=-0.98, max=0.98)
        # g = torch.clip(g, min=-torch.quantile(g.detach().abs(), 0.99), max=torch.quantile(g.detach().abs(), 0.99))
        # print(f'min_grad: {g.min()} max_grad: {g.max()}')


        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None, None
