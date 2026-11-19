"""Shared reduced-Marmousi setup for independent validation scripts."""

import numpy as np
import torch
from examples.seismic import (AcquisitionGeometry, Model, RickerSource,
                              TimeAxis)


TRUE_PATH = 'data/velocity_model/marmousi2-model-true-large_freq_5.pth'
INITIAL_PATH = 'data/velocity_model/marmousi2-model-smooth-linear_freq_5.pth'


def load_vp(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return (np.asarray(payload['vp'], dtype=np.float64),
            float(payload['dx']), float(payload['dy']))


def build_case(vp_true, dx, dy, tmax_ms=2000.0, nbl=50, nrec=100,
               space_order=8):
    model = Model(space_order=space_order, vp=vp_true, origin=(0.0, 0.0),
                  shape=vp_true.shape, spacing=(dx, dy), nbl=nbl,
                  bcs='damp', dtype=np.float64)
    dt = float(model.critical_dt)
    time = TimeAxis(start=0.0, stop=tmax_ms, step=dt)
    rec_x = np.linspace(100.0, model.domain_size[0] - 100.0, nrec)
    rec = np.column_stack([rec_x, np.full(nrec, 2.0 * dy, dtype=np.float64)])
    src = RickerSource(name='src', grid=model.grid, f0=0.005,
                       time_range=time, npoint=1)
    src.coordinates.data[:, 0] = model.domain_size[0] / 2.0
    src.coordinates.data[:, 1] = 2.0 * dy
    geometry = AcquisitionGeometry(model, rec, src.coordinates.data,
                                   t0=0.0, tn=tmax_ms,
                                   src_type='Ricker', f0=0.005)
    return model, geometry, dt, time.num


def fresh_forward(base_model, geometry, vp, dt):
    """Forward with a solver compiled for the model passed in."""
    initialize = __import__('devito').initialize_function
    initialize(base_model.vp, vp, base_model.nbl)
    from examples.seismic.acoustic import AcousticWaveSolver
    solver = AcousticWaveSolver(base_model, geometry, kernel='OT2',
                                space_order=base_model.space_order)
    rec = solver.forward(save=False, dt=dt)[0]
    return np.array(rec.data, dtype=np.float64)


def direct_adjoint(base_model, geometry, vp, residual, dt,
                   checkpointing=False):
    from devito import Function, initialize_function
    from examples.seismic.acoustic import AcousticWaveSolver
    initialize_function(base_model.vp, vp, base_model.nbl)
    solver = AcousticWaveSolver(base_model, geometry, kernel='OT2',
                                space_order=base_model.space_order)
    _, u0, _ = solver.forward(save=True, dt=dt)
    rec = geometry.rec
    rec.data[:] = residual
    grad = Function(name='grad_check', grid=base_model.grid)
    solver.jacobian_adjoint(rec=rec, u=u0, vp=base_model.vp, grad=grad,
                            checkpointing=checkpointing, dt=dt)
    nb = base_model.nbl
    return np.array(grad.data[nb:-nb, nb:-nb], dtype=np.float64)
