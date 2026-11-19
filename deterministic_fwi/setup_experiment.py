"""Acoustic setup for deterministic FWI, isolated from the legacy package."""

import argparse

import numpy as np
import torch
from examples.seismic import (AcquisitionGeometry, Model, RickerSource,
                              TimeAxis)
from examples.seismic.acoustic import AcousticWaveSolver

from dp4inversion.utils import load_torch_file

from .devito_layer import ForwardModelingLayer


class DeterministicSetup:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        model_payload = load_torch_file(args.model_path)
        background_payload = load_torch_file(args.background_path)
        vp_true = np.asarray(model_payload['vp'], dtype=np.float32)
        vp_bg = np.asarray(background_payload['vp'], dtype=np.float32)
        if vp_true.shape != vp_bg.shape:
            raise ValueError('true and background velocity shapes differ')
        self.spacing = (float(model_payload['dx']), float(model_payload['dy']))
        self.shape = (int(model_payload['nx']), int(model_payload['ny']))
        self.origin = (0.0, 0.0)
        if vp_true.shape != self.shape:
            raise ValueError('velocity shape disagrees with nx/ny metadata')

        water_end = int(float(args.water_depth_m) / self.spacing[1]) + 1
        if not 0 < water_end < self.shape[1]:
            raise ValueError('water layer exceeds the model')
        vp_mask = np.ones_like(vp_true)
        vp_mask[:, :water_end] = 0
        vp_true = vp_mask * vp_true + (1.0 - vp_mask) * vp_true
        if int(args.bg_linear) == 1:
            vp_bg = np.mean(vp_bg, axis=0, keepdims=True).repeat(
                vp_bg.shape[0], axis=0)
        vp_bg = vp_mask * vp_bg + (1.0 - vp_mask) * vp_true

        self.vp_true = torch.from_numpy(vp_true).unsqueeze(0).unsqueeze(0)
        self.dm = torch.from_numpy(1.0 / vp_true**2).unsqueeze(0).unsqueeze(0)
        self.dm_init = torch.from_numpy(
            1.0 / vp_bg**2).unsqueeze(0).unsqueeze(0)
        self.vp_mask = torch.from_numpy(vp_mask).unsqueeze(0).unsqueeze(0)

        # Same convention as the validated conventional loop: clamp the total
        # squared slowness to the true model's minimum and maximum.
        self.dm_min = float(self.dm.min())
        self.dm_max = float(self.dm.max())
        self.vp_min = float(self.dm_max ** -0.5)
        self.vp_max = float(self.dm_min ** -0.5)

        self.model = Model(space_order=int(args.space_order),
                           vp=vp_true, origin=self.origin, shape=self.shape,
                           dtype=np.float32, spacing=self.spacing,
                           nbl=int(args.nbl), bcs='damp')
        self.dt = self.model.critical_dt
        if args.tmax_ms is not None:
            self.tn = float(args.tmax_ms)
        else:
            diagonal = np.sqrt((self.shape[0] * self.spacing[0])**2
                               + (self.shape[1] * self.spacing[1])**2)
            self.tn = diagonal / float(vp_true.mean()) + 1.0
        self.nsrc = int(args.nsrc)
        self.nrec = int(args.nrec)
        self.coord_src = np.linspace(float(args.edge_margin_m),
                                     self.model.domain_size[0] - float(args.edge_margin_m),
                                     num=self.nsrc)
        self.forward_op = self.wrap_op()

    def wrap_op(self):
        time_range = TimeAxis(start=0.0, stop=self.tn, step=self.dt)
        self.nt = time_range.num
        margin = float(self.args.edge_margin_m)
        depth = float(self.args.deth_rec) * self.spacing[1]
        rec_x = np.linspace(margin, self.model.domain_size[0] - margin,
                            num=self.nrec, dtype=np.float32)
        rec_coordinates = np.column_stack(
            [rec_x, np.full(self.nrec, depth, dtype=np.float32)])
        src = RickerSource(name='src', grid=self.model.grid,
                           f0=float(self.args.freq) * 1e-3,
                           time_range=time_range, npoint=1)
        src.coordinates.data[:, 0] = self.coord_src[0]
        src.coordinates.data[:, -1] = float(self.args.deth_src) * self.spacing[1]
        geometry = AcquisitionGeometry(
            self.model, rec_coordinates, src.coordinates.data, t0=0.0,
            tn=self.tn, src_type='Ricker', f0=float(self.args.freq) * 1e-3)
        solver = AcousticWaveSolver(self.model, geometry, kernel='OT2',
                                    space_order=int(self.args.space_order))
        return ForwardModelingLayer(
            self.model, src, geometry, solver, torch.device('cpu'),
            checkpointing=bool(self.args.checkpointing),
            num_checkpoints=int(self.args.num_checkpoints), dt=self.dt)

    def create_op(self, src_idx):
        self.forward_op.src.coordinates.data[:, 0] = self.coord_src[src_idx]

        def J(x):
            return self.forward_op(x)

        return J
