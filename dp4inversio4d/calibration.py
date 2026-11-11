"""Calibrate one pooling-consistent white-noise SIGMA per survey."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dp4inversion.setup_experiment import SeismicSetup, mute

from .io import sha256

LEGACY_SOURCE_SEED = 42


def _args(model_path, background_path, nsrc, nrec, nsimsrc, freq):
    return argparse.Namespace(
        mode='legacy', vel_model='marmousi', model_path=str(model_path),
        background_path=str(background_path), bg_linear=1,
        nsrc=nsrc, nrec=nrec, nsimsrc=nsimsrc, freq=freq, sigma=0.0,
        cuda=0, dv=0.0, seed=LEGACY_SOURCE_SEED, tmax=None,
        noise_type='white_gaussian', file_result_figs='4d/calibration')


def calibrate_survey(truth, survey, source_mode='simult', nsrc=100, nrec=400,
                     frequency_hz=5.0, target_snr_db=6.0, nsimsrc=None):
    nsimsrc = nsrc if nsimsrc is None else nsimsrc
    truth = Path(truth)
    models = truth/'truth'/'models'
    model_path, background_path = models/f'{survey}.pth', models/'background.pth'
    if not model_path.is_file() or not background_path.is_file():
        raise FileNotFoundError('run build-models first: missing legacy PTH inputs')
    cfg_args = _args(model_path, background_path, nsrc, nrec, nsimsrc, frequency_hz)
    setup = SeismicSetup(cfg_args, device=False, sim_source=False)
    receivers_x = np.linspace(0, setup.model.domain_size[0], num=setup.nrec)
    gathers = np.empty((setup.nsrc, setup.nt, setup.nrec), dtype=np.float32)
    for index in range(setup.nsrc):
        gathers[index] = setup.create_op(index)(setup.dm).detach().numpy().astype(np.float32)
    if source_mode == 'sequential':
        energy = 0.0
        for index in range(setup.nsrc):
            mask = mute(torch.ones((setup.nt, setup.nrec)), float(setup.coord_src[index]),
                        receivers_x, setup.dt, setup.nt, 1.5, flat_len=300,
                        taper_len=300, device='cpu').numpy()
            energy += float(np.sum((gathers[index].astype(np.float64)*mask)**2))
        snr_reference = 'muted clean physical shot, pooled survey'
    elif source_mode == 'simult':
        codes = np.random.RandomState(LEGACY_SOURCE_SEED).randn(
            setup.nsrc, nsimsrc).astype(np.float32)/np.sqrt(nsimsrc)
        energy = 0.0
        for index in range(setup.nsrc):
            coded = np.tensordot(codes[index], gathers, axes=1)
            energy += float(np.sum(coded.astype(np.float64)**2))
        # Noise is added to each physical shot and then coded. For iid unit
        # variance per shot, each coded experiment has variance sum_j c_ij**2.
        noise_gain = float(np.mean(np.sum(codes.astype(np.float64)**2, axis=1)))
        snr_reference = 'complete clean coded experiment; coded independent shot noise'
    else:
        raise ValueError('source_mode must be sequential or simult')
    sample_count = setup.nsrc*setup.nt*setup.nrec
    signal_power = energy/sample_count
    gain = 1.0 if source_mode == 'sequential' else noise_gain
    sigma = float(np.sqrt(signal_power/(gain*10**(target_snr_db/10))))
    report = dict(survey=survey, source_mode=source_mode, sigma=sigma,
                  target_snr_db=float(target_snr_db), snr_reference=snr_reference,
                  signal_power=float(signal_power), noise_gain=gain,
                  sample_count=int(sample_count),
                  nsrc=setup.nsrc, nrec=setup.nrec, nt=setup.nt,
                  dt_ms=float(setup.dt), duration_s=float(setup.tn/1000.0),
                  model_sha256=sha256(model_path),
                  background_sha256=sha256(background_path),
                  calibration_method='expected_unit_variance_white_gaussian')
    destination = models/f'sigma_{survey}.json'
    destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report
