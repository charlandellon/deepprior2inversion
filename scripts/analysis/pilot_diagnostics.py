"""Compare the short sampler pilots from their checkpoints and sample files."""
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

TRUTH = Path('outputs/4d/marmousi4d_expanded_oil_water_white_E2/truth/models.npz')
ROOT = Path('data/checkpoints/4d')
DEFAULT_PILOTS = (
    'pilot_a_scale10_lr1e-4',
    'pilot_b_scale1_lr1e-4',
    'pilot_c_scale1_lr1e-5',
)
PILOTS = tuple(sys.argv[1:]) or DEFAULT_PILOTS

with np.load(TRUTH) as data:
    masks = {'R1': data['mask_R1'], 'R2': data['mask_R2']}


def lag1_ess(values):
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return np.nan, np.nan
    rho = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    rho = min(max(rho, 0.0), 0.999999)
    return rho, len(values) * (1 - rho) / (1 + rho)


for name in PILOTS:
    directory = ROOT/name/name
    checkpoint = torch.load(directory/'checkpoint.pth', map_location='cpu',
                            weights_only=False)
    print('===', name)
    print('iteration', checkpoint['check_point_itr'])
    print('optimizer scale', [g.get('scale') for g in
                              checkpoint['optimizer_state']['param_groups']])
    for key in ('obj_log', 'error_log', 'rmse_seismogram', 'rmse_model'):
        values = np.asarray(checkpoint[key], dtype=float).ravel()[25000:]
        if values.size == 0:
            continue
        jumps = np.abs(np.diff(values))
        print(f'{key}: n={values.size} mean={values.mean():.6g} '
              f'std={values.std():.6g} min={values.min():.6g} '
              f'max={values.max():.6g} max_jump={jumps.max() if jumps.size else 0:.6g}')

    samples_path = directory/'samples.hdf5'
    if not samples_path.exists():
        continue
    with h5py.File(samples_path, 'r') as handle:
        count = handle['samples'].shape[0]
        indices = np.arange(count)
        samples = handle['samples'][indices].transpose(0, 2, 1)
    print('samples', samples.shape)
    for region, mask in masks.items():
        means = samples[:, mask].mean(axis=1)
        rho, ess = lag1_ess(means)
        print(f'{region}: first={means[:len(means)//2].mean():.6g} '
              f'second={means[len(means)//2:].mean():.6g} '
              f'lag1={rho:.4f} ESS~{ess:.2f}')
