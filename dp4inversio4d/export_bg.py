"""Export a shared latent variable and a baseline posterior velocity sample.

The exported velocity sample is used as the 2D background for the monitor
chains; the latent tensor is reused unchanged by baseline, monitor1 and
monitor2.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', default='data/checkpoints')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--survey', default='baseline')
    parser.add_argument(
        '--truth',
        default='outputs/4d/marmousi4d_expanded_oil_water_white_E2',
    )
    parser.add_argument('--burn-in', type=int, default=17500)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--bg-output', required=True)
    parser.add_argument('--z-output', required=True)
    args = parser.parse_args()

    experiment = f'{args.tag}_{args.survey}'
    checkpoint_dir = Path(args.checkpoints) / '4d' / experiment / experiment
    checkpoint_path = checkpoint_dir / 'checkpoint.pth'
    samples_path = checkpoint_dir / 'samples.hdf5'
    if not checkpoint_path.is_file():
        raise SystemExit(f'checkpoint not found: {checkpoint_path}')
    if not samples_path.is_file():
        raise SystemExit(f'samples not found: {samples_path}')

    truth_dir = Path(args.truth) / 'truth' / 'models'
    truth = torch.load(
        truth_dir / f'{args.survey}.pth', map_location='cpu', weights_only=False
    )
    background = torch.load(
        truth_dir / 'background.pth', map_location='cpu', weights_only=False
    )

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False
    )
    z = checkpoint['z'].detach().cpu()
    z_output = Path(args.z_output)
    z_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z, z_output)

    with h5py.File(samples_path, 'r') as handle:
        count = int(handle['samples'].shape[0])
        if args.burn_in >= count:
            raise SystemExit(
                f'burn-in {args.burn_in} >= available samples {count}'
            )
        indices = np.arange(args.burn_in, count, args.stride, dtype=np.int64)
        rng = np.random.default_rng(args.seed)
        index = int(rng.choice(indices))
        vp = handle['samples'][index].astype(np.float32)

    vp = np.clip(vp, float(truth['vp'].min()), float(truth['vp'].max()))
    if vp.shape != background['vp'].shape:
        raise SystemExit(
            f'velocity shape {vp.shape} does not match background '
            f"{background['vp'].shape}"
        )

    background = dict(background)
    background['vp'] = vp
    bg_output = Path(args.bg_output)
    bg_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(background, bg_output)

    summary = {
        'experiment': experiment,
        'checkpoint': str(checkpoint_path),
        'samples': str(samples_path),
        'sample_index': index,
        'sample_iteration': index,
        'available_samples': count,
        'post_burnin_indices': int(indices.size),
        'bg_output': str(bg_output),
        'z_output': str(z_output),
        'z_shape': list(z.shape),
        'vp_range_km_s': [float(vp.min()), float(vp.max())],
    }

    truth_npz_path = Path(args.truth) / 'truth' / 'models.npz'
    if truth_npz_path.is_file():
        with np.load(truth_npz_path) as truth_npz:
            sample_xz = vp.T
            for name in ('R1', 'R2'):
                mask = truth_npz[f'mask_{name}']
                summary[f'{name}_sample_mean_m_s'] = float(
                    sample_xz[mask].mean() * 1000.0
                )
                summary[f'{name}_truth_mean_m_s'] = float(
                    truth_npz['v_true'][0][mask].mean()
                )

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
