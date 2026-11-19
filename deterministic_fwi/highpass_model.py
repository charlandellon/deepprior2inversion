"""Export a high-passed FWI update as a Deep Prior background.

The very-long-wavelength part of the deterministic FWI update is not
constrained reliably by the surface data (it carries a global bias).  This
script keeps the initial background at long wavelengths and adds only the
mid/short-scale part of the update:

    v_hp = (vp_final - vp_init) - gaussian_filter(vp_final - vp_init, sigma)
    vp_bg = vp_init + v_hp
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .plot_models import load_velocity, plot_one


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial_path', required=True)
    parser.add_argument('--final_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--sigma', type=float, default=12.0)
    parser.add_argument('--water_depth_m', type=float, default=500.0)
    parser.add_argument('--truth_path', default=None)
    parser.add_argument('--figures_dir', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    initial, dx, dy = load_velocity(args.initial_path)
    final, _, _ = load_velocity(args.final_path)
    if initial.shape != final.shape:
        raise ValueError('initial and final velocity shapes differ')

    update = final.astype(np.float64) - initial.astype(np.float64)
    update_hp = update - gaussian_filter(update, float(args.sigma))
    model = (initial.astype(np.float64) + update_hp).astype(np.float32)

    # The Gaussian tail leaks the update into the water column; restore it.
    water_end = int(float(args.water_depth_m) / dy) + 1
    model[:, :water_end] = initial[:, :water_end]

    payload = dict(
        vp=model,
        nx=int(model.shape[0]),
        ny=int(model.shape[1]),
        dx=float(dx),
        dy=float(dy),
        origin_xz_m=(0.0, 0.0),
        velocity_units='km/s',
        axes='xz',
        source='deterministic_fwi_highpass',
        sigma_cells=float(args.sigma),
        initial_path=str(args.initial_path),
        final_path=str(args.final_path),
    )
    torch.save(payload, args.output_path)

    if args.figures_dir:
        if args.truth_path is None:
            raise ValueError('--truth_path is required when --figures_dir is set')
        truth, _, _ = load_velocity(args.truth_path)
        output = Path(args.figures_dir)
        output.mkdir(parents=True, exist_ok=True)
        clip = (float(truth.min()), float(truth.max()))
        plot_one(model, initial, truth, dx, dy, clip, 'final_hp', output)

    rms_update = float(np.sqrt(np.mean(update ** 2)))
    rms_hp = float(np.sqrt(np.mean(update_hp ** 2)))
    print(f'wrote {args.output_path} sigma={args.sigma} '
          f'rms_update={rms_update:.4f} rms_hp={rms_hp:.4f} '
          f'vp_min={model.min():.4f} vp_max={model.max():.4f}')


if __name__ == '__main__':
    main()
