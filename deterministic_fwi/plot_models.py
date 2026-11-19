"""Legacy-style figures for every deterministic-FWI velocity update.

Velocity panels use the project's ``plot_velocity`` helper (jet, 15x6,
Length/Depth labels, dpi 300).  Update panels follow the same layout with a
symmetric seismic colormap.
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from examples.seismic import Model
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dp4inversion.utils import plot_velocity


def load_velocity(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    vp = np.asarray(payload['vp'], dtype=np.float32)
    return vp, float(payload['dx']), float(payload['dy'])


def as_model(vp, dx, dy):
    return Model(space_order=8, vp=vp.astype(np.float32), origin=(0.0, 0.0),
                 shape=vp.shape, spacing=(dx, dy), nbl=1, bcs='damp')


def plot_update(update, dx, dy, clip, file_name):
    extent = [0.0, update.shape[0] * dx * 1e-3,
              update.shape[1] * dy * 1e-3, 0.0]
    fig = plt.figure(figsize=(15, 6))
    image = plt.imshow(np.transpose(update), cmap='seismic',
                       vmin=-clip, vmax=clip, extent=extent)
    plt.xlabel('Length (km)')
    plt.ylabel('Depth (km)')
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='3%', pad=0.05)
    plt.colorbar(image, cax=cax).set_label('Velocity update (km/s)')
    plt.savefig(file_name, dpi=300)
    plt.close(fig)


def plot_one(vp, vp_init, vp_true, dx, dy, clip, stem, output):
    plot_velocity(as_model(vp, dx, dy), output / f'velocity_{stem}.png',
                  clip=clip)
    update = vp - vp_init
    limit = max(float(np.quantile(np.abs(update), 0.99)), 1e-6)
    plot_update(update, dx, dy, limit, output / f'update_{stem}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial_path', required=True)
    parser.add_argument('--truth_path', required=True)
    parser.add_argument('--models_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--watch', type=int, default=0)
    parser.add_argument('--interval', type=float, default=30.0)
    args = parser.parse_args()

    initial, dx, dy = load_velocity(args.initial_path)
    truth, _, _ = load_velocity(args.truth_path)
    if initial.shape != truth.shape:
        raise ValueError('initial and truth shapes differ')
    clip = (float(truth.min()), float(truth.max()))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    seen = set()

    if not (output / 'velocity_initial.png').is_file():
        plot_velocity(as_model(initial, dx, dy),
                      output / 'velocity_initial.png', clip=clip)
    if not (output / 'velocity_true.png').is_file():
        plot_velocity(as_model(truth, dx, dy),
                      output / 'velocity_true.png', clip=clip)

    while True:
        files = sorted(models_dir.glob('model_iter_*.pth'))
        final = models_dir / 'model_final.pth'
        if final.is_file():
            files.append(final)
        for path in files:
            if path.name in seen:
                continue
            vp, _, _ = load_velocity(path)
            if vp.shape != initial.shape:
                print(f'skip shape mismatch: {path}', flush=True)
                continue
            stem = path.stem.replace('model_iter_', 'iter_')
            plot_one(vp, initial, truth, dx, dy, clip, stem, output)
            seen.add(path.name)
            print(f'wrote {output / ("velocity_" + stem + ".png")}',
                  flush=True)
        if not args.watch:
            break
        time.sleep(max(1.0, args.interval))


if __name__ == '__main__':
    main()
