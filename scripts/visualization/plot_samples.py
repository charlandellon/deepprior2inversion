import argparse
from pathlib import Path

import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def mypallete():
    cores = np.loadtxt(
        Path(__file__).with_name('palleta.txt'),
        skiprows=4,
        max_rows=50,
    ).tolist()
    return colors.LinearSegmentedColormap.from_list(
        'my_colormap', cores, N=256, gamma=1.0
    )


def main():
    parser = argparse.ArgumentParser(description='Export velocity samples to PNG.')
    parser.add_argument('--samples', required=True, help='HDF5 file containing the samples dataset')
    parser.add_argument('--output_dir', required=True, help='Directory for the generated PNG files')
    parser.add_argument('--stride', type=int, default=200, help='Sampling interval between saved images')
    parser.add_argument('--max_samples', type=int, default=10001, help='Maximum sample index to export')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.samples, 'r') as handle:
        samples = handle['samples']
        for i, sample in enumerate(samples):
            if i % args.stride != 0 or i >= args.max_samples:
                continue
            vmax = np.quantile(sample, 0.95)
            vmin = np.quantile(sample, 0.05)
            plt.imsave(
                output_dir / f'imagem_salva_{i}.png',
                sample.T,
                vmin=vmin,
                vmax=vmax,
                cmap='jet',
                dpi=300,
            )


if __name__ == '__main__':
    main()
