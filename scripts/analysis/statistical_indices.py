import os

import argparse
import numpy as np
import torch
from tqdm import tqdm

from dp4inversion.deep_prior import DeepPrior, Generator, Generator_, Model_vp, Model_v, NN_FWI
from dp4inversion.psgld import PreconditionedSGLD
from dp4inversion.setup_experiment import SeismicSetup, simulate_sequential_data, mute
from dp4inversion.utils import get_velocity_marmousi, load_checkpoint, plot_velocity, plot_error, plot_objective, plot_reflectivity
from dp4inversion.project_path import checkpointsdir, datadir, imagesdir
from examples.seismic import Model
import h5py
from devito.builtins import initialize_function
from devito import Function

parser = argparse.ArgumentParser(description='')
parser.add_argument('--max_itr',
                    dest='max_itr',
                    type=int,
                    default=10001,
                    help='maximum number of updates to weights')
parser.add_argument('--sigma',
                    dest='sigma',
                    type=float,
                    default=0.25,
                    help='noise standard variation')
parser.add_argument('--lr',
                    dest='lr',
                    type=float,
                    default=0.01,
                    help='initial learning rate for pSGLD')
parser.add_argument('--lr_final',
                    dest='lr_final',
                    type=float,
                    default=0.005,
                    help='final learning rate for pSGLD')
parser.add_argument('--wd',
                    dest='wd',
                    type=float,
                    default=30.0,
                    help='weight decay coefficient')
parser.add_argument('--gamma',
                    dest='gamma',
                    type=float,
                    default=-0.3333,
                    help='learning rate decay rate')
parser.add_argument('--experiment',
                    dest='experiment',
                    default='deep-bayesian-inference',
                    help='experiment name')
parser.add_argument('--cuda',
                    dest='cuda',
                    type=int,
                    default=0,
                    help='set to 1 for running on GPU, 0 for CPU')

parser.add_argument('--seed',
                    dest='seed',
                    type=int,
                    default=42,
                    help='set seed for aleatory sample latent space')

parser.add_argument('--phase',
                    dest='phase',
                    default='sample',
                    help='sample or inference')

parser.add_argument('--file_simut_data',
                    dest='file_simult_data',
                    default='sim_src_data_marmousi.h5',
                    help='File name for obseved data with multi-source')

parser.add_argument('--file_seq_data',
                    dest='file_seq_data',
                    default='observed_data_marmousi.h5',
                    help='File name for obseved data with sequential-source')

parser.add_argument('--file_figs',
                    dest='file_result_figs',
                    default='4Layers',
                    help='File name for obseved data with sequential-source')

parser.add_argument('--run_all',
                    dest='run_all',
                    type=int,
                    default=0,
                    help='set to 1 for run all process of modelling seismic dataset')

args = parser.parse_args()

def main():

    # Ensure that noise standard deviation is greater than zero.
    if args.sigma <= 0.0:
        ValueError(f'`sigma` must be greater than zero but is {args.sigma}')

    # Ensure that initial learning rate is not less than final learning rate.
    if args.lr < args.lr_final:
        ValueError(f'`lr` ({args.lr}) must not be smaller than '
                   f'`lr_final ({args.lr_final}).')

    # Experiment name according to input arguments.
    args.experiment = (os.sep.join([args.file_result_figs, f'{args.experiment}_max_itr-{args.max_itr}_sigma'
                       f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_'
                       f'wd-{args.wd}_gamma-{args.gamma}']))

    vp, spacing, shape, origin = get_velocity_marmousi()
    nbl = 50
    space_order = 4
    model = Model(space_order=space_order,
                  vp=vp,
                  origin=origin,
                  shape=shape,
                  dtype=np.float32,
                  spacing=spacing,
                  nbl=nbl,
                  bcs='damp')

    samples = h5py.File(
            os.path.join(checkpointsdir(args.experiment), 'samples.hdf5'),
            'r+')['samples']

    attrs = load_checkpoint(args)


    obj_log = np.array(attrs['obj_log'])

    plot_objective(obj_log / np.linalg.norm(obj_log), os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'convergence_image_test'])),
                                         'objective_function.png']), y_label='Normalized Error')

    return
    group = -1000

    std_base = np.std(samples[group:], axis=0)

    import matplotlib.pyplot as plt

    imaging_setup = SeismicSetup(args, device=False, sim_source=True)


    src_idx = 25
    forward_op = imaging_setup.create_op(src_idx=src_idx)
    d = forward_op(imaging_setup.dm)

    plt.imshow(d.detach().cpu().numpy(), cmap='gray', vmin=-.3, vmax=.3, aspect='auto' ); plt.show()


    for i, _ in enumerate(samples):
        slices = (i + 2) * group
        std = np.std(samples[slices:], axis=0)
        norm_std = np.linalg.norm(std_base - std)
        print(f'A máxima norm é: {norm_std}')
        std_base = std.copy()
        std_index = (std.max() - std.copy()) / (std.max() - std.min())

        if norm_std > 10:
            print(f'O post-burnin inicia em: {slices}')
            break

        vp = Function(name='vp', grid=model.grid,
                    space_order=model.space_order)
        initialize_function(vp, std_index, model.nbl)
        model.vp = vp
        plot_velocity(model, os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'velocity_statistical_images'])),
                                                    f'imagem_salva_{slices}.png']), set_label='confidence index')

if __name__ == '__main__':
    main()
