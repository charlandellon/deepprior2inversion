import os

import argparse
import numpy as np
import torch
from tqdm import tqdm

from dp4inversion.project_path import checkpointsdir, datadir, imagesdir
from examples.seismic import Model
import h5py
from devito.builtins import initialize_function
from devito import Function
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
                    default='deep_prior',
                    help='File name for obseved data with sequential-source')

parser.add_argument('--run_all',
                    dest='run_all',
                    type=int,
                    default=0,
                    help='set to 1 for run all process of modelling seismic dataset')

parser.add_argument('--nsrc',
                    dest='nsrc',
                    type=int,
                    default=50,
                    help='set the sorce number for modelling')

parser.add_argument('--vel_model',
                    dest='vel_model',
                    default='4layers',
                    help='Velocities models available (4layers, marmousi or bp)')

parser.add_argument('--nrec',
                    dest='nrec',
                    type=int,
                    default=100,
                    help='set the receptor number for modelling')

parser.add_argument('--scale',
                    dest='scale',
                    type=float,
                    default=1.0,
                    help='Scale of noise in the gradients parameters of the CNN')

parser.add_argument('--simult',
                    dest='sim_source',
                    type=int,
                    default=1,
                    help='Consirering simultaneous source')

parser.add_argument('--upsample_scale',
                    dest='upsample_scale',
                    type=int,
                    default=1,
                    help='Latente variable scale')

parser.add_argument('--ch_encoding',
                    dest='ch_encoding',
                    type=int,
                    default=[16, 32, 64],
                    nargs=3,
                    help='Define the encoding layers')

parser.add_argument('--ch_decoding',
                    dest='ch_decoding',
                    type=int,
                    default=[16, 32, 64],
                    nargs=3,
                    help='Define the decoding layers')

parser.add_argument('--ch_skip',
                    dest='ch_skip',
                    type=int,
                    default=[0, 0, 64],
                    nargs=3,
                    help='Define the skip connection layers')


parser.add_argument('--time_seismic',
                    dest='time_seismic',
                    type=float,
                    default=1000,
                    help='Define seismic modelling time')

parser.add_argument('--freq',
                    dest='freq',
                    type=float,
                    default=5.0,
                    help='Define the frequence of the modelling')

parser.add_argument('--nsimsrc',
                    dest='nsimsrc',
                    type=int,
                    default=1,
                    help='Simultaneus sources to modelling.')

parser.add_argument('--bg_linear',
                    dest='bg_linear',
                    type=int,
                    default=1,
                    help='Background linear to modelling.')


args = parser.parse_args()


def plot_rmse(error, file_name, extents=None, y_label='RMSE'):

    colors = ['blue', 'red']
    label = ['Conventional', 'Deep Prior']

    if extents is not None:
        itr = np.arange(extents.start, extents.stop)

    # plt.figure(figsize=(15, 4))

    f, ax = plt.subplots(figsize=(11, 4))

    for i, rmse in enumerate(error):
        ax.plot(itr, rmse, color=colors[i], label=label[i] )

    ax.legend(fontsize='x-small', loc='upper right')

    ax.set_xlabel('Iteration')
    ax.set_ylabel(y_label)
    #ax.set_yscale('log')
    # plt.xscale('log')
    # ax.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

def main():

    # Ensure that noise standard deviation is greater than zero.
    if args.sigma <= 0.0:
        ValueError(f'`sigma` must be greater than zero but is {args.sigma}')

    # Ensure that initial learning rate is not less than final learning rate.
    if args.lr < args.lr_final:
        ValueError(f'`lr` ({args.lr}) must not be smaller than '
                   f'`lr_final ({args.lr_final}).')

    rmses, rmse_seis = [], []

    f_fig = args.file_result_figs
    experiment = args.experiment

    for i in range(2):

        args.file_result_figs = f_fig
        args.experiment = experiment


        if i == 1:
            args.file_result_figs = f'ch_encoding_{args.ch_encoding}_ch_decoding_{args.ch_decoding}_ch_skip_{args.ch_skip}_Latente_scale_{args.upsample_scale}-scale_{args.scale}_{args.file_result_figs}_time_{args.time_seismic}_max_itr-{args.max_itr}_sigma' \
                                f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_' \
                                f'wd-{args.wd}_gamma-{args.gamma}_seed-{args.seed}_nsrc-{args.nsrc}_nrec-{args.nrec}_freq-{args.freq}Hz'


            # Experiment name according to input arguments.
            args.experiment = (os.sep.join([args.file_result_figs, f'{args.experiment}_max_itr-{args.max_itr}_sigma'
                            f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_'
                            f'wd-{args.wd}_gamma-{args.gamma}_nsrc-{args.nsrc}_nrec-{args.nrec}']))

        else:
            args.file_result_figs = f'convencional-scale_{args.scale}_{args.file_result_figs}_time_{args.time_seismic}_max_itr-{args.max_itr}_sigma' \
                                f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_' \
                                f'wd-{args.wd}_gamma-{args.gamma}_seed-{args.seed}_nsrc-{args.nsrc}_nrec-{args.nrec}_freq-{args.freq}Hz'

            # Experiment name according to input arguments.
            args.experiment = (os.sep.join([args.file_result_figs, f'{args.experiment}_max_itr-{args.max_itr}_sigma'
                            f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_'
                            f'wd-{args.wd}_gamma-{args.gamma}_nsrc-{args.nsrc}_nrec-{args.nrec}']))


        attrs = torch.load(os.path.join(checkpointsdir(args.experiment), 'checkpoint.pth'))

        group = slice(100, 50001)

        rmse_model = np.array(attrs['rmse_model'])[group.start:group.stop]
        rmse_seismogram = np.array(attrs['rmse_seismogram'])[group.start:group.stop]

        rmse_seis.append(rmse_seismogram)

        rmses.append(rmse_model)



    plot_rmse(rmses,
            os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'velocity_statistical_images'])),
                        f'rmse_model_{group.start}_{group.stop}.png']), extents=group, y_label='RMSE (km/s)')

    plot_rmse(rmse_seis,
              os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'velocity_statistical_images'])),
                           f'rmse_seismogram_{group.start}_{group.stop}.png']), extents=group, y_label='RMSE (N/m²)')

if __name__ == '__main__':

    main()
