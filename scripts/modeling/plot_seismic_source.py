import os

import argparse
import numpy as np
import torch
from tqdm import tqdm

from dp4inversion.deep_prior import DeepPrior, Generator, Generator_, Model_vp, Model_v, NN_FWI
from dp4inversion.psgld import PreconditionedSGLD
from dp4inversion.setup_experiment import SeismicSetup, simulate_sequential_data, mute
from dp4inversion.utils import get_velocity_marmousi, plot_velocity, plot_error, plot_objective, plot_reflectivity
from dp4inversion.project_path import checkpointsdir, datadir, imagesdir
from examples.seismic import Model, TimeAxis
import h5py
from devito.builtins import initialize_function
from devito import Function
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
                    default='4Layers',
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

parser.add_argument('--nrec',
                    dest='nrec',
                    type=int,
                    default=100,
                    help='set the receptor number for modelling')

parser.add_argument('--vel_model',
                    dest='vel_model',
                    default='4layers',
                    help='Velocities models available (4layers, marmousi or bp)')

parser.add_argument('--retrain',
                    dest='retrain',
                    type=int,
                    default=0,
                    help='Train model of initial.')

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


parser.add_argument('--scale',
                    dest='scale',
                    type=float,
                    default=1.0,
                    help='Scale of noise in the gradients parameters of the CNN')

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

parser.add_argument('--tmax',
                    dest='tmax',
                    type=float,
                    default=None,
                    help='Maximum acquisition time in seconds')

parser.add_argument('--snr',
                    dest='snr',
                    type=float,
                    default=None,
                    help='Target signal-to-noise ratio in dB (overrides --sigma)')

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

parser.add_argument('--idx_source',
                    dest='idx_source',
                    type=int,
                    default=0,
                    help='Simultaneus sources to modelling.')


parser.add_argument('--dv',
                    dest='dv',
                    type=float,
                    default=0.0,
                    help='small pad in maximum velocity')


args = parser.parse_args()

def main():

    # Ensure that noise standard deviation is greater than zero.
    if args.snr is None and args.sigma <= 0.0:
        ValueError(f'`sigma` must be greater than zero but is {args.sigma}')

    # Ensure that initial learning rate is not less than final learning rate.
    if args.lr < args.lr_final:
        ValueError(f'`lr` ({args.lr}) must not be smaller than '
                   f'`lr_final ({args.lr_final}).')

    # args.file_result_figs = f'ch_encoding_{args.ch_encoding}_ch_decoding_{args.ch_decoding}_ch_skip_{args.ch_skip}_Latente_scale_{args.upsample_scale}-scale_{args.scale}_{args.file_result_figs}_time_{args.time_seismic}_max_itr-{args.max_itr}_sigma' \
    #                         f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_' \
    #                         f'wd-{args.wd}_gamma-{args.gamma}_seed-{args.seed}_nsrc-{args.nsrc}_nrec-{args.nrec}'

    tmax_tag = f'_tmax-{args.tmax}s' if args.tmax is not None else ''
    noise_tag = f'_snr-{args.snr}dB' if args.snr is not None else f'_sigma-{args.sigma}'
    args.file_result_figs = f'ch_encoding_{args.ch_encoding}_ch_decoding_{args.ch_decoding}_ch_skip_{args.ch_skip}_Latente_scale_{args.upsample_scale}-scale_{args.scale}_{args.file_result_figs}{tmax_tag}_max_itr-{args.max_itr}{noise_tag}_lr-{args.lr}_lr_final-{args.lr_final}_' \
                            f'wd-{args.wd}_gamma-{args.gamma}_seed-{args.seed}_nsrc-{args.nsrc}_nrec-{args.nrec}_freq-{args.freq}Hz'



    import matplotlib.pyplot as plt
    from scipy import signal

    imaging_setup = SeismicSetup(args, device=False, sim_source=False)


    src_idx = args.idx_source
    forward_op = imaging_setup.create_op(src_idx=src_idx)
    d = forward_op(imaging_setup.dm)

    print(f'vmin: {imaging_setup.dm.min()} vmax: {imaging_setup.dm.max()}')

    mask = mute(d, imaging_setup.coord_src[src_idx],
                np.linspace(0, imaging_setup.model.domain_size[0], num=imaging_setup.nrec),
                imaging_setup.dt, imaging_setup.nt, 1.5, flat_len = 300,
                taper_len = 300, device='cpu')
    # d *= mask
    d_base = (d * mask).detach().numpy().astype(np.float32)
    d_lin = d.detach().numpy().astype(np.float32)

    # plt.plot(imaging_setup.wavelet[:imaging_setup.np_kernel]); plt.show()

    time_range = np.arange(start=0.0, stop=imaging_setup.tn, step=imaging_setup.dt)

    P_signal = np.mean(d_base**2)
    if args.snr is not None:
        sigma = np.sqrt(P_signal * 10**(-args.snr / 10))
    else:
        sigma = args.sigma
    kernel = np.outer(imaging_setup.wavelet[:imaging_setup.np_kernel],
                      imaging_setup.wavelet[:imaging_setup.np_kernel])
    e = sigma * np.random.randn(*d_lin.shape)
    e = signal.fftconvolve(e, kernel, mode='same')
    e *= 1 / np.linalg.norm(e)
    e *= np.sqrt(np.prod(e.shape)) * sigma

    P_noise = np.mean(e**2)
    # Cálculo da SNR em dB

    d_lin += e
    snr = 10 * np.log10(P_signal / ( P_noise))

    print(f'O desvio padrão de {sigma} equivale a snr de {snr}')
    print(f'shape: {imaging_setup.dm.shape}')

    print(f'shape data: {d_lin.shape}')

    cmap= np.quantile(d, .95)

    # plot_shotrecord(d_lin, imaging_setup.model, 0.0, imaging_setup.tn)

    plt.figure(figsize=(10, 6))  # Defina o tamanho da figura (largura, altura)

    # Subplot 1: Primeira imagem
    plt.subplot(1, 2, 1)  # 1 linha, 2 colunas, primeiro subplot

    plt.xlabel('Length (km)')
    plt.ylabel('Time (s)')

    domain_size = 1.e-3 * np.array(imaging_setup.model.domain_size)
    extent = [imaging_setup.model.origin[0], imaging_setup.model.origin[0] + domain_size[0],
              imaging_setup.tn * 1e-3, 0.0]
    # Ensure axis limits
    plt.xlim(imaging_setup.model.origin[0], imaging_setup.model.origin[0] + domain_size[0])
    plt.ylim(imaging_setup.tn * 1e-3, 0.0)

    plt.imshow(-d, cmap='gray', vmin=-cmap, vmax=cmap, aspect='auto', extent=extent)
    plt.title('Seismic clean', fontsize='medium')  # Adicione um título opcional
    # plt.text(10, 10, 'Legenda da Imagem 1', color='white', fontsize=10)

    plt.subplot(1, 2, 2)  # 1 linha, 2 colunas, primeiro subplot

    plot = plt.imshow(-d_lin, cmap='gray', vmin=-cmap, vmax=cmap, aspect='auto', extent=extent)
    plt.title(f'Seismic noisy - SNR: {snr:.2f} dB', fontsize='medium')  # Adicione um título opcional
    # plt.text(10, 10, 'Legenda da Imagem 1', color='white', fontsize=10)
    plt.xlabel('Length (km)')

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.set_label('Amplitude (N/m²)')


    # plt.tight_layout()
    # Create aligned colorbar on the right

    plt.tight_layout()
    plt.savefig(os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'seismic_images'])),
                           f'seismic_image_{snr:.2f}.png']), dpi=300)

    plt.show()



if __name__ == '__main__':
    main()

#python plot_seismic_source.py --sigma 0.5 --seed 42 --nsrc 50
