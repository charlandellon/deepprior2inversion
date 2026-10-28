import os

import argparse
import numpy as np
from tqdm import tqdm

from dp4inversion.project_path import checkpointsdir, datadir, imagesdir
from dp4inversion.utils import load_torch_file
from examples.seismic import Model
import h5py
from devito.builtins import initialize_function
from devito import Function
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd


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


args = parser.parse_args()

def plot_velocity(model, file_name, colorbar=True, clip=None,
                  cmap="jet", set_label='Velocity (km/s)', wells_pos=None):
    """
    Plot a two-dimensional velocity field from a seismic `Model`
    object. Optionally also includes point markers for sources and receivers.

    Parameters
    ----------
    model : Model
        Object that holds the velocity model.
    source : array_like or float
        Coordinates of the source point.
    receiver : array_like or float
        Coordinates of the receiver points.
    colorbar : bool
        Option to plot the colorbar.
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    depth = np.linspace(model.origin[1], model.origin[1] + domain_size[1], num=model.shape[1])

    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    if getattr(model, 'vp', None) is not None:
        field = np.array(model.vp.data[slices])
    else:
        field = np.array(model.lam.data[slices])

    if clip is not None:
        vmin, vmax = clip[0], clip[1]
    else:
        # vmax = np.quantile(field, 0.9999)
        # vmin = np.quantile(field, 0)

        vmax = field.max()
        vmin = field.min()

    plt.figure(figsize=(15, 6))
    plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                      vmin=vmin, vmax=vmax,
                      extent=extent)

    if wells_pos is not None:
        for curva in wells_pos:
            plt.plot(curva * np.ones_like(depth), depth, '--', color='red')

    plt.xlabel('Length (km)')
    plt.ylabel('Depth (km)')

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])
    # plt.tight_layout()
    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        # ticks = np.linspace(np.ceil(vmin), np.floor(vmax), num=6)  # 0, 2, 4, 6, 8, 10
        # cbar.set_ticks(ticks)
        cbar.set_label(set_label)
    plt.savefig(file_name, dpi=300)
    # plt.show()

def SSIM(x, y):
    k1 = 0.01
    k2 = 0.03
    vmax = np.max(x)
    vmin = np.min(x)
    μx = np.mean(x)
    μy = np.mean(y)
    σx = np.sqrt(np.mean((x-μx)**2))
    σy = np.sqrt(np.mean((y-μy)**2))
    σxy = np.mean((x-μx)*(y-μy))
    c1 = ( k1 * (vmax-vmin) )**2
    c2 = ( k2 * (vmax-vmin) )**2
    ssim = (2*μx*μy + c1) * (2*σxy + c2) / ((μx**2 + μy**2 + c1) * (σx**2 + σy**2 + c2))
#     print((2*μx*μy + c1), (μx**2 + μy**2 + c1),(2*σxy + c2), (σx**2 + σy**2 + c2))
    print(f"SSIM = {ssim}")
    return ssim

def PSNR(x, y):
    mse = np.sqrt(np.mean((x - y)**2))
    vmax = np.max(x)
    vmin = np.min(x)
    maxI = vmax - vmin
    psnr = 20.0 * np.log10(maxI) - 10.0 * np.log10(mse)
    print(f"PSNR = {psnr}")
    return psnr

def MSE(x, y):
    mse = np.sqrt(np.mean((x - y)**2))
    print(f"MSE = {mse}")
    return mse

def plot_rmse(error, file_name, extents=None, y_label='RMSE'):

    if extents is not None:
        itr = np.arange(extents.start, extents.stop)

    plt.figure(figsize=(15, 4))
    plt.plot(itr, error)
    # plt.title('')
    plt.xlabel('Step')
    plt.ylabel(y_label)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()

    plt.savefig(file_name)
    plt.close()

def main():

    # Ensure that noise standard deviation is greater than zero.
    if args.sigma <= 0.0:
        ValueError(f'`sigma` must be greater than zero but is {args.sigma}')

    # Ensure that initial learning rate is not less than final learning rate.
    if args.lr < args.lr_final:
        ValueError(f'`lr` ({args.lr}) must not be smaller than '
                   f'`lr_final ({args.lr_final}).')
    # args.file_result_figs = f'{args.file_result_figs}_max_itr-{args.max_itr}_sigma' \
    #                         f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_' \
    #                         f'wd-{args.wd}_gamma-{args.gamma}_seed-{args.seed}_nsrc-{args.nsrc}_nrec-{args.nrec}'

    # args.file_result_figs = f'scale_{args.scale}_{args.file_result_figs}_max_itr-{args.max_itr}_sigma' \
    #                         f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_' \
    #                         f'wd-{args.wd}_gamma-{args.gamma}_seed-{args.seed}_nsrc-{args.nsrc}_nrec-{args.nrec}'

    # args.file_result_figs = f'ch_encoding_{args.ch_encoding}_ch_decoding_{args.ch_decoding}_ch_skip_{args.ch_skip}_Latente_scale_{args.upsample_scale}-scale_{args.scale}_{args.file_result_figs}_max_itr-{args.max_itr}_sigma' \
    #                         f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_' \
    #                         f'wd-{args.wd}_gamma-{args.gamma}_seed-{args.seed}_nsrc-{args.nsrc}_nrec-{args.nrec}'

    # args.file_result_figs = f'ch_encoding_{args.ch_encoding}_ch_decoding_{args.ch_decoding}_ch_skip_{args.ch_skip}_Latente_scale_{args.upsample_scale}-scale_{args.scale}_{args.file_result_figs}_time_{args.time_seismic}_max_itr-{args.max_itr}_sigma' \
    #                         f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_' \
    #                         f'wd-{args.wd}_gamma-{args.gamma}_seed-{args.seed}_nsrc-{args.nsrc}_nrec-{args.nrec}'

    args.file_result_figs = f'ch_encoding_{args.ch_encoding}_ch_decoding_{args.ch_decoding}_ch_skip_{args.ch_skip}_Latente_scale_{args.upsample_scale}-scale_{args.scale}_{args.file_result_figs}_max_itr-{args.max_itr}_sigma' \
                            f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_' \
                            f'wd-{args.wd}_gamma-{args.gamma}_seed-{args.seed}_nsrc-{args.nsrc}_nrec-{args.nrec}_freq-{args.freq}Hz'


    # Experiment name according to input arguments.
    args.experiment = (os.sep.join([args.file_result_figs, f'{args.experiment}_max_itr-{args.max_itr}_sigma'
                       f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_'
                       f'wd-{args.wd}_gamma-{args.gamma}_nsrc-{args.nsrc}_nrec-{args.nrec}']))


    samples = h5py.File(
            os.path.join(checkpointsdir(args.experiment), 'samples.hdf5'),
            'r+')['samples']

    print(f'samples shape: {samples.shape}')

    attrs = load_torch_file(os.path.join(checkpointsdir(args.experiment), 'checkpoint.pth'))


    if args.vel_model.lower() == '4layers':
        shape=(100, 100)
        spacing=(10., 10.)
        origin = (0, 0)
        nbl = 30
        space_order = 8

        from examples.seismic import demo_model

        layers = demo_model('layers-isotropic', origin=(0., 0.), shape=shape,
                           spacing=spacing, nbl=nbl, nlayers=4)

        vp = np.array(layers.vp.data[nbl:-nbl, nbl:-nbl])

        group = slice(4500, 10000)
        corte = 75 #31 #32 159 268 351
        wells_pos = (np.round(25*spacing[0] + spacing[0]) * 1e-3,
                     np.round(50*spacing[0] + spacing[0]) * 1e-3,
                     np.round(75*spacing[0] + spacing[0]) * 1e-3)
        rmse_model = np.array(attrs['rmse_model'])[group.start:group.stop]
        rmse_seismogram = np.array(attrs['rmse_seismogram'])[group.start:group.stop]



    elif args.vel_model.lower() == 'marmousi':
        # self.setup_model()
        f0 = 5
        vel_path = datadir("velocity_model")
        path = os.path.join(vel_path, f'marmousi2-model-true-large_freq_{f0}.pth')
        model_vp = load_torch_file(path)
        spacing = (model_vp['dx'], model_vp['dy'])
        shape = (model_vp['nx'], model_vp['ny'])
        origin = (0, 0)
        nbl = 50
        space_order = 8
        nrec = model_vp['nrcv']
        print(f'nrec: {nrec}')
        plot_samples = False
        group = slice(0, 1000, 1) #slice(45000, 50000)
        corte = 268 #31 #32 159 268 351
        wells_pos = (np.round(corte*spacing[0] + spacing[0]) * 1e-3,) #(1.0, 5.0, 8.400, 11.0)
        rmse_model = np.array(attrs['rmse_model'])[group.start:group.stop]
        rmse_seismogram = np.array(attrs['rmse_seismogram'])[group.start:group.stop]
        vp = model_vp['vp']
    elif args.vel_model.lower() == 'bp':
        f0 = 3
        vel_path = datadir("velocity_model")
        path = os.path.join(vel_path, f'BP_model_freq_{f0}_v1.pth')
        model_vp = load_torch_file(path)
        spacing = (model_vp['dx'], model_vp['dy'])
        shape = (model_vp['nx'], model_vp['ny'])
        origin = (0, 0)
        nbl = 50
        space_order = 8
        nrec = model_vp['nrcv']
        print(f'nrec: {nrec}')
        plot_samples = False
        group = slice(100, 50001, 1) #slice(45000, 50000)
        corte = 350 #31 #32 159 268 351
        wells_pos = (np.round(150*spacing[0] + spacing[0]) * 1e-3,
                     np.round(268*spacing[0] + spacing[0]) * 1e-3,
                     np.round(350*spacing[0] + spacing[0]) * 1e-3)
        rmse_model = np.array(attrs['rmse_model'])[group.start:group.stop]
        rmse_seismogram = np.array(attrs['rmse_seismogram'])[group.start:group.stop]
        vp = model_vp['vp']
    else:
        print('Escolha um modelo de velocidade válido! (4layers, marmousi ou bp)')


    print(f'spacing: {spacing}')
    print(f'shape: {shape}')

    clip = (vp.min(), vp.max())
    model = Model(space_order=space_order,
                  vp=vp,
                  origin=origin,
                  shape=shape,
                  dtype=np.float32,
                  spacing=spacing,
                  nbl=nbl,
                  bcs='damp')

    plot_velocity(model, os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'velocity_statistical_images'])),
                                      f'{args.vel_model.lower()}_real_with_wells_position_{wells_pos[0]}.png']),
                                      cmap="jet", set_label='Velocity (km/s)', wells_pos=wells_pos, clip=clip)
    if plot_samples:
        for i in range(samples.shape[0]):
            if i % 100 == 0 or i == args.max_itr - 1:
                vp_ = Function(name='vp', grid=model.grid,
                                space_order=model.space_order)
                initialize_function(vp_, samples[i], model.nbl)
                model.vp = vp_
                plot_velocity(model, os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'velocity_statistical_images'])),
                                            f'{args.vel_model.lower()}_sample_{i}.png']),
                                            cmap="jet", set_label='Velocity (km/s)', clip=clip)



    std_base = np.std(samples[group], axis=0)
    print(f'std_base: ({std_base.min()}, {std_base.max()})')
    mean_base = np.mean(samples[group], axis=0)
    print(f'mean_base: ({mean_base.min()}, {mean_base.max()})')


    std_index = 100 * std_base / mean_base

    p1 = np.percentile(std_index, 1)
    p99 = np.percentile(std_index, 99.2)

    # Aplicação do clip
    std_index = np.clip(std_index, p1, p99)
    # std_index = np.clip(std_index, 0, 10)

    print(f'std_index: ({std_index.min()}, {std_index.max()})')

    print(f'model.domain_size: {model.domain_size}')



    # std_index = (std_base.max() - std_base.copy()) / (std_base.max() - std_base.min())

    vp_ = Function(name='vp', grid=model.grid,
                    space_order=model.space_order)
    initialize_function(vp_, std_index, model.nbl)
    model.vp = vp_
    plot_velocity(model, os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'velocity_statistical_images'])),
                                      f'variability_index_{group.start}_{group.stop}_novo.png']),
                                      cmap="jet", set_label='Variability Index (%)')

    plot_rmse(rmse_model,
              os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'velocity_statistical_images'])),
                           f'rmse_model_{group.start}_{group.stop}_novo.png']), extents=group, y_label='RMSE (km/s)')

    rm = (rmse_seismogram - rmse_seismogram.min()) / (rmse_seismogram.max() - rmse_seismogram.min())
    plot_rmse(rmse_seismogram,
              os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'velocity_statistical_images'])),
                           f'rmse_seismogram_{group.start}_{group.stop}_novo.png']), extents=group, y_label='RMSE (N/m²)')

    num_curvas = samples[group].shape[0]
    n_samples = samples[group].shape[2]

    print(f'num_curvas: {num_curvas}')
    print(f'n_samples: {n_samples}')

    std = std_base[corte, :] #np.std(samples, axis=0)[corte, :]
    media = mean_base[corte, :]
    vp_corte = vp[corte, :]

    # plt.hist(samples[group][31, 35])
    # plt.show()


    zona_confianca_curvas = 2.576 * std * np.sqrt( 1.0 / num_curvas + (vp_corte - media) ** 2)
    zona_confianca_curvas = 2.576 * std
    zona_min = media - zona_confianca_curvas
    zona_max = media + zona_confianca_curvas

    print(f'zona_min: {zona_min.min()}')
    print(f'zona_max: {zona_max.max()}')
    print(f'zona_confianca_curvas: {zona_confianca_curvas.min(), zona_confianca_curvas.max()}')


    domain_size = 1.e-3 * np.array(model.domain_size)
    depth = np.linspace(model.origin[1], model.origin[1] + domain_size[1], num=shape[1])

    f, ax = plt.subplots(figsize=(4, 12))
    # for i in range(5):
    #     idx = np.random.choice(num_curvas, 1, replace=False)[0]
    #     vp_sample = samples[group][idx][corte, :]
    # #    ax.plot(samples[slice(25000,26000)][i][100, :], depth, '-b', alpha=0.3)
    #     ax.plot(vp_sample, depth, '-k', label=f'Sample {idx} - Vp')

    ax.plot(media, depth, '-g', label='Mean velocity')
    ax.plot(vp_corte, depth, '-b', label='True velocity')


    # Adicionar zona de confiança em torno da média
    ax.fill_betweenx(depth, zona_min, zona_max, color='red', alpha=0.5, label='%99 confidence interval')


    ax.set_ylim(depth.min(), depth.max())
    ax.invert_yaxis()
    ax.grid(True)
    ax.locator_params(axis='x', nbins=5)
    ax.legend(fontsize='x-small', loc='upper right')
    ax.set_xlabel("Velocity (km/s)", fontsize='small')
    ax.set_xlim(1., 6.0)
    ax.set_ylabel("Depth (km)", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'velocity_statistical_images'])),
                           f'view_well_position_{np.round(corte*spacing[0]+spacing[0])}_samples_{group.start}_{group.stop}.png']))
    # plt.show()

    idx_sample = int((group.start + group.stop) / 2)
    metrics = {'SSIM': [SSIM(vp*1e3, samples[idx_sample]*1e3)],
                    'PSNR': [PSNR(vp*1e3, samples[idx_sample]*1e3)],
                    'MSE': [MSE(vp*1e3, samples[idx_sample]*1e3)]}
    # Converter o dicionário para DataFrame
    df = pd.DataFrame(metrics)

    # Salvar o DataFrame em um arquivo CSV
    df.to_excel(os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'velocity_statistical_images'])), 'metrics.xlsx']), index=False)

if __name__ == '__main__':

    main()
