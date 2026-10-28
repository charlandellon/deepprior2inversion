"""Utility functions for the training process.
"""

import argparse
import h5py
import scipy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d, RegularGridInterpolator
import numpy as np
import os
import torch
from typing import List, Optional, Tuple
import obspy

from .project_path import checkpointsdir, datadir

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.ticker as ticker
def plot_objective(error, file_name, y_label='Normalized Error'):
    plt.figure(figsize=(15, 6))
    plt.plot(error / error.max())
    # plt.title('Convergence')
    plt.xlabel('Iteration')
    plt.ylabel(y_label)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_error(error, file_name, y_label='Normalized Error'):
    plt.figure(figsize=(15, 6))
    plt.plot(error / error.max())
    # plt.title('')
    plt.xlabel('Iteration')
    plt.ylabel(y_label)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()

    plt.savefig(file_name)
    plt.close()

def plot_rmse(error, file_name, y_label='RMSE'):
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(len(error)), error)
    # plt.title('')
    plt.xlabel('Iteration')
    plt.ylabel(y_label)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()

    plt.savefig(file_name)
    plt.close()



def plot_velocity(model, file_name, clip=None, colorbar=True,
                  cmap="jet", set_label='Velocity (km/s)'):
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
        vmax = np.quantile(field, 0.9999)
        vmin = np.quantile(field, 0)

    plt.figure(figsize=(15, 6))
    plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                      vmin=vmin, vmax=vmax,
                      extent=extent)
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
        cbar.set_label(set_label)
    plt.savefig(file_name, dpi=300)
    plt.close()


def plot_reflectivity(image, model, file_name, label_data='Migrated Image (km/s)', colorbar=True, cmap="gray"):
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
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    field = image
    vmax = np.quantile(field, 0.99)
    vmin = np.quantile(field, 0.01)

    plt.figure(figsize=(15, 6))
    plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                      vmin=vmin, vmax=vmax,
                      extent=extent)
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
        cbar.set_label(label_data)
    plt.savefig(file_name, dpi=300)
    plt.close()

def plot_latente(image, file_name, label_data='Latente Image', colorbar=True, cmap="gray"):
    """
    Plot a two-dimensional latente field object.
    Optionally also includes point markers for sources and receivers.

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

    field = image
    vmax = np.quantile(field, 0.99)
    vmin = np.quantile(field, 0.01)

    plt.figure(figsize=(15, 6))
    plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                      vmin=vmin, vmax=vmax)
    plt.xlabel('Length')
    plt.ylabel('Depth')

    # Ensure axis limits
    # plt.tight_layout()
    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label(label_data)
    plt.savefig(file_name, dpi=300)
    plt.close()


def calc_dx_dt(f0, vmax, space_order=4):

    operador = {4 : (5, 0.2), 6 : (3.8, 0.13), 8 : (3.33, 0.11),
                10 : (3.03, 0.087), 12 : (2.94, 0.078),
                14 : (2.77, 0.072), 16 : (2.7, 0.067) }

    # vmin = np.min(vp)
    # vmax = np.max(vp)
    # print(f'vmin: {vmin} vmax: {vmax}')
    dx =  vmax * 1/f0 / 10#operador[space_order][0]
    dy = dx
    dt = dx / vmax / 3.5 #operador[space_order][1]
    return dx, dt


def extract_data(meta):
    data = []
    for trace in meta:
        data.append(trace.data)
    return np.array(data)

def load_marmousi_model(in_dir):
    if not os.path.exists(os.path.join(in_dir, "vp_marmousi-ii.segy.gz")):
        os.system("wget {} -P {}".format("http://www.agl.uh.edu/downloads/vp_marmousi-ii.segy.gz", in_dir))
    if not os.path.exists(os.path.join(in_dir, "vs_marmousi-ii.segy.gz")):
        os.system("wget {} -P {}".format("http://www.agl.uh.edu/downloads/vs_marmousi-ii.segy.gz", in_dir))
    if not os.path.exists(os.path.join(in_dir, "density_marmousi-ii.segy.gz")):
        os.system("wget {} -P {}".format("http://www.agl.uh.edu/downloads/density_marmousi-ii.segy.gz", in_dir))
    meta = obspy.read(os.path.join(in_dir, "vs_marmousi-ii.segy.gz"), format='segy')
    vs = extract_data(meta) * 1e3 #m/s^2
    meta = obspy.read(os.path.join(in_dir, "vp_marmousi-ii.segy.gz"), format='segy')
    vp = extract_data(meta) * 1e3 #m/s^2
    meta = obspy.read(os.path.join(in_dir, "density_marmousi-ii.segy.gz"), format='segy')
    rho = extract_data(meta) * 1e3 #kg/m^3

    x_range = [0, 17000] #m
    y_range = [0, 3500] #m
    nx, ny = vp.shape
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # x = np.arange(x_range[0], x_range[1], dx)
    # y = np.arange(y_range[0], y_range[1], dy)

    x_range = [0, 17000] #m
    y_range = [0, 3500] #m

    indices_x = np.where(np.logical_and(x >= x_range[0], x <= x_range[1]))[0]
    indices_y = np.where(np.logical_and(y >= y_range[0], y <= y_range[1]))[0]

    print(f'shape: {indices_x}')


    marmousi_model = {'vp': vp[indices_x[:, np.newaxis], indices_y],
                      'vs':vs[indices_x[:, np.newaxis], indices_y],
                      'rho':rho[indices_x[:, np.newaxis], indices_y],
                      'x': x[indices_x], 'y': y[indices_y],
                      'dx': dx, 'dy': dy}

    print(f"nx = {nx}, ny = {ny}")
    print(f"dx = {dx:.2f} m, dy = {dy:.2f} m")


    return marmousi_model

def load_BP_model(in_dir):
    if not os.path.exists(os.path.join(in_dir, "vel_z6.25m_x12.5m_exact.segy.gz")):
        os.system("wget {} -P {}".format("http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/vel_z6.25m_x12.5m_exact.segy.gz", in_dir))

    meta = obspy.read(os.path.join(in_dir, "vel_z6.25m_x12.5m_exact.segy.gz"), format='segy')
    vp = extract_data(meta)  #kg/m^3

    nx, ny = vp.shape
    dx = 12.5
    dy = 6.25
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy

    vp = gaussian_filter(vp, [5, 5], mode='reflect')

    model = {"vp": vp, "x": x, "y":y, "dx":dx, "dy":dy, "nx":nx, "ny":ny}

    print(f"nx = {nx}, ny = {ny}")
    print(f"dx = {dx:.2f} m, dy = {dy:.2f} m")

    return model

def resample_model(x, y, model):

    ratio_x = (x[1] - x[0])/(model['x'][1] - model['x'][0])
    ratio_y = (y[1] - y[0])/(model['y'][1] - model['y'][0])
    ratio = 3
    vp = gaussian_filter(model['vp'], [ratio_x/ratio, ratio_y/ratio], mode='reflect')
    vs = gaussian_filter(model['vs'], [ratio_x/ratio, ratio_y/ratio], mode='reflect')
    rho = gaussian_filter(model['rho'], [ratio_x/ratio, ratio_y/ratio], mode='reflect')

    # vp = interp2d(model['y'], model['x'], model['vp'], kind='cubic')(y, x)
    # vs = interp2d(model['y'], model['x'], model['vs'], kind='cubic')(y, x)
    # rho = interp2d(model['y'], model['x'], model['rho'], kind='cubic')(y, x)

    # Cria interpoladores para vp, vs, e rho
    vp_interpolator = RegularGridInterpolator((model['x'], model['y']), model['vp'], method='cubic')
    vs_interpolator = RegularGridInterpolator((model['x'], model['y']), model['vs'], method='cubic')
    rho_interpolator = RegularGridInterpolator((model['x'], model['y']), model['rho'], method='cubic')

    # Cria os pontos onde a interpolação será realizada
    new_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # Interpola os valores
    vp = vp_interpolator(new_points).reshape(len(x), len(y))
    vs = vs_interpolator(new_points).reshape(len(x), len(y))
    rho = rho_interpolator(new_points).reshape(len(x), len(y))


    new_model = {"vp":vp, "vs":vs, "rho": rho, "x":x, "y":y}
    new_model['dx'] = x[1] - x[0]
    new_model['dy'] = y[1] - y[0]

    return new_model

def resample_model_BP(x, y, model):

    # ratio_x = (x[1] - x[0])/(model['x'][1] - model['x'][0])
    # ratio_y = (y[1] - y[0])/(model['y'][1] - model['y'][0])
    # ratio = 3
    # vp = gaussian_filter(model['vp'], [ratio_x/ratio, ratio_y/ratio], mode='reflect')

    # vp = interp2d(model['y'], model['x'], vp, kind='cubic')(y, x)

    # Cria interpoladores para vp, vs, e rho
    vp_interpolator = RegularGridInterpolator((model['x'], model['y']), model['vp'], method='cubic')

    # Cria os pontos onde a interpolação será realizada
    new_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # Interpola os valores
    vp = vp_interpolator(new_points).reshape(len(x), len(y))
    print(f' max(vp): {np.max(vp)}')

    new_model = {"vp":vp, "x":x, "y":y}
    new_model['dx'] = x[1] - x[0]
    new_model['dy'] = y[1] - y[0]

    return new_model

def get_smooth_model(model, src_rcv_depth, mask_extra_detph, kernel=(15, 15)):
    nx = len(model['x'])
    if mask_extra_detph > 0:
        vp = model['vp'].copy()
        vp[:,src_rcv_depth+mask_extra_detph:] = gaussian_filter(model['vp'][:, src_rcv_depth+mask_extra_detph:], kernel, mode='reflect')
        vs = model['vs'].copy()
        vs[:,src_rcv_depth+mask_extra_detph:] = gaussian_filter(model['vs'][:, src_rcv_depth+mask_extra_detph:], kernel, mode='reflect')
        rho = model['rho'].copy()
        rho[:,src_rcv_depth+mask_extra_detph:] = gaussian_filter(model['rho'][:, src_rcv_depth+mask_extra_detph:], kernel, mode='reflect')
    else:
        vp = model['vp'].copy()
        vp = gaussian_filter(model['vp'], kernel, mode='reflect')
        vs = model['vs'].copy()
        vs = gaussian_filter(model['vs'], kernel, mode='reflect')
        rho = model['rho'].copy()
        rho = gaussian_filter(model['rho'], kernel, mode='reflect')

    # 1D model
    #vp = np.mean(vp, axis=0, keepdims=True).repeat(nx, axis=0)
    #vs = np.mean(vs, axis=0, keepdims=True).repeat(nx, axis=0)
    #rho = np.mean(rho, axis=0, keepdims=True).repeat(nx, axis=0)

    mask = np.ones_like(vp)
    if mask_extra_detph > 0:
        mask[:,:src_rcv_depth+mask_extra_detph]=0
        vp[:,:src_rcv_depth+mask_extra_detph] = model['vp'][:,:src_rcv_depth+mask_extra_detph]
        vs[:,:src_rcv_depth+mask_extra_detph] = model['vs'][:,:src_rcv_depth+mask_extra_detph]
        rho[:,:src_rcv_depth+mask_extra_detph] = model['rho'][:,:src_rcv_depth+mask_extra_detph]

    new_model = {"vp":vp, "vs":vs, "rho":rho, "mask":mask}

    return new_model

def get_smooth_model_BP(model, src_rcv_depth, mask_extra_detph, kernel=(15, 15)):
    nx = len(model['x'])
    if mask_extra_detph > 0:
        vp = model['vp'].copy()
        vp[:,src_rcv_depth+mask_extra_detph:] = gaussian_filter(model['vp'][:, src_rcv_depth+mask_extra_detph:], kernel, mode='reflect')
    else:
        vp = model['vp'].copy()
        vp = gaussian_filter(model['vp'], kernel, mode='reflect')

    # 1D model
    # vp = np.mean(vp, axis=0, keepdims=True).repeat(nx, axis=0)

    mask = np.ones_like(vp)
    if mask_extra_detph > 0:
        mask[:,:src_rcv_depth+mask_extra_detph]=0
        vp[:,:src_rcv_depth+mask_extra_detph] = model['vp'][:,:src_rcv_depth+mask_extra_detph]

    new_model = {"vp":vp, "mask":mask}

    return new_model


def get_velocity(vel_path: Optional[str] = datadir("velocity_model")) -> Tuple[
        np.ndarray, np.ndarray, Tuple[float], Tuple[int], Tuple[float]]:
    """Downloads and returns a velocity model.

    Args:
        vel_path: A string containing path to the velocity model.

    Returns:
        m0: A numpy array containing the smooth background squared-slowness
            model.
        dm: A numpy array containing the true perturbation model (seismic
            image).
        spacing: A tuple containing the grid spacing of the model.
        shape: A tuple containing the shape of the model.
        origin: A tuple containing the origin of the model.
    """
    # Path to velocity model.
    vel_file = os.path.join(vel_path, 'parihaka_model_high-freq.h5')

    if not os.path.exists(vel_file):
        os.system("wget https://www.dropbox.com/s/eouo2awl156vc94/"
                  "parihaka_model_high-freq.h5 --no-check-certificate -O" + vel_file)

    spacing = (25.0, 12.5)

    m0 = np.transpose(h5py.File(vel_file, 'r')['m0'][...])
    dm = np.transpose(h5py.File(vel_file, 'r')['dm'][...])
    shape = h5py.File(vel_file, 'r')['n'][...][::-1]
    origin = (0., 0.)

    return m0, dm, spacing, shape, origin

def get_velocity_(vel_path: str) -> Tuple[
        np.ndarray, np.ndarray, Tuple[float], Tuple[int], Tuple[float]]:
    """Downloads and returns a velocity model.

    Args:
        vel_path: A string containing path to the velocity model.

    Returns:
        m0: A numpy array containing the smooth background squared-slowness
            model.
        dm: A numpy array containing the true perturbation model (seismic
            image).
        spacing: A tuple containing the grid spacing of the model.
        shape: A tuple containing the shape of the model.
        origin: A tuple containing the origin of the model.
    """
    # Path to velocity model.
    vel_file = os.path.join(datadir("velocity_model"), vel_path)

    model = torch.load(vel_file)

    spacing = (model['dx'].astype(np.float32), model['dy'].astype(np.float32))
    shape = (model['nx'], model['ny'])
    origin = (0., 0.)


    return model, origin, spacing, shape

def get_velocity_marmousi(vel_path: Optional[str] = datadir("velocity_model")) -> Tuple[
        np.ndarray, Tuple[float], Tuple[int], Tuple[float]]:
    shape = (1601, 401)
    spacing = (7.5, 7.5)
    origin = (0., 0.)

    # Read 2D Marmousi model from devitocodes/data repo
    # Path to velocity model.
    path = os.path.join(vel_path, 'vp_marmousi_bi')
    v = np.fromfile(path, dtype='float32', sep="")
    v = v.reshape(shape)
    v = v[:-1, :-1] #v[:shape[0] - 1:4, :shape[1] - 1:4]#v[501:-500, :400]
    # v: = v[:shape[0] - 1, :shape[1] - 1]
    # print(f'velocity max: {v.max()}')

    # Cut the model to make it slightly cheaper
    return v, spacing, v.shape, origin

def setup_sample_file_(args: argparse.Namespace, shape: Tuple[int], max_samples: int):
    """Setting up an HDF5 file to write samples.

    Args:
        args: An argparse.Namespace, containing command line arguments.
        shape: A tuple containing the shape of the model.
        max_samples: An integer containing the maximum number of samples that
            will be stored.
    """
    # Define o caminho para o arquivo HDF5
    hdf5_file_path = os.path.join(checkpointsdir(args.experiment), 'samples.hdf5')

    if args.retrain == 1:
        chp = os.path.join(checkpointsdir(args.experiment), 'checkpoint.pth')

        if os.path.exists(hdf5_file_path):
            os.remove(hdf5_file_path)

        if os.path.exists(chp):
            os.remove(chp)



    # Verifica se o arquivo já existe
    if not os.path.isfile(hdf5_file_path):
        # O arquivo não existe, então crie-o
        with h5py.File(hdf5_file_path, 'w') as samples_file:
            dataset_shape = (max_samples, shape[0], shape[1])

            # Cria o HDF5 dataset para amostras
            samples_file.create_dataset('samples', dataset_shape, dtype=np.float32)

            # Cria o HDF5 dataset para acompanhar o número de amostras salvas
            num_samples_saved = samples_file.create_dataset('num_samples', [1], dtype=int)
            num_samples_saved[0] = 0
            samples_file.close()
    else:
        print("O arquivo HDF5 já existe. Não é necessário criar um novo.")




def setup_sample_file(args: argparse.Namespace, shape: Tuple[int], max_samples: int):
    hdf5_file_path = os.path.join(checkpointsdir(args.experiment), 'samples.hdf5')

    if args.retrain == 1:
        chp = os.path.join(checkpointsdir(args.experiment), 'checkpoint.pth')
        if os.path.exists(hdf5_file_path):
            os.remove(hdf5_file_path)
        if os.path.exists(chp):
            os.remove(chp)

    if not os.path.isfile(hdf5_file_path):
        with h5py.File(hdf5_file_path, 'w') as samples_file:
            # cria um dataset redimensionável no 1º eixo (n amostras ilimitado)
            item_shape = (shape[0], shape[1])  # ajuste se for diferente
            samples_file.create_dataset(
                'samples',
                shape=(0,) + item_shape,           # começa vazio
                maxshape=(None,) + item_shape,     # permite crescimento no eixo 0
                dtype=np.float32,
                chunks=(1,) + item_shape           # chunks recomendados para resize
            )

            # contador (dataset shape (1,))
            num_samples_saved = samples_file.create_dataset('num_samples', (1,), dtype='int64')
            num_samples_saved[0] = 0
    else:
        print("O arquivo HDF5 já existe. Não é necessário criar um novo.")



def save_checkpoint(args: argparse.Namespace, obj_log: list,
                    error_log: list, rmse_seimogram: list,
                    rmse_model: list, itr: int,
                    state: Tuple[torch.nn.Module, torch.optim.Optimizer],
                    z: torch.Tensor, samples_buffer: list):
    # salva checkpoint do torch (sua parte existente)
    state_dict = None if state[0] is None else state[0].state_dict()

    torch.save({
        'obj_log': obj_log,
        'error_log': error_log,
        'rmse_seismogram': rmse_seimogram,
        'rmse_model': rmse_model,
        'model_state_dict': state_dict,
        'optimizer_state': state[1].state_dict(),
        'z': z,
        'check_point_itr': itr,
    }, os.path.join(checkpointsdir(args.experiment), 'checkpoint.pth'))

    # escreve amostras
    if len(samples_buffer) == 0:
        return

    h5path = os.path.join(checkpointsdir(args.experiment), 'samples.hdf5')
    samples_buffer = np.asarray(samples_buffer)
    shape = samples_buffer.shape[3:]
    samples_buffer = samples_buffer.reshape(-1, shape[0], shape[1])

    n_new = samples_buffer.shape[0]


    with h5py.File(h5path, 'a') as samples_file:
        if "num_samples" not in samples_file:
            # caso raro: criar contador se ausente
            samples_file.create_dataset('num_samples', (1,), dtype='int64', data=np.array([0], dtype='int64'))
        num_ds = samples_file['num_samples']

        if "samples" not in samples_file:
            # caso raro: criar dataset com base no buffer (se você confiar no buffer como referência)
            item_shape = samples_buffer.shape[1:]
            samples_file.create_dataset('samples', shape=(0,) + item_shape,
                                        maxshape=(None,) + item_shape,
                                        dtype=samples_buffer.dtype,
                                        chunks=(1,) + item_shape)

        ds = samples_file['samples']

        # garantir que o buffer tenha o mesmo item shape do dataset
        item_shape_ds = ds.shape[1:]
        if samples_buffer.shape[1:] != item_shape_ds:
            # tente rearranjar apenas se fizer sentido; senão lance erro claro
            raise ValueError(f"item shape mismatch: dataset {item_shape_ds}, buffer {samples_buffer.shape[1:]}")

        # verificar se podemos crescer; se dataset foi criado sem maxshape, ds.maxshape[0] será ds.shape[0]
        can_resize = (ds.maxshape[0] is None) or (ds.maxshape[0] >= (ds.shape[0] + n_new))

        current = int(num_ds[0])

        if ds.maxshape[0] is None:
            # redimensionável: aumenta para caber novas amostras
            ds.resize((current + n_new,) + ds.shape[1:])
        else:
            # não-redimensionável: verificar overflow
            if current + n_new > ds.shape[0]:
                raise RuntimeError(f"Not enough capacity in 'samples' dataset: current={current}, need={n_new}, capacity={ds.shape[0]}. Consider recreating file with larger max_samples or using maxshape=(None, ...).")

        # escrever e atualizar contador explicitamente
        ds[current: current + n_new, ...] = samples_buffer
        num_ds[0] = current + n_new

        samples_file.flush()


def save_checkpoint_(args: argparse.Namespace, obj_log: List[float],
                    error_log: List[float], rmse_seimogram: List[float],
                    rmse_model: List[float], itr: int,
                    G: torch.nn.Module, z: torch.Tensor,
                    samples_buffer: List[np.ndarray]):
    """Saves the current state of the training/sampling process.

    Saves intermediate results, such as network parameters and training logs. It
    also takes in samples that are ready to be written to an HDF5 file.

    Args:
        args: An argparse.Namespace, containing command line arguments.
        obj_log: A list containing the objective function values.
        error_log: A list containing the error values.
        itr: A float containind the iteration value of the current state
        g: A torch.nn.Module containing the generator network.
        z: A torch.Tensor containing the fixed latent variable.
        samples_buffer: A list containing the samples that are ready to be
            written to file
    """

    if G is None:
        state_dict = None
    else:
        state_dict = G.state_dict()

    # Save intermediate network weights and training logs
    torch.save(
        {
            'obj_log': obj_log,
            'error_log': error_log,
            'rmse_seismogram' : rmse_seimogram,
            'rmse_model' : rmse_model,
            'model_state_dict': state_dict,
            'z': z,
            'check_point_itr': itr,
        }, os.path.join(checkpointsdir(args.experiment), 'checkpoint.pth'))

    # Save smaples in buffer
    if len(samples_buffer) > 0:
        samples_file = h5py.File(
            os.path.join(checkpointsdir(args.experiment), 'samples.hdf5'),
            'a')
        num_samples_saved = samples_file["num_samples"]

        shape = samples_file['samples'].shape[1:]
        samples_buffer = np.array(samples_buffer)
        samples_buffer = samples_buffer.reshape(-1, shape[0], shape[1])
        dataset = samples_file['samples']

        dataset[num_samples_saved[0]:(num_samples_saved[0] +
                                                      samples_buffer.shape[0]),
                                ...] = samples_buffer

        num_samples_saved[0] += samples_buffer.shape[0]


        samples_file.close()

def save_checkpoint_convencional(args: argparse.Namespace, obj_log: List[float],
                    error_log: List[float], rmse_seimogram: List[float],
                    rmse_model: List[float], itr: int,
                    state: Tuple[torch.Tensor, torch.optim.Optimizer],
                    samples_buffer: List[np.ndarray]):
    """Saves the current state of the training/sampling process.

    Saves intermediate results, such as network parameters and training logs. It
    also takes in samples that are ready to be written to an HDF5 file.

    Args:
        args: An argparse.Namespace, containing command line arguments.
        obj_log: A list containing the objective function values.
        error_log: A list containing the error values.
        itr: A float containind the iteration value of the current state
        g: A torch.nn.Module containing the generator network.
        z: A torch.Tensor containing the fixed latent variable.
        samples_buffer: A list containing the samples that are ready to be
            written to file
    """

    # Save intermediate network weights and training logs
    torch.save(
        {
            'obj_log': obj_log,
            'error_log': error_log,
            'rmse_seismogram' : rmse_seimogram,
            'rmse_model' : rmse_model,
            'v': state[0].detach().cpu(),
            'optimizer_state': state[1].state_dict(),
            'check_point_itr': itr,
        }, os.path.join(checkpointsdir(args.experiment), 'checkpoint.pth'))

    # Save smaples in buffer
    if len(samples_buffer) > 0:
        samples_file = h5py.File(
            os.path.join(checkpointsdir(args.experiment), 'samples.hdf5'),
            'a')
        num_samples_saved = samples_file["num_samples"]

        shape = samples_file['samples'].shape[1:]
        samples_buffer = np.array(samples_buffer)
        samples_buffer = samples_buffer.reshape(-1, shape[0], shape[1])
        dataset = samples_file['samples']

        dataset[num_samples_saved[0]:(num_samples_saved[0] +
                                                      samples_buffer.shape[0]),
                                ...] = samples_buffer

        num_samples_saved[0] += samples_buffer.shape[0]


        samples_file.close()


def save_checkpoint_(args: argparse.Namespace, obj_log: List[float],
                    error_log: List[float], itr: int,
                    G: torch.nn.Module, z: torch.Tensor,
                    samples_buffer: List[np.ndarray]):
    """Saves the current state of the training/sampling process.

    Saves intermediate results, such as network parameters and training logs. It
    also takes in samples that are ready to be written to an HDF5 file.

    Args:
        args: An argparse.Namespace, containing command line arguments.
        obj_log: A list containing the objective function values.
        error_log: A list containing the error values.
        itr: A float containind the iteration value of the current state
        g: A torch.nn.Module containing the generator network.
        z: A torch.Tensor containing the fixed latent variable.
        samples_buffer: A list containing the samples that are ready to be
            written to file
    """

    if G is None:
        state_dict = None
    else:
        state_dict = G.state_dict()

    # Save smaples in buffer
    try:
        if len(samples_buffer) > 0:
            samples_file = h5py.File(
                os.path.join(checkpointsdir(args.experiment), 'samples.hdf5'),
                'r+')
            num_samples_saved = samples_file["num_samples"]

            shape = samples_file['samples'].shape[1:]
            samples_buffer = np.array(samples_buffer)
            samples_buffer = samples_buffer.reshape(-1, shape[0], shape[1])

            samples_file['samples'][num_samples_saved[0]:(num_samples_saved[0] +
                                                        samples_buffer.shape[0]),
                                    ...] = samples_buffer

            num_samples_saved[0] += samples_buffer.shape[0]
            samples_file.close()
    except:
        return


    # Save intermediate network weights and training logs
    torch.save(
        {
            'obj_log': obj_log,
            'error_log': error_log,
            'model_state_dict': state_dict,
            'z': z,
            'check_point_itr': itr,
        }, os.path.join(checkpointsdir(args.experiment), 'checkpoint.pth'))


def decay_fn(args: argparse.Namespace):
    """
    Learning rate scheduler

    Returns:
        A function that takes in the current epoch and returns the learning
            rate.
    """
    if args.lr == args.lr_final:

        def lr_fn(t):
            return args.lr
    else:
        lag = 0
        max_itr_lag = args.max_itr - lag
        b = max_itr_lag / ((args.lr_final / args.lr)**(1 / args.gamma) - 1.0)
        a = args.lr / (b**args.gamma)
        print(f'a: {a}')
        print(f'b: {b}')
        def lr_fn(t, a=a, b=b, gamma=args.gamma, lag=lag):
            if t < lag:
                return args.lr
            else:
                return a * ((b + t)**gamma)

    return lr_fn

def load_checkpoint(args: argparse.Namespace):
    """Load the current state of the training/sampling process.

    Load intermediate results, such as network parameters and training logs.

    Args:
        args: An argparse.Namespace, containing command line arguments.

    """

    # Load intermediate network weights and training logs
    attrs = torch.load(os.path.join(checkpointsdir(args.experiment), 'checkpoint.pth'))

    return attrs
