from devito import configuration, gaussian_smooth
from examples.seismic import Model, AcquisitionGeometry, TimeAxis, RickerSource, plot_velocity
from examples.seismic.acoustic import AcousticWaveSolver
import h5py
import numpy as np
import os
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import torch
from tqdm import tqdm

from .devito_layer import ForwardBorn, ForwardModeling
from .project_path import datadir, imagesdir
from .utils import get_velocity, get_velocity_marmousi, get_velocity_, load_torch_file, plot_velocity
import logging
import argparse

configuration['log-level'] = 'WARNING'
logging.basicConfig(level=logging.INFO)


class ForwardBornLayer(torch.nn.Module):

    def __init__(self, model, src, geometry, device):
        super(ForwardBornLayer, self).__init__()
        self.forward_born = ForwardBorn()
        self.model = model
        self.src = src
        self.geometry = geometry
        self.device = device
        self.solver = AcousticWaveSolver(self.model,
                                         self.geometry,
                                         space_order=16)

    def forward(self, x):
        return self.forward_born.apply(x, self.src, self.model, self.geometry,
                                       self.solver, self.device)

class ForwardModelingLayer(torch.nn.Module):

    def __init__(self, model, src, geometry, device):
        super(ForwardModelingLayer, self).__init__()
        self.forward_born = ForwardModeling()
        self.model = model
        self.src = src
        self.geometry = geometry
        self.device = device
        self.solver = AcousticWaveSolver(self.model,
                                         self.geometry, kernel= 'OT2',
                                         space_order=self.model.space_order)

    def forward(self, x):
        return self.forward_born.apply(x, self.src, self.model, self.geometry,
                                       self.solver, self.device)


class SeismicSetup(object):
    """Creating Devito wrapper and utilities.
    """

    def __init__(self, args: argparse.Namespace, device=False, sim_source=False):
        super(SeismicSetup, self).__init__()

        # Setting device and default data type.
        if torch.cuda.is_available() and args.cuda and device:
            self.device = torch.device('cuda')
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.set_default_dtype(torch.float32)
        else:
            self.device = torch.device('cpu')
            # torch.set_default_tensor_type('torch.FloatTensor')
            torch.set_default_dtype(torch.float32)

        self.args = args

        if self.args.vel_model.lower() == '4layers':
            self.setup_model_layers()

        elif self.args.vel_model.lower() == 'marmousi':
            # self.setup_model()
            self.setup_model_marmousi()
        elif self.args.vel_model.lower() == 'bp':
            self.setup_model_BP()
        else:
            print('Escolha um modelo de velocidade válido! (4layers, marmousi ou bp)')
        # self.setup_model_marmousi()


        tmax = getattr(args, 'tmax', None)
        if tmax is not None:
            self.tn = min(self.tn, float(tmax) * 1e3)

        self.sigma = args.sigma
        self.sim_source = sim_source
        if self.sim_source:
            self.nsimsrc = self.args.nsimsrc
            self.src_enc = np.random.randn(self.nsrc, self.nsimsrc)
            self.src_enc *= 1.0 / np.sqrt(self.nsimsrc)
        else:
            self.nsimsrc = 1
            self.src_enc = None

        self.forward_op = self.wrap_op()

    def mute_op(self, dm, end=10, length=5):
        start = end - length
        damp = torch.zeros([dm.shape[-2], dm.shape[-1]], device=dm.device)
        damp[..., end:] = 1.
        damp[..., start:end] = (1. + torch.sin(
            (np.pi / 2.0 * torch.arange(0, length)) / (length))) / 2.
        return damp * dm

    def setup_model(self):
        """
        Load velocity and setup Devito model
        """
        from scipy.ndimage import gaussian_filter

        vp, self.spacing, shape, origin = get_velocity_marmousi()

        vp_mask = np.ones_like(vp)
        vp_mask = np.where((vp >= 1.2) & (vp <= 1.6), 0, vp_mask)

        # vp_smooth = 1. / gaussian_filter( 1. / vp, 50)
        vp_smooth = gaussian_filter(vp, [7, 7], mode='reflect')
        vp_smooth = np.mean(vp_smooth, axis=0, keepdims=True).repeat(vp.shape[0], axis=0)

        vp_smooth = vp_mask * vp_smooth + (1.0 - vp_mask) * vp

        t_range = np.sqrt((shape[0]*self.spacing[0])**2 + (shape[1]*self.spacing[1])**2)/np.mean(vp)

        self.tn = t_range
        self.f0 = 0.020
        self.np_kernel = 130
        self.nsrc = self.args.nsrc #int(shape[0] // 32) #200#300
        self.nrec = int(shape[0] // 2) #400#600
        self.deth_rec = 2
        self.deth_src = 2
        self.flat_len = 250 # retirar onda direta
        self.taper_len = 100 # filtrar a região de onda direta


        self.dm = torch.from_numpy(1. / vp**2).unsqueeze(0).unsqueeze(0)
        nbl = 50
        space_order = 4
        self.model = Model(space_order=space_order,
                           vp=vp,
                           origin=origin,
                           shape=shape,
                           dtype=np.float32,
                           spacing=self.spacing,
                           nbl=nbl,
                           bcs='damp')

        # self.model.dt_scale = 1.73

        self.model_smooth = Model(space_order=space_order,
                           vp=vp_smooth,
                           origin=origin,
                           shape=shape,
                           dtype=np.float32,
                           spacing=self.spacing,
                           nbl=nbl,
                           bcs='damp')

        self.dt = self.model.critical_dt
        self.coord_src = np.linspace(0, self.model.domain_size[0],
                                     num=self.nsrc)



        plot_velocity(self.model, os.sep.join([imagesdir(self.args.file_result_figs), f'marmousi_real.png']))
        plot_velocity(self.model_smooth, os.sep.join([imagesdir(self.args.file_result_figs), f'marmousi_inicial.png']))
        nb = self.model_smooth.nbl
        self.vp_init = torch.from_numpy(np.array(1. / self.model_smooth.vp.data[nb:-nb, nb:-nb]**2, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        self.vp_mask = torch.from_numpy(vp_mask).unsqueeze(0).unsqueeze(0)

        del self.model_smooth
        del vp
        del vp_mask
        del vp_smooth

    def setup_model_BP(self):
        """
        Load velocity and setup Marmousi model
        """
        f0 = 3
        vel_path = datadir("velocity_model")
        path = os.path.join(vel_path, f'BP_model_freq_{f0}_v1.pth')
        model_vp = load_torch_file(path)
        vp = model_vp['vp']
        self.spacing = (model_vp['dx'], model_vp['dy'])
        shape = (model_vp['nx'], model_vp['ny'])
        origin = (0, 0)



        vp_mask = np.ones_like(vp)
        # vp_mask[:, :int(500 / model_vp['dx'])] = 0

        vp_mask = np.where(vp <= 1.5, 0, vp_mask)

        path = os.path.join(vel_path, f'BP-smooth-linear_freq_{f0}_v1.pth')
        model_vp_smooth = load_torch_file(path)
        vp_smooth = model_vp_smooth['vp']
        # vp_smooth = np.mean(vp_smooth, axis=0, keepdims=True).repeat(vp.shape[0], axis=0)
        # vp_smooth = np.mean(vp[::10, :], axis=0, keepdims=True).repeat(vp.shape[0], axis=0)
        # vp_smooth = np.ones_like(vp) * ((vp.min() + vp.max()) / 2) + 0.5

        if self.args.bg_linear == 1:
            vp_smooth = np.mean(vp_smooth, axis=0, keepdims=True).repeat(vp.shape[0], axis=0)
        elif self.args.bg_linear == 2:

            vp_smooth = gaussian_filter1d(np.mean(vp_smooth, axis=0, keepdims=True), sigma=50).repeat(vp.shape[0], axis=0)



        vp_smooth = vp_mask * vp_smooth + (1.0 - vp_mask) * vp

        t_range = (np.sqrt((shape[0]*self.spacing[0])**2 + (shape[1]*self.spacing[1])**2)/np.mean(vp)) * 2.0

        print(f't_max: {t_range}')

        self.tn = t_range
        self.f0 = model_vp_smooth['f0'] * 1e-3
        self.np_kernel = 100
        self.nsrc = self.args.nsrc #int(shape[0] // 32) #200#300
        self.nrec = self.args.nrec #model_vp_smooth['nrcv'] #int(shape[0] // 2) #400#600
        self.deth_rec = 2
        self.deth_src = 2
        self.dm = torch.from_numpy(1. / vp**2).unsqueeze(0).unsqueeze(0)
        self.dm_min = torch.from_numpy(1. / (vp + self.args.dv)**2).min().item()
        self.dm_max = self.dm.max().item()
        nbl = 50
        space_order = 8
        self.model = Model(space_order=space_order,
                           vp=vp,
                           origin=origin,
                           shape=shape,
                           dtype=np.float32,
                           spacing=self.spacing,
                           nbl=nbl,
                           bcs='damp')

        # self.model.dt_scale = 1.73

        self.model_smooth = Model(space_order=space_order,
                           vp=vp_smooth,
                           origin=origin,
                           shape=shape,
                           dtype=np.float32,
                           spacing=self.spacing,
                           nbl=nbl,
                           bcs='damp')

        self.dt = self.model.critical_dt

        print(f'dt_critical: {self.dt}')
        self.coord_src = np.linspace(0, self.model.domain_size[0],
                                     num=self.nsrc)

        del model_vp_smooth
        del model_vp

        plot_velocity(self.model, os.sep.join([imagesdir(self.args.file_result_figs), f'BP_real.png']), clip=(self.dm_max**(-0.5), self.dm_min**(-0.5)))
        plot_velocity(self.model_smooth, os.sep.join([imagesdir(self.args.file_result_figs), f'BP_inicial.png']), clip=(self.dm_max**(-0.5), self.dm_min**(-0.5)))
        nb = self.model_smooth.nbl
        self.vp_init = torch.from_numpy(np.array(1. / self.model_smooth.vp.data[nb:-nb, nb:-nb]**2, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        self.vp_mask = torch.from_numpy(vp_mask).unsqueeze(0).unsqueeze(0)
        del self.model_smooth
        del vp
        del vp_mask
        del vp_smooth

    def setup_model_marmousi(self):
        """
        Load velocity and setup Marmousi model
        """
        f0 = 5
        vel_path = datadir("velocity_model")
        path = os.path.join(vel_path, f'marmousi2-model-true-large_freq_{f0}.pth')
        model_vp = load_torch_file(path)
        vp = model_vp['vp']
        self.spacing = (model_vp['dx'], model_vp['dy'])
        shape = (model_vp['nx'], model_vp['ny'])
        origin = (0, 0)

        vp_mask = np.ones_like(vp)
        vp_mask[:, :int(500 / model_vp['dx']) + 1] = 0

        vp = vp_mask * model_vp['vp'] + (1.0 - vp_mask) * model_vp['vp']

        # vp_mask = np.where(vp == 1.5, 0, vp_mask)

        path = os.path.join(vel_path, f'marmousi2-model-smooth-linear_freq_{f0}.pth')
        model_vp_smooth = load_torch_file(path)
        vp_smooth = model_vp_smooth['vp']

        if self.args.bg_linear == 1:
            vp_smooth = np.mean(vp_smooth, axis=0, keepdims=True).repeat(vp.shape[0], axis=0)
        elif self.args.bg_linear == 2:
            from scipy.ndimage import gaussian_filter1d

            vp_smooth = gaussian_filter1d(np.mean(vp_smooth, axis=0, keepdims=True), sigma=50).repeat(vp.shape[0], axis=0)



        vp_smooth = vp_mask * vp_smooth + (1.0 - vp_mask) * vp

        t_range = np.sqrt((shape[0]*self.spacing[0])**2 + (shape[1]*self.spacing[1])**2)/np.mean(vp)

        # f0 = 2.5
        self.tn = t_range + 1.0
        self.f0 = f0 * 1e-3 #model_vp_smooth['f0'] * 1e-3
        self.nsrc = self.args.nsrc #int(shape[0] // 32) #200#300
        self.nrec = self.args.nrec #model_vp_smooth['nrcv'] #int(shape[0] // 2) #400#600
        self.deth_rec = 2
        self.deth_src = 2
        self.np_kernel = 120

        self.dm = torch.from_numpy(1. / vp**2).unsqueeze(0).unsqueeze(0)
        self.dm_min = torch.from_numpy(1. / (vp + self.args.dv)**2).min().item()
        self.dm_max = self.dm.max().item()
        nbl = 50
        space_order = 8
        self.model = Model(space_order=space_order,
                           vp=vp,
                           origin=origin,
                           shape=shape,
                           dtype=np.float32,
                           spacing=self.spacing,
                           nbl=nbl,
                           bcs='damp')

        # self.model.dt_scale = 1.73

        self.model_smooth = Model(space_order=space_order,
                           vp=vp_smooth,
                           origin=origin,
                           shape=shape,
                           dtype=np.float32,
                           spacing=self.spacing,
                           nbl=nbl,
                           bcs='damp')

        self.dt = self.model.critical_dt
        self.coord_src = np.linspace(0, self.model.domain_size[0],
                                     num=self.nsrc)

        del model_vp_smooth
        del model_vp


        plot_velocity(self.model, os.sep.join([imagesdir(self.args.file_result_figs), f'marmousi_real.png']), clip=(vp.min(), vp.max()))
        plot_velocity(self.model_smooth, os.sep.join([imagesdir(self.args.file_result_figs), f'marmousi_inicial.png']), clip=(vp.min(), vp.max()))
        nb = self.model_smooth.nbl
        self.vp_init = torch.from_numpy(np.array(1. / self.model_smooth.vp.data[nb:-nb, nb:-nb]**2, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        self.vp_mask = torch.from_numpy(vp_mask).unsqueeze(0).unsqueeze(0)
        del self.model_smooth
        del vp
        del vp_mask
        del vp_smooth

    def setup_model_layers(self):
        """
        Load velocity and setup Devito model
        """
        from scipy.ndimage import gaussian_filter
        from examples.seismic import demo_model

        layers = demo_model('layers-isotropic', origin=(0., 0.), shape=(100, 100),
                           spacing=(10., 10.), nbl=30, nlayers=4, vp_bottom=3.5)

        # plot_velocity(layers)
        nbl = layers.nbl
        vp, self.spacing, shape, origin = np.array(layers.vp.data[nbl:-nbl, nbl:-nbl]), layers.spacing, layers.shape, layers.origin
        # vp_unique = np.unique(vp)
        # vp[vp == vp_unique[-2]] = 0
        # vp[vp == vp_unique[-1]] = vp_unique[-2]
        # vp[vp == 0] = vp_unique[-1]
        vp_mask = np.ones_like(vp)
        vp_mask = np.where(vp == 1.5, 0, vp_mask)

        # vp_smooth = np.tile(np.linspace(vp.min(), vp.max(), shape[0]), (shape[1], 1)) #np.mean(vp) * np.ones_like(vp) #1. / gaussian_filter( 1. / vp, 5)
        vp_smooth = np.mean(vp) * np.ones_like(vp) #1. / gaussian_filter( 1. / vp, 5)

        #vp_smooth = 1. / gaussian_filter( 1. / vp, 7)

        vp_smooth = vp_mask * vp_smooth + (1.0 - vp_mask) * vp

        t_range = (np.sqrt((shape[0]*self.spacing[0])**2 + (shape[1]*self.spacing[1])**2)/np.mean(vp)) * 2.0

        self.tn = t_range #self.args.time_seismic
        self.f0 = self.args.freq * 1e-3 #0.005
        self.np_kernel = 130
        self.nsrc = self.args.nsrc #31
        self.nrec = self.args.nrec #101 #int(shape[0] // 2)
        self.deth_rec = 2
        self.deth_src = 2
        self.flat_len = 250 # retirar onda direta
        self.taper_len = 100 # filtrar a região de onda direta

        self.dm = torch.from_numpy(1. / vp**2).unsqueeze(0).unsqueeze(0)
        self.dm_min = torch.from_numpy(1. / (vp + self.args.dv)**2).min().item()
        self.dm_max = self.dm.max().item()

        space_order = 8
        self.model = Model(space_order=space_order,
                           vp=vp,
                           origin=origin,
                           shape=shape,
                           dtype=np.float32,
                           spacing=self.spacing,
                           nbl=nbl,
                           bcs='damp')

        # self.model.dt_scale = 1.73

        self.model_smooth = Model(space_order=space_order,
                           vp=vp_smooth,
                           origin=origin,
                           shape=shape,
                           dtype=np.float32,
                           spacing=self.spacing,
                           nbl=nbl,
                           bcs='damp')

        self.dt = self.model.critical_dt
        self.coord_src = np.linspace(0, self.model.domain_size[0],
                                     num=self.nsrc)

        plot_velocity(self.model, os.sep.join([imagesdir(self.args.file_result_figs), f'4Layers_real.png']), clip=(vp.min(), vp.max()))
        plot_velocity(self.model_smooth, os.sep.join([imagesdir(self.args.file_result_figs), f'4Layers_inicial.png']), clip=(vp.min(), vp.max()))
        nb = self.model_smooth.nbl
        self.vp_init = torch.from_numpy(np.array(1. / self.model_smooth.vp.data[nb:-nb, nb:-nb]**2, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        self.vp_mask = torch.from_numpy(vp_mask).unsqueeze(0).unsqueeze(0)

    def setup_model_(self):
        """
        Load velocity and setup Devito model
        """
        model_true, origin, self.spacing, shape = get_velocity_('marmousi2-model-true-large.pth')

        model_smt, _, _, _ = get_velocity_('marmousi2-model-smooth-large.pth')

        self.tn = model_true['tmax'].astype(np.float32)
        self.f0 = model_true['f0'] / 1000.
        self.dt = model_true['dt'].astype(np.float32)
        self.nsrc = int(shape[0] // 2) #200#300
        self.nrec = model_true['nrcv'] #shape[0] #400#600
        self.deth_rec = model_true['src_rcv_depth']
        self.deth_src = model_true['src_rcv_depth']

        # dm = 1.0 / (vp ** 2)
        # self.dm = torch.from_numpy(dm).unsqueeze(0).unsqueeze(0)
        vp = model_true['vp'].astype(np.float32)
        vp_smooth = model_smt['vp'].astype(np.float32)
        vp_mask = model_smt['mask']
        self.dm = torch.from_numpy(vp).unsqueeze(0).unsqueeze(0)
        nbl = 50
        space_order = 4
        self.model = Model(space_order=space_order,
                           dt=self.dt,
                           vp=vp,
                           origin=origin,
                           shape=shape,
                           dtype=np.float32,
                           spacing=self.spacing,
                           nbl=nbl,
                           bcs='damp')

        # self.model.dt_scale = 1.73

        self.model_smooth = Model(space_order=space_order,
                            dt=self.dt,
                           vp=vp_smooth,
                           origin=origin,
                           shape=shape,
                           dtype=np.float32,
                           spacing=self.spacing,
                           nbl=nbl,
                           bcs='damp')

        # self.dt = self.model.critical_dt
        self.coord_src = np.linspace(0, self.model.domain_size[0],
                                     num=self.nsrc)

        # plot_velocity(self.model)
        # plot_velocity(self.model_smooth)
        nb = self.model_smooth.nbl
        self.vp_init = torch.from_numpy(np.array(self.model_smooth.vp.data[nb:-nb, nb:-nb], dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        self.vp_mask = torch.from_numpy(vp_mask).unsqueeze(0).unsqueeze(0)




    def wrap_op(self):
        """
        Initialize the wrapped Devito operator
        """

        time_range = TimeAxis(start=0.0, stop=self.tn, step=self.dt)
        self.nt = time_range.num

        rec_coordinates = np.empty((self.nrec, len(self.model.spacing)),
                                   dtype=np.float32)
        rec_coordinates[:, 0] = np.linspace(0,
                                            self.model.domain_size[0],
                                            num=self.nrec)
        rec_coordinates[:, 1] = self.deth_rec * self.model.spacing[1]

        # Source geometry
        src = RickerSource(name='src',
                           grid=self.model.grid,
                           f0=self.f0,
                           time_range=time_range,
                           npoint=self.nsimsrc)

        # src.data[:, 0:1] /= 5e5

        self.wavelet = np.array(src.data[:, 0:1])
        # print(f'fonnte min: {self.wavelet.min()} max: {self.wavelet.max()}')
        src.coordinates.data[:, 0] = np.linspace(0,
                                                 self.model.domain_size[0],
                                                 num=self.nsimsrc)
        src.coordinates.data[:, -1] = self.deth_src * self.model.spacing[1]

        # Create geometry
        geometry = AcquisitionGeometry(self.model,
                                       rec_coordinates,
                                       src.coordinates.data,
                                       t0=0.0,
                                       tn=self.tn,
                                       src_type='Ricker',
                                       f0=self.f0)
        return ForwardModelingLayer(self.model, src, geometry, self.device)

    def create_op(self, src_idx=0):
        """
        Update the operator with the source index
        """
        if not self.sim_source:
            self.forward_op.src.coordinates_data[:, 0] = self.coord_src[src_idx]
        else:
            self.forward_op.src.data[:] = self.wavelet
            self.forward_op.src.data[:] *= self.src_enc[src_idx, :]

        def J(x):
            return self.forward_op(x)

        return J

    def create_sim_src_data(self, args: argparse.Namespace):
        """Create simultaneous source data.
        """

        sim_data_path = os.path.join(datadir("observed_data"), args.file_simult_data)

        if (not os.path.exists(sim_data_path)) or (args.run_all == 1):
            d_lin = simulate_sequential_data(args)[...]

            sim_data_file = h5py.File(sim_data_path, 'w')
            sim_data = sim_data_file.create_dataset(
                'data', [self.nsrc, self.nt, self.nrec], dtype=np.float32)

            logging.info('Creating simultaneous source experiments')
            for i in tqdm(range(self.nsrc)):
                for j in range(self.nsimsrc):
                    sim_data[i, ...] += self.src_enc[i, j] * d_lin[j, ...]

            sim_data_file.close()

        return h5py.File(sim_data_path, 'r')["data"]


def simulate_sequential_data(args: argparse.Namespace):
    """Simulates simultaneous source data and adds noise.
    """

    data_path = os.path.join(datadir("observed_data"), args.file_seq_data)

    if (not os.path.exists(data_path)) or (args.run_all == 1):
        imaging_setup = SeismicSetup(args, device=False, sim_source=False)
        dm = imaging_setup.dm

        data_file = h5py.File(data_path, 'w')
        d_lin = data_file.create_dataset(
            'data', [imaging_setup.nsrc, imaging_setup.nt, imaging_setup.nrec],
            dtype=np.float32)

        logging.info('Simulating observed data')
        for idx in tqdm(range(imaging_setup.nsrc)):
            forward_op = imaging_setup.create_op(src_idx=idx)
            d = forward_op(dm)
            # mask = mute(d, imaging_setup.coord_src[idx],
            #                 np.linspace(0, imaging_setup.model.domain_size[0], num=imaging_setup.nrec),
            #                 imaging_setup.dt, imaging_setup.nt, 1.5, flat_len = imaging_setup.flat_len,
            #                 taper_len = imaging_setup.taper_len, device='cpu')
            # d *= mask
            d_lin[idx, :, :] = d.detach().numpy().astype(np.float32)

        # Add noise to get observed data (-8.7466 dB)
        kernel = np.outer(imaging_setup.wavelet[:imaging_setup.np_kernel],
                            imaging_setup.wavelet[:imaging_setup.np_kernel])
        for j in range(d_lin.shape[0]):
            e = args.sigma * np.random.randn(*d_lin.shape[1:])
            e = signal.fftconvolve(e, kernel, mode='same')
            e *= 1 / np.linalg.norm(e)
            e *= np.sqrt(np.prod(e.shape)) * args.sigma
            d_lin[j, ...] += e

        data_file.close()

    return h5py.File(data_path, 'r')["data"]



def mute(observed_data, source_position,
         receivers_per_shot, dt, nt, v_direct,
         flat_len = 100, taper_len = 200, device='cuda'):

        mask = torch.ones_like(observed_data)

        taper = torch.cos(torch.arange(taper_len) / taper_len * np.pi / 2)
        mute_len = flat_len + 2 * taper_len
        mute = torch.zeros(mute_len, device=device)
        mute[:taper_len] = taper
        mute[-taper_len:] = taper.flip(0)

        sx = source_position
        for receiver_idx, rx in enumerate(receivers_per_shot):
            dist = abs(sx - rx)
            arrival_time = dist / v_direct / dt
            mute_start = int(arrival_time) - mute_len // 2
            mute_end = mute_start + mute_len
            if (mute_start > nt):
                continue
            actual_mute_start = max(mute_start, 0)
            actual_mute_end = min(mute_end, nt)
            mask[actual_mute_start:actual_mute_end, receiver_idx] = \
                mute[actual_mute_start-mute_start: actual_mute_end-mute_start]

        return mask
