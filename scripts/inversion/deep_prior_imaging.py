"""Implementation of deep Bayesian inference for seismic imaging algorithm.

Typical usage example:

imaging_instance = DeepPriorImaging(args)
imaging_instance.sample(args)
"""

import os

import argparse
import numpy as np
import torch
from tqdm import tqdm

from dp4inversion.deep_prior import DeepPrior, Generator, Generator_, Model_vp, Model_v, NN_FWI
from dp4inversion.psgld import PreconditionedSGLD
from dp4inversion.setup_experiment import SeismicSetup, simulate_sequential_data, mute
from dp4inversion.utils import plot_latente, plot_rmse, decay_fn, save_checkpoint, load_checkpoint, setup_sample_file, plot_velocity, plot_error, plot_objective, plot_reflectivity
from dp4inversion.project_path import datadir, imagesdir

from scipy import ndimage


class DeepPriorImaging(object):
    """Seismic imaging and uncertainty quantification with deep priors.

    This class implements the seismic imaging and uncertainty quantification
    approach with deep priors. Sets up the seismic experiment, creates the
    observed data and forward operators, and initializes the deep prior. During
    training, the deep prior is updated to fit the observed data. The code saves
    all the seismic images throughout posterior sampling.

    Attributes:
        device: A torch.device object indicating the device to use.
        imaging_setup: A SeismicSetup object, containing the seismic experiment
            acquisition setup.
        obj_log: A list of objective function values.
        error_log: A list of prediction error values.
    """

    def __init__(self, args: argparse.Namespace):
        """Initializes a DeepPriorImaging object.

        Args:
            args: An argparse.Namespace, containing command line arguments.
        """
        super().__init__()

        # Setting device and default data type.
        if torch.cuda.is_available() and args.cuda:
            self.device = torch.device('cuda')
            # self.device = torch.set_default_device('cuda:0')
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.set_default_dtype(torch.float32)

        else:
            self.device = torch.device('cpu')
            # torch.set_default_tensor_type('torch.FloatTensor')
            # self.device = torch.set_default_device('cpu')
            torch.set_default_dtype(torch.float32)


        torch.manual_seed(args.seed)
        np.random.seed(42)

        # Setup seismic experiment, including data acquisition setup and forward
        # operators.
        self.imaging_setup = SeismicSetup(args,
                                          device=True,
                                          sim_source=args.sim_source)

        # Set up HDF5 file to store posterior samples.
        setup_sample_file(args, self.imaging_setup.dm.shape[2:], args.max_itr)

        # Initialize some book keeping lists.
        self.obj_log = []
        self.error_log = []

        self.rmse_seimogram = []
        self.rmse_model = []

        print(f'{args.file_result_figs}')

    def negative_log_likelihood(self, d_pred: torch.Tensor,
                                d_obs: torch.Tensor,
                                sigma: float) -> torch.Tensor:
        """Negative-log likelihood under Gaussian noise assumption.

        Args:
            d_pred: A torch.Tensor object, containing the predicted data. d_obs:
            A torch.Tensor object, containing the observed data. sigma: A float,
            containing the noise standard deviation.

        Returns:
            A torch.Tensor containing a float value, the negative log
                likelihood.
        """
        return (self.imaging_setup.nsrc / 2.0) * torch.norm(
            (d_pred - d_obs) / sigma)**2

    def sample(self, args: argparse.Namespace):
        """Main deep-prior based seismic imaging posterior sampling loop.

        Args:
            args: An argparse.Namespace, containing command line hyperparameters
                and data paths.
        """
        # Create observed data (HDF5 dataset, not fully loaded into memory).

        if args.sim_source:
            d_obs = self.imaging_setup.create_sim_src_data(args)
        else:
            d_obs = simulate_sequential_data(args)


        # True seismic image (perturbation model) to be used for book keeping.
        dm = (self.imaging_setup.dm).to(self.device)
        vp_init = (self.imaging_setup.vp_init).to(self.device)
        vp_mask = (self.imaging_setup.vp_mask).to(self.device)

        # vp_init = torch.ones_like(vp_init) / (2.5**2)

        # vp_init_ = torch.randn(vp_init.shape) * vp_init.std() + vp_init.mean()
        # vp_init = torch.clamp(vp_init_, min=vp_init.min(), max=vp_init.max()).to(self.device)

        # vp_init = vp_mask * vp_init + (1.0 - vp_mask) * dm

        scaling_factor_dvp = ((self.imaging_setup.dm - self.imaging_setup.vp_init).min().item(),
                          (self.imaging_setup.dm - self.imaging_setup.vp_init).max().item())

        print(f'scaling_factor_dvp: {scaling_factor_dvp}')

        print(f'vp_init: {vp_init.detach().cpu().numpy().shape}')
        print(f'vp_mask: {vp_mask.detach().cpu().numpy().shape}')
        print(f'dm: {dm.size()}')

        scaling_factor = (self.imaging_setup.dm_min, self.imaging_setup.dm_max)
        # scaling_factor = (1./3.6**2, self.imaging_setup.dm.max().item())

        print(f'scaling_factor: {scaling_factor}')

        # args.wd = 30
        # args.sigma = 0.1
        # args.lr_final = 1e-4
        # args.lr = 1e-3


        # g = NN_FWI(vp_init, Mask=vp_mask, Vp_min=scaling_factor[0],
        #            Vp_max=scaling_factor[1], l0=8, h0=6, w0=13).to(self.device)

        # input_shape = 8 #w * h * num_layer
        # z = torch.randn(1, input_shape).to(self.device)


        # input_shape = 100
        # num_layer = 4
        # w, h = 50, 50 #dm.size()[-2], dm.size()[-1]
        # g = Generator(input_shape=input_shape, Vp=vp_init, Mask=vp_mask,
        #               Vp_min=scaling_factor[0], Vp_max=scaling_factor[1],
        #               kernel_size=4, w0=w, h0=h, num_layer=num_layer).to(self.device)

        # z = torch.randn(1, input_shape).to(self.device)

        # Deep prior network and its fixed random input.
        g = DeepPrior(dm.size(), vp_init, Mask=None, ch_in=1, ch_encoding=args.ch_encoding,
                    ch_decoding=args.ch_decoding,
                    activation_fn=torch.nn.PReLU(init=0.2),
                    ch_skip=args.ch_skip, bias=True,
                    kernel_size_encoding=5, kernel_size_decoding=5,
                    upsample_scale=args.upsample_scale,
                    coarse_output_scale=args.coarse_output_scale,
                    scaling_factor=scaling_factor).to(self.device)

        # Stochastic gradient Langevin dynamics sampling sub-routine.
        self.sampler = PreconditionedSGLD(g.parameters(),
                                          args.lr,
                                          weight_decay=args.wd,
                                          scale=args.scale)

        try:
            attrs = load_checkpoint(args)
            g.load_state_dict(attrs['model_state_dict'])
            z = attrs['z'].to(self.device)
            currrent_it = attrs['check_point_itr'] + 1
            self.obj_log = attrs['obj_log']
            self.error_log = attrs['error_log']
            self.rmse_seimogram = attrs['rmse_seismogram']
            self.rmse_model = attrs['rmse_model']
            # Stochastic gradient Langevin dynamics sampling sub-routine.
            self.sampler = PreconditionedSGLD(g.parameters(),
                                            args.lr,
                                            weight_decay=args.wd,
                                            scale=args.scale)
            if 'optimizer_state' in attrs:
                self.sampler.load_state_dict(attrs['optimizer_state'])

        except Exception as e:
            print(f'Não foi possível carregar o modelo! Iniciando treinamento do início. Motivo: {e}')

            # z = torch.randn(g.get_latent_shape()).to(self.device)
            z = torch.randn((1, 1, dm.size()[-2] // args.upsample_scale,
                            dm.size()[-1] // args.upsample_scale)).to(self.device)
            currrent_it = 0


        total_params = sum(p.numel() for p in g.parameters())
        print("Número total de pesos na rede:", sum(p.numel() for p in g.parameters()))

        print(f'latente shape: {z.size()}')



        # Learning rate decay function.
        self.lr_fn = decay_fn(args)

        if args.upsample_scale == 1:
            plot_reflectivity(z.squeeze().detach().cpu().numpy(), self.imaging_setup.model,
                                    os.sep.join([imagesdir(args.file_result_figs), f'latente.png']),
                                    label_data='Latente sample', cmap='jet')
        else:
            plot_latente(z.squeeze().detach().cpu().numpy(), os.sep.join([imagesdir(args.file_result_figs), f'latente.png']),
                                    label_data='Latente sample', cmap='jet')

        # A buffer for samples to be written to file every 100 iterations.
        samples_buffer = []
        import time
        total_time = 0

        print(f'currrent_it: {currrent_it}')
        # Sampling loop, run for `args.max_itr` iterations.
        with tqdm(range(currrent_it, args.max_itr), unit='iteration', colour='#B5F2A9') as pb:
            for itr in pb:
                # print(f'itr: {itr}')
                start_time = time.time()

                # Update the learning rate.
                self.lr_decay(itr)

                # Compute predicted seismic image via the deep prior
                # reparameterization.
                # z = torch.randn(g.get_latent_shape()).to(self.device)

                dm_est = g(z)
                # dm_est = g()

                # dm_est = dm_est_ + vp_init

                # dm_est = torch.where(dm_est < vp_min,
                #                      torch.full_like(dm_est, vp_min), dm_est)

                # import matplotlib.pyplot as plt

                # plt.imshow(self.imaging_setup.model.vp.data[100:-100, 100:-100].T, cmap='jet')
                # plt.show()

                # dm_est = vp_mask * dm_est + (1.0 - vp_mask) * dm


                # plt.imshow(dm_est.detach().cpu().numpy()[0, 0, :,:].T, cmap='jet')
                # plt.show()

                # print(f'o minimo:{dm_est.detach().cpu().numpy().min()}\no máximo: {dm_est.detach().cpu().numpy().max()}')

                # obj = 0.0
                # for _ in range(5):
                # Randomly pick a source experiment
                idx = np.random.choice(self.imaging_setup.nsrc,
                                    1,
                                    replace=False)[0]

                # Create Devito-based Born scattering forward modeling operator
                # for source position index `idx`. This operator is wrapped as a
                # pytorch layer to facilitate gradient computation via automatic
                # differentiation while exploiting Devito's highly optimized
                # stencil code.
                # obj = 0.0

                # for idx in range(self.imaging_setup.nsrc):

                forward_op = self.imaging_setup.create_op(src_idx=idx)

                # Compute predicted data to match the observed data.
                d_pred = forward_op(dm_est)

                #mascara de mute
                # mask = mute(d_pred, self.imaging_setup.coord_src[idx],
                #             np.linspace(0, self.imaging_setup.model.domain_size[0], num=self.imaging_setup.nrec),
                #             self.imaging_setup.dt, self.imaging_setup.nt, 1.5, flat_len = self.imaging_setup.flat_len,
                #             taper_len = self.imaging_setup.taper_len, device=self.device)

                # d_pred *=  mask
                # Compute the negative-log likelihood.

                # args.wd = (5e-3)
                obj = self.negative_log_likelihood(
                    d_pred,
                    torch.from_numpy(d_obs[idx]).to(self.device),
                    args.sigma) #+ 0.5 * args.wd * torch.norm(torch.cat([p.view(-1) for p in g.parameters()]))**2


                # Compute the gradient of negative-log likelihood with respect
                # to deep prior weights. The gradient contribution of the
                # Gaussian prior term on deep prior weights will be included
                # internally in the sampling sub-routine.
                grad = torch.autograd.grad(obj, g.parameters())
                for param, grad in zip(g.parameters(), grad):
                    param.grad = grad

                # obj.backward()
                # Update the deep prior weights. The gradient of the Gaussian
                # prior is included internally in `sampler`.
                self.sampler.step()

                # Write summary Of this iterations.

                # dm_est = vp_mask * dm_est + (1.0 - vp_mask) * dm

                self.write_summary(args, itr, obj.item(), dm_est, dm, pb)

                self.write_summary_rmse(dm_est.detach().cpu().numpy() **(-0.5),
                                        dm.detach().cpu().numpy() ** (-0.5),
                                        d_pred.detach().cpu().numpy(),
                                        d_obs[idx])

                samples_buffer.append(dm_est.detach().cpu().numpy() **(-0.5))


                if itr % 100 == 0 or itr == args.max_itr - 1:
                    # print(f'len: {len(samples_buffer)}')
                    plot_reflectivity(ndimage.gaussian_laplace((dm_est**(-0.5) - vp_init**(-0.5)).squeeze().detach().cpu().numpy(), sigma=(5, 3)),
                                      self.imaging_setup.model, os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'migrated_images'])), f'migrated_salva_{itr}.png']),
                                      label_data='Migrated Image')
                    plot_velocity(self.imaging_setup.model,
                                  os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'velocity_images'])),
                                               f'imagem_salva_{itr}.png']), clip=(scaling_factor[1]**(-0.5), scaling_factor[0]**(-0.5)))
                    plot_error(np.array(self.error_log),
                               os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'convergence_image'])),
                                            'quadratic_norm_error.png']))
                    plot_objective(np.array(self.obj_log),
                                   os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'convergence_image'])),
                                                'objective_function.png']))

                    plot_rmse(np.array(self.rmse_seimogram),
                               os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'convergence_image'])),
                                            'rmse_seismogram.png']), y_label='RMSE (N/m²)')

                    plot_rmse(np.array(self.rmse_model),
                                   os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'convergence_image'])),
                                                'rmse_model.png']), y_label='RMSE (km/s)')

                    save_checkpoint(args, self.obj_log, self.error_log,
                                    self.rmse_seimogram, self.rmse_model,
                                    itr, (g, self.sampler), z, samples_buffer)
                    samples_buffer = []

                end_time = time.time()

                # Calcular o tempo de execução
                execution_time = end_time - start_time
                total_time += execution_time

        plot_rmse(np.array(self.rmse_seimogram[int(args.max_itr//2):]),
                    os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'convergence_image'])),
                    'rmse_seismogram_pos_burnin.png']), y_label='RMSE (N/m²)')

        plot_rmse(np.array(self.rmse_model[int(args.max_itr//2):]),
                    os.sep.join([imagesdir(os.sep.join([args.file_result_figs, 'convergence_image'])),
                    'rmse_model_pos_burnin.png']), y_label='RMSE (km/s)')

                # print(f"Tempo de execução da iteração {itr}: {execution_time:.6f} segundos")


        # Calcular o tempo médio de execução por iteração
        average_time = total_time / args.max_itr

        print(f"Tempo médio de execução por iteração: {average_time:.6f} segundos")

    def write_summary(self, args: argparse.Namespace, itr: int, obj: float,
                      dm_est: np.ndarray, dm: np.ndarray, pb: tqdm):
        """Writes the summary of one SGLD iteration to file.

        Args:
            args: An argparse.Namespace, containing command line hyperparameters
                and data paths.
            itr: An int, containing the current iteration number.
            obj: A float, containing the negative-log likelihood.
            dm_est: A numpy.ndarray, containing the estimated seismic image.
            dm: A numpy.ndarray, containing the true seismic image.
            pb: A tqdm.tqdm, containing the progress bar.
        """
        # Prediction error.
        model_error = ((dm_est - dm).norm()**2).item()

        # Print current objective value.
        pb.set_postfix(itr=f'{itr + 1}/{args.max_itr}',
                       obj=f'{obj:.4f}',
                       misfit=f'{model_error:.4f}')

        # Append current objective value and perdition error to log.
        self.obj_log.append(obj)
        self.error_log.append(model_error)

    def write_summary_rmse(self, dm_est: np.ndarray, dm: np.ndarray,
                      d_pred: np.ndarray, d_obs: np.ndarray):
        rmse_simogram = np.sqrt(np.mean((d_pred - d_obs)**2))
        rmse_model = np.sqrt(np.mean((dm_est - dm)**2))

        self.rmse_seimogram.append(rmse_simogram)
        self.rmse_model.append(rmse_model)

    def lr_decay(self, itr: int):
        """Reduces the learning rate of sampling sub-routine.

        Args:
            itr: An int, containing the current iteration number.
        """
        for param_group in self.sampler.param_groups:
            param_group['lr'] = self.lr_fn(itr)
            # print(f'minha real lr {param_group['lr']} - it {itr}')
