"""Central clean/noisy gather figures in the project seismic convention."""

from argparse import Namespace
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dp4inversion.setup_experiment import SeismicSetup, mute

from .plotting import DPI


def _setup_args(truth, survey, source_mode, nsrc, nrec, nsimsrc, freq):
    models = truth/'truth'/'models'
    return Namespace(
        mode='legacy', vel_model='marmousi', model_path=str(models/f'{survey}.pth'),
        background_path=str(models/'background.pth'), bg_linear=1,
        nsrc=nsrc, nrec=nrec, nsimsrc=nsimsrc, freq=freq, sigma=0.0,
        cuda=0, dv=0.0, seed=42, tmax=None, noise_type='white_gaussian',
        file_result_figs='4d/gather_qc')


def _central_clean(truth, survey, source_mode, nsrc=100, nrec=400, nsimsrc=100,
                   frequency_hz=5.0):
    args = _setup_args(truth, survey, source_mode, nsrc, nrec, nsimsrc, frequency_hz)
    # DeepPriorImaging seeds NumPy with 42 before creating SeismicSetup, which
    # draws the simultaneous codes from the global stream.
    np.random.seed(42)
    physical = SeismicSetup(args, device=False, sim_source=False)
    index = physical.nsrc//2
    if source_mode == 'sequential':
        clean = physical.create_op(index)(physical.dm).detach().numpy()
        source_x = float(physical.coord_src[index])
    else:
        encoded = SeismicSetup(args, device=False, sim_source=True)
        gathers = np.stack([physical.create_op(j)(physical.dm).detach().numpy()
                            for j in range(encoded.nsimsrc)])
        clean = np.tensordot(encoded.src_enc[index], gathers, axes=1)
        source_x = None
    return clean.astype(np.float64), index, physical, source_x


def _limit(clean):
    value = float(np.quantile(clean, .95))
    return value if value > 0 else max(float(np.quantile(np.abs(clean), .95)), 1e-12)


def _pair(clean, observed, extent, limit, snr_db, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    images = []
    for axis, field, panel in zip(axes, (clean, observed),
                                  ('Seismic clean', 'Seismic noisy')):
        images.append(axis.imshow(-field, cmap='gray', vmin=-limit, vmax=limit,
                                  aspect='auto', extent=extent))
        axis.set_xlabel('Length (km)')
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_title(f'{panel} - SNR: {snr_db:.2f} dB', fontsize='medium')
    axes[0].set_ylabel('Time (s)')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.07)
    fig.colorbar(images[1], cax=cax).set_label('Amplitude (N/m²)')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_central_gather(truth, survey, data_path, source_mode='simult',
                        nsrc=100, nrec=400, nsimsrc=100, frequency_hz=5.0,
                        output=None):
    truth, data_path = Path(truth), Path(data_path)
    clean, index, setup, source_x = _central_clean(
        truth, survey, source_mode, nsrc, nrec, nsimsrc, frequency_hz)
    with h5py.File(data_path, 'r') as handle:
        observed = np.asarray(handle['data'][index], dtype=np.float64)
    if observed.shape != clean.shape:
        raise ValueError(f'observed {observed.shape} and clean {clean.shape} differ; '
                         'check acquisition geometry/duration')
    noise_power = float(np.mean((observed-clean)**2))
    if source_mode == 'sequential':
        mask = mute(__import__('torch').ones(clean.shape), source_x,
                    np.linspace(0, setup.model.domain_size[0], nrec), setup.dt,
                    clean.shape[0], 1.5, flat_len=300, taper_len=300,
                    device='cpu').numpy()
        signal_power = float(np.mean((clean*mask)**2))
    else:
        signal_power = float(np.mean(clean**2))
    snr_db = float(10*np.log10(signal_power/noise_power))
    extent = [0, setup.model.domain_size[0]/1000, (clean.shape[0]-1)*setup.dt/1000, 0]
    kind = ('coded experiment' if source_mode == 'simult' else 'physical shot')
    title = f'{survey}: central {kind} {index}; white Gaussian noise'
    figure = _pair(clean, observed, extent, _limit(clean), snr_db, title)
    output = Path(output) if output is not None else truth/'figures'/f'gather_{survey}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=DPI)
    plt.close(figure)
    return {'survey': survey, 'data': str(data_path), 'index': index,
            'source_mode': source_mode, 'snr_db': snr_db, 'figure': str(output)}
