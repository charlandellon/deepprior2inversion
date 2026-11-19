"""Deterministic cyclic FWI used to build a better Deep Prior background.

The unknown is a correction v to the squared slowness of the background:
dm = clamp(dm_init + v).  Sources are visited in a fixed order; there is no
Langevin noise.  The final velocity is exported in the legacy PTH format.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from dp4inversion.psgld import PreconditionedSGLDConvencional

from .setup_experiment import DeterministicSetup


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--background_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--freq', type=float, required=True)
    parser.add_argument('--tmax_ms', type=float, default=None)
    parser.add_argument('--nbl', type=int, default=150)
    parser.add_argument('--space_order', type=int, default=8)
    parser.add_argument('--deth_rec', type=int, default=5)
    parser.add_argument('--deth_src', type=int, default=5)
    parser.add_argument('--edge_margin_m', type=float, default=62.5)
    parser.add_argument('--water_depth_m', type=float, default=500.0)
    parser.add_argument('--bg_linear', type=int, default=0)
    parser.add_argument('--nsrc', type=int, default=100)
    parser.add_argument('--nrec', type=int, default=800)
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--checkpointing', type=int, default=1)
    parser.add_argument('--num_checkpoints', type=int, default=120)
    parser.add_argument('--max_itr', type=int, default=100)
    parser.add_argument('--shots_per_iter', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=30.0)
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--mask_water', type=int, default=0)
    parser.add_argument('--grad_smooth', type=float, default=0.0)
    parser.add_argument('--data_lowpass_hz', type=float, default=0.0)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--export_every', type=int, default=10)
    parser.add_argument('--resume', type=int, default=1)
    parser.add_argument('--retrain', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def make_optimizer(args, parameter):
    return PreconditionedSGLDConvencional(
        [parameter], lr=args.lr, weight_decay=args.wd, scale=args.scale)


def save_checkpoint(path, v, optimizer, iteration, history):
    torch.save(dict(
        v=v.detach().cpu(),
        optimizer_state=optimizer.state_dict(),
        iteration=int(iteration),
        history=history,
    ), path)


def export_velocity(path, setup, v, iteration):
    with torch.no_grad():
        dm_est = torch.clamp(setup.dm_init + v, setup.dm_min, setup.dm_max)
        vp = dm_est[0, 0].pow(-0.5).cpu().numpy().astype(np.float32)
    payload = dict(
        vp=vp,
        nx=int(setup.shape[0]),
        ny=int(setup.shape[1]),
        dx=float(setup.spacing[0]),
        dy=float(setup.spacing[1]),
        origin_xz_m=(0.0, 0.0),
        velocity_units='km/s',
        axes='xz',
        source='deterministic_fwi',
        iteration=int(iteration),
    )
    torch.save(payload, path)
    return payload


def main():
    args = parse_args()
    if args.sigma <= 0:
        raise ValueError('sigma must be positive')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    setup = DeterministicSetup(args)
    data_file = h5py.File(args.data_path, 'r')
    data = data_file['data']
    if tuple(data.shape[1:]) != (setup.nt, setup.nrec):
        raise ValueError(
            f'data shape {data.shape} incompatible with nt={setup.nt}, '
            f'nrec={setup.nrec}')

    v = torch.nn.Parameter(torch.zeros_like(setup.dm_init))
    optimizer = make_optimizer(args, v)
    history = []
    start = 0
    checkpoint_path = output / 'checkpoint.pth'
    if args.resume and not args.retrain and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location='cpu',
                           weights_only=False)
        if state['v'].shape == v.shape:
            v.data.copy_(state['v'])
            optimizer.load_state_dict(state['optimizer_state'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr
            start = int(state['iteration'])
            history = list(state.get('history', []))
            print(f'resuming from iteration {start}', flush=True)
        else:
            print('checkpoint shape differs; starting from zero', flush=True)

    print(json.dumps(dict(
        start=start, max_itr=args.max_itr, nsrc=setup.nsrc,
        nrec=setup.nrec, nt=setup.nt, dt_ms=float(setup.dt),
        nbl=int(args.nbl), checkpointing=bool(args.checkpointing),
        num_checkpoints=int(args.num_checkpoints),
        sigma=float(args.sigma), dm_min=float(setup.dm_min),
        dm_max=float(setup.dm_max), vp_min=setup.vp_min,
        vp_max=setup.vp_max,
        data_lowpass_hz=float(args.data_lowpass_hz))), flush=True)

    lowpass_mask = None
    if args.data_lowpass_hz > 0:
        freqs = torch.fft.rfftfreq(setup.nt, d=float(setup.dt))
        cutoff = float(args.data_lowpass_hz) * 1e-3   # cycles per ms
        lowpass_mask = 1.0 / (1.0 + (freqs / cutoff) ** 8)

    for itr in range(start, args.max_itr):
        started = time.perf_counter()
        optimizer.zero_grad()
        loss_sum, misfit_sum = 0.0, 0.0
        for k in range(args.shots_per_iter):
            idx = (itr * args.shots_per_iter + k) % setup.nsrc
            dm_est = torch.clamp(setup.dm_init + v,
                                 setup.dm_min, setup.dm_max)
            d_pred = setup.create_op(idx)(dm_est)
            d_obs = torch.from_numpy(
                np.asarray(data[idx], dtype=np.float32))
            diff = d_pred - d_obs
            if lowpass_mask is not None:
                diff = torch.fft.irfft(
                    torch.fft.rfft(diff, dim=0) * lowpass_mask[:, None],
                    n=setup.nt, dim=0)
                if not torch.isfinite(diff).all():
                    raise FloatingPointError(
                        'non-finite residual after low-pass')
            residual = diff / float(args.sigma)
            loss = 0.5 * residual.pow(2).sum() / args.shots_per_iter
            loss.backward()
            loss_sum += float(loss.detach()) * args.shots_per_iter
            misfit_sum += float(diff.pow(2).mean().detach())

        if args.grad_smooth > 0:
            smoothed = gaussian_filter(
                v.grad[0, 0].detach().cpu().numpy(),
                float(args.grad_smooth), mode='nearest')
            v.grad[0, 0] = torch.from_numpy(smoothed).to(v.grad.dtype)
        if args.mask_water:
            v.grad.mul_(setup.vp_mask)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([v], args.grad_clip)
        optimizer.step()
        if args.mask_water:
            with torch.no_grad():
                v.data.mul_(setup.vp_mask)


        elapsed = time.perf_counter() - started
        record = dict(iteration=itr + 1, shot=int(idx),
                      loss=loss_sum / args.shots_per_iter,
                      misfit=misfit_sum / args.shots_per_iter,
                      seconds=elapsed)
        history.append(record)
        print(json.dumps(record), flush=True)

        if (itr + 1) % args.save_every == 0 or itr + 1 == args.max_itr:
            save_checkpoint(checkpoint_path, v, optimizer, itr + 1, history)
        if (itr + 1) % args.export_every == 0 or itr + 1 == args.max_itr:
            export_velocity(output / f'model_iter_{itr + 1:05d}.pth',
                            setup, v, itr + 1)

    export_velocity(output / 'model_final.pth', setup, v, args.max_itr)
    (output / 'history.json').write_text(
        json.dumps(history, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {output / "model_final.pth"}', flush=True)


if __name__ == '__main__':
    main()
