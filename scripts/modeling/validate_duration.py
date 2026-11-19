"""Check that the acquisition window records all relevant events.

A shot is propagated with the configured duration T and again with a longer
window (factor*T).  The extra energy beyond T measures how much the shorter
run truncates.  The test does not store the full wavefield.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from examples.seismic import (AcquisitionGeometry, Model, RickerSource,
                              TimeAxis)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--factor', type=float, default=1.25)
    parser.add_argument('--shot-fractions', type=float, nargs='+',
                        default=[0.1, 0.5, 0.9])
    parser.add_argument('--nrec', type=int, default=400)
    parser.add_argument('--nbl', type=int, default=50)
    parser.add_argument('--space-order', type=int, default=8)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def load_payload(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def run_shot(payload, duration_ms, shot_x, nrec, nbl, space_order,
             edge_margin, depth_m):
    from examples.seismic.acoustic import AcousticWaveSolver

    vp = np.asarray(payload['vp'], dtype=np.float32)
    shape = (int(payload['nx']), int(payload['ny']))
    model = Model(space_order=space_order, vp=vp, origin=(0.0, 0.0),
                  shape=shape, dtype=np.float32,
                  spacing=(float(payload['dx']), float(payload['dy'])),
                  nbl=int(nbl), bcs='damp')
    x = np.linspace(edge_margin, model.domain_size[0] - edge_margin,
                    num=nrec, dtype=np.float32)
    rec_coordinates = np.column_stack(
        [x, np.full(nrec, depth_m, dtype=np.float32)])
    f0 = float(payload['f0']) * 1e-3
    time_range = TimeAxis(start=0.0, stop=float(duration_ms),
                          step=float(model.critical_dt))
    src = RickerSource(name='src', grid=model.grid, f0=f0,
                       time_range=time_range, npoint=1)
    src.coordinates.data[:, 0] = shot_x
    src.coordinates.data[:, -1] = depth_m
    geometry = AcquisitionGeometry(model, rec_coordinates,
                                   src.coordinates.data, t0=0.0,
                                   tn=float(duration_ms), src_type='Ricker',
                                   f0=f0)
    solver = AcousticWaveSolver(model, geometry, kernel='OT2',
                                space_order=space_order)
    rec, _, _ = solver.forward(src=src, rec=geometry.rec, vp=model.vp,
                               save=False)
    return np.array(rec.data, dtype=np.float32), float(model.critical_dt)


def main():
    args = parse_args()
    if args.factor <= 1.0:
        raise ValueError('factor must be greater than one')
    payload = load_payload(args.model_path)
    base_ms = float(payload['tmax'])
    extended_ms = base_ms*args.factor
    edge_margin = float(payload.get('edge_margin_m', 0.0))
    depth_m = float(payload.get('src_rcv_depth_m', 2.0*payload['dy']))
    x0 = edge_margin
    x1 = float(payload['nx'])*float(payload['dx']) - edge_margin

    output = Path(args.output) if args.output else (
        Path('data/images/duration_qc')/Path(args.model_path).stem)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(model_path=str(args.model_path), base_duration_ms=base_ms,
                   extended_duration_ms=extended_ms, factor=args.factor,
                   shots=[])

    for fraction in args.shot_fractions:
        shot_x = x0 + fraction*(x1 - x0)
        base, dt_ms = run_shot(payload, base_ms, shot_x, args.nrec, args.nbl,
                               args.space_order, edge_margin, depth_m)
        extended, _ = run_shot(payload, extended_ms, shot_x, args.nrec,
                               args.nbl, args.space_order, edge_margin, depth_m)
        base_samples = base.shape[0]
        if extended.shape[0] < base_samples:
            raise ValueError('extended gather is shorter than the configured window')
        base_energy = float(np.sum(base.astype(np.float64)**2))
        tail = np.sum(base[int(0.9*base_samples):].astype(np.float64)**2)
        # The extended run starts at the same t0 and covers the base window
        # exactly, so the extra samples are the additional recording time.
        extra = np.sum(extended[base_samples:].astype(np.float64)**2)
        ratio = float(extra/max(base_energy, 1e-30))
        tail_ratio = float(tail/max(base_energy, 1e-30))
        # A large tail is expected near the end of the window and is not a
        # truncation; only energy that keeps arriving after T is decisive.
        status = 'ok' if ratio < 1e-3 else 'review'
        shot_dir = output/f'shot_x{shot_x:08.1f}'
        shot_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shot_dir/'gathers.npz', base=base,
                            extended=extended)
        time_s = np.arange(extended.shape[0])*dt_ms/1000
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.plot(time_s, np.sqrt(np.mean(extended.astype(np.float64)**2, axis=1)),
                  linewidth=0.8, label='extended RMS')
        axis.axvline(base_ms/1000, color='tab:red', linestyle='--',
                     label='configured T')
        axis.set_xlabel('Time (s)')
        axis.set_ylabel('Receiver RMS')
        axis.set_title(f'Duration check, shot x={shot_x/1000:.2f} km')
        axis.legend()
        fig.tight_layout()
        figure = shot_dir/'duration.png'
        fig.savefig(figure, dpi=140)
        plt.close(fig)
        summary['shots'].append(dict(
            shot_x_m=float(shot_x), base_energy=base_energy,
            base_tail_energy_ratio=tail_ratio,
            energy_beyond_base_over_base=ratio,
            status=status, figure=str(figure)))
        print(f'shot x={shot_x:.1f}: tail={tail_ratio:.3e} '
              f'beyond={ratio:.3e} status={status}')

    report = output/'duration_qc.json'
    report.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(f'wrote {report}')
    return summary


if __name__ == '__main__':
    main()
