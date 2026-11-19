"""Test absorbing-boundary thickness without storing the full wavefield.

The same physical model, acquisition geometry, wavelet and duration are used
for every ``nbl``; only the padding changes.  A thick reference boundary is
simulated first, then each candidate is compared against it in the complete
and late time windows.  Reflections that move when ``nbl`` changes are the
numerical artefacts this test is designed to expose.

Example (inside the Docker image, where Devito is installed):

    python scripts/modeling/test_absorber_boundaries.py \
        --model_path data/velocity_model/marmousi2-model-true-large_dx12.5.pth \
        --nbl 25 50 75 100 150 --shot-fractions 0.1 0.5 0.9
"""

import argparse
import json
import time
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
    parser.add_argument('--nbl', type=int, nargs='+', default=[25, 50, 75, 100, 150])
    parser.add_argument('--reference-nbl', type=int, default=None,
                        help='defaults to the largest requested nbl')
    parser.add_argument('--shot-fractions', type=float, nargs='+',
                        default=[0.1, 0.5, 0.9])
    parser.add_argument('--nrec', type=int, default=400)
    parser.add_argument('--space-order', type=int, default=8)
    parser.add_argument('--f0-hz', type=float, default=None,
                        help='defaults to the PTH metadata')
    parser.add_argument('--tmax-ms', type=float, default=None,
                        help='defaults to the PTH metadata')
    parser.add_argument('--edge-margin-m', type=float, default=None)
    parser.add_argument('--depth-m', type=float, default=None)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def load_payload(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def boundary_model(payload, nbl, space_order):
    vp = np.asarray(payload['vp'], dtype=np.float32)
    dx, dz = float(payload['dx']), float(payload['dy'])
    shape = (int(payload['nx']), int(payload['ny']))
    if vp.shape != shape:
        raise ValueError(f'vp shape {vp.shape} disagrees with metadata {shape}')
    return Model(space_order=space_order, vp=vp, origin=(0.0, 0.0),
                 shape=shape, dtype=np.float32, spacing=(dx, dz),
                 nbl=int(nbl), bcs='damp')


def simulate_shot(payload, model, shot_x, nrec, edge_margin, depth_m,
                  f0_hz, tmax_ms):
    dx = float(payload['dx'])
    dz = float(payload['dy'])
    x = np.linspace(edge_margin, model.domain_size[0] - edge_margin, num=nrec,
                    dtype=np.float32)
    rec_coordinates = np.column_stack(
        [x, np.full(nrec, depth_m, dtype=np.float32)])
    time_range = TimeAxis(start=0.0, stop=float(tmax_ms),
                          step=float(model.critical_dt))
    f0_devito = float(f0_hz) * 1e-3  # Hz -> 1/ms
    src = RickerSource(name='src', grid=model.grid, f0=f0_devito,
                       time_range=time_range, npoint=1)
    src.coordinates.data[:, 0] = shot_x
    src.coordinates.data[:, -1] = depth_m
    geometry = AcquisitionGeometry(model, rec_coordinates,
                                   src.coordinates.data, t0=0.0,
                                   tn=float(tmax_ms), src_type='Ricker',
                                   f0=f0_devito)
    from examples.seismic.acoustic import AcousticWaveSolver
    solver = AcousticWaveSolver(model, geometry, kernel='OT2',
                                space_order=model.space_order)
    started = time.perf_counter()
    rec, _, _ = solver.forward(src=src, rec=geometry.rec, vp=model.vp,
                               save=False)
    elapsed = time.perf_counter() - started
    return np.array(rec.data, dtype=np.float32), time_range.num, elapsed


def relative_metrics(data, reference, time_ms):
    residual = data.astype(np.float64) - reference.astype(np.float64)
    ref = reference.astype(np.float64)
    full = dict(
        relative_l2=float(np.linalg.norm(residual)/max(np.linalg.norm(ref), 1e-30)),
        max_abs_over_ref_max=float(np.max(np.abs(residual))/max(np.max(np.abs(ref)), 1e-30)))
    late = time_ms >= 0.7*time_ms[-1]
    late_residual, late_ref = residual[late], ref[late]
    full['late_relative_l2'] = float(
        np.linalg.norm(late_residual)/max(np.linalg.norm(late_ref), 1e-30))
    full['late_max_abs_over_ref_max'] = float(
        np.max(np.abs(late_residual))/max(np.max(np.abs(ref)), 1e-30))
    return full


def plot_differences(output, shot_x, results, reference_nbl, dt_ms):
    names = [f'nbl={item["nbl"]}' for item in results]
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.1*len(results)),
                             squeeze=False, sharex=True, sharey=True)
    limit = None
    for item in results:
        difference = item['data'].astype(np.float64) - item['reference'].astype(np.float64)
        scale = float(np.quantile(np.abs(difference), 0.99)) or 1.0
        limit = max(limit or 0.0, scale)
    for axis, item, name in zip(axes[:, 0], results, names):
        difference = item['data'].astype(np.float64) - item['reference'].astype(np.float64)
        image = axis.imshow(difference.T, cmap='seismic', vmin=-limit, vmax=limit,
                            aspect='auto', extent=[0, difference.shape[0]*dt_ms/1000,
                                                   0, difference.shape[1]])
        axis.set_title(f'{name} minus reference nbl={reference_nbl}')
        axis.set_ylabel('Receiver')
    axes[-1, 0].set_xlabel('Time (s)')
    fig.suptitle(f'Boundary residual, shot x={shot_x/1000:.2f} km')
    fig.tight_layout()
    path = output/f'boundary_difference_shot_x{shot_x:08.1f}.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_traces(output, shot_x, results, dt_ms):
    fig, axis = plt.subplots(figsize=(10, 5))
    time_s = np.arange(results[0]['data'].shape[0])*dt_ms/1000
    receiver = results[0]['data'].shape[1] - 1
    for item in results:
        axis.plot(time_s, item['data'][:, receiver], linewidth=0.8,
                  label=f'nbl={item["nbl"]}')
    axis.set_xlabel('Time (s)')
    axis.set_ylabel('Amplitude (N/m²)')
    axis.set_title(f'Far-offset trace, shot x={shot_x/1000:.2f} km')
    axis.legend()
    fig.tight_layout()
    path = output/f'boundary_traces_shot_x{shot_x:08.1f}.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    payload = load_payload(args.model_path)
    f0_hz = float(args.f0_hz or payload['f0'])
    tmax_ms = float(args.tmax_ms or payload['tmax'])
    edge_margin = float(args.edge_margin_m if args.edge_margin_m is not None
                       else payload.get('edge_margin_m', 0.0))
    depth_m = float(args.depth_m if args.depth_m is not None
                    else payload.get('src_rcv_depth_m', 2.0*payload['dy']))
    nbl_values = sorted(set(int(value) for value in args.nbl))
    reference_nbl = (int(args.reference_nbl) if args.reference_nbl is not None
                     else max(nbl_values))
    if reference_nbl not in nbl_values:
        nbl_values.append(reference_nbl)
        nbl_values.sort()
    if min(nbl_values) < 10:
        raise ValueError('nbl must be at least 10 cells')

    stem = Path(args.model_path).stem
    output = Path(args.output) if args.output else (
        Path('data/images/boundary_qc')/stem)
    output.mkdir(parents=True, exist_ok=True)

    x0, x1 = edge_margin, float(payload['nx'])*float(payload['dx']) - edge_margin
    shots = [x0 + fraction*(x1 - x0) for fraction in args.shot_fractions]
    summary = dict(model_path=str(args.model_path), f0_hz=f0_hz,
                   tmax_ms=tmax_ms, nrec=args.nrec, space_order=args.space_order,
                   edge_margin_m=edge_margin, depth_m=depth_m,
                   reference_nbl=reference_nbl, shots=[])
    print(f'model={args.model_path} f0={f0_hz:.6f} Hz tmax={tmax_ms:.3f} ms')
    print(f'edge_margin={edge_margin:.3f} m depth={depth_m:.3f} m '
          f'reference_nbl={reference_nbl}')

    for shot_x in shots:
        gathers, timings, dt_ms = {}, {}, None
        for nbl in nbl_values:
            model = boundary_model(payload, nbl, args.space_order)
            gather, nt, elapsed = simulate_shot(
                payload, model, shot_x, args.nrec, edge_margin, depth_m,
                f0_hz, tmax_ms)
            gathers[nbl] = gather
            dt_ms = float(model.critical_dt)
            timings[nbl] = dict(seconds=float(elapsed), nt=int(nt),
                                cells=int((int(payload['nx']) + 2*nbl)
                                          * (int(payload['ny']) + 2*nbl)),
                                full_wavefield_gib=float(
                                    (int(payload['nx']) + 2*nbl)
                                    * (int(payload['ny']) + 2*nbl)
                                    * nt * 4 / 2**30))
            print(f'shot x={shot_x:.1f} nbl={nbl} nt={nt} dt={dt_ms:.4f} ms '
                  f'cells={timings[nbl]["cells"]} '
                  f'wavefield={timings[nbl]["full_wavefield_gib"]:.2f} GiB '
                  f'time={elapsed:.1f} s')

        reference = gathers[reference_nbl]
        time_ms = np.arange(reference.shape[0])*dt_ms
        results = []
        for nbl in nbl_values:
            metrics = relative_metrics(gathers[nbl], reference, time_ms)
            results.append(dict(nbl=nbl, metrics=metrics,
                                data=gathers[nbl], reference=reference))
        shots_dir = output/f'shot_x{shot_x:08.1f}'
        shots_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            shots_dir/'gathers.npz',
            **{f'nbl_{nbl}': gathers[nbl] for nbl in nbl_values})
        difference_figure = plot_differences(
            shots_dir, shot_x, results, reference_nbl, dt_ms)
        traces_figure = plot_traces(shots_dir, shot_x, results, dt_ms)
        for item in results:
            item.pop('data')
            item.pop('reference')
        summary['shots'].append(dict(
            shot_x_m=float(shot_x), dt_ms=dt_ms,
            timings={str(key): value for key, value in timings.items()},
            metrics={str(item['nbl']): item['metrics'] for item in results},
            difference_figure=str(difference_figure),
            traces_figure=str(traces_figure)))

    report = output/'boundary_qc.json'
    report.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(f'wrote {report}')
    return summary


if __name__ == '__main__':
    main()
