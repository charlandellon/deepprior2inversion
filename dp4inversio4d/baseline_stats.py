"""Baseline-only velocity and saturation statistics.

Useful while the monitor chains are still running: computes the same
region summaries as :mod:`dp4inversio4d.stats` for a single survey.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from .rock_physics import PetroParameters
from .stats import _invert_saturation, _region_summary, _saturation_summary, load_posterior


def _flatten(prefix, value, rows):
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f'{prefix}.{key}' if prefix else key, item, rows)
    else:
        rows.append((prefix, value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--checkpoints', default='data/checkpoints')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--survey', default='baseline')
    parser.add_argument('--burn-in', type=int, default=15000)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    truth_dir = Path(args.truth)
    checkpoint_root = Path(args.checkpoints)
    survey = args.survey
    prefix = f'{args.tag}_{survey}'
    matches = [
        path for path in checkpoint_root.rglob(f'{prefix}*/samples.hdf5')
        if path.parent.name == prefix
        or path.parent.name.startswith(prefix + '_max_itr-')
    ]
    if len(matches) != 1:
        raise SystemExit(
            f'expected exactly one samples.hdf5 for {survey}, found {matches}'
        )
    samples_path = matches[0]

    with np.load(truth_dir / 'truth' / 'models.npz') as data:
        truth = {
            key: data[key] for key in
            ('v_true', 'So_true', 'mask_R1', 'mask_R2', 'K_dry')
        }
    with (truth_dir / 'config_resolved.yaml').open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)

    posterior = load_posterior(samples_path, args.burn_in, args.stride)
    index = {'baseline': 0, 'monitor1': 1, 'monitor2': 2}[survey]
    masks = {'R1': truth['mask_R1'], 'R2': truth['mask_R2']}
    control = ~(truth['mask_R1'] | truth['mask_R2'])

    summary = {
        'burn_in': int(args.burn_in),
        'stride': int(args.stride),
        'survey': survey,
        'samples_used': int(posterior.shape[0]),
        'samples_file': str(samples_path),
        'vp_global': {
            'mean_m_s': float(posterior.mean()),
            'std_m_s': float(posterior.std()),
            'p05_m_s': float(np.percentile(posterior, 5)),
            'p95_m_s': float(np.percentile(posterior, 95)),
        },
        'reservoirs': {
            name: _region_summary(posterior[..., mask], truth['v_true'][index][mask])
            for name, mask in masks.items()
        },
        'control': _region_summary(posterior[..., control],
                                   truth['v_true'][index][control]),
        'saturation': {},
    }

    for name, mask in masks.items():
        params = PetroParameters(**config['rock_physics'][name])
        so_posterior = _invert_saturation(
            posterior[..., mask], truth['K_dry'][mask], params
        )
        summary['saturation'][name] = _saturation_summary(
            so_posterior, truth['So_true'][index][mask]
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'baseline_statistics.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8'
    )

    rows = []
    _flatten('vp_global', summary['vp_global'], rows)
    for region, metrics in summary['reservoirs'].items():
        _flatten(region, metrics, rows)
    _flatten('control', summary['control'], rows)
    for region, metrics in summary['saturation'].items():
        _flatten(f'saturation.{region}', metrics, rows)
    with (output / 'baseline_statistics.csv').open(
        'w', newline='', encoding='utf-8'
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric', 'value'])
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
