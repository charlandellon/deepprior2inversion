"""Command line entry point for the independent 4D workflow."""

import argparse
import json
from pathlib import Path

from .build import build_models
from .calibration import calibrate_survey
from .gathers import plot_central_gather
from .stats import SURVEYS, compute_statistics


def main():
    parser = argparse.ArgumentParser(prog='python -m dp4inversio4d')
    commands = parser.add_subparsers(dest='command', required=True)

    build = commands.add_parser('build', help='build the three velocity models and legacy PTH inputs')
    build.add_argument('--config', default='dp4inversio4d/configs/marmousi4d_oil_water.yaml')
    build.add_argument('--output', default=None)
    build.add_argument('--seed', type=int, default=None)
    build.add_argument('--scenario', choices=['E0', 'E1', 'E2', 'E3'], default=None)

    calibrate = commands.add_parser('calibrate', help='write one white-noise SIGMA per survey')
    calibrate.add_argument('--truth', required=True)
    calibrate.add_argument('--survey', choices=[*SURVEYS, 'all'], default='all')
    calibrate.add_argument('--source-mode', choices=['sequential', 'simult'], default='simult')
    calibrate.add_argument('--nsrc', type=int, default=100)
    calibrate.add_argument('--nrec', type=int, default=400)
    calibrate.add_argument('--freq', type=float, default=5.0)
    calibrate.add_argument('--nsimsrc', type=int, default=None)
    calibrate.add_argument('--target-snr-db', type=float, default=6.0)

    stats = commands.add_parser('stats', help='quantify 4D posterior statistics after inversion')
    stats.add_argument('--truth', required=True)
    stats.add_argument('--checkpoints', default='data/checkpoints')
    stats.add_argument('--tag', required=True, help='experiment prefix; expects <tag>_<survey>')
    stats.add_argument('--output', default=None)
    stats.add_argument('--burn-in', type=int, default=17500)
    stats.add_argument('--stride', type=int, default=10)

    gathers = commands.add_parser('gathers', help='plot central clean/noisy gathers')
    gathers.add_argument('--truth', required=True)
    gathers.add_argument('--survey', choices=[*SURVEYS, 'all'], default='all')
    gathers.add_argument('--source-mode', choices=['sequential', 'simult'], default='simult')
    gathers.add_argument('--data-dir', default='data/observed_data')
    gathers.add_argument('--tag', default='expanded4d')
    gathers.add_argument('--data', action='append', default=[],
                         help='optional survey=path override')
    gathers.add_argument('--nsrc', type=int, default=100)
    gathers.add_argument('--nrec', type=int, default=400)
    gathers.add_argument('--nsimsrc', type=int, default=100)
    gathers.add_argument('--freq', type=float, default=5.0)

    args = parser.parse_args()
    if args.command == 'build':
        result = build_models(args.config, args.output, args.seed, args.scenario)
        print(json.dumps(result, indent=2))
    elif args.command == 'calibrate':
        surveys = SURVEYS if args.survey == 'all' else (args.survey,)
        for survey in surveys:
            report = calibrate_survey(args.truth, survey, args.source_mode,
                                      args.nsrc, args.nrec, args.freq,
                                      args.target_snr_db, args.nsimsrc)
            print(json.dumps(report, indent=2))
    elif args.command == 'gathers':
        overrides = dict(item.split('=', 1) for item in args.data)
        surveys = SURVEYS if args.survey == 'all' else (args.survey,)
        data_dir = Path(args.data_dir)
        for survey in surveys:
            if survey in overrides:
                data_path = Path(overrides[survey])
            else:
                pattern = f'freq_*_{args.tag}_{survey}_{args.source_mode}*_sigma_*.h5'
                matches = sorted(data_dir.glob(pattern))
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f'expected one data file for {survey} matching {data_dir/pattern}, '
                        f'found {matches}; pass --data {survey}=<path>')
                data_path = matches[0]
            print(json.dumps(plot_central_gather(
                args.truth, survey, data_path, args.source_mode, args.nsrc,
                args.nrec, args.nsimsrc, args.freq), indent=2))
    else:
        report = compute_statistics(args.truth, args.checkpoints, args.tag,
                                    args.output, args.burn_in, args.stride)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
