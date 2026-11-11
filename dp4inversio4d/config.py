"""Configuration validation for implemented truth-building phases only."""

from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path, seed=None, scenario=None):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        raise ValueError('configuration must be a mapping')
    cfg = deepcopy(cfg)
    for key in ('experiment', 'model', 'reservoirs', 'output'):
        if not isinstance(cfg.get(key), dict):
            raise ValueError(f'{key} must be configured as a mapping')
    def placeholders(value, path='config'):
        if value is None or isinstance(value, str) and ('<' in value or '>' in value):
            raise ValueError(f'{path}: unresolved placeholder/null')
        if isinstance(value, dict):
            for key, item in value.items():
                placeholders(item, f'{path}.{key}')
        elif isinstance(value, list):
            for i, item in enumerate(value):
                placeholders(item, f'{path}[{i}]')
    placeholders(cfg)
    if seed is not None:
        cfg['experiment']['seed'] = seed
    if scenario is not None:
        cfg['experiment']['scenario'] = scenario
    if not isinstance(cfg['experiment'].get('seed'), int) or cfg['experiment']['seed'] < 0:
        raise ValueError('experiment.seed must be a nonnegative integer')
    if cfg['model'].get('input_units') != 'km/s' or cfg['model'].get('input_axes') != 'xz':
        raise ValueError('legacy input requires explicit km/s and xz axes')
    mode = cfg['model'].get('baseline_mode', 'calibrate_to_original')
    if mode not in ('calibrate_to_original', 'replace_reservoirs_oil_water'):
        raise ValueError('unknown baseline_mode')
    input_keys = ('marmousi_path', 'original_vp_path') if mode == 'replace_reservoirs_oil_water' else ('marmousi_path', 'background_path')
    for key in input_keys:
        if not Path(cfg['model'][key]).is_file():
            raise ValueError(f'model.{key}: file not found: {cfg["model"][key]}')
    scenario = cfg['experiment'].get('scenario')
    if scenario not in ('E0', 'E1', 'E2', 'E3'):
        raise ValueError('truth scenario must be E0, E1, E2 or E3')
    for name, reservoir in cfg['reservoirs'].items():
        if scenario == 'E0' or scenario == 'E1' and name == 'R2':
            reservoir['active'] = False
        elif scenario in ('E2', 'E3'):
            reservoir['active'] = True
    return cfg
