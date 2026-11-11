"""Build baseline/monitor velocity models and export them to the legacy format."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .build_models import build_velocity_truth
from .config import load_config
from .io import load_legacy_model, load_original_marmousi_vp, write_manifest
from .plotting import plot_scenario, plot_velocity_truth
from .reservoirs import extract_body, expand_body_vertically, wells_along_body
from .saturation import build_saturations


def save_legacy_pth(path, velocity_zx_m_s, grid):
    """Write the PTH fields read by dp4inversion.setup_model_marmousi."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(vp=(velocity_zx_m_s.T/1000).astype(np.float32),
                   nx=grid.shape_zx[1], ny=grid.shape_zx[0],
                   dx=grid.spacing_xz_m[0], dy=grid.spacing_xz_m[1],
                   origin_xz_m=grid.origin_xz_m,
                   velocity_units='km/s', axes='xz')
    torch.save(payload, path)
    return payload


def build_models(config_path, output=None, seed=None, scenario=None):
    cfg = load_config(config_path, seed, scenario)
    original, grid = load_legacy_model(cfg['model']['marmousi_path'], cfg['model']['origin_xz_m'])
    if cfg['model'].get('baseline_mode') != 'replace_reservoirs_oil_water':
        raise ValueError('the 4D module implements only the oil-water replacement baseline')
    if cfg['model']['reservoir_selection'] != 'native_low_velocity_components':
        raise ValueError('reservoir selection must be native_low_velocity_components')
    native, native_grid = load_original_marmousi_vp(
        cfg['model']['original_vp_path'], cfg['model']['original_to_local_x_offset_m'])
    selected, selection_qc, native_masks, original_masks = {}, {}, {}, {}
    for name, reservoir in cfg['reservoirs'].items():
        if reservoir['well_placement'] != 'body_x_quantiles_20_80':
            raise ValueError('unsupported well placement rule')
        mask, metadata, cropped = extract_body(
            native, native_grid, grid, reservoir['search_roi_xz_m'],
            reservoir['low_velocity_threshold_m_s'], reservoir.get('body_seed_xz_m'))
        original_masks[name] = mask.copy()
        reservoir['injection_well_xz_m'], reservoir['production_well_xz_m'] = wells_along_body(mask, grid)
        if 'enlargement' in reservoir:
            enlargement = reservoir['enlargement']
            if enlargement.get('method') != 'native_vertical_dilation':
                raise ValueError('unsupported reservoir enlargement method')
            mask, expansion_qc, cropped = expand_body_vertically(
                cropped, grid, enlargement['padding_z_m'])
            if np.any(original_masks[name] & ~mask):
                raise ValueError('enlargement removed original reservoir cells')
            metadata['enlargement'] = expansion_qc
            metadata['well_placement_reference'] = 'original_selected_body'
        selected[name] = mask
        selection_qc[name] = metadata
        native_masks.update({name+'_'+key: value for key, value in cropped.items()})
    del native

    so, sw, masks = build_saturations(
        grid, cfg['reservoirs'], cfg['experiment']['times_years'], selected)
    models = build_velocity_truth(
        original, so, masks, cfg['rock_physics'], cfg['dry_bulk_modulus_Pa'])
    velocity = models.v_true[0].copy()
    background = gaussian_filter(velocity, cfg['model']['background_smoothing_sigma_cells'], mode='reflect')
    # Same legacy background rule as the original generator, but applied once.
    background = np.repeat(gaussian_filter1d(background.mean(axis=1), sigma=50)[:, None],
                           background.shape[1], axis=1)
    water_end = int(500/grid.spacing_xz_m[0])+1
    background[:water_end] = velocity[:water_end]
    union = masks['R1'] | masks['R2']
    if not np.array_equal(models.v_true[:, ~union],
                          np.broadcast_to(original[~union], (3, int((~union).sum())))):
        raise ValueError('velocity changed outside the reservoirs')

    run_id = (cfg['experiment']['id']+'_'+cfg['experiment']['scenario']+'_'
              + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output = Path(output) if output is not None else Path(cfg['output']['directory'])/run_id
    output.mkdir(parents=True, exist_ok=False)
    truth = output/'truth'
    figures = output/'figures'
    models_dir = truth/'models'
    models_dir.mkdir(parents=True)
    figures.mkdir()

    artifact = truth/'models.npz'
    np.savez_compressed(
        artifact, So_true=so, Sw_true=sw, v_baseline=velocity, v_original=original,
        static_baseline_change=velocity-original, v_true=models.v_true, v_bg=background,
        K_dry=models.k_dry, dv_true=models.v_true-background,
        delta_true=np.stack([models.v_true[1]-models.v_true[0],
                             models.v_true[2]-models.v_true[0],
                             models.v_true[2]-models.v_true[1]]),
        mask_R1=masks['R1'], mask_R2=masks['R2'],
        **{'mask_'+name+'_original': mask for name, mask in original_masks.items()},
        spacing_xz_m=grid.spacing_xz_m, origin_xz_m=grid.origin_xz_m)

    pth_paths = {}
    for name, velocity_zx in [('baseline', models.v_true[0]), ('monitor1', models.v_true[1]),
                              ('monitor2', models.v_true[2]), ('background', background)]:
        path = models_dir/f'{name}.pth'
        save_legacy_pth(path, velocity_zx, grid)
        pth_paths[name] = path

    native_path = truth/'native_reservoir_masks.npz'
    np.savez_compressed(native_path, **native_masks)

    plot_scenario(figures, grid, velocity, so, cfg['reservoirs'], masks,
                  cfg['experiment']['times_years'])
    plot_velocity_truth(figures, grid, models.v_true)
    from .plotting import plot_baseline_replacement
    plot_baseline_replacement(figures, grid, original, velocity, masks, cfg['reservoirs'])

    summary = {}
    for name, mask in masks.items():
        summary[name] = {
            'pixels': int(mask.sum()),
            'vp_range_m_s': [float(velocity[mask].min()), float(velocity[mask].max())],
            'So_means': [float(values[mask].mean()) for values in so],
            'baseline_max_error_m_s': models.calibration_error_m_s[name],
            'baseline_reference': 'synthetic_oil_water',
            'original_vp_range_m_s': [float(original[mask].min()), float(original[mask].max())],
            'selection': selection_qc[name],
            'K_dry_range_Pa': [float(models.k_dry[mask].min()), float(models.k_dry[mask].max())],
        }
        for a, b in [(1, 0), (2, 0), (2, 1)]:
            delta = models.v_true[a, mask]-models.v_true[b, mask]
            summary[name][f'delta_{a}{b}_range_m_s'] = [float(delta.min()), float(delta.max())]
            summary[name][f'delta_{a}{b}_range_percent'] = [
                float((100*delta/models.v_true[b, mask]).min()),
                float((100*delta/models.v_true[b, mask]).max())]
        before = original_masks[name]
        cell_area = float(np.prod(grid.spacing_xz_m))
        thickness = lambda field: (field.sum(axis=0)[field.any(axis=0)]*grid.spacing_xz_m[1]).tolist()
        summary[name]['geometry_comparison'] = dict(
            original_pixels=int(before.sum()), expanded_pixels=int(mask.sum()),
            original_area_m2=float(before.sum()*cell_area),
            expanded_area_m2=float(mask.sum()*cell_area),
            area_factor=float(mask.sum()/before.sum()),
            original_vertical_thickness_per_column_m=thickness(before),
            expanded_vertical_thickness_per_column_m=thickness(mask))

    (output/'scenario_qc.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    resolved = output/'config_resolved.yaml'
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    write_manifest(output, cfg,
                   [config_path, cfg['model']['marmousi_path'], cfg['model']['original_vp_path']],
                   [artifact, native_path, *pth_paths.values(), output/'scenario_qc.json',
                    resolved, *figures.glob('*.png')])
    return {'output': str(output),
            'models': {name: str(path) for name, path in pth_paths.items()},
            'summary': summary}
