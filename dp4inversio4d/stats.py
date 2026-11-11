"""Post-inversion 4D statistics from the legacy samples.hdf5 files."""

import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .plotting import DPI, FIGSIZE, _extent_km, _image, _save
from .rock_physics import PetroParameters, gassmann

SURVEYS = ('baseline', 'monitor1', 'monitor2')


def load_posterior(path, burn_in=17500, stride=10):
    """Read km/s (sample,x,z) samples and return SI (sample,z,x) after burn-in."""
    with h5py.File(path, 'r') as handle:
        count = handle['samples'].shape[0]
        indices = np.arange(burn_in, count, stride)
        if indices.size == 0:
            raise ValueError(f'{path}: no samples after burn-in {burn_in} (count={count})')
        samples = handle['samples'][indices].transpose(0, 2, 1).astype(np.float32)*1000.0
    return samples


def _region_summary(selected, truth=None):
    """Summarize an already-masked (samples, pixels) array."""
    posterior_mean = selected.mean(axis=0)
    posterior_std = selected.std(axis=0)
    p05, p50, p95 = np.percentile(selected, [5, 50, 95], axis=0)
    summary = dict(
        pixels=int(selected.shape[1]), samples=int(selected.shape[0]),
        mean_of_means_m_s=float(posterior_mean.mean()),
        spatial_std_of_means_m_s=float(posterior_mean.std()),
        mean_posterior_std_m_s=float(posterior_std.mean()),
        mean_p05_m_s=float(p05.mean()), mean_median_m_s=float(p50.mean()),
        mean_p95_m_s=float(p95.mean()))
    if truth is not None:
        error = posterior_mean-truth
        summary['posterior_mean_rmse_m_s'] = float(np.sqrt(np.mean(error**2)))
        summary['posterior_mean_bias_m_s'] = float(np.mean(error))
        summary['truth_inside_5_95_fraction'] = float(np.mean(
            (truth >= p05) & (truth <= p95)))
    return summary


def _difference_summary(values, reference, rng, draws=2000, max_pixels=5000):
    """Monte Carlo summary of monitor-baseline differences for independent chains.

    Chains are not paired: sample indices have no common-random-number meaning
    after reverting to the legacy sampler. Pixels are subsampled only to bound
    memory in the control region.
    """
    flat_values = values.reshape(values.shape[0], -1)
    flat_reference = reference.reshape(reference.shape[0], -1)
    if flat_values.shape[1] > max_pixels:
        pixels = rng.choice(flat_values.shape[1], max_pixels, replace=False)
        flat_values, flat_reference = flat_values[:, pixels], flat_reference[:, pixels]
    index_values = rng.integers(0, flat_values.shape[0], size=draws)
    index_reference = rng.integers(0, flat_reference.shape[0], size=draws)
    difference = flat_values[index_values] - flat_reference[index_reference]
    probability = (difference > 0).mean(axis=0)
    posterior_std = difference.std(axis=0)
    return dict(
        mean_delta_m_s=float(difference.mean()), mean_posterior_std_m_s=float(posterior_std.mean()),
        mean_probability_increase=float(probability.mean()),
        probability_increase_p05=float(np.percentile(probability, 5)),
        probability_increase_p95=float(np.percentile(probability, 95)),
        delta_p05_m_s=float(np.percentile(difference, 5, axis=0).mean()),
        delta_p95_m_s=float(np.percentile(difference, 95, axis=0).mean()),
        pairing='monte_carlo_independent_chains', draws=int(draws),
        pixels_used=int(flat_values.shape[1]))


def _difference_probability(values, reference, rng, draws=1000, chunk=4000):
    """P(monitor > baseline) per pixel for independent chains, chunked in memory."""
    flat_values = values.reshape(values.shape[0], -1)
    flat_reference = reference.reshape(reference.shape[0], -1)
    output = np.empty(flat_values.shape[1], dtype=np.float32)
    for start in range(0, flat_values.shape[1], chunk):
        current_values = flat_values[:, start:start+chunk]
        current_reference = flat_reference[:, start:start+chunk]
        index_values = rng.integers(0, current_values.shape[0], size=draws)
        index_reference = rng.integers(0, current_reference.shape[0], size=draws)
        output[start:start+current_values.shape[1]] = (
            current_values[index_values] > current_reference[index_reference]).mean(axis=0)
    return output.reshape(values.shape[1:])


def _saturation_probability_decrease(values, reference, rng, draws=2000):
    index_values = rng.integers(0, values.shape[0], size=draws)
    index_reference = rng.integers(0, reference.shape[0], size=draws)
    return (values[index_values] < reference[index_reference]).mean(axis=0)


def _invert_saturation(vp_samples, k_dry, params, grid_size=1001):
    """Invert the monotonic Wood-Gassmann Vp(So) relation per pixel."""
    so_grid = np.linspace(0.0, 1.0, grid_size)
    vp_curve = gassmann(so_grid[None, :], np.asarray(k_dry)[:, None], params)[0]
    result = np.empty_like(vp_samples, dtype=np.float32)
    for index in range(vp_curve.shape[0]):
        result[:, index] = np.interp(vp_samples[:, index],
                                     vp_curve[index][::-1], so_grid[::-1])
    return result


def _saturation_summary(selected, truth):
    """Summarize an already-masked (samples, pixels) saturation array."""
    posterior_mean = selected.mean(axis=0)
    posterior_std = selected.std(axis=0)
    return dict(pixels=int(selected.shape[1]), mean_of_means=float(posterior_mean.mean()),
                spatial_std_of_means=float(posterior_mean.std()),
                mean_posterior_std=float(posterior_std.mean()),
                posterior_mean_rmse=float(np.sqrt(np.mean((posterior_mean-truth)**2))))


def _saturation_difference(values, reference, rng, draws=2000):
    index_values = rng.integers(0, values.shape[0], size=draws)
    index_reference = rng.integers(0, reference.shape[0], size=draws)
    difference = values[index_values]-reference[index_reference]
    probability = (difference < 0).mean(axis=0)
    return dict(mean_difference=float(difference.mean()),
                mean_probability_decrease=float(probability.mean()),
                probability_decrease_p05=float(np.percentile(probability, 5)),
                probability_decrease_p95=float(np.percentile(probability, 95)),
                difference_p05=float(np.percentile(difference, 5, axis=0).mean()),
                difference_p95=float(np.percentile(difference, 95, axis=0).mean()),
                pairing='monte_carlo_independent_chains', draws=int(draws))


def _robust_symmetric_limit(fields, percentile):
    values = np.concatenate([np.abs(field[np.isfinite(field)]).ravel() for field in fields])
    if values.size == 0:
        return 1e-12
    return max(float(np.percentile(values, percentile)), 1e-12)


def _robust_upper_limit(fields, percentile):
    values = np.concatenate([field[np.isfinite(field)].ravel() for field in fields])
    if values.size == 0:
        return 1.0
    return max(float(np.percentile(values, percentile)), 1e-12)


def _plot_fields(directory, grid, fields, titles, filename, cmap, limits, label):
    extent = _extent_km(grid)
    # Vertical stacking fills the figure and gives each colour bar enough height.
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    for axis, field, title in zip(axes, fields, titles):
        _image(axis, field, extent, cmap, limits, label, title,
               label_size=22, tick_size=18, nbins=5, colorbar_size='6%')
    fig.subplots_adjust(left=0.06, right=0.86, top=0.97, bottom=0.05, hspace=0.28)
    _save(fig, directory/filename)


def _reservoir_crop(truth, name):
    x, z = truth['grid'].axes
    mask = truth['mask_'+name]
    rows, columns = np.where(mask)
    ix = np.arange(max(int(columns.min())-4, 0), min(int(columns.max())+5, len(x)))
    iz = np.arange(max(int(rows.min())-4, 0), min(int(rows.max())+5, len(z)))
    extent = [x[ix[0]]/1000, x[ix[-1]]/1000, z[iz[-1]]/1000, z[iz[0]]/1000]
    return np.ix_(iz, ix), extent


def _plot_zoom(directory, truth, row_specs, columns, fields, filename, cmap, limits, label):
    """Zoomed reservoir panels with stretched aspect for legible colour bars.

    limits may be a single (vmin, vmax) or one tuple per row (per reservoir).
    """
    per_row = isinstance(limits, list)
    fig, axes = plt.subplots(len(row_specs), len(columns),
                             figsize=(5.8*len(columns), 4.4*len(row_specs)), squeeze=False)
    for row_index, (reservoir, row_label) in enumerate(row_specs):
        crop, extent = _reservoir_crop(truth, reservoir)
        row_limits = limits[row_index] if per_row else limits
        for column_index, title in enumerate(columns):
            _image(axes[row_index, column_index], fields[row_index][column_index][crop],
                   extent, cmap, row_limits, label, f'{row_label} — {title}',
                   label_size=16, tick_size=14, nbins=4, aspect='auto',
                   colorbar_size='6%')
    fig.tight_layout()
    _save(fig, directory/filename)


def _plot_saturation(directory, truth, fields, titles, filename, label, limits=(0, 1)):
    x, z = truth['grid'].axes
    ncols = len(titles)
    fig, axes = plt.subplots(2, ncols, figsize=(6*ncols, 9), squeeze=False)
    for row, name in enumerate(('R1', 'R2')):
        mask = truth['mask_'+name]
        rows, columns = np.where(mask)
        ix = np.arange(max(int(columns.min())-4, 0), min(int(columns.max())+5, len(x)))
        iz = np.arange(max(int(rows.min())-4, 0), min(int(rows.max())+5, len(z)))
        extent = [x[ix[0]]/1000, x[ix[-1]]/1000, z[iz[-1]]/1000, z[iz[0]]/1000]
        for column, (field, title) in enumerate(zip(fields[name], titles)):
            _image(axes[row, column], field[np.ix_(iz, ix)], extent, 'jet', limits,
                   label, f'{name} — {title}', label_size=16, tick_size=14, nbins=4,
                   aspect='auto', colorbar_size='6%')
    fig.tight_layout()
    _save(fig, directory/filename)


def compute_statistics(truth_dir, checkpoint_root, tag, output=None,
                       burn_in=17500, stride=10, clip_percentile=99):
    if not 50 <= clip_percentile < 100:
        raise ValueError('clip_percentile must be in [50, 100)')
    truth_dir, checkpoint_root = Path(truth_dir), Path(checkpoint_root)
    with np.load(truth_dir/'truth'/'models.npz') as data:
        truth = {key: data[key] for key in
                 ('v_true', 'So_true', 'mask_R1', 'mask_R2', 'spacing_xz_m', 'origin_xz_m', 'K_dry')}
    with (truth_dir/'config_resolved.yaml').open(encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    from .geometry import Grid2D
    grid = Grid2D(truth['v_true'].shape[-2:], tuple(truth['spacing_xz_m']),
                  tuple(truth['origin_xz_m']))
    truth['grid'] = grid

    paths, posterior = {}, {}
    for survey in SURVEYS:
        # Legacy names nest the run under a ch_encoding prefix; the immediate
        # parent of samples.hdf5 is <experiment>_max_itr-... .
        prefix = f'{tag}_{survey}'
        matches = [path for path in checkpoint_root.rglob(f'{prefix}*/samples.hdf5')
                   if path.parent.name == prefix
                   or path.parent.name.startswith(prefix+'_max_itr-')]
        if len(matches) != 1:
            raise FileNotFoundError(
                f'expected exactly one samples.hdf5 for {survey} under {checkpoint_root}, found {matches}')
        paths[survey] = matches[0]
        posterior[survey] = load_posterior(matches[0], burn_in, stride)
        if posterior[survey].shape != (posterior[survey].shape[0],)+truth['v_true'].shape[1:]:
            raise ValueError(f'{survey}: posterior shape {posterior[survey].shape} differs from truth')
    available = {survey: values.shape[0] for survey, values in posterior.items()}
    common = min(available.values())
    if common < 1:
        raise ValueError('no common posterior samples after burn-in/stride')
    for survey in posterior:
        posterior[survey] = posterior[survey][:common]

    masks = {'R1': truth['mask_R1'], 'R2': truth['mask_R2']}
    control = ~(truth['mask_R1'] | truth['mask_R2'])
    report = {'burn_in': int(burn_in), 'stride': int(stride),
              'samples_used': int(common), 'samples_available': available,
              'samples_file': {key: str(value) for key, value in paths.items()},
              'surveys': {}, 'regions': {}, 'notes': []}
    rng = np.random.default_rng(20261111)
    report['notes'].append('chains truncated to the shortest available one for a common comparison')
    report['notes'].append('baseline and monitors are independent legacy chains; '
                           'differences use Monte Carlo independent pairing, not sample-index pairing')
    report['notes'].append('saturation is conditional on the configured Wood-Gassmann petrophysics')

    for index, survey in enumerate(SURVEYS):
        values = posterior[survey]
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        p05, p50, p95 = np.percentile(values, [5, 50, 95], axis=0)
        report['surveys'][survey] = {
            'vp_global': dict(mean_m_s=float(mean.mean()), std_m_s=float(std.mean()),
                              p05_m_s=float(p05.mean()), p95_m_s=float(p95.mean())),
            'reservoirs': {name: _region_summary(values[..., mask], truth['v_true'][index][mask])
                           for name, mask in masks.items()},
            'control': _region_summary(values[..., control], truth['v_true'][index][control])}
        if survey != 'baseline':
            report['surveys'][survey]['difference_vs_baseline'] = {
                name: _difference_summary(values[..., mask], posterior['baseline'][..., mask], rng)
                for name, mask in {**masks, 'control': control}.items()}

    for name, mask in masks.items():
        params = PetroParameters(**cfg['rock_physics'][name])
        so_posterior = {}
        for survey in SURVEYS:
            vp = posterior[survey][..., mask]
            so_posterior[survey] = _invert_saturation(vp, truth['K_dry'][mask], params)
            report['surveys'][survey].setdefault('saturation', {})[name] = _saturation_summary(
                so_posterior[survey], truth['So_true'][SURVEYS.index(survey)][mask])
        for survey in ('monitor1', 'monitor2'):
            report['surveys'][survey]['saturation'][name]['difference_vs_baseline'] = \
                _saturation_difference(so_posterior[survey], so_posterior['baseline'], rng)

    output = Path(output) if output is not None else truth_dir/'statistics'
    output.mkdir(parents=True, exist_ok=True)
    (output/'statistics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    _write_csv(output/'region_statistics.csv', report)

    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    means = [posterior[survey].mean(axis=0) for survey in SURVEYS]
    stds = [posterior[survey].std(axis=0) for survey in SURVEYS]
    stds_km = [item/1000 for item in stds]
    reservoir_union = truth['mask_R1'] | truth['mask_R2']
    vp_limits = (float(min(item.min() for item in means)/1000),
                 float(max(item.max() for item in means)/1000))
    _plot_fields(figures, grid, [item/1000 for item in means], SURVEYS,
                 'vp_posterior_mean.png', 'jet', vp_limits, 'Vp (km/s)')
    std_limit = (0, _robust_upper_limit(
        [item[reservoir_union] for item in stds_km], clip_percentile))
    _plot_fields(figures, grid, stds_km, SURVEYS, 'vp_posterior_std.png', 'viridis',
                 std_limit, 'σ(Vp) (km/s)')
    deltas = [(posterior['monitor1']-posterior['baseline']).mean(axis=0),
              (posterior['monitor2']-posterior['baseline']).mean(axis=0),
              (posterior['monitor2']-posterior['monitor1']).mean(axis=0)]
    deltas_km = [item/1000 for item in deltas]
    delta_limit = _robust_symmetric_limit(
        [item[reservoir_union] for item in deltas_km], clip_percentile)
    _plot_fields(figures, grid, deltas_km, ['M1 - baseline', 'M2 - baseline',
                                            'M2 - M1'],
                 'vp_delta_mean.png', 'RdBu_r', (-delta_limit, delta_limit), 'ΔVp (km/s)')
    probabilities = [_difference_probability(posterior['monitor1'], posterior['baseline'], rng),
                     _difference_probability(posterior['monitor2'], posterior['baseline'], rng),
                     _difference_probability(posterior['monitor2'], posterior['monitor1'], rng)]
    _plot_fields(figures, grid, probabilities, ['P(M1 > B)', 'P(M2 > B)', 'P(M2 > M1)'],
                 'vp_delta_probability_increase.png', 'jet', (0, 1), 'P(ΔVp > 0)')
    report['display'] = {'velocity_units': 'km/s', 'clip_percentile': float(clip_percentile),
                         'clip_region': 'R1|R2', 'std_limit_km_s': float(std_limit[1]),
                         'delta_limit_km_s': float(delta_limit)}
    report['notes'].append('full-section colour limits use reservoir-union percentiles; '
                           'control-region values may saturate')

    # Zoomed panels: R1/R2 are too small to assess on the full section.
    for name in masks:
        truth_fields = [truth['v_true'][k]/1000 for k in range(3)]
        mean_fields = [means[k]/1000 for k in range(3)]
        mask = truth['mask_'+name]
        local = np.concatenate([field[mask] for field in truth_fields]
                               + [field[mask] for field in mean_fields])
        low, high = float(np.percentile(local, 1)), float(np.percentile(local, 99))
        span = max(high-low, 1e-6)
        local_limits = (low-0.05*span, high+0.05*span)
        _plot_zoom(figures, truth, [(name, 'Truth'), (name, 'Posterior mean')], SURVEYS,
                   [truth_fields, mean_fields], f'vp_truth_vs_posterior_{name}.png',
                   'jet', local_limits, 'Vp (km/s)')
    # Per-reservoir zoom limits; the reservoirs have very different amplitudes.
    std_row_limits, delta_row_limits = [], []
    for name in ('R1', 'R2'):
        mask = truth['mask_'+name]
        std_row_limits.append((0, _robust_upper_limit(
            [stds_km[k][mask] for k in range(3)], clip_percentile)))
        row_limit = _robust_symmetric_limit(
            [deltas_km[k][mask] for k in range(3)], clip_percentile)
        delta_row_limits.append((-row_limit, row_limit))
    _plot_zoom(figures, truth, [('R1', 'R1'), ('R2', 'R2')], SURVEYS,
               [[stds_km[k] for k in range(3)], [stds_km[k] for k in range(3)]],
               'vp_posterior_std_zoom.png', 'viridis', std_row_limits, 'σ(Vp) (km/s)')
    _plot_zoom(figures, truth, [('R1', 'R1'), ('R2', 'R2')],
               ['M1 - B', 'M2 - B', 'M2 - M1'],
               [deltas_km, deltas_km], 'vp_delta_mean_zoom.png', 'RdBu_r',
               delta_row_limits, 'ΔVp (km/s)')
    _plot_zoom(figures, truth, [('R1', 'R1'), ('R2', 'R2')],
               ['P(M1 > B)', 'P(M2 > B)', 'P(M2 > M1)'],
               [probabilities, probabilities], 'vp_delta_probability_zoom.png', 'jet',
               (0, 1), 'P(ΔVp > 0)')
    report['display']['zoom_std_limits_km_s'] = {
        name: float(limit[1]) for name, limit in zip(('R1', 'R2'), std_row_limits)}
    report['display']['zoom_delta_limits_km_s'] = {
        name: float(limit[1]) for name, limit in zip(('R1', 'R2'), delta_row_limits)}
    so_fields = {name: [None, None, None] for name in masks}
    for name, mask in masks.items():
        params = PetroParameters(**cfg['rock_physics'][name])
        for index, survey in enumerate(SURVEYS):
            vp = posterior[survey][..., mask]
            so = _invert_saturation(vp, truth['K_dry'][mask], params)
            field = np.full(grid.shape_zx, np.nan)
            field[mask] = so.mean(axis=0)
            so_fields[name][index] = field
    _plot_saturation(figures, truth, so_fields, SURVEYS, 'saturation_posterior_mean.png',
                     'So')
    decrease = {name: [None, None] for name in masks}
    for name, mask in masks.items():
        params = PetroParameters(**cfg['rock_physics'][name])
        baseline_so = _invert_saturation(posterior['baseline'][..., mask],
                                         truth['K_dry'][mask], params)
        for index, survey in enumerate(('monitor1', 'monitor2')):
            so = _invert_saturation(posterior[survey][..., mask], truth['K_dry'][mask], params)
            field = np.full(grid.shape_zx, np.nan)
            field[mask] = _saturation_probability_decrease(so, baseline_so, rng)
            decrease[name][index] = field
    _plot_saturation(figures, truth, decrease, ['P(So M1 < base)', 'P(So M2 < base)'],
                     'saturation_probability_decrease.png', 'P(So ↓)')
    report['outputs'] = {'statistics': str(output/'statistics.json'),
                         'csv': str(output/'region_statistics.csv'),
                         'figures': str(figures)}
    (output/'statistics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def _write_csv(path, report):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['survey', 'region', 'metric', 'value'])
        for survey, values in report['surveys'].items():
            for region, metrics in values.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            writer.writerow([survey, region, f'{metric}.{sub_metric}', sub_value])
                    else:
                        writer.writerow([survey, region, metric, value])
