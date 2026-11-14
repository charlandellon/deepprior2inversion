"""Reservoir-focused statistical indices for 4D posterior samples.

The script is run-independent: it discovers the samples by ``tag`` and survey
under a checkpoint root and uses the masks stored in the truth archive.  It
keeps the variability index and vertical confidence profiles of the legacy
``statistical_indice.py`` but uses explicit standard definitions for SSIM,
PSNR and MSE/RMSE.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

from .geometry import Grid2D
from .plotting import DPI, _image
from .stats import SURVEYS, SURVEY_LABELS, load_posterior


def _sample_paths(checkpoints: Path, tag: str, surveys):
    paths = {}
    for survey in surveys:
        prefix = f'{tag}_{survey}'
        matches = [path for path in checkpoints.rglob(f'{prefix}*/samples.hdf5')
                   if path.parent.name == prefix
                   or path.parent.name.startswith(prefix + '_max_itr-')]
        if not matches:
            raise FileNotFoundError(f'no samples.hdf5 for {prefix} under {checkpoints}')
        paths[survey] = max(matches, key=lambda path: path.stat().st_mtime)
    return paths


def _masked_ssim(truth, estimate, mask, data_range):
    """Standard local-window SSIM evaluated inside the reservoir mask.

    Outside the mask the estimate is replaced by the truth so that only the
    reservoir contributes to the comparison.
    """
    reference = truth.copy()
    candidate = truth.copy()
    candidate[mask] = estimate[mask]
    sigma = 1.5
    mu_x = gaussian_filter(reference, sigma, mode='nearest')
    mu_y = gaussian_filter(candidate, sigma, mode='nearest')
    sigma_x2 = gaussian_filter(reference**2, sigma, mode='nearest') - mu_x**2
    sigma_y2 = gaussian_filter(candidate**2, sigma, mode='nearest') - mu_y**2
    sigma_xy = gaussian_filter(reference*candidate, sigma, mode='nearest') - mu_x*mu_y
    c1 = (0.01*data_range)**2
    c2 = (0.03*data_range)**2
    numerator = (2*mu_x*mu_y + c1) * (2*sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / denominator
    return float(np.mean(ssim_map[mask]))


def _safe_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {'pearson': None, 'spearman': None}
    return {'pearson': float(pearsonr(x, y).statistic),
            'spearman': float(spearmanr(x, y).statistic)}


def _profile(truth, mean, std, mask, grid):
    rows, columns = np.where(mask)
    counts = np.bincount(columns, minlength=mask.shape[1])
    column = int(np.argmax(counts))
    # Legacy-style complete 1D profile: all depths at the reservoir column.
    selected = np.arange(mask.shape[0])
    z = grid.axes[1][selected]
    return {
        'column_m': float(grid.axes[0][column]),
        'z_km': (z/1000).tolist(),
        'truth_m_s': truth[selected, column].tolist(),
        'mean_m_s': mean[selected, column].tolist(),
        'low_m_s': (mean[selected, column] - 2.576*std[selected, column]).tolist(),
        'high_m_s': (mean[selected, column] + 2.576*std[selected, column]).tolist(),
    }


def _plot_samples(directory, truth, posterior, survey, masks, grid, quantiles):
    reference = masks.get('R1', next(iter(masks.values())))
    score = posterior[survey][:, reference].mean(axis=1)
    indices = [int(np.argmin(np.abs(score - np.percentile(score, q))))
               for q in quantiles]
    fig, axes = plt.subplots(len(masks), len(quantiles), figsize=(17, 9), squeeze=False)
    for row, (region, mask) in enumerate(masks.items()):
        fields = [posterior[survey][index] for index in indices]
        values = [truth['v_true'][SURVEYS.index(survey), mask]] + \
                 [field[mask] for field in fields]
        local = np.concatenate([item.ravel() for item in values])
        low, high = np.percentile(local, [1, 99])
        span = max(high - low, 1e-6)
        limits = (low - 0.05*span, high + 0.05*span)
        rows, columns = np.where(mask)
        ix = np.arange(max(int(columns.min())-4, 0), min(int(columns.max())+5, len(grid.axes[0])))
        iz = np.arange(max(int(rows.min())-4, 0), min(int(rows.max())+5, len(grid.axes[1])))
        crop = np.ix_(iz, ix)
        extent = [grid.axes[0][ix[0]]/1000, grid.axes[0][ix[-1]]/1000,
                  grid.axes[1][iz[-1]]/1000, grid.axes[1][iz[0]]/1000]
        for column_index, (sample, q) in enumerate(zip(fields, quantiles)):
            masked = np.full(sample.shape, np.nan)
            masked[mask] = sample[mask]
            axis = axes[row, column_index]
            _image(axis, masked[crop], extent, 'jet', limits,
                   'Vp (km/s)', f'{region} — P{q}', label_size=16, tick_size=14,
                   nbins=4, aspect='auto', colorbar_size='5%')
            axis.tick_params(axis='both', labelsize=14)
            axis.xaxis.label.set_size(16)
            axis.yaxis.label.set_size(16)
            axis.title.set_size(16)
            axis.locator_params(axis='both', nbins=4)
    fig.tight_layout()
    fig.savefig(directory/f'samples_{survey}.png', dpi=DPI)
    plt.close(fig)
    return indices


def _plot_profiles(directory, profiles, masks, grid, titles):
    """Track-style profiles: depth vertical, Vp in narrow columns."""
    n_surveys = len(titles)
    n_regions = len(masks)
    fig, axes = plt.subplots(
        n_surveys, n_regions,
        figsize=(2.9 * n_regions, 7.0 * n_surveys),
        squeeze=False, sharey='row',
    )
    region_limits = {}
    for region in masks:
        values = []
        for survey in titles:
            item = profiles[survey][region]
            values.extend(np.asarray(item['truth_m_s']) / 1000.0)
            values.extend(np.asarray(item['mean_m_s']) / 1000.0)
            values.extend(np.asarray(item['low_m_s']) / 1000.0)
            values.extend(np.asarray(item['high_m_s']) / 1000.0)
        region_limits[region] = (min(values) - 0.01, max(values) + 0.01)

    for row, survey in enumerate(titles):
        for column, region in enumerate(masks):
            item = profiles[survey][region]
            axis = axes[row, column]
            truth = np.asarray(item['truth_m_s']) / 1000.0
            mean = np.asarray(item['mean_m_s']) / 1000.0
            low_ci = np.asarray(item['low_m_s']) / 1000.0
            high_ci = np.asarray(item['high_m_s']) / 1000.0
            axis.fill_betweenx(item['z_km'], low_ci, high_ci,
                               color='tab:red', alpha=0.25, label='99% CI')
            axis.plot(mean, item['z_km'], color='tab:green', linewidth=1.2,
                      label='Posterior mean')
            axis.plot(truth, item['z_km'], color='tab:blue', linewidth=1.0,
                      label='Truth')
            axis.set_xlim(*region_limits[region])
            axis.set_ylim(max(item['z_km']), min(item['z_km']))
            axis.set_box_aspect(2.6)
            axis.set_title(
                f'{region} — {SURVEY_LABELS.get(survey, survey)}',
                fontsize=11, pad=4,
            )
            axis.set_xlabel('Vp (km/s)', fontsize=10)
            axis.set_ylabel('Depth (km)' if column == 0 else '', fontsize=10)
            axis.tick_params(labelsize=8.5)
            axis.grid(alpha=0.3)
            axis.locator_params(axis='both', nbins=5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(directory/'profiles.png', dpi=DPI)
    plt.close(fig)


def _plot_full_vi(directory, fields, grid, surveys, limit):
    extent = [grid.axes[0][0]/1000, grid.axes[0][-1]/1000,
              grid.axes[1][-1]/1000, grid.axes[1][0]/1000]
    fig, axes = plt.subplots(len(surveys), 1, figsize=(3.3, 2.55))
    axes = np.atleast_1d(axes)
    for axis, survey in zip(axes, surveys):
        image = axis.imshow(fields[survey], cmap='viridis', vmin=0, vmax=limit,
                            extent=extent, aspect='auto')
        axis.set_title(SURVEY_LABELS[survey], fontsize=7, pad=2)
        axis.set_xlabel('Length (km)', fontsize=6.5, labelpad=1)
        axis.set_ylabel('Depth (km)', fontsize=6.5, labelpad=1)
        axis.tick_params(labelsize=5.5, length=2, pad=1)
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size='4%', pad=0.04)
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_label('IV (%)', fontsize=6, labelpad=1)
        colorbar.ax.tick_params(labelsize=5.5, length=2, pad=1)
    fig.subplots_adjust(left=0.14, right=0.82, top=0.94, bottom=0.11, hspace=0.8)
    fig.savefig(directory/'vi_full_map.png', dpi=400)
    plt.close(fig)

    for survey in surveys:
        fig, axis = plt.subplots(figsize=(3.4, 1.15))
        image = axis.imshow(fields[survey], cmap='viridis', vmin=0, vmax=limit,
                            extent=extent, aspect='auto')
        axis.set_title(SURVEY_LABELS[survey], fontsize=7, pad=2)
        axis.set_xlabel('Length (km)', fontsize=6.5, labelpad=1)
        axis.set_ylabel('Depth (km)', fontsize=6.5, labelpad=1)
        axis.tick_params(labelsize=5.5, length=2, pad=1)
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size='4%', pad=0.04)
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_label('IV (%)', fontsize=6, labelpad=1)
        colorbar.ax.tick_params(labelsize=5.5, length=2, pad=1)
        fig.subplots_adjust(left=0.16, right=0.82, top=0.86, bottom=0.22)
        fig.savefig(directory/f'vi_full_{survey}.png', dpi=400)
        plt.close(fig)


def _plot_correlation(directory, points):
    fig, axis = plt.subplots(figsize=(6, 4))
    colors = {'R1': 'tab:blue', 'R2': 'tab:red'}
    for region, (x, y) in points.items():
        axis.scatter(x, y, s=10, alpha=0.55, color=colors.get(region, None),
                     label=region, edgecolors='none')
    axis.set_xlabel('Variability index IV = 100 σ/μ (%)', fontsize=16)
    axis.set_ylabel('Absolute error |mean − truth| (m/s)', fontsize=16)
    axis.tick_params(labelsize=14)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(directory/'vi_vs_error.png', dpi=DPI)
    plt.close(fig)


def compute_indices(truth_dir, checkpoints, tag, surveys, burn_in, stride,
                    clip_percentile, sample_quantiles, output=None):
    truth_dir = Path(truth_dir)
    checkpoints = Path(checkpoints)
    with np.load(truth_dir/'truth'/'models.npz') as data:
        truth = {key: data[key] for key in
                 ('v_true', 'mask_R1', 'mask_R2', 'spacing_xz_m', 'origin_xz_m')}
    grid = Grid2D(truth['v_true'].shape[-2:], tuple(truth['spacing_xz_m']),
                  tuple(truth['origin_xz_m']))
    masks = {'R1': truth['mask_R1'], 'R2': truth['mask_R2']}

    paths = _sample_paths(checkpoints, tag, surveys)
    posterior = {survey: load_posterior(path, burn_in, stride)
                 for survey, path in paths.items()}
    common = min(item.shape[0] for item in posterior.values())
    for survey in posterior:
        posterior[survey] = posterior[survey][:common]

    output = Path(output) if output else truth_dir/f'statistics_{tag}_indices'
    output.mkdir(parents=True, exist_ok=True)
    figures = output/'figures'
    figures.mkdir(exist_ok=True)

    report = {'tag': tag, 'burn_in': int(burn_in), 'stride': int(stride),
              'samples_used': int(common), 'surveys': {},
              'sample_quantiles': list(sample_quantiles), 'samples': {}}
    rows = []
    vi_fields = []
    full_vi = {}
    profiles = {}
    points = {region: ([], []) for region in masks}
    sample_indices = {}

    for survey in surveys:
        values = posterior[survey]
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        full_vi[survey] = 100.0*std/mean
        p05, p50, p95 = np.percentile(values, [5, 50, 95], axis=0)
        survey_index = SURVEYS.index(survey)
        report['surveys'][survey] = {}
        sample_indices[survey] = _plot_samples(
            figures, truth, posterior, survey, masks, grid, sample_quantiles)
        report['samples'][survey] = {
            'indices': sample_indices[survey],
            'quantiles': list(sample_quantiles),
        }
        vi_survey = {}
        for region, mask in masks.items():
            truth_values = truth['v_true'][survey_index][mask]
            mean_values = mean[mask]
            error = mean_values - truth_values
            mse = float(np.mean(error**2))
            rmse = float(np.sqrt(mse))
            bias = float(np.mean(error))
            coverage = float(np.mean((truth_values >= p05[mask]) &
                                     (truth_values <= p95[mask])))
            std_values = std[mask]
            vi = 100.0*std_values/mean_values
            rows_, columns_ = np.where(mask)
            ix = np.arange(max(int(columns_.min())-4, 0),
                           min(int(columns_.max())+5, values.shape[2]))
            iz = np.arange(max(int(rows_.min())-4, 0),
                           min(int(rows_.max())+5, values.shape[1]))
            crop = np.ix_(iz, ix)
            mask_crop = mask[crop]
            truth_crop = truth['v_true'][survey_index][crop]
            mean_crop = mean[crop]
            data_range_crop = float(truth_crop.max() - truth_crop.min())
            psnr = (float(10*np.log10(data_range_crop**2/mse))
                    if mse > 0 and data_range_crop > 0 else None)
            ssim = _masked_ssim(truth_crop, mean_crop, mask_crop, data_range_crop)
            summary = {
                'pixels': int(mask.sum()),
                'mean_m_s': float(mean_values.mean()),
                'std_m_s': float(std_values.mean()),
                'vi_percent_mean': float(vi.mean()),
                'vi_percent_median': float(np.median(vi)),
                'rmse_m_s': rmse,
                'mae_m_s': float(np.mean(np.abs(error))),
                'bias_m_s': bias,
                'coverage_5_95': coverage,
                'ssim': ssim,
                'psnr_db': psnr,
                'correlation_vi_abs_error': _safe_correlation(vi, np.abs(error)),
                'profile': _profile(truth['v_true'][survey_index], mean, std,
                                    mask, grid),
            }
            report['surveys'][survey][region] = summary
            rows.append([survey, region, summary['vi_percent_mean'],
                         summary['vi_percent_median'], summary['ssim'],
                         summary['psnr_db'], summary['rmse_m_s'],
                         summary['bias_m_s'], summary['coverage_5_95']])
            vi_survey[region] = vi
            profiles.setdefault(survey, {})[region] = summary['profile']
            points[region][0].extend(vi.tolist())
            points[region][1].extend(np.abs(error).tolist())
        vi_fields.append(vi_survey)

    # Clipped VI maps, shared limits across reservoirs/surveys.
    all_vi = np.concatenate([
        np.concatenate([np.asarray(fields[region]).ravel()
                        for fields in vi_fields for region in masks])
    ])
    vi_limit = max(float(np.percentile(all_vi, clip_percentile)), 1e-12)
    clipped = []
    for fields in vi_fields:
        clipped.append({region: np.clip(np.asarray(fields[region]), 0, vi_limit)
                        for region in masks})
    # Build full VI maps as 2 x nsurveys panels.
    fig, axes = plt.subplots(len(masks), len(surveys), figsize=(17, 9), squeeze=False)
    for row, (region, mask) in enumerate(masks.items()):
        rows_, columns_ = np.where(mask)
        ix = np.arange(max(int(columns_.min())-4, 0),
                       min(int(columns_.max())+5, grid.shape_zx[1]))
        iz = np.arange(max(int(rows_.min())-4, 0),
                       min(int(rows_.max())+5, grid.shape_zx[0]))
        crop = np.ix_(iz, ix)
        extent = [grid.axes[0][ix[0]]/1000, grid.axes[0][ix[-1]]/1000,
                  grid.axes[1][iz[-1]]/1000, grid.axes[1][iz[0]]/1000]
        for column_index, survey in enumerate(surveys):
            values = clipped[column_index][region]
            field = np.full(mask.shape, np.nan)
            field[mask] = values
            axis = axes[row, column_index]
            _image(axis, field[crop], extent, 'viridis',
                   (0, vi_limit), 'IV (%)', f'{region} — {SURVEY_LABELS[survey]}',
                   label_size=16, tick_size=14, nbins=4, aspect='auto',
                   colorbar_size='5%')
            axis.tick_params(axis='both', labelsize=14)
            axis.xaxis.label.set_size(16)
            axis.yaxis.label.set_size(16)
            axis.title.set_size(16)
            axis.locator_params(axis='both', nbins=4)
    fig.tight_layout()
    fig.savefig(figures/'vi_map.png', dpi=DPI)
    plt.close(fig)

    all_full = np.concatenate([full_vi[survey].ravel() for survey in surveys])
    full_limit = max(float(np.percentile(all_full, clip_percentile)), 1e-12)
    clipped_full = {survey: np.clip(full_vi[survey], 0, full_limit)
                    for survey in surveys}
    _plot_full_vi(figures, clipped_full, grid, surveys, full_limit)
    report['display'] = {'vi_limit_percent': vi_limit,
                         'vi_full_limit_percent': full_limit}
    report['full_field_vi'] = {
        survey: {'mean_percent': float(full_vi[survey].mean()),
                 'median_percent': float(np.median(full_vi[survey]))}
        for survey in surveys
    }

    _plot_profiles(figures, profiles, masks, grid, surveys)
    _plot_correlation(figures, points)

    (output/'statistical_indices.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (output/'statistical_indices.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['survey', 'region', 'vi_mean_percent', 'vi_median_percent',
                         'ssim', 'psnr_db', 'rmse_m_s', 'bias_m_s', 'coverage_5_95'])
        writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--checkpoints', default='data/checkpoints')
    parser.add_argument('--tag', default='expanded4d_s1')
    parser.add_argument('--surveys', nargs='+', default=list(SURVEYS))
    parser.add_argument('--burn-in', type=int, default=17500)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--clip-percentile', type=float, default=99.0)
    parser.add_argument('--sample-quantiles', type=float, nargs='+',
                        default=[10, 50, 90])
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    compute_indices(args.truth, args.checkpoints, args.tag, args.surveys,
                    args.burn_in, args.stride, args.clip_percentile,
                    args.sample_quantiles, args.output)


if __name__ == '__main__':
    main()
