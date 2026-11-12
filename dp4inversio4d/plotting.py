"""Project plotting conventions (utils.plot_velocity/plot_reflectivity).

Physics/storage remain SI and (z,x). Only display converts to km and km/s.
Each model panel preserves physical aspect, jet/gray palettes, right-aligned
colour bars and 300 dpi. Temporal comparisons always share colour limits.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


FIGSIZE = (15, 6)
DPI = 300


def _extent_km(grid):
    x, z = grid.axes
    return [x[0]/1000, x[-1]/1000, z[-1]/1000, z[0]/1000]


def _image(ax, field, extent, cmap, limits, label, title='', seismic=False,
           label_size=10, tick_size=10, nbins=5, aspect=None, colorbar_size='3%'):
    image = ax.imshow(field, animated=True, cmap=cmap, vmin=limits[0], vmax=limits[1],
                      extent=extent,
                      aspect='auto' if seismic else ('equal' if aspect is None else aspect))
    ax.set_xlabel('Length (km)')
    ax.set_ylabel('Time (s)' if seismic else 'Depth (km)')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if title:
        ax.set_title(title, fontsize='medium')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=colorbar_size, pad=0.05)
    colorbar = ax.figure.colorbar(image, cax=cax)
    colorbar.set_label(label, fontsize=label_size)
    colorbar.ax.tick_params(labelsize=tick_size)
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    return image


def _save(fig, path):
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _panels(directory, filename, fields, titles, extent, cmap, limits, label, seismic=False):
    # Full-size panels match the legacy 15x6 figures; the combined version stacks
    # them rather than squeezing three physical sections horizontally.
    for k, (field, title) in enumerate(zip(fields, titles)):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _image(ax, field, extent, cmap, limits, label, title, seismic)
        _save(fig, directory/f'{filename.removesuffix(".png")}_{k}.png')
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    for ax, field, title in zip(axes, fields, titles):
        _image(ax, field, extent, cmap, limits, label, title, seismic)
    fig.subplots_adjust(hspace=0.25)
    _save(fig, directory/filename)


def plot_scenario(directory, grid, velocity, so, reservoirs, masks, times):
    extent = _extent_km(grid)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _image(ax, velocity/1000, extent, 'jet', (float(velocity.min()/1000), float(velocity.max()/1000)),
           'Velocity (km/s)')
    for (name, cfg), colour in zip(reservoirs.items(), ('white', 'magenta')):
        if 'polygon_xz_m' in cfg:
            polygon = np.asarray(cfg['polygon_xz_m'])/1000
            polygon = np.vstack((polygon, polygon[0]))
            ax.plot(*polygon.T, color=colour, label=name)
        else:
            x, z = grid.axes
            ax.contour(x/1000, z/1000, masks[name], levels=[0.5], colors=[colour], linewidths=1)
            ax.plot([], [], color=colour, label=name)
        for key, marker in [('injection_well_xz_m', 'v'), ('production_well_xz_m', '^')]:
            ax.scatter(*(np.asarray(cfg[key])/1000), marker=marker,
                       color=colour, edgecolor='black')
    ax.legend()
    _save(fig, directory/'marmousi_masks_wells.png')
    _panels(directory, 'saturation_truth.png', so,
            [f't{k}: {times[k]} synthetic years' for k in range(3)],
            extent, 'jet', (0, 1), 'Oil saturation (fraction)')


def plot_velocity_truth(directory, grid, velocity):
    extent = _extent_km(grid)
    pairs = [(1, 0), (2, 0), (2, 1)]
    deltas = [(velocity[a]-velocity[b])/1000 for a, b in pairs]
    limit = max(float(np.max(np.abs(deltas))), 1e-12)
    _panels(directory, 'velocity_truth.png', velocity/1000, ['t0', 't1', 't2'],
            extent, 'jet', (float(velocity.min()/1000), float(velocity.max()/1000)),
            'Velocity (km/s)')
    _panels(directory, 'delta_velocity_truth.png', deltas,
            [f'Delta {a}{b}' for a, b in pairs], extent, 'gray', (-limit, limit),
            'Velocity difference (km/s)')
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for values, (a, b) in zip(deltas, pairs):
        changed = values[values != 0]
        if changed.size:
            ax.hist(changed, bins=30, histtype='step', label=f'{a}{b}')
    ax.set(xlabel='Velocity difference (km/s)', ylabel='Pixel count')
    if any(np.any(values != 0) for values in deltas):
        ax.legend()
    _save(fig, directory/'delta_velocity_histogram.png')


def plot_baseline_replacement(directory, grid, original, baseline, masks, reservoirs):
    """Separate the static model adaptation from subsequent time-lapse changes."""
    extent = _extent_km(grid)
    limits = (float(min(original.min(), baseline.min())/1000),
              float(max(original.max(), baseline.max())/1000))
    for name, field in [('original', original), ('modified', baseline)]:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _image(ax, field/1000, extent, 'jet', limits, 'Velocity (km/s)',
               f'Marmousi {name} baseline')
        _save(fig, directory/f'baseline_{name}.png')
    change = (baseline-original)/1000
    limit = max(float(np.max(abs(change))), 1e-12)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _image(ax, change, extent, 'RdBu_r', (-limit, limit), 'ΔVp (km/s)')
    _save(fig, directory/'static_baseline_change.png')
    x, z = grid.axes
    for name, cfg in reservoirs.items():
        a, b, c, d = cfg['search_roi_xz_m']
        ix = np.flatnonzero((x >= a) & (x <= b))
        iz = np.flatnonzero((z >= c) & (z <= d))
        crop = np.ix_(iz, ix)
        crop_extent = [x[ix[0]]/1000, x[ix[-1]]/1000, z[iz[-1]]/1000, z[iz[0]]/1000]
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
        for ax, field, title in zip(axes, (original, baseline), ('Original', 'Oil-water baseline')):
            _image(ax, field[crop]/1000, crop_extent, 'jet', limits, 'Velocity (km/s)', f'{name}: {title}')
            ax.contour(x[ix]/1000, z[iz]/1000, masks[name][crop], levels=[.5], colors=['white'])
        _save(fig, directory/f'reservoir_{name}_replacement_zoom.png')


def plot_reservoir_expansion(directory, grid, before, after, old_masks, masks, velocity, so):
    """Preview synthetic enlargement with identical scales and physical aspect."""
    from matplotlib.lines import Line2D

    x, z = grid.axes
    extent = _extent_km(grid)
    limits = (float(min(before.min(), after.min())/1000),
              float(max(before.max(), after.max())/1000))
    legend = [Line2D([0], [0], color='white', linestyle='--', label='Original outline'),
              Line2D([0], [0], color='black', label='Expanded outline')]

    def outlines(ax, old, new, xx, zz):
        ax.contour(xx/1000, zz/1000, new, levels=[.5], colors=['black'], linewidths=1.2)
        ax.contour(xx/1000, zz/1000, old, levels=[.5], colors=['white'],
                   linestyles='--', linewidths=1.2)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    for ax, field, title in zip(axes, (before, after),
                                ('Previous oil-water baseline', 'Expanded oil-water baseline — preview')):
        _image(ax, field/1000, extent, 'jet', limits, 'Velocity (km/s)', title)
        for name in masks:
            outlines(ax, old_masks[name], masks[name], x, z)
            rows, columns = np.where(masks[name])
            ax.text(float(x[columns].mean()/1000), float(z[rows].min()/1000)-.1,
                    name, color='black', ha='center', fontsize=11,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
        ax.legend(handles=legend, loc='lower right', facecolor='0.6', framealpha=1)
    fig.tight_layout()
    _save(fig, directory/'reservoir_expansion_overview.png')

    crops = {}
    cell_area = float(np.prod(grid.spacing_xz_m))
    for name, mask in masks.items():
        rows, columns = np.where(mask)
        ix = np.arange(max(int(columns.min())-4, 0), min(int(columns.max())+5, len(x)))
        iz = np.arange(max(int(rows.min())-4, 0), min(int(rows.max())+5, len(z)))
        crop = np.ix_(iz, ix)
        crop_extent = [x[ix[0]]/1000, x[ix[-1]]/1000, z[iz[-1]]/1000, z[iz[0]]/1000]
        crops[name] = (ix, iz, crop, crop_extent)
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
        for ax, field, selected, label in zip(axes, (before, after), (old_masks[name], mask),
                                              ('Previous', 'Expanded')):
            title = f'{name} — {label}: {selected.sum()} cells, {selected.sum()*cell_area/1e6:.4f} km²'
            _image(ax, field[crop]/1000, crop_extent, 'jet', limits, 'Velocity (km/s)', title)
            outlines(ax, old_masks[name][crop], mask[crop], x[ix], z[iz])
            ax.legend(handles=legend, loc='lower right', facecolor='0.6', framealpha=1)
        fig.tight_layout()
        _save(fig, directory/f'reservoir_{name}_expansion_zoom.png')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for row, name in enumerate(masks):
        ix, iz, crop, crop_extent = crops[name]
        for k, label in enumerate(('Baseline', 'M1', 'M2')):
            _image(axes[row, k], so[k][crop], crop_extent, 'jet', (0, 1),
                   'Oil saturation (fraction)', f'{name} — {label}')
            outlines(axes[row, k], old_masks[name][crop], masks[name][crop], x[ix], z[iz])
    fig.tight_layout()
    _save(fig, directory/'reservoir_expansion_saturation_zoom.png')

    deltas = (velocity[1:]-velocity[0])/1000
    limit = max(float(np.max(np.abs(deltas))), 1e-12)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
    for row, name in enumerate(masks):
        ix, iz, crop, crop_extent = crops[name]
        for k in range(2):
            _image(axes[row, k], deltas[k][crop], crop_extent, 'gray', (-limit, limit),
                   'Velocity difference (km/s)', f'{name} — Monitor {k+1} minus baseline')
            outlines(axes[row, k], old_masks[name][crop], masks[name][crop], x[ix], z[iz])
    fig.tight_layout()
    _save(fig, directory/'reservoir_expansion_delta_zoom.png')


