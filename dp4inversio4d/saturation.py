"""Oil-water kinematic fronts; this is not a Darcy flow simulator."""

import numpy as np
from scipy.special import expit

from .geometry import reservoir_masks


def build_saturations(grid, reservoirs, times_years=(0.0, 2.0, 5.0), predefined_masks=None):
    """Return So, Sw (time,z,x), masks; NaN outside the reservoirs.

    An anisotropic radial distance is clipped by each mask. Connectivity and
    mass balance are not modeled. A control reservoir has active=False or speed=0.
    """
    times = np.asarray(times_years, dtype=float)
    if (times.shape != (3,) or not np.all(np.isfinite(times)) or times[0] != 0
            or np.any(np.diff(times) <= 0)):
        raise ValueError('times_years must contain 0 followed by two increasing times')
    masks = reservoir_masks(grid, reservoirs, predefined_masks)
    sw = np.full((3,) + grid.shape_zx, np.nan, dtype=np.float64)
    xx, zz = np.meshgrid(*grid.axes)
    for name, cfg in reservoirs.items():
        swi = float(cfg['Sw_initial'])
        sor = float(cfg['So_residual'])
        speed = float(cfg['front_speed_m_per_year'])
        width = float(cfg['front_width_m'])
        delay = float(cfg.get('front_delay_years', 0.0))
        anisotropy = np.asarray(cfg.get('distance_scale_xz', [1.0, 1.0]), dtype=float)
        if not np.all(np.isfinite([swi, sor, speed, width, delay])):
            raise ValueError(f'{name}: saturation/front parameters must be finite')
        if not (0 <= swi <= 1 and 0 <= sor <= 1 and swi+sor <= 1):
            raise ValueError(f'{name}: invalid initial/residual saturations')
        if width <= 0 or speed < 0 or delay < 0:
            raise ValueError(f'{name}: width must be positive, speed/delay nonnegative')
        if anisotropy.shape != (2,) or not np.all(np.isfinite(anisotropy)) or min(anisotropy) <= 0:
            raise ValueError(f'{name}: distance_scale_xz must be finite and positive')
        active = cfg.get('active', True)
        if not isinstance(active, bool):
            raise ValueError(f'{name}: active must be a boolean')
        ix, iz = cfg['injection_well_xz_m']
        distance = np.hypot((xx-ix)/anisotropy[0], (zz-iz)/anisotropy[1])
        for k, time in enumerate(times):
            front = np.zeros(grid.shape_zx)
            if active and speed > 0 and time > delay:
                front = expit((speed*(time-delay) - distance)/width)
            mask = masks[name]
            sw[k, mask] = swi + (1-swi-sor)*front[mask]
    return 1-sw, sw, masks
