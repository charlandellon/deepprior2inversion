"""Extract the user-identified low-Vp bodies from original Marmousi geometry.

ROIs identify the bodies; a threshold and connected component define their
actual outlines. They are not evidence for a particular original fluid type.
"""

import numpy as np
from scipy.ndimage import label, maximum_filter1d


def extract_body(native_velocity, native_grid, target_grid, roi_xz_m, threshold_m_s, seed_xz_m=None):
    if native_velocity.shape != native_grid.shape_zx or not np.isfinite(native_velocity).all():
        raise ValueError('native velocity must be finite and match the native grid')
    bounds = np.asarray(roi_xz_m, dtype=float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError('ROI must be finite [xmin,xmax,zmin,zmax]')
    if not np.isfinite(threshold_m_s) or threshold_m_s <= 0:
        raise ValueError('velocity threshold must be positive')
    nx, nz = native_grid.axes
    if bounds[0] < nx[0] or bounds[1] > nx[-1] or bounds[2] < nz[0] or bounds[3] > nz[-1]:
        raise ValueError('ROI exceeds native grid')
    ix = np.flatnonzero((nx >= bounds[0]) & (nx <= bounds[1]))
    iz = np.flatnonzero((nz >= bounds[2]) & (nz <= bounds[3]))
    if not ix.size or not iz.size:
        raise ValueError('empty native ROI')
    values = native_velocity[np.ix_(iz, ix)]
    labels, count = label(values < threshold_m_s)  # 4-connected in the native grid
    if count == 0:
        raise ValueError('no low-velocity body found in ROI')
    sizes = np.bincount(labels.ravel())
    if seed_xz_m is None:
        component = int(np.argmax(sizes[1:])+1)
    else:
        sz, sx = native_grid.point_index(seed_xz_m)
        if sx not in ix or sz not in iz:
            raise ValueError('reservoir seed is outside its search ROI')
        component = int(labels[sz-iz[0], sx-ix[0]])
        if component == 0:
            raise ValueError('reservoir seed is not inside a low-velocity component')
    body = labels == component
    if np.any(body[0]) or np.any(body[-1]) or np.any(body[:, 0]) or np.any(body[:, -1]):
        rz, rx = np.where(body)
        raise ValueError(f'selected body touches ROI boundary; enlarge ROI before extracting; '
                         f'ROI={bounds.tolist()}, body bounds x=[{nx[ix[rx.min()]]}, {nx[ix[rx.max()]]}], '
                         f'z=[{nz[iz[rz.min()]]}, {nz[iz[rz.max()]]}], '
                         f'Vp=[{values[body].min()}, {values[body].max()}]')
    tx, tz = target_grid.axes
    xi = np.rint((tx-nx[ix[0]])/native_grid.spacing_xz_m[0]).astype(int)
    zi = np.rint((tz-nz[iz[0]])/native_grid.spacing_xz_m[1]).astype(int)
    validx = (xi >= 0) & (xi < len(ix))
    validz = (zi >= 0) & (zi < len(iz))
    mask = np.zeros(target_grid.shape_zx, dtype=bool)
    mask[np.ix_(validz, validx)] = body[np.ix_(zi[validz], xi[validx])]
    if not mask.any():
        raise ValueError('body vanished on target grid; refine grid instead of dilating geometry')
    rz, rx = np.where(body)
    metadata = dict(selection='seeded_4_connected_low_vp_body' if seed_xz_m is not None else 'largest_4_connected_low_vp_body_in_user_roi',
                    seed_xz_m=list(seed_xz_m) if seed_xz_m is not None else None,
                    threshold_m_s=float(threshold_m_s), roi_xz_m=bounds.tolist(),
                    native_pixels=int(body.sum()), target_pixels=int(mask.sum()),
                    native_bounds_xz_m=[float(nx[ix[rx.min()]]), float(nx[ix[rx.max()]]),
                                        float(nz[iz[rz.min()]]), float(nz[iz[rz.max()]] )],
                    native_area_m2=float(body.sum()*np.prod(native_grid.spacing_xz_m)),
                    target_area_m2=float(mask.sum()*np.prod(target_grid.spacing_xz_m)),
                    target_components=int(label(mask)[1]),
                    target_components_8_connected=int(label(mask, structure=np.ones((3, 3)))[1]))
    return mask, metadata, {'mask': body, 'x_m': nx[ix], 'z_m': nz[iz]}


def wells_along_body(mask, grid):
    """Deterministic 20%/80% x positions on actual selected target nodes."""
    x, z = grid.axes
    rows, columns = np.where(mask)
    if not rows.size:
        raise ValueError('cannot place wells in empty body')
    wells = []
    for fraction in (0.2, 0.8):
        column = int(columns[np.argmin(abs(x[columns]-np.quantile(x[columns], fraction)))])
        available = rows[columns == column]
        row = int(available[len(available)//2])
        wells.append([float(x[column]), float(z[row])])
    if wells[0] == wells[1]:
        raise ValueError('target body too small for distinct injector/producer nodes')
    return wells


def expand_body_vertically(native_body, target_grid, padding_z_m):
    """Synthetic vertical dilation on the native grid, without lateral growth.

    Keep the native centerline/dip and x support. Padding is symmetric above and
    below every occupied native column. Refuse clipped geometry, rather than
    silently cutting the enlarged body at the extraction ROI.
    """
    body = np.asarray(native_body['mask'], dtype=bool)
    x = np.asarray(native_body['x_m'], dtype=float)
    z = np.asarray(native_body['z_m'], dtype=float)
    if (body.shape != (z.size, x.size) or min(x.size, z.size) < 2
            or not body.any() or not np.isfinite(x).all() or not np.isfinite(z).all()):
        raise ValueError('native body must have a nonempty mask and matching finite axes')
    if not np.isfinite(padding_z_m) or padding_z_m <= 0:
        raise ValueError('vertical padding must be finite and positive')
    dx, dz = np.diff(x), np.diff(z)
    if (min(dx) <= 0 or min(dz) <= 0 or not np.allclose(dx, dx[0])
            or not np.allclose(dz, dz[0])):
        raise ValueError('native axes must be increasing and uniformly sampled')
    radius = int(np.ceil(padding_z_m/dz[0]))
    rows = np.flatnonzero(body.any(axis=1))
    if rows[0]-radius < 0 or rows[-1]+radius >= z.size:
        raise ValueError('expanded body exceeds native ROI; enlarge its vertical bounds')
    expanded = maximum_filter1d(body, size=2*radius+1, axis=0, mode='constant', cval=0)
    tx, tz = target_grid.axes
    xi = np.rint((tx-x[0])/dx[0]).astype(int)
    zi = np.rint((tz-z[0])/dz[0]).astype(int)
    validx, validz = (xi >= 0) & (xi < len(x)), (zi >= 0) & (zi < len(z))
    mask = np.zeros(target_grid.shape_zx, dtype=bool)
    mask[np.ix_(validz, validx)] = expanded[np.ix_(zi[validz], xi[validx])]
    if not mask.any():
        raise ValueError('expanded body vanished on target grid')
    metadata = dict(method='native_vertical_dilation', padding_z_requested_m=float(padding_z_m),
                    padding_z_actual_m=float(radius*dz[0]), lateral_padding_m=0.0,
                    original_native_pixels=int(body.sum()), expanded_native_pixels=int(expanded.sum()),
                    expanded_native_area_m2=float(expanded.sum()*dx[0]*dz[0]),
                    target_pixels=int(mask.sum()),
                    target_area_m2=float(mask.sum()*np.prod(target_grid.spacing_xz_m)),
                    target_components=int(label(mask)[1]),
                    target_components_8_connected=int(label(mask, structure=np.ones((3, 3)))[1]))
    return mask, metadata, {**native_body, 'mask_original': body.copy(), 'mask': expanded}
