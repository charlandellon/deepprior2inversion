"""Physical grids and configurable reservoir polygons, independent of Devito."""

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path


@dataclass(frozen=True)
class Grid2D:
    shape_zx: tuple[int, int]
    spacing_xz_m: tuple[float, float]
    origin_xz_m: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if len(self.shape_zx) != 2 or any(
            isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2
            for n in self.shape_zx
        ):
            raise ValueError('shape_zx must contain two integers >= 2')
        for name, values in [('spacing_xz_m', self.spacing_xz_m),
                             ('origin_xz_m', self.origin_xz_m)]:
            if len(values) != 2 or not np.all(np.isfinite(values)):
                raise ValueError(f'{name} must contain two finite SI coordinates')
        if min(self.spacing_xz_m) <= 0:
            raise ValueError('spacing_xz_m must be positive')

    @property
    def axes(self):
        nz, nx = self.shape_zx
        dx, dz = self.spacing_xz_m
        ox, oz = self.origin_xz_m
        return ox + np.arange(nx) * dx, oz + np.arange(nz) * dz

    def point_index(self, point_xz_m):
        point = np.asarray(point_xz_m, dtype=float)
        x, z = self.axes
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError('well must contain two finite coordinates (x, z)')
        if not (x[0] <= point[0] <= x[-1] and z[0] <= point[1] <= z[-1]):
            raise ValueError('well lies outside the grid')
        return int(np.argmin(abs(z - point[1]))), int(np.argmin(abs(x - point[0])))


def polygon_mask(grid, polygon_xz_m):
    """Rasterize a simple polygon at grid nodes, including its edges.

    Reject crossed/degenerate polygons rather than relying on winding ambiguity.
    """
    vertices = np.asarray(polygon_xz_m, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1:] != (2,) or len(vertices) < 3:
        raise ValueError('polygon_xz_m needs at least three (x, z) vertices')
    if not np.all(np.isfinite(vertices)):
        raise ValueError('polygon coordinates must be finite')
    if np.array_equal(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if len(vertices) < 3 or len(np.unique(vertices, axis=0)) != len(vertices):
        raise ValueError('polygon needs distinct vertices')
    x, z = grid.axes
    if np.any(vertices < (x[0], z[0])) or np.any(vertices > (x[-1], z[-1])):
        raise ValueError('polygon lies outside the grid')
    edges = np.roll(vertices, -1, axis=0) - vertices
    area = np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1)
                  - vertices[:, 1] * np.roll(vertices[:, 0], -1))
    if abs(area) <= np.finfo(float).eps * max(1.0, np.max(abs(vertices))) ** 2:
        raise ValueError('polygon has zero area')
    # Non-adjacent edges must not intersect (including touching).
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def intersects(a, b, c, d):
        if np.any(np.maximum(a, b) < np.minimum(c, d)) or np.any(
            np.maximum(c, d) < np.minimum(a, b)
        ):
            return False
        return (cross(b-a, c-a) * cross(b-a, d-a) <= 0
                and cross(d-c, a-c) * cross(d-c, b-c) <= 0)

    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if intersects(vertices[i], vertices[(i+1) % n],
                          vertices[j], vertices[(j+1) % n]):
                raise ValueError('polygon must be simple (no crossed edges)')
    xx, zz = np.meshgrid(x, z)
    points = np.column_stack((xx.ravel(), zz.ravel()))
    mask = Path(np.vstack((vertices, vertices[0]))).contains_points(points)
    tolerance = min(grid.spacing_xz_m) * 1e-9
    for a, edge in zip(vertices, edges):
        fraction = np.clip((points-a) @ edge / (edge @ edge), 0, 1)
        mask |= np.linalg.norm(points - a - fraction[:, None]*edge, axis=1) <= tolerance
    mask = mask.reshape(grid.shape_zx)
    if not mask.any():
        raise ValueError('polygon contains no grid nodes')
    return mask


def reservoir_masks(grid, reservoirs, predefined_masks=None):
    if set(reservoirs) != {'R1', 'R2'}:
        raise ValueError('exactly R1 and R2 must be configured')
    masks = {}
    occupied = np.zeros(grid.shape_zx, dtype=bool)
    for name, cfg in reservoirs.items():
        if predefined_masks is None:
            mask = polygon_mask(grid, cfg['polygon_xz_m'])
        else:
            mask = np.asarray(predefined_masks[name])
            if mask.shape != grid.shape_zx or mask.dtype != bool or not mask.any():
                raise ValueError('extracted masks must be nonempty boolean arrays matching grid')
        if np.any(occupied & mask):
            raise ValueError('reservoir masks must not overlap')
        for key in ('injection_well_xz_m', 'production_well_xz_m'):
            point = cfg[key]
            index = grid.point_index(point)
            if predefined_masks is not None:
                if not mask[index]:
                    raise ValueError(f'{name}.{key} is outside its extracted reservoir')
                continue
            # Require the physical well itself inside/on the polygon, not only its node.
            polygon = np.asarray(cfg['polygon_xz_m'], dtype=float)
            path = Path(np.vstack((polygon, polygon[0])))
            on_edge = False
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                edge = b-a
                if edge @ edge == 0:
                    continue
                t = np.clip((np.asarray(point)-a) @ edge / (edge @ edge), 0, 1)
                on_edge |= np.linalg.norm(np.asarray(point)-a-t*edge) <= 1e-8
            if not mask[index] or not (path.contains_point(point) or on_edge):
                raise ValueError(f'{name}.{key} is outside its reservoir')
        masks[name] = mask
        occupied |= mask
    return masks
