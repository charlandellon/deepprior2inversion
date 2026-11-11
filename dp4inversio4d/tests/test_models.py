import numpy as np
import pytest

from dp4inversio4d.build_models import build_velocity_truth
from dp4inversio4d.geometry import Grid2D, reservoir_masks
from dp4inversio4d.reservoirs import expand_body_vertically
from dp4inversio4d.rock_physics import PetroParameters, gassmann
from dp4inversio4d.saturation import build_saturations

PARAMS = dict(K_o=1e9, K_w=2.2e9, rho_o=800., rho_w=1030., K_s=37e9,
              rho_s=2650., G_dry=1e9, phi=.3)


def test_expansion_is_vertical_and_preserves_old_cells():
    grid = Grid2D((61, 61), (1., 1.))
    body = np.zeros(grid.shape_zx, dtype=bool)
    for column in range(15, 41):
        body[20+column//4:23+column//4, column] = True
    expanded, metadata, _ = expand_body_vertically(
        {'mask': body, 'x_m': grid.axes[0], 'z_m': grid.axes[1]}, grid, 4.)
    assert np.all(expanded[body])
    np.testing.assert_array_equal(expanded.any(axis=0), body.any(axis=0))
    assert metadata['lateral_padding_m'] == 0.
    assert metadata['padding_z_actual_m'] == 4.


def test_saturation_and_gassmann_are_local():
    grid = Grid2D((31, 41), (25., 25.))
    reservoirs = {
        'R1': dict(Sw_initial=.2, So_residual=.2, front_speed_m_per_year=80.,
                   front_width_m=35., distance_scale_xz=[1., .5], active=True,
                   injection_well_xz_m=[250., 275.], production_well_xz_m=[425., 300.]),
        'R2': dict(Sw_initial=.2, So_residual=.2, front_speed_m_per_year=40.,
                   front_width_m=35., distance_scale_xz=[1., .7], active=True,
                   injection_well_xz_m=[650., 275.], production_well_xz_m=[800., 300.]),
    }
    masks = {'R1': np.zeros(grid.shape_zx, bool), 'R2': np.zeros(grid.shape_zx, bool)}
    masks['R1'][10:14, 8:20] = True
    masks['R2'][10:14, 24:36] = True
    so, sw, built = build_saturations(grid, reservoirs, [0., 2., 5.], masks)
    np.testing.assert_allclose(so[0][masks['R1']], .8)
    assert np.all(so[2][masks['R1']] < so[1][masks['R1']])
    assert np.isnan(so[:, 0, 0]).all()
    truth = build_velocity_truth(np.full(grid.shape_zx, 2500.), so, built,
                                 {'R1': PARAMS, 'R2': PARAMS}, {'R1': 3e9, 'R2': 3e9})
    union = built['R1'] | built['R2']
    np.testing.assert_array_equal(truth.v_true[:, ~union], 2500.)
    assert truth.v_true[2, built['R1']].mean() > truth.v_true[0, built['R1']].mean()


@pytest.mark.parametrize('bad', [-1., 0., float('nan')])
def test_expansion_rejects_invalid_padding(bad):
    grid = Grid2D((31, 31), (1., 1.))
    body = np.zeros(grid.shape_zx, dtype=bool)
    body[10:15, 10:15] = True
    with pytest.raises(ValueError, match='padding'):
        expand_body_vertically({'mask': body, 'x_m': grid.axes[0], 'z_m': grid.axes[1]},
                               grid, bad)


def test_wood_gassmann_monotonic_in_oil_saturation():
    params = PetroParameters(**PARAMS)
    so = np.linspace(0, 1, 101)
    vp = gassmann(so, np.full_like(so, 3e9), params)[0]
    assert np.all(np.diff(vp) < 0)
