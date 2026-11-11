import json

import h5py
import numpy as np
import yaml

from dp4inversio4d.rock_physics import PetroParameters, gassmann
from dp4inversio4d.stats import SURVEYS, compute_statistics, load_posterior

PARAMS_R1 = dict(K_o=1e9, K_w=2.2e9, rho_o=800., rho_w=1030., K_s=37e9,
                 rho_s=2650., G_dry=1e9, phi=.3)


def _truth_and_samples(tmp_path):
    nz, nx = 10, 12
    masks = {'R1': np.zeros((nz, nx), bool), 'R2': np.zeros((nz, nx), bool)}
    masks['R1'][3:6, 2:6] = True
    masks['R2'][4:7, 7:11] = True
    so_true = np.full((3, nz, nx), np.nan)
    k_dry = np.full((nz, nx), np.nan)
    params = PetroParameters(**PARAMS_R1)
    v_true = np.full((3, nz, nx), 2500.)
    for name, mask in masks.items():
        k_dry[mask] = 3e9
        for index, value in enumerate((.8, .5, .2)):
            so_true[index, mask] = value
            v_true[index, mask] = gassmann(value, 3e9, params)[0]
    truth = tmp_path/'truth'
    (truth/'models').mkdir(parents=True)
    np.savez(truth/'models.npz', v_true=v_true, v_bg=v_true[0], So_true=so_true,
             mask_R1=masks['R1'], mask_R2=masks['R2'], K_dry=k_dry,
             spacing_xz_m=np.array([25., 25.]), origin_xz_m=np.array([0., 0.]))
    cfg = {'rock_physics': {'R1': PARAMS_R1, 'R2': PARAMS_R1}}
    (tmp_path/'config_resolved.yaml').write_text(yaml.safe_dump(cfg))
    root = tmp_path/'checkpoints'
    rng = np.random.default_rng(0)
    for index, survey in enumerate(SURVEYS):
        folder = root/'4d'/f'tag_{survey}_max_itr-5'
        folder.mkdir(parents=True)
        samples = v_true[index].T[None] + rng.normal(0, 2., (5, nx, nz))
        with h5py.File(folder/'samples.hdf5', 'w') as handle:
            handle.create_dataset('samples', data=(samples/1000).astype(np.float32))
    return tmp_path, root, masks, v_true, so_true


def test_posterior_reading_applies_burn_in_and_axes(tmp_path):
    path = tmp_path/'samples.hdf5'
    data = np.arange(4*2*3, dtype=np.float32).reshape(4, 2, 3)
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('samples', data=data)
    result = load_posterior(path, burn_in=1, stride=2)
    assert result.shape == (2, 3, 2)
    np.testing.assert_allclose(result[0, 0], data[1, :, 0]*1000)


def test_statistics_detect_velocity_and_saturation_changes(tmp_path):
    case, root, masks, v_true, so_true = _truth_and_samples(tmp_path)
    report = compute_statistics(case, root, 'tag', burn_in=0, stride=1)
    assert report['samples_used'] == 5
    for survey in SURVEYS:
        region = report['surveys'][survey]['reservoirs']
        assert set(region) == {'R1', 'R2'}
        assert region['R1']['pixels'] == int(masks['R1'].sum())
    delta = report['surveys']['monitor1']['difference_vs_baseline']
    assert delta['R1']['mean_delta_m_s'] > 0
    assert delta['R1']['mean_probability_increase'] > .95
    saturation = report['surveys']['monitor2']['saturation']['R1']
    assert .1 < saturation['mean_of_means'] < .9
    assert saturation['difference_vs_baseline']['mean_probability_decrease'] > .95
    assert saturation['posterior_mean_rmse'] < .1
    outputs = report['outputs']
    assert (case/'statistics'/'region_statistics.csv').is_file()
    figures = case/'statistics'/'figures'
    for name in ('vp_posterior_mean.png', 'vp_posterior_std.png',
                 'vp_delta_mean.png', 'vp_delta_probability_increase.png',
                 'vp_truth_vs_posterior_R1.png', 'vp_truth_vs_posterior_R2.png',
                 'vp_posterior_std_zoom.png', 'vp_delta_mean_zoom.png',
                 'vp_delta_probability_zoom.png',
                 'saturation_posterior_mean.png', 'saturation_probability_decrease.png'):
        assert (figures/name).is_file(), name
    saved = json.loads((case/'statistics'/'statistics.json').read_text())
    assert saved['surveys']['baseline']['control']['pixels'] > 0
