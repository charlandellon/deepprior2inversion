"""Plot the short pilot traces for sampler diagnosis."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path('data/checkpoints/4d')
OUT = Path('outputs/4d/marmousi4d_expanded_oil_water_white_E2/pilot_diagnostics')
OUT.mkdir(parents=True, exist_ok=True)
PILOTS = {
    'scale=10, lr=1e-4': 'pilot_a_scale10_lr1e-4',
    'scale=1, lr=1e-4': 'pilot_b_scale1_lr1e-4',
    'scale=1, lr=3e-5': 'pilot_d_scale1_lr3e-5',
    'scale=1, lr=1e-5': 'pilot_c_scale1_lr1e-5',
}

for key, label in (('rmse_model', 'RMSE model (km/s)'),
                   ('objective', 'objective')):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, pilot in PILOTS.items():
        checkpoint = torch.load(ROOT/pilot/pilot/'checkpoint.pth',
                                map_location='cpu', weights_only=False)
        source = 'obj_log' if key == 'objective' else key
        values = np.asarray(checkpoint[source], dtype=float)[25000:]
        ax.plot(np.arange(25000, 25000 + len(values)), values, label=name,
                linewidth=0.9)
    ax.set(xlabel='iteration', ylabel=label)
    ax.legend()
    if key == 'objective':
        ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(OUT/f'pilot_{key}.png', dpi=200)
    plt.close(fig)
print(OUT)
