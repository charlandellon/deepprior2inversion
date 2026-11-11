"""Explicit legacy-file boundary and reproducibility metadata."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from .geometry import Grid2D


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def load_legacy_model(path, origin_xz_m=(0.0, 0.0)):
    """Load trusted project PTH (km/s, xz); return SI array (zx) and grid."""
    from dp4inversion.utils import load_torch_file

    data = load_torch_file(path, map_location='cpu')
    velocity = np.asarray(data['vp'], dtype=np.float64)
    if velocity.shape != (int(data['nx']), int(data['ny'])):
        raise ValueError('legacy vp shape disagrees with nx/ny metadata')
    if not np.isfinite(velocity).all() or np.any(velocity <= 0):
        raise ValueError('legacy velocity must be finite and positive')
    grid = Grid2D(velocity.T.shape, (float(data['dx']), float(data['dy'])),
                  tuple(origin_xz_m))
    return velocity.T.copy()*1000.0, grid


def load_original_marmousi_vp(path, local_x_offset_m=-2000.0):
    """Read original AGL SEG-Y; original x=0..17000 m, z=0..3500 m.

    Coordinates follow utils.load_marmousi_model; values are km/s in SEG-Y.
    Only Vp is loaded; input is never changed or downloaded implicitly.
    """
    import obspy
    from dp4inversion.utils import extract_data

    stream = obspy.read(str(path), format='SEGY')
    velocity = extract_data(stream).T.astype(np.float64)*1000
    grid = Grid2D(velocity.shape, (17000/(velocity.shape[1]-1), 3500/(velocity.shape[0]-1)),
                  (float(local_x_offset_m), 0.0))
    return velocity, grid


def write_manifest(directory, cfg, inputs, outputs):
    import torch

    application_root = Path(__file__).resolve().parent
    repository_root = application_root.parent
    versions = {}
    for package in ('torch', 'devito', 'numpy', 'scipy', 'matplotlib', 'PyYAML'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    # The repository checkout may be unavailable inside a minimal runtime.
    commit, dirty = None, None
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repository_root,
                                  capture_output=True, text=True, check=False)
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=repository_root,
                                capture_output=True, text=True, check=False)
        if revision.returncode == 0:
            commit = revision.stdout.strip()
        if status.returncode == 0:
            dirty = bool(status.stdout.strip())
    except FileNotFoundError:
        pass
    source_paths = sorted(application_root.glob('*.py'))
    manifest = dict(run_id=directory.name, created_utc=datetime.now(timezone.utc).isoformat(),
                    commit=commit, working_tree_dirty=dirty,
                    source_sha256={str(p.relative_to(application_root)): sha256(p) for p in source_paths},
                    python=platform.python_version(), platform=platform.platform(),
                    versions=versions, cuda_available=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    seeds={'scenario': cfg['experiment']['seed']},
                    array_axes='zx', velocity_units='m/s', dtype='float64',
                    inputs={str(p): sha256(p) for p in inputs},
                    outputs={str(p.relative_to(directory)): sha256(p) for p in outputs})
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
