import os
import os
from dp4inversion.utils import (calc_dx_dt, extract_data,
                    load_marmousi_model, resample_model,
                    get_smooth_model)

from dp4inversion.project_path import datadir
import numpy as np

import torch

datadir("velocity_model")
in_dir = datadir("velocity_model")
out_dir = datadir("velocity_model")

if not os.path.exists(in_dir):
    os.makedirs(in_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


marmousi_model = load_marmousi_model(in_dir)
# print(f"nx = {nx}, ny = {ny}, nt = {nt}")
# print(f"dx = {dx:.2f} m, dy = {dy:.2f} m, dt = {dt:.2e} s")

x_range = [2000, 14500] #m
y_range = [0, 3500] #m


f0 = 5
vp0 = np.max(marmousi_model['vp'])
dx, dt = calc_dx_dt(f0 * 3, vp0, space_order=4)
dy = dx
n_pml = 10
t_range = [0, np.sqrt((x_range[1] - x_range[0])**2 + (y_range[1] - y_range[0])**2)/np.mean(marmousi_model['vp']) * 1.6]
x = np.arange(x_range[0], x_range[1], dx)
y = np.arange(y_range[0], y_range[1], dy)
t = np.arange(t_range[0], t_range[1], dt)
nx = len(x)
ny = len(y)
nt = len(t)
src_rcv_depth = 0 #n_pml + 5
mask_extra_detph = 0 #

print(f"nx = {nx}, ny = {ny}, nt = {nt}")
print(f"dx = {dx:.2f} m, dy = {dy:.2f} m, dt = {dt:.2e} s")


model = resample_model(x, y, marmousi_model)

# vp0 = np.max(model['vp'])
# _, dt = calc_dx_dt(f0, vp0)

nrcv = int(nx*dx / (int(vp0/f0)/6)) #numeros de receptores

smooth_model = get_smooth_model(model, src_rcv_depth, mask_extra_detph)


torch.save(
         {"vp" : model['vp'] * 1e-3,
         "vs" : model['vs'] * 1e-3,
         "rho" : model['rho'],
         "dx" : dx,
         "dy" : dy,
         "dt" : dt * 1e3,
         "nx" : nx,
         "ny" : ny,
         "nt" : nt,
         "f0" : f0,
         "tmax": t_range[1] * 1e3,
         "src_rcv_depth": src_rcv_depth,
         "nrcv": nrcv},
         os.path.join(out_dir, f'marmousi2-model-true-large_freq_{f0}.pth'))


torch.save(
         {"vp" : smooth_model['vp'] * 1e-3,
         "vs" : smooth_model['vs'] * 1e-3,
         "rho" : smooth_model['rho'],
         "mask": smooth_model["mask"],
         "dx" : dx,
         "dy" : dy,
         "dt" : dt * 1e3,
         "nx" : nx,
         "ny" : ny,
         "nt" : nt,
         "f0" : f0,
         "tmax": t_range[1] * 1e3,
         "src_rcv_depth": src_rcv_depth,
         "nrcv": nrcv},
         os.path.join(out_dir, f'marmousi2-model-smooth-linear_freq_{f0}.pth'))
