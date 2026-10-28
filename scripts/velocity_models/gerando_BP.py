import os
from dp4inversion.utils import (calc_dx_dt, load_BP_model,
                                resample_model_BP, get_smooth_model_BP)

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


BP_model = load_BP_model(in_dir)
# print(f"nx = {nx}, ny = {ny}, nt = {nt}")
# print(f"dx = {dx:.2f} m, dy = {dy:.2f} m, dt = {dt:.2e} s")
print(f'BP_model max(vp): {np.max(BP_model['vp'])}')
x_range = [0, 25000] #m
y_range = [0, 9000] #m


f0 = 3
vp0 = 4510
dx, dt = calc_dx_dt(f0 * 3, vp0, space_order=4)
dy = dx
n_pml = 10


# dx = BP_model['dx']
# dy = BP_model['dy']

x = np.arange(x_range[0], x_range[1], dx)
y = np.arange(y_range[0], y_range[1], dy)


model = resample_model_BP(x, y, BP_model)

t_range = [0, np.sqrt((x_range[1] - x_range[0])**2 + (y_range[1] - y_range[0])**2)/np.mean(model['vp']) * 2.0]

t = np.arange(t_range[0], t_range[1], dt)
nx = len(x)
ny = len(y)
nt = len(t)
src_rcv_depth = 0 #n_pml + 5
mask_extra_detph = 0 #

print(f'model max(vp): {np.max(model['vp'])}')

print(f"nx = {nx}, ny = {ny}, nt = {nt}, tmax = {t[-1]}")
print(f"dx = {dx:.2f} m, dy = {dy:.2f} m, dt = {dt:.2e} s")



# vp0 = np.max(model['vp'])
# _, dt = calc_dx_dt(f0, vp0)

nrcv = int(nx*dx / (int(vp0/f0)/6)) #numeros de receptores

smooth_model = get_smooth_model_BP(model, src_rcv_depth, mask_extra_detph, kernel=(11, 11))

# nx = nx - 3
# ny = ny - 2

torch.save(
         {"vp" : model['vp'][:nx, :ny] * 1e-3,
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
         os.path.join(out_dir, f'BP_model_freq_{f0}_v1.pth'))


torch.save(
         {"vp" : smooth_model['vp'][:nx, :ny] * 1e-3,
         "mask": smooth_model["mask"][:nx, :ny],
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
         os.path.join(out_dir, f'BP-smooth-linear_freq_{f0}_v1.pth'))
