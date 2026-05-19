"""extract_aluminum_z0.py
=======================
Pre-compute the aluminum PINN model's XY plane at z_idx=0 (hottest z-slice)
and save to /assets/pinn_heatsink_aluminum_z0.npy.

Aluminum model (cycle_n_0011) coordinate domain:
  x: (-0.70, 0.70)  — flow direction (board Y axis)
  y: (-0.39, 0.39)  — fin height (averaged out for top-down view)
  z: (-0.98, 0.77)  — chip side at z=-0.98 (z_idx=0, hottest)

Run inside physicsnemo Docker:
  docker run --rm --gpus all \
    -v /home/jack/thermal_helper/backends/Thermal_inference:/workspace \
    -v /home/jack/thermal_helper/assets:/assets \
    -w /workspace \
    data-service.inventec.com:1443/physicsnemo:latest \
    python extract_aluminum_z0.py
"""

import numpy as np
import torch
from pathlib import Path
from infer_thermal_solid import build_net, load_checkpoint, run_inference, TEMP_SCALE

# Aluminum model domain bounds (non-dimensional, from cycle_n_0011)
AL_X = (-0.70, 0.70)   # flow direction
AL_Y = (-0.39, 0.39)   # fin height
AL_Z = (-0.98, 0.77)   # channel depth; chip at z_idx=0 (z=-0.98)

nx, ny = 64, 32
nz_orig = 32
z_nd = float(np.linspace(AL_Z[0], AL_Z[1], nz_orig)[0])   # = -0.9800
print(f"z_nd={z_nd:.4f}  (z_idx=0 of nz={nz_orig})")

xs = np.linspace(AL_X[0], AL_X[1], nx, dtype=np.float32)
ys = np.linspace(AL_Y[0], AL_Y[1], ny, dtype=np.float32)
gx, gy = np.meshgrid(xs, ys, indexing='ij')
gz = np.full_like(gx, z_nd)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = build_net(dev)
net = load_checkpoint(
    net,
    Path('model/NV_heatsink_aluminum/thermal_solid_network.0.pth'),
    dev,
)

theta = run_inference(net, gx.ravel(), gy.ravel(), gz.ravel(), dev)
T_C = (theta * TEMP_SCALE).reshape(nx, ny)

result = {
    'T_C': T_C,
    'xs_nd': xs,
    'ys_nd': ys,
    'z_nd': z_nd,
    'description': (
        'Aluminum model XY plane at z_idx=0 (z_nd=-0.9800). '
        'Domain: x=(-0.70,0.70) flow dir, y=(-0.39,0.39) fin height. '
        'T_C[ix, iy]: ix along PINN x (flow dir), iy along PINN y (fin height).'
    ),
}
np.save('/assets/pinn_heatsink_aluminum_z0.npy', result)
print(f"T_C shape: {T_C.shape}")
print(f"T range: {T_C.min():.3f} - {T_C.max():.3f} C   mean: {T_C.mean():.3f} C")
print("Saved to /assets/pinn_heatsink_aluminum_z0.npy")
