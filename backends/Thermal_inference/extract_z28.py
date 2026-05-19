import numpy as np, torch
from pathlib import Path
from infer_thermal_solid import (
    build_net, load_checkpoint, run_inference,
    HEAT_SINK_X, DOMAIN_Y, DOMAIN_Z, TEMP_SCALE,
)

nx, ny = 64, 32
nz_orig = 32
z_nd = float(np.linspace(DOMAIN_Z[0], DOMAIN_Z[1], nz_orig)[28])
print(f"z_nd={z_nd:.4f}  (z_idx=28 of nz={nz_orig})")

xs = np.linspace(HEAT_SINK_X[0], HEAT_SINK_X[1], nx, dtype=np.float32)
ys = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], ny, dtype=np.float32)
gx, gy = np.meshgrid(xs, ys, indexing='ij')
gz = np.full_like(gx, z_nd)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = build_net(dev)
net = load_checkpoint(net, Path('model/NV_heatsink_copper/thermal_solid_network.0.pth'), dev)
theta = run_inference(net, gx.ravel(), gy.ravel(), gz.ravel(), dev)
T_C = (theta * TEMP_SCALE).reshape(nx, ny)

out = {
    'T_C': T_C,
    'xs_nd': xs,
    'ys_nd': ys,
    'z_nd': z_nd,
    'description': (
        'Copper model XY plane at z_idx=28 (z_nd=5.148). '
        'T_C[ix, iy]: ix along PINN x (flow dir), iy along PINN y (fin height).'
    ),
}
np.save('/assets/pinn_heatsink_z28.npy', out)
print(f"T_C shape: {T_C.shape}")
print(f"T range: {T_C.min():.3f} - {T_C.max():.3f} C   mean: {T_C.mean():.3f} C")
print("Saved to /assets/pinn_heatsink_z28.npy")
