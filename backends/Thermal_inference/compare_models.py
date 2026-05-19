"""
compare_models.py
=================
Run both NV_heatsink_copper and NV_heatsink_aluminum checkpoints on the same
grid, then report per-XY-plane (z-slice) mean temperatures for each model.

Usage (inside physicsnemo Docker container):
    python compare_models.py
    python compare_models.py --nx 64 --ny 32 --nz 64
    python compare_models.py --save-npy comparison.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from infer_thermal_solid import (
    build_net,
    load_checkpoint,
    run_inference,
    make_grid,
    LENGTH_SCALE,
    TEMP_SCALE,
    DOMAIN_Z,
    HEAT_SINK_X,
    DOMAIN_Y,
)

SCRIPT_DIR    = Path(__file__).parent
MODEL_DIR     = SCRIPT_DIR / "model"
COPPER_CKPT   = MODEL_DIR / "NV_heatsink_copper"   / "thermal_solid_network.0.pth"
ALUMINUM_CKPT = MODEL_DIR / "NV_heatsink_aluminum" / "thermal_solid_network.0.pth"


def _parse_args():
    p = argparse.ArgumentParser(description="Compare copper vs aluminum – XY-plane temperature")
    p.add_argument("--device", default=None)
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--ny", type=int, default=10)
    p.add_argument("--nz", type=int, default=20)
    p.add_argument("--batch", type=int, default=65536)
    p.add_argument("--save-npy", default=None)
    return p.parse_args()


def _infer_model(ckpt: Path, label: str, x_nd, y_nd, z_nd, dev, batch):
    print(f"\n── {label} ──────────────────────────────────")
    import torch
    net = build_net(dev)
    net = load_checkpoint(net, ckpt, dev)
    theta = run_inference(net, x_nd, y_nd, z_nd, dev, batch)
    del net
    T_C = theta * TEMP_SCALE
    print(f"   Overall: min={T_C.min():.3f} C  max={T_C.max():.3f} C  mean={T_C.mean():.3f} C")
    return T_C


def main():
    args = _parse_args()

    import torch
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    nx, ny, nz = args.nx, args.ny, args.nz
    print(f"[Device] {dev}")
    print(f"[Grid]   {nx} x {ny} x {nz} = {nx*ny*nz:,} points  "
          f"(x: heat-sink {HEAT_SINK_X}, y: {DOMAIN_Y}, z: {DOMAIN_Z})")

    # Build flat coordinate arrays (same order as make_grid: indexing='ij')
    xs = np.linspace(HEAT_SINK_X[0], HEAT_SINK_X[1], nx, dtype=np.float32)
    ys = np.linspace(DOMAIN_Y[0],    DOMAIN_Y[1],    ny, dtype=np.float32)
    zs = np.linspace(DOMAIN_Z[0],    DOMAIN_Z[1],    nz, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    x_nd = gx.ravel()
    y_nd = gy.ravel()
    z_nd = gz.ravel()

    T_cu = _infer_model(COPPER_CKPT,   "Copper   (NV_heatsink_copper)",   x_nd, y_nd, z_nd, dev, args.batch)
    T_al = _infer_model(ALUMINUM_CKPT, "Aluminum (NV_heatsink_aluminum)", x_nd, y_nd, z_nd, dev, args.batch)

    # Reshape to (nx, ny, nz)  so axis-2 = z slices = XY planes
    T_cu_vol = T_cu.reshape(nx, ny, nz)
    T_al_vol = T_al.reshape(nx, ny, nz)

    # Mean temperature of each XY plane (average over x and y, one value per z)
    cu_plane_mean  = T_cu_vol.mean(axis=(0, 1))   # shape (nz,)
    al_plane_mean  = T_al_vol.mean(axis=(0, 1))
    cu_plane_max   = T_cu_vol.max(axis=(0, 1))
    al_plane_max   = T_al_vol.max(axis=(0, 1))

    diff_mean = cu_plane_mean - al_plane_mean

    z_phys_mm = zs * LENGTH_SCALE * 1000   # non-dim -> mm

    print("\n══════════════════════════════════════════════════════════════════════════════════════")
    print("  XY-PLANE (z-slice) TEMPERATURE COMPARISON")
    print(f"  {'z_idx':>5}  {'z [mm]':>8}  {'Cu mean [C]':>12}  {'Al mean [C]':>12}  "
          f"{'Cu max [C]':>11}  {'Al max [C]':>11}  {'Δmean Cu−Al':>12}")
    print("  " + "─" * 82)
    for i, (zm, cu_m, al_m, cu_x, al_x, dm) in enumerate(
            zip(z_phys_mm, cu_plane_mean, al_plane_mean, cu_plane_max, al_plane_max, diff_mean)):
        marker = " <-- hottest Cu" if i == cu_plane_mean.argmax() else (
                 " <-- hottest Al" if i == al_plane_mean.argmax() else "")
        print(f"  {i:>5}  {zm:>8.2f}  {cu_m:>12.3f}  {al_m:>12.3f}  "
              f"{cu_x:>11.3f}  {al_x:>11.3f}  {dm:>+12.3f}{marker}")
    print("  " + "─" * 82)

    cu_hot_idx = int(cu_plane_mean.argmax())
    al_hot_idx = int(al_plane_mean.argmax())
    print(f"\n  Hottest XY-plane (by mean T):")
    print(f"    Copper   -> z_idx={cu_hot_idx}  z={z_phys_mm[cu_hot_idx]:.2f} mm  "
          f"mean={cu_plane_mean[cu_hot_idx]:.3f} C  max={cu_plane_max[cu_hot_idx]:.3f} C")
    print(f"    Aluminum -> z_idx={al_hot_idx}  z={z_phys_mm[al_hot_idx]:.2f} mm  "
          f"mean={al_plane_mean[al_hot_idx]:.3f} C  max={al_plane_max[al_hot_idx]:.3f} C")

    diff_vol = T_cu_vol - T_al_vol
    print(f"\n  Overall Cu − Al difference:")
    print(f"    mean={diff_vol.mean():+.3f} C   max={diff_vol.max():+.3f} C   "
          f"min={diff_vol.min():+.3f} C")
    print("══════════════════════════════════════════════════════════════════════════════════════")

    if args.save_npy:
        np.save(args.save_npy, {
            "xs_nd": xs, "ys_nd": ys, "zs_nd": zs,
            "z_phys_mm": z_phys_mm,
            "T_copper_C": T_cu_vol,
            "T_aluminum_C": T_al_vol,
            "diff_C": diff_vol,
        })
        print(f"\n[Saved] {args.save_npy}")


if __name__ == "__main__":
    main()
