"""
compare_models.py
=================
Run NV_heatsink_copper and NV_heatsink_aluminum checkpoints, each queried
within their OWN training coordinate domain, then compare temperatures.

Root cause of earlier wrong results
─────────────────────────────────────
The two models were trained on DIFFERENT STL geometries with different
translations/scales, giving completely different non-dimensional coordinate
ranges:

  Copper   (OMEN1526):  x∈[0.15,1.15]  y∈[0,0.60]      z∈[0,5.70]
  Aluminum (NV model):  x∈[-0.70,0.70] y∈[-0.39,0.39]  z∈[-0.98,0.77]

Querying the aluminum model with copper coordinates causes massive
extrapolation errors (values up to 886,000 °C).

Usage (inside physicsnemo Docker container):
    python compare_models.py
    python compare_models.py --nx 32 --ny 16 --nz 32
    python compare_models.py --save-npy comparison.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from infer_thermal_solid import build_net, load_checkpoint, run_inference, TEMP_SCALE

SCRIPT_DIR    = Path(__file__).parent
MODEL_DIR     = SCRIPT_DIR / "model"
COPPER_CKPT   = MODEL_DIR / "NV_heatsink_copper"   / "thermal_solid_network.0.pth"
ALUMINUM_CKPT = MODEL_DIR / "NV_heatsink_aluminum" / "thermal_solid_network.0.pth"

# ── Per-model coordinate domains (non-dimensional) ────────────────────────────
# Copper: from infer_thermal_solid.py / limerock_geometry.py (OMEN1526)
COPPER_DOMAIN = dict(
    x=( 0.15,  1.15),   # flow direction, heat-sink region
    y=( 0.00,  0.60),   # fin height
    z=( 0.00,  5.70),   # channel width
)

# Aluminum: from solid_temperature.csv actual data + visualization.py comment
# heat_sink_bounds = (-0.70, 0.70); geo bounds read from STL (different geometry)
ALUMINUM_DOMAIN = dict(
    x=(-0.70,  0.70),   # flow direction
    y=(-0.39,  0.39),   # fin height
    z=(-0.98,  0.77),   # channel width
)


def _parse_args():
    p = argparse.ArgumentParser(description="Compare copper vs aluminum with correct per-model domains")
    p.add_argument("--device", default=None)
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--ny", type=int, default=10)
    p.add_argument("--nz", type=int, default=20)
    p.add_argument("--batch", type=int, default=65536)
    p.add_argument("--save-npy", default=None)
    return p.parse_args()


def _make_grid(domain: dict, nx: int, ny: int, nz: int):
    xs = np.linspace(domain["x"][0], domain["x"][1], nx, dtype=np.float32)
    ys = np.linspace(domain["y"][0], domain["y"][1], ny, dtype=np.float32)
    zs = np.linspace(domain["z"][0], domain["z"][1], nz, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return xs, ys, zs, gx.ravel(), gy.ravel(), gz.ravel()


def _infer_model(ckpt: Path, label: str, domain: dict,
                 nx: int, ny: int, nz: int, dev, batch: int):
    import torch
    print(f"\n── {label}")
    print(f"   Domain  x:{domain['x']}  y:{domain['y']}  z:{domain['z']}")
    xs, ys, zs, x_nd, y_nd, z_nd = _make_grid(domain, nx, ny, nz)
    net = build_net(dev)
    net = load_checkpoint(net, ckpt, dev)
    theta = run_inference(net, x_nd, y_nd, z_nd, dev, batch)
    del net
    T_C = (theta * TEMP_SCALE).reshape(nx, ny, nz)
    print(f"   Overall T_solid: min={T_C.min():.3f} C  max={T_C.max():.3f} C  mean={T_C.mean():.3f} C")
    return xs, ys, zs, T_C


def _print_plane_table(label: str, zs: np.ndarray, T_vol: np.ndarray):
    """Print per-XY-plane (z-slice) statistics."""
    plane_mean = T_vol.mean(axis=(0, 1))
    plane_max  = T_vol.max(axis=(0, 1))
    hot_idx    = int(plane_mean.argmax())
    print(f"\n  {label} — XY-plane breakdown:")
    print(f"  {'z_idx':>5}  {'z_nd':>7}  {'mean [C]':>10}  {'max [C]':>10}")
    print("  " + "─" * 38)
    for i, (zv, pm, px) in enumerate(zip(zs, plane_mean, plane_max)):
        marker = "  ← hottest" if i == hot_idx else ""
        print(f"  {i:>5}  {zv:>7.4f}  {pm:>10.3f}  {px:>10.3f}{marker}")
    print(f"\n  Hottest z_idx={hot_idx}  z_nd={zs[hot_idx]:.4f}  "
          f"mean={plane_mean[hot_idx]:.3f} C  max={plane_max[hot_idx]:.3f} C")


def main():
    args = _parse_args()

    import torch
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    nx, ny, nz = args.nx, args.ny, args.nz

    print(f"[Device] {dev}")
    print(f"[Grid]   {nx} × {ny} × {nz} = {nx*ny*nz:,} points per model")
    print("NOTE: each model is queried within its own training coordinate domain.")

    xs_cu, ys_cu, zs_cu, T_cu = _infer_model(
        COPPER_CKPT, "Copper   (NV_heatsink_copper)",   COPPER_DOMAIN,   nx, ny, nz, dev, args.batch)
    xs_al, ys_al, zs_al, T_al = _infer_model(
        ALUMINUM_CKPT, "Aluminum (NV_heatsink_aluminum)", ALUMINUM_DOMAIN, nx, ny, nz, dev, args.batch)

    print("\n" + "═" * 70)
    _print_plane_table("Copper",   zs_cu, T_cu)
    print()
    _print_plane_table("Aluminum", zs_al, T_al)

    print("\n" + "═" * 70)
    print("  OVERALL SUMMARY")
    print(f"  Copper   T_solid: min={T_cu.min():.3f} C  max={T_cu.max():.3f} C  mean={T_cu.mean():.3f} C")
    print(f"  Aluminum T_solid: min={T_al.min():.3f} C  max={T_al.max():.3f} C  mean={T_al.mean():.3f} C")
    print("═" * 70)

    if args.save_npy:
        np.save(args.save_npy, {
            "copper":   {"xs": xs_cu, "ys": ys_cu, "zs": zs_cu, "T_C": T_cu},
            "aluminum": {"xs": xs_al, "ys": ys_al, "zs": zs_al, "T_C": T_al},
        })
        print(f"\n[Saved] {args.save_npy}")


if __name__ == "__main__":
    main()

