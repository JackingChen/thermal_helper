"""
infer_thermal_solid.py
======================================================================================
Inference script for the OMEN1526 thermal_solid_network.

Loads the pre-trained FourierNetArch checkpoint (thermal_solid_network.0.pth)
and queries solid temperature at arbitrary (x, y, z) points.

DO I NEED THE FLOW / PRESSURE NETWORK?
  No.  thermal_solid_network predicts theta_s purely from spatial coordinates
  (x, y, z).  Flow (u, v, w, p) and fluid temperature (theta_f) are only
  needed as boundary-condition couplings *during training* -- at inference the
  solid-temperature network is fully standalone.

Usage
-----
    # inside the physicsnemo Docker container:
    python infer_thermal_solid.py                       # default 20x10x20 grid
    python infer_thermal_solid.py --nx 64 --ny 32 --nz 64
    python infer_thermal_solid.py --point 1.0 0.03 2.7  # single physical point [m]
    python infer_thermal_solid.py --save-npy results.npy

REQUIRED INPUTS
---------------
1.  CHECKPOINT FILE
      model/thermal_solid_network.0.pth  (default: model/ subdirectory)

2.  QUERY COORDINATES (x, y, z) -- non-dimensional
      length_scale = 0.01 m  ->  x_nd = x_physical_m / 0.01

      Domain bounds (non-dim, from STL + translate=(80,5,160) mm, scale=1/100):
        x : [0.0, ~3.25]    flow direction (heat-sink at [0.15, 1.15])
        y : [0.0, ~0.60]    fin height
        z : [0.0, ~5.70]    channel width  (chip at z ~ 2.55-2.85)

3.  OUTPUT
      T_solid [C] = theta_s_nd * 273.15
      (reference: inlet coolant at 0 C)
======================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# PhysicsNeMo (same imports as limerock_thermal.py)
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.key import Key

# Non-dimensionalisation (from limerock_properties.py)
LENGTH_SCALE = 0.01    # m   (scale_value = 100)
TEMP_SCALE   = 273.15  # K   (inlet_temp = 0 C reference)

# Domain bounds (non-dimensional)
DOMAIN_X    = (0.0,  3.25)
DOMAIN_Y    = (0.0,  0.60)
DOMAIN_Z    = (0.0,  5.70)
HEAT_SINK_X = (0.15, 1.15)   # from limerock_geometry.py heat_sink_bounds


def build_net(device: torch.device) -> FourierNetArch:
    """
    Exact copy of the thermal_solid_network definition from limerock_thermal.py:

        thermal_s_net = FourierNetArch(
            input_keys=[Key("x"), Key("y"), Key("z")],
            output_keys=[Key("theta_s")]
        )

    FourierNetArch defaults (match training): layer_size=512, nr_layers=6.
    """
    net = FourierNetArch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("theta_s")],
    )
    return net.to(device).eval()


def load_checkpoint(net: FourierNetArch, ckpt_path: Path, device: torch.device) -> FourierNetArch:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state_dict)
    print(f"[OK] Loaded: {ckpt_path}")
    return net


def run_inference(
    net: FourierNetArch,
    x_nd: np.ndarray,
    y_nd: np.ndarray,
    z_nd: np.ndarray,
    device: torch.device,
    batch_size: int = 65_536,
) -> np.ndarray:
    """
    Query thermal_solid_network at N spatial points.

    Inputs
    ------
    x_nd, y_nd, z_nd : np.ndarray shape (N,)  -- non-dimensional coordinates
        x_nd = x_m / 0.01,  y_nd = y_m / 0.01,  z_nd = z_m / 0.01

    Returns
    -------
    theta_s_nd : np.ndarray shape (N,)
        Convert to Celsius:  T_solid_C = theta_s_nd * 273.15
    """
    x_nd = np.asarray(x_nd, dtype=np.float32).ravel()
    y_nd = np.asarray(y_nd, dtype=np.float32).ravel()
    z_nd = np.asarray(z_nd, dtype=np.float32).ravel()
    N = len(x_nd)
    theta_s = np.empty(N, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            out = net({
                "x": torch.tensor(x_nd[start:end, None], device=device),
                "y": torch.tensor(y_nd[start:end, None], device=device),
                "z": torch.tensor(z_nd[start:end, None], device=device),
            })
            theta_s[start:end] = out["theta_s"].cpu().numpy().ravel()

    return theta_s


def make_grid(nx: int, ny: int, nz: int):
    xs = np.linspace(HEAT_SINK_X[0], HEAT_SINK_X[1], nx, dtype=np.float32)
    ys = np.linspace(DOMAIN_Y[0],    DOMAIN_Y[1],    ny, dtype=np.float32)
    zs = np.linspace(DOMAIN_Z[0],    DOMAIN_Z[1],    nz, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return gx.ravel(), gy.ravel(), gz.ravel()


def _parse_args():
    p = argparse.ArgumentParser(description="OMEN1526 thermal_solid_network inference")
    p.add_argument("--ckpt", default=str(Path(__file__).parent / "model" / "thermal_solid_network.0.pth"))
    p.add_argument("--device", default=None, help="cuda or cpu")
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--ny", type=int, default=10)
    p.add_argument("--nz", type=int, default=20)
    p.add_argument("--batch", type=int, default=65536)
    p.add_argument("--point", nargs=3, type=float, metavar=("X_M", "Y_M", "Z_M"),
                   help="Query a single physical point in metres")
    p.add_argument("--save-npy", default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    dev  = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = Path(args.ckpt)

    print(f"[Device]     {dev}")
    print(f"[Checkpoint] {ckpt}")

    net = build_net(dev)
    net = load_checkpoint(net, ckpt, dev)

    if args.point:
        xm, ym, zm = args.point
        x_nd = np.array([xm / LENGTH_SCALE], dtype=np.float32)
        y_nd = np.array([ym / LENGTH_SCALE], dtype=np.float32)
        z_nd = np.array([zm / LENGTH_SCALE], dtype=np.float32)
        print(f"\n[Single point] ({xm} m, {ym} m, {zm} m) -> "
              f"nd=({x_nd[0]:.2f}, {y_nd[0]:.2f}, {z_nd[0]:.2f})")
    else:
        print(f"\n[Grid] {args.nx}x{args.ny}x{args.nz} = {args.nx*args.ny*args.nz:,} points")
        x_nd, y_nd, z_nd = make_grid(args.nx, args.ny, args.nz)

    theta_s_nd = run_inference(net, x_nd, y_nd, z_nd, dev, args.batch)
    T_solid_C  = theta_s_nd * TEMP_SCALE

    print(f"\n[Results]")
    print(f"  theta_s (nd) : min={theta_s_nd.min():.5f}  max={theta_s_nd.max():.5f}  mean={theta_s_nd.mean():.5f}")
    print(f"  T_solid  [C] : min={T_solid_C.min():.2f}  max={T_solid_C.max():.2f}  mean={T_solid_C.mean():.2f}")

    if args.point:
        xm, ym, zm = args.point
        print(f"\n  -> T_solid at ({xm} m, {ym} m, {zm} m) = {T_solid_C[0]:.2f} C")

    if args.save_npy:
        np.save(args.save_npy, {"x_nd": x_nd, "y_nd": y_nd, "z_nd": z_nd,
                                "theta_s_nd": theta_s_nd, "T_solid_C": T_solid_C})
        print(f"[Saved] {args.save_npy}")


if __name__ == "__main__":
    main()
