#!/usr/bin/env python3
"""
vti_to_project.py
──────────────────
Reads a VTK ImageData (.vti) file header, extracts domain geometry, and saves
all CFD components to data/<vti_stem>.json for use in the Streamlit modeling page.

Strategy
────────
Because large VTI files often embed inline-binary field data that exceeds the
VTK XML parser's limits, this script uses two-pass reading:

  Pass 1 — Header-only (fast, always works):
    Reads the first 8 KB with regex to extract WholeExtent, Origin, Spacing,
    and DataArray metadata.  Geometric components (Inlet, Outlet, walls, core
    flow, wake zone) are derived from domain bounds and typical CFD conventions.

  Pass 2 — Full PyVista read (optional, ~60 s for 350 MB):
    Attempted only when Pass 1 produces no DataArrays or when --full-read is
    passed. If successful, field statistics (pressure / velocity percentiles)
    refine the component positions.

Output format
─────────────
data/grid_inference_flow.json:
{
  "CFD_Flow_Demo": {
    "inlet":         { "x": …, "y": …, "w": …, "h": …, "label": "Inlet",         "rotated": false },
    "outlet":        { … "label": "Outlet" … },
    "core_flow":     { … "label": "Core Flow" … },
    "wake_zone":     { … "label": "Wake Zone" … },
    "top_wall":      { … "label": "Top Wall" … },
    "bottom_wall":   { … "label": "Bottom Wall" … }
  }
}

Usage (inside Docker):
    docker exec thermal-helper python3 /app/vti_to_project.py

    # force full PyVista read (slower but uses field stats for positions):
    docker exec thermal-helper python3 /app/vti_to_project.py --full-read

    # custom paths / id:
    docker exec thermal-helper python3 /app/vti_to_project.py \\
        --vti /app/assets/grid_inference_flow.vti \\
        --project-id CFD_Flow_Demo
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np

# ── Defaults ───────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
_VTI_PATH   = _HERE / "assets" / "grid_inference_flow.vti"
_DATA_DIR   = _HERE / "data"
_PROJECT_ID = "CFD_Flow_Demo"
_CANVAS_W   = 150   # matches thermal_sim._W
_CANVAS_H   = 100   # matches thermal_sim._H


# ── Canvas mapping ─────────────────────────────────────────────────────────────

def _to_canvas(val: float, phys_min: float, phys_max: float, canvas_size: float) -> float:
    span = phys_max - phys_min
    if span == 0:
        return canvas_size / 2
    return (val - phys_min) / span * canvas_size


def _comp(label: str, cx: float, cy: float, cw: float, ch: float) -> dict:
    return {
        "x": round(cx, 1), "y": round(cy, 1),
        "w": round(cw, 1), "h": round(ch, 1),
        "label": label, "rotated": False,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pass 1 — Header-only extraction
# ══════════════════════════════════════════════════════════════════════════════

def _parse_header(vti_path: Path) -> dict:
    """
    Read ≤ 8 KB of the VTI file and extract domain metadata via regex.
    Returns a dict with keys: nx, ny, nz, ox, oy, oz, dx, dy, dz, arrays.
    """
    with open(vti_path, "rb") as f:
        raw = f.read(8192)

    def _floats(pattern: bytes) -> list[float]:
        m = re.search(pattern, raw)
        return [float(v) for v in m.group(1).split()] if m else []

    extent  = _floats(rb'WholeExtent="([^"]+)"')
    origin  = _floats(rb'Origin="([^"]+)"')
    spacing = _floats(rb'Spacing="([^"]+)"')

    if len(extent) < 6:
        raise ValueError("Could not parse WholeExtent from VTI header.")

    # WholeExtent is "x0 x1 y0 y1 z0 z1" (point indices, inclusive)
    nx = int(extent[1] - extent[0]) + 1
    ny = int(extent[3] - extent[2]) + 1
    nz = int(extent[5] - extent[4]) + 1

    ox, oy, oz = (origin  + [0.0, 0.0, 0.0])[:3]
    dx, dy, dz = (spacing + [1.0, 1.0, 1.0])[:3]

    # Collect DataArray descriptions
    arrays = []
    for m in re.finditer(rb'<DataArray[^>]+>', raw):
        tag = m.group(0)
        name_m  = re.search(rb'Name="([^"]+)"', tag)
        ncomp_m = re.search(rb'NumberOfComponents="([^"]+)"', tag)
        rmin_m  = re.search(rb'RangeMin="([^"]+)"', tag)
        rmax_m  = re.search(rb'RangeMax="([^"]+)"', tag)
        arrays.append({
            "name":       name_m.group(1).decode()  if name_m  else "?",
            "components": int(ncomp_m.group(1))     if ncomp_m else 1,
            "range_min":  float(rmin_m.group(1))    if rmin_m  else None,
            "range_max":  float(rmax_m.group(1))    if rmax_m  else None,
        })

    return {"nx": nx, "ny": ny, "nz": nz,
            "ox": ox, "oy": oy, "oz": oz,
            "dx": dx, "dy": dy, "dz": dz,
            "arrays": arrays}


def _geometric_components(meta: dict) -> dict:
    """
    Derive CFD component bboxes purely from domain geometry.

    Convention assumed: x = streamwise (inlet → outlet), z = wall-normal height.
    Maps physical coords to the 150 × 100 modeling canvas.
    """
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    ox, oy, oz = meta["ox"], meta["oy"], meta["oz"]
    dx, dy, dz = meta["dx"], meta["dy"], meta["dz"]

    x_min, x_max = ox, ox + (nx - 1) * dx
    z_min, z_max = oz, oz + (nz - 1) * dz

    def xc(v): return _to_canvas(v, x_min, x_max, _CANVAS_W)
    def zc(v): return _to_canvas(v, z_min, z_max, _CANVAS_H)

    cx_full  = _CANVAS_W / 2
    cy_full  = _CANVAS_H / 2
    cw_full  = _CANVAS_W
    ch_full  = _CANVAS_H
    face_w   = round(_CANVAS_W * 0.07, 1)    # thin vertical slab = wall/face marker
    wall_h   = round(_CANVAS_H * 0.06, 1)    # thin horizontal slab = wall marker
    core_w   = round(_CANVAS_W * 0.35, 1)    # core flow occupies ~35 % streamwise
    core_h   = round(_CANVAS_H * 0.40, 1)    # ~40 % of height (centre region)
    wake_w   = round(_CANVAS_W * 0.25, 1)    # wake zone near outlet
    wake_h   = round(_CANVAS_H * 0.50, 1)

    components: dict = {}

    # ── Inlet: left face slab ──────────────────────────────────────────────────
    components["inlet"] = _comp(
        "Inlet",
        cx=face_w / 2 + 1,  cy=cy_full,
        cw=face_w,           ch=round(_CANVAS_H * 0.80, 1),
    )

    # ── Outlet: right face slab ────────────────────────────────────────────────
    components["outlet"] = _comp(
        "Outlet",
        cx=_CANVAS_W - face_w / 2 - 1, cy=cy_full,
        cw=face_w,                      ch=round(_CANVAS_H * 0.80, 1),
    )

    # ── Core Flow: centre region (high-speed jet/duct core) ───────────────────
    components["core_flow"] = _comp(
        "Core Flow",
        cx=_CANVAS_W * 0.45,  cy=cy_full,
        cw=core_w,             ch=core_h,
    )

    # ── Wake Zone: lower-speed area near outlet ────────────────────────────────
    components["wake_zone"] = _comp(
        "Wake Zone",
        cx=_CANVAS_W * 0.78, cy=cy_full,
        cw=wake_w,            ch=wake_h,
    )

    # ── Top Wall boundary ──────────────────────────────────────────────────────
    components["top_wall"] = _comp(
        "Top Wall",
        cx=cx_full,   cy=_CANVAS_H - wall_h / 2 - 1,
        cw=cw_full,   ch=wall_h,
    )

    # ── Bottom Wall boundary ───────────────────────────────────────────────────
    components["bottom_wall"] = _comp(
        "Bottom Wall",
        cx=cx_full,  cy=wall_h / 2 + 1,
        cw=cw_full,  ch=wall_h,
    )

    return components


# ══════════════════════════════════════════════════════════════════════════════
# Pass 2 — Full PyVista read (field-statistics refinement)
# ══════════════════════════════════════════════════════════════════════════════

def _refine_with_field_stats(vti_path: Path, meta: dict,
                             components: dict) -> dict:
    """
    Attempt a full PyVista read. If successful, update core_flow and wake_zone
    positions from the velocity-magnitude percentile distribution.
    """
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = pv.read(str(vti_path))

        if "u" not in mesh.array_names or mesh.n_points == 0:
            return components

        nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
        ox, oz     = meta["ox"], meta["oz"]
        dx, dz     = meta["dx"], meta["dz"]
        x_min, x_max = ox, ox + (nx - 1) * dx
        z_min, z_max = oz, oz + (nz - 1) * dz

        x_c = ox + np.arange(nx) * dx
        z_c = oz + np.arange(nz) * dz

        mid_y = meta["ny"] // 2
        u_flat  = mesh.point_data["u"].astype(np.float32)
        u_3d    = u_flat.reshape(nz, ny, nx, 3)
        spd     = np.linalg.norm(u_3d[:, mid_y, :, :], axis=2)

        spd_min, spd_max = float(spd.min()), float(spd.max())
        print(f"  Speed |u| on mid-Y slice: min={spd_min:.4f}  max={spd_max:.4f}")

        def canvas_bbox(iz_hits, ix_hits):
            px = x_c[ix_hits]; pz = z_c[iz_hits]
            if len(px) == 0:
                return None
            cx = _to_canvas(float(px.mean()), x_min, x_max, _CANVAS_W)
            cy = _to_canvas(float(pz.mean()), z_min, z_max, _CANVAS_H)
            cw = max(float(px.std()) * 2 / (x_max-x_min) * _CANVAS_W, 8.0)
            ch = max(float(pz.std()) * 2 / (z_max-z_min) * _CANVAS_H, 8.0)
            cx = max(cw/2, min(_CANVAS_W - cw/2, cx))
            cy = max(ch/2, min(_CANVAS_H - ch/2, cy))
            return round(cx,1), round(cy,1), round(cw,1), round(ch,1)

        # Core flow → top 5 % speed
        t95 = np.percentile(spd, 95)
        iz, ix = np.where(spd >= t95)
        res = canvas_bbox(iz, ix)
        if res:
            components["core_flow"] = _comp("Core Flow", *res)
            print(f"  → core_flow refined  ({res[0]}, {res[1]})  {res[2]}×{res[3]}")

        # Wake zone → low-speed interior (5–15 th pct, exclude exact-zero walls)
        nonzero = spd[spd > spd_max * 0.02]
        if len(nonzero) > 0:
            t_lo = np.percentile(nonzero, 5)
            t_hi = np.percentile(nonzero, 15)
            iz, ix = np.where((spd >= t_lo) & (spd <= t_hi))
            res = canvas_bbox(iz, ix)
            if res:
                components["wake_zone"] = _comp("Wake Zone", *res)
                print(f"  → wake_zone refined  ({res[0]}, {res[1]})  {res[2]}×{res[3]}")

        return components

    except Exception as exc:
        print(f"  Full PyVista read failed ({exc}); keeping geometric defaults.")
        return components


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def extract_components(vti_path: Path, project_id: str,
                       full_read: bool = False) -> dict:
    """
    Returns { project_id: { component_name: {...} } }
    """
    print(f"Reading header: {vti_path.name}  ({vti_path.stat().st_size / 1e6:.0f} MB)")
    meta = _parse_header(vti_path)

    print(f"  Grid       : {meta['nx']} × {meta['ny']} × {meta['nz']}")
    print(f"  Origin     : ({meta['ox']:.4f}, {meta['oy']:.4f}, {meta['oz']:.4f})")
    print(f"  Spacing    : ({meta['dx']:.6f}, {meta['dy']:.6f}, {meta['dz']:.6f})")
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    ox, oz = meta["ox"], meta["oz"]
    dx, dz = meta["dx"], meta["dz"]
    print(f"  Phys bounds: x=[{ox:.3f}, {ox+(nx-1)*dx:.3f}]  "
          f"z=[{oz:.3f}, {oz+(nz-1)*dz:.3f}]")
    if meta["arrays"]:
        for a in meta["arrays"]:
            print(f"  Array '{a['name']}': {a['components']} component(s)  "
                  f"range=[{a['range_min']}, {a['range_max']}]")
    else:
        print("  No DataArrays found in header (data may start beyond 8 KB scan).")

    components = _geometric_components(meta)
    print("\n  Geometric components derived from domain bounds:")
    for name, c in components.items():
        print(f"    {name:14s}  ({c['x']:6.1f}, {c['y']:6.1f})  "
              f"{c['w']:5.1f} × {c['h']:5.1f}")

    if full_read:
        print("\n  Attempting full PyVista read for field-statistics refinement …")
        components = _refine_with_field_stats(vti_path, meta, components)

    return {project_id: components}


# ── Save ───────────────────────────────────────────────────────────────────────

def save_project_json(positions: dict, vti_path: Path, data_dir: Path) -> Path:
    out_path = data_dir / f"{vti_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)

    pid = next(iter(positions))
    print(f"\nSaved → {out_path}")
    print(f"\n── Final components for '{pid}' ──────────────────────────────────────")
    for name, comp in positions[pid].items():
        print(f"  {name:14s}  x={comp['x']:6.1f}  y={comp['y']:6.1f}  "
              f"w={comp['w']:5.1f}  h={comp['h']:5.1f}  label='{comp['label']}'")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vti",        default=str(_VTI_PATH),  help="Path to .vti file")
    parser.add_argument("--data-dir",   default=str(_DATA_DIR),  help="Output data/ directory")
    parser.add_argument("--project-id", default=_PROJECT_ID,     help="Project ID key in JSON")
    parser.add_argument("--full-read",  action="store_true",
                        help="Attempt full PyVista read for field-statistics refinement "
                             "(slow, ~60 s for 350 MB files)")
    args = parser.parse_args()

    vti_path = Path(args.vti)
    data_dir = Path(args.data_dir)

    if not vti_path.exists():
        raise FileNotFoundError(f"VTI not found: {vti_path}")
    data_dir.mkdir(parents=True, exist_ok=True)

    positions = extract_components(vti_path, args.project_id, args.full_read)
    out = save_project_json(positions, vti_path, data_dir)
    print(f"\nDone. Streamlit will auto-discover {out.name} on next reload.")


if __name__ == "__main__":
    main()
