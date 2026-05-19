"""
thermal_viz_app.py
==================
Streamlit UI for OMEN1526 thermal-solid-network inference & visualisation.

Run INSIDE the physicsnemo Docker container (see run_viz.sh):
    streamlit run thermal_viz_app.py --server.port 8502 --server.address 0.0.0.0

Inference is executed IN-PROCESS (no subprocess Docker call needed because
the app itself runs inside the physicsnemo image which has all ML deps).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
WORKSPACE     = Path(__file__).parent
CKPT_PATH     = WORKSPACE / "model" / "thermal_solid_network.0.pth"
RESULTS_NPY   = WORKSPACE / "results.npy"

LENGTH_SCALE  = 0.01    # m  (1 non-dim unit = 10 mm)
TEMP_SCALE    = 273.15  # K  (T_C = theta_s_nd * 273.15)

# Domain (physical, metres)
DOMAIN_X_M   = (0.0, 0.0325)
DOMAIN_Y_M   = (0.0, 0.006)
DOMAIN_Z_M   = (0.0, 0.057)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thermal Field Inference",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ OMEN1526 — Thermal Field Inference & Visualisation")
st.caption(
    f"FourierNetArch PINN · checkpoint: `{CKPT_PATH}`"
)

# ── Lazy-load the model (cached across reruns) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading PhysicsNeMo model…")
def _load_model(ckpt: str, device_str: str):
    """Import physicsnemo and load the checkpoint once."""
    import torch
    from infer_thermal_solid import build_net, load_checkpoint
    dev = torch.device(device_str)
    net = build_net(dev)
    net = load_checkpoint(net, Path(ckpt), dev)
    return net, dev


# ── Sidebar — parameter controls ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Inference Parameters")

    mode = st.radio("Query mode", ["Grid", "Single point"], horizontal=True)

    st.divider()

    if mode == "Grid":
        st.subheader("Grid resolution")
        nx = st.slider("nx  (flow, x)",     5, 128, 30,
                       help="Heat-sink x region (15–115 mm)")
        ny = st.slider("ny  (fin height, y)", 5,  64, 12,
                       help="0–6 mm")
        nz = st.slider("nz  (channel, z)",   5, 128, 30,
                       help="0–57 mm")
        st.caption(f"Total points: **{nx * ny * nz:,}**")
    else:
        st.subheader("Physical coordinates [m]")
        x_m = st.number_input("X [m]", 0.0, DOMAIN_X_M[1], 0.065, step=0.001, format="%.4f")
        y_m = st.number_input("Y [m]", 0.0, DOMAIN_Y_M[1], 0.003, step=0.001, format="%.4f")
        z_m = st.number_input("Z [m]", 0.0, DOMAIN_Z_M[1], 0.027, step=0.001, format="%.4f")

    st.divider()
    st.subheader("Runtime")
    import subprocess, shutil
    _has_cuda = shutil.which("nvidia-smi") is not None
    device_str = st.selectbox("Device", ["cuda", "cpu"],
                              index=0 if _has_cuda else 1)
    batch_sz   = st.select_slider(
        "Batch size", options=[4096, 16384, 65536, 131072], value=65536
    )

    st.divider()
    run_btn = st.button("▶  Run Inference", type="primary", use_container_width=True)


# ── Run inference ──────────────────────────────────────────────────────────────
if run_btn:
    try:
        import torch
        from infer_thermal_solid import run_inference, make_grid
    except ImportError as e:
        st.error(
            f"**PhysicsNeMo not found**: {e}\n\n"
            "Make sure you're running this app **inside** the physicsnemo Docker container "
            "(see `run_viz.sh`)."
        )
        st.stop()

    with st.spinner("Loading model…"):
        net, dev = _load_model(str(CKPT_PATH), device_str)

    t0 = time.perf_counter()
    if mode == "Grid":
        with st.spinner(f"Running inference on {nx}×{ny}×{nz} = {nx*ny*nz:,} points…"):
            x_nd, y_nd, z_nd = make_grid(nx, ny, nz)
            theta_s_nd = run_inference(net, x_nd, y_nd, z_nd, dev, batch_sz)
        st.session_state["grid_shape"] = (nx, ny, nz)
    else:
        x_nd = np.array([x_m / LENGTH_SCALE], dtype=np.float32)
        y_nd = np.array([y_m / LENGTH_SCALE], dtype=np.float32)
        z_nd = np.array([z_m / LENGTH_SCALE], dtype=np.float32)
        theta_s_nd = run_inference(net, x_nd, y_nd, z_nd, dev, batch_sz)
        st.session_state.pop("grid_shape", None)

    T_solid_C = theta_s_nd * TEMP_SCALE
    elapsed   = time.perf_counter() - t0

    np.save(RESULTS_NPY, {
        "x_nd": x_nd, "y_nd": y_nd, "z_nd": z_nd,
        "theta_s_nd": theta_s_nd, "T_solid_C": T_solid_C,
    })
    st.session_state["last_mode"] = mode
    st.success(f"Done in {elapsed:.2f} s — results saved to `results.npy`")


# ── Load and display results ───────────────────────────────────────────────────
if not RESULTS_NPY.exists():
    st.info("No results yet — set parameters in the sidebar and click **▶ Run Inference**.")
    st.stop()

try:
    data = np.load(RESULTS_NPY, allow_pickle=True).item()
except Exception as e:
    st.error(f"Could not load results.npy: {e}")
    st.stop()

x_nd      = data["x_nd"]
y_nd      = data["y_nd"]
z_nd      = data["z_nd"]
T_solid_C = data["T_solid_C"]

x_mm = x_nd * LENGTH_SCALE * 1000
y_mm = y_nd * LENGTH_SCALE * 1000
z_mm = z_nd * LENGTH_SCALE * 1000

T_min, T_max = float(T_solid_C.min()), float(T_solid_C.max())
last_mode    = st.session_state.get("last_mode", "Grid" if len(x_nd) > 1 else "Single point")
grid_shape   = st.session_state.get("grid_shape", None)

# ── Header metrics ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Points",    f"{len(T_solid_C):,}")
c2.metric("T min",     f"{T_min:.2f} °C")
c3.metric("T max",     f"{T_max:.2f} °C")
c4.metric("T mean",    f"{T_solid_C.mean():.2f} °C")

# ── Single-point shortcut ──────────────────────────────────────────────────────
if last_mode == "Single point" or len(x_nd) == 1:
    st.header("📍 Single-point result")
    st.metric("T solid", f"{T_solid_C[0]:.2f} °C",
              help=f"({x_mm[0]:.1f} mm, {y_mm[0]:.1f} mm, {z_mm[0]:.1f} mm)")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────
st.header("📊 Temperature Field")
tab3d, tabXZ, tabXY, tabYZ = st.tabs(
    ["3-D Scatter", "XZ Slice (y-cut)", "XY Slice (z-cut)", "YZ Slice (x-cut)"]
)

# ── 3-D scatter ───────────────────────────────────────────────────────────────
with tab3d:
    MAX_PTS = 40_000
    if len(x_nd) > MAX_PTS:
        idx = np.random.choice(len(x_nd), MAX_PTS, replace=False)
        st.caption(f"Showing {MAX_PTS:,} of {len(x_nd):,} points (random sample).")
    else:
        idx = np.arange(len(x_nd))

    fig3d = go.Figure(go.Scatter3d(
        x=x_mm[idx], y=y_mm[idx], z=z_mm[idx],
        mode="markers",
        marker=dict(
            size=3,
            color=T_solid_C[idx],
            colorscale="Thermal",
            cmin=T_min, cmax=T_max,
            colorbar=dict(title="T [°C]", thickness=15),
            opacity=0.85,
        ),
        text=[f"T={t:.2f} °C" for t in T_solid_C[idx]],
        hovertemplate=(
            "x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>z=%{z:.1f} mm<br>%{text}<extra></extra>"
        ),
    ))
    fig3d.update_layout(
        scene=dict(
            xaxis_title="x [mm]  (flow)",
            yaxis_title="y [mm]  (fin height)",
            zaxis_title="z [mm]  (channel)",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
    )
    st.plotly_chart(fig3d, use_container_width=True)

# ── Structured-grid slices ────────────────────────────────────────────────────
if grid_shape is None:
    for tab in (tabXZ, tabXY, tabYZ):
        with tab:
            st.info("Re-run a **Grid** inference to enable slice views.")
else:
    GNX, GNY, GNZ = grid_shape
    try:
        X3 = x_mm.reshape(GNX, GNY, GNZ)
        Y3 = y_mm.reshape(GNX, GNY, GNZ)
        Z3 = z_mm.reshape(GNX, GNY, GNZ)
        T3 = T_solid_C.reshape(GNX, GNY, GNZ)
    except ValueError as e:
        for tab in (tabXZ, tabXY, tabYZ):
            with tab:
                st.warning(f"Shape mismatch ({e}). Re-run inference to refresh.")
        st.stop()

    # XZ slice — fixed y
    with tabXZ:
        y_idx = st.slider("y index (fin height)", 0, GNY - 1, GNY // 2, key="y_idx")
        y_val = float(Y3[0, y_idx, 0])
        st.caption(f"y ≈ **{y_val:.2f} mm**")
        fig_xz = px.imshow(
            T3[:, y_idx, :].T,
            x=X3[:, y_idx, 0], y=Z3[0, y_idx, :],
            labels=dict(x="x [mm] (flow)", y="z [mm] (channel)", color="T [°C]"),
            color_continuous_scale="Thermal", zmin=T_min, zmax=T_max, aspect="auto",
            title=f"T solid — XZ plane @ y={y_val:.2f} mm",
        )
        fig_xz.update_layout(height=450)
        st.plotly_chart(fig_xz, use_container_width=True)

    # XY slice — fixed z
    with tabXY:
        z_idx = st.slider("z index (channel width)", 0, GNZ - 1, GNZ // 2, key="z_idx")
        z_val = float(Z3[0, 0, z_idx])
        st.caption(f"z ≈ **{z_val:.2f} mm**")
        fig_xy = px.imshow(
            T3[:, :, z_idx].T,
            x=X3[:, 0, z_idx], y=Y3[0, :, z_idx],
            labels=dict(x="x [mm] (flow)", y="y [mm] (fin height)", color="T [°C]"),
            color_continuous_scale="Thermal", zmin=T_min, zmax=T_max, aspect="auto",
            title=f"T solid — XY plane @ z={z_val:.2f} mm",
        )
        fig_xy.update_layout(height=450)
        st.plotly_chart(fig_xy, use_container_width=True)

    # YZ slice — fixed x
    with tabYZ:
        x_idx = st.slider("x index (flow direction)", 0, GNX - 1, GNX // 2, key="x_idx")
        x_val = float(X3[x_idx, 0, 0])
        st.caption(f"x ≈ **{x_val:.2f} mm**")
        fig_yz = px.imshow(
            T3[x_idx, :, :],
            x=Z3[x_idx, 0, :], y=Y3[x_idx, :, 0],
            labels=dict(x="z [mm] (channel)", y="y [mm] (fin height)", color="T [°C]"),
            color_continuous_scale="Thermal", zmin=T_min, zmax=T_max, aspect="auto",
            title=f"T solid — YZ plane @ x={x_val:.2f} mm",
        )
        fig_yz.update_layout(height=450)
        st.plotly_chart(fig_yz, use_container_width=True)

# ── Download ───────────────────────────────────────────────────────────────────
st.divider()
with st.expander("⬇️  Download results"):
    import io, csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["x_mm", "y_mm", "z_mm", "T_solid_C"])
    for row in zip(x_mm.tolist(), y_mm.tolist(), z_mm.tolist(), T_solid_C.tolist()):
        writer.writerow(row)
    st.download_button("Download CSV", buf.getvalue(),
                       "thermal_results.csv", "text/csv")
    st.download_button("Download .npy", RESULTS_NPY.read_bytes(),
                       "thermal_results.npy", "application/octet-stream")
