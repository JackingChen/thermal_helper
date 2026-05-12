# Thermal Inference — OMEN1526

PINN inference for the OMEN1526 heat-sink solid temperature field, trained
with PhysicsNeMo-Sym using a `FourierNetArch` network.

## Directory layout

```
Thermal_inference/
    infer_thermal_solid.py      inference script
    requirements_infer.txt      pip requirements (venv install)
    model/
        thermal_solid_network.0.pth   pre-trained checkpoint
    training_script_4_reference/      original training scripts (read-only reference)
```

## Do I need the flow or pressure network?

No. `thermal_solid_network` maps `(x, y, z)` → `theta_s` (non-dimensional
solid temperature) directly. The flow / pressure networks are only coupled
during training; at inference they are not needed.

---

## Option A — Docker (recommended, no local setup required)

The `data-service.inventec.com:1443/physicsnemo:latest` image ships PhysicsNeMo
and all dependencies pre-installed.

```bash
# default 20x10x20 grid over the heat-sink region
docker run --rm --gpus all -it \
  -v /home/jack/thermal_helper/backends/Thermal_inference:/workspace \
  -w /workspace \
  data-service.inventec.com:1443/physicsnemo:latest \
  /bin/bash

# custom grid resolution
docker run --rm --gpus all \
  -v /home/jack/thermal_helper/backends/Thermal_inference:/workspace \
  -w /workspace \
  data-service.inventec.com:1443/physicsnemo:latest \
  python infer_thermal_solid.py --nx 64 --ny 32 --nz 64

# single physical point (metres)
docker run --rm --gpus all \
  -v /home/jack/thermal_helper/backends/Thermal_inference:/workspace \
  -w /workspace \
  data-service.inventec.com:1443/physicsnemo:latest \
  python infer_thermal_solid.py --point 0.09 0.025 0.027

# save results as .npy
docker run --rm --gpus all \
  -v /home/jack/thermal_helper/backends/Thermal_inference:/workspace \
  -w /workspace \
  data-service.inventec.com:1443/physicsnemo:latest \
  python infer_thermal_solid.py --nx 64 --ny 32 --nz 64 --save-npy results.npy
```

> **Tip:** add `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864` for
> large grids to avoid SHMEM allocation warnings from PhysicsNeMo.

---

## Option B — Local Python venv

### Requirements

- Python 3.10–3.12
- CUDA-capable GPU (tested: RTX 2060 Super, CUDA 12.4)
- `nvidia-physicsnemo-sym` has a build-time dependency on **Cython** and
  **numpoly**; these must be installed before the main package.

### Install

```bash
cd Thermal_inference

python3 -m venv venv
source venv/bin/activate

# 1. PyTorch (CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Build dependencies first (order matters)
pip install numpy Cython
pip install numpoly --no-build-isolation

# 3. PhysicsNeMo-Sym
pip install nvidia-physicsnemo-sym --no-build-isolation
```

### Run

```bash
source venv/bin/activate

python infer_thermal_solid.py                        # default 20x10x20 grid
python infer_thermal_solid.py --nx 64 --ny 32 --nz 64
python infer_thermal_solid.py --point 0.09 0.025 0.027
python infer_thermal_solid.py --save-npy results.npy
```

---

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt PATH` | `model/thermal_solid_network.0.pth` | Path to checkpoint |
| `--device` | auto (cuda if available) | `cuda` or `cpu` |
| `--nx INT` | 20 | Grid points along x (flow direction) |
| `--ny INT` | 10 | Grid points along y (fin height) |
| `--nz INT` | 20 | Grid points along z (channel width) |
| `--batch INT` | 65536 | Inference batch size |
| `--point X Y Z` | — | Single physical point in metres |
| `--save-npy PATH` | — | Save results to `.npy` file |

---

## Coordinate system and non-dimensionalisation

| Quantity | Physical | Non-dimensional |
|----------|----------|-----------------|
| Length scale | 0.01 m | 1.0 |
| x (flow dir) | 0 – 32.5 mm | 0.0 – 3.25 |
| y (fin height) | 0 – 6 mm | 0.0 – 0.60 |
| z (channel width) | 0 – 57 mm | 0.0 – 5.70 |
| Heat sink x range | 15 – 115 mm | 0.15 – 1.15 |

The grid defaults to the heat-sink x range. Single `--point` arguments are
given in **metres** and converted internally.

### Output conversion

```
T_solid [degC] = theta_s_nd * 273.15
```

The training reference temperature is 0 °C (inlet coolant).
