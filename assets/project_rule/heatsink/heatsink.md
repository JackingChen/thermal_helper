# Heatsink Project — Thermal Design Rules

## Project overview
Air-cooled CPU heatsink module with copper base plate, five sintered copper heat pipes (HP-1 to HP-5),
five cross-bar connectors (HP-xbar-1 to HP-xbar-5), and an aluminum fin array (FIN-array).
The heatsink sits on a 60 × 72 mm board; the CPU die is centred under the base plate.

## Baseline (copper)
- CU-BASE, HP-1 to HP-5, HP-xbar-1 to HP-xbar-5: material = copper (conductivity ~400 W/m·K)
- FIN-array: material = aluminum (conductivity ~160 W/m·K)
- Thermal prediction: PINN model (copper, z_idx=28) — T range 25–29°C across fins

## Optimized (aluminum upgrade)
- All copper heatsink parts replaced with high-grade aluminum alloy
- Reduces weight by ~65 % with only ~5 % reduction in thermal conductivity at component level
- PINN model switches to the aluminum model (z_idx=0) — T range 22–35°C; higher peak reflects higher
  chip-side temperature gradient but lighter thermal mass and faster transient response
- Recommended when weight, cost, or vibration resistance is the primary concern

## Design constraints
- FIN-array always aluminum (unchanged between baseline and optimized)
- CPU-Substrate, CPU-Lid, CPU-Die, TIM1, PTM7900 positions are fixed — do not move
- Fin orientation must remain parallel to airflow (Y-axis of board)
- Heatsink base clearance from board edge ≥ 2 mm
