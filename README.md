# WMM Isaac Sim — Wheeled Mobile Manipulator Controller

Controllers for a 4-DOF wheeled mobile manipulator (WMM) in NVIDIA Isaac Sim 5.0.  
Two control laws are implemented and compared: a simplified baseline and a Barrier Lyapunov Function (BLF) variant that enforces hard tracking-error bounds.

---

## Repository structure

```
wmm-isaac-sim/
├── controller/
│   ├── wmm_simple_isaacsim.py   # Simplified controller (baseline)
│   ├── wmm_blf_isaacsim.py      # BLF controller with error bounds
│   └── wmm_isaacsim.py          # Adaptive RBF-NN controller
├── robot/
│   ├── urdf/                    # URDF + STL meshes
│   └── usd/                     # USD scene files for Isaac Sim
├── matlab_reference/            # Reference MATLAB simulation scripts
├── results/
│   ├── simple/                  # Plots from baseline controller
│   └── blf/                     # Plots from BLF controller
└── docs/
    └── WMM_Controller_Report.md # Full technical report
```

---

## Requirements

- NVIDIA Isaac Sim 5.0 (with bundled Python at `<isaac_root>/python.sh`)
- NumPy, Matplotlib (included in Isaac Sim's Python environment)

---

## Running the controllers

From the Isaac Sim installation root (e.g. `/home/<user>/isaacsim5.0`):

```bash
# Baseline simplified controller
./python.sh <path-to-repo>/controller/wmm_simple_isaacsim.py

# BLF controller (enforces hard error bounds)
./python.sh <path-to-repo>/controller/wmm_blf_isaacsim.py

# Adaptive RBF-NN controller
./python.sh <path-to-repo>/controller/wmm_isaacsim.py
```

Add `--headless` to run without the GUI (supported by `wmm_simple_isaacsim.py` and `wmm_blf_isaacsim.py`).

Plots are saved to `results/simple/` or `results/blf/` inside this repo.  
Console output is also written to `/tmp/wmm_simple.txt` or `/tmp/wmm_blf.txt`.

---

## Key results

| Metric | Simple controller | BLF controller |
|--------|------------------|----------------|
| EE-x RMSE | 6.9 cm | 6.7 cm |
| EE-y RMSE | 5.0 cm | 5.2 cm |
| Base-x RMSE | 5.9 cm | 6.2 cm |
| Base-y RMSE | 1.5 cm | 1.6 cm |
| Error bound violated? | N/A | No (max 51% usage) |

The BLF controller guarantees `|e1(t)| < k_b(t)` for all time, with the time-varying barrier decaying from 0.8 m to 0.35 m over ≈15 s.

---

## Robot

4-DOF task space: `X = [EE_x, EE_y, base_x, base_y]`  
Actuators: 2 differential-drive wheel pairs + 2-link planar arm  
URDF: `robot/urdf/robot_wheel_0904.urdf`
