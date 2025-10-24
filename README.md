# Constant-Time Quantum State Transfer via Edge Localization

Code accompanying the paper:

**"Constant-time quantum state transfer in static nearest-neighbor spin chains via edge localization"**  
by Robert Mereau  
University of Calgary

## Overview

This repository contains Python code to reproduce the results demonstrating that quantum state transfer in static, nearest-neighbor spin chains can occur in **constant time** (independent of chain length). The mechanism relies on edge-state localization induced by √N-scaled boundary impurities.

## Key Results

- Arrival times: **t* ≈ 22 ± 2** (in units ℏ/J₀)
- Fidelities: **F ≥ 0.95** for N ≥ 41, **F ≥ 0.99** for N ≥ 61
- Edge localization: **>99%** spatial confinement
- Scaling regime: Constant time across N = 21 to 201

## Construction Details

The network combines three synergistic elements:
1. **Dual opposite-sign impurities** at sites 3 and 5 from each end with w₃ = +c₃√N, w₅ = -c₅√N
2. **Short edge taper** using Krawtchouk-polynomial-inspired bond rescaling
3. **Weak parity-alternating rungs** connecting mirror sites

## Files

### Core Simulation
- `timelean_arrival_scan.py` - Main simulation script for scanning arrival times across chain lengths
- `timelean_arrival_summary.csv` - Pre-computed results data

### Validation & Testing
- `validate_physics.py` - Physics validation tests (Hermiticity, norm conservation, edge localization)
- `final_verification.py` - Referee verification tests (large N scaling, disorder robustness)

## Installation

### Requirements
- Python 3.7+
- NumPy
- Matplotlib (optional, for figure generation)

### Quick Install
```bash
pip install numpy matplotlib
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Scan
Run the main simulation across default chain lengths (N = 21, 41, ..., 201):

```bash
python timelean_arrival_scan.py
```

This generates `timelean_arrival_summary.csv` with arrival times and fidelities.

### Custom Parameters
Scan specific chain lengths with custom parameters:

```bash
python timelean_arrival_scan.py --Ns 21 41 61 81 101 \
    --c3 0.72 --c5 0.44 --eps 0.14 \
    --threshold 0.95 --save-fig
```

### Generate Figure
Create the scaling analysis figure:

```bash
python timelean_arrival_scan.py --save-fig --figure scaling_analysis.pdf
```

### Run Validation Tests
Verify physics constraints and edge localization:

```bash
python validate_physics.py
```

Expected output:
```
✅ Test 1: Hermiticity passed.
✅ Test 2: Norm conservation passed.
✅ Test 3: F(0) is zero as expected.
✅ Test 4: Max Fidelity is not > 1.
✅ Smoking Gun Test: Dominant transport mode is highly edge-localized.
```

### Run Verification Tests
Test large-N scaling and disorder robustness:

```bash
python final_verification.py
```

This confirms:
- Constant arrival times for N = 301, 401
- Robustness to 2% coupling disorder

## Parameters

Default tuned parameters (optimized for constant-time transfer):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `c3` | 0.72 | Impurity coupling coefficient at site 3 |
| `c5` | 0.44 | Impurity coupling coefficient at site 5 (opposite sign) |
| `eps` | 0.14 | Micro-rung coupling strength |
| `gamma` | 1.15 | Edge taper scale factor |
| `taper_len` | 4 | Number of tapered bonds from each end |
| `threshold` | 0.95 | Fidelity threshold for peak detection |

## Key Functions

### `build_timelean_tuned(N, c3, c5, eps, taper_len, gamma)`
Constructs the time-lean Hamiltonian matrix for chain length N.

### `earliest_strong_peak(H, N_chain, t_max, num_pts, threshold)`
Finds the earliest arrival time exceeding the fidelity threshold.

### `scan_sizes(Ns, ...)`
Scans multiple chain lengths and returns arrival statistics.

## Example: Single Chain Analysis

```python
import numpy as np
from timelean_arrival_scan import build_timelean_tuned, earliest_strong_peak

# Build Hamiltonian for N=101
N = 101
H = build_timelean_tuned(N, c3=0.72, c5=0.44, eps=0.14)

# Find arrival time
t_max = 100.0  # Maximum time window
t_peak, F_peak, mode = earliest_strong_peak(H, N, t_max, threshold=0.95)

print(f"N={N}: t*={t_peak:.2f}, F={F_peak:.4f}")
# Expected: N=101: t*≈22, F≈0.99
```

## Units and Conventions

- **Energy units**: All couplings in units of bulk coupling J₀ = 1
- **Time units**: ℏ/J₀ (with ℏ = 1)
- **No N-dependent renormalization**: Times reported in absolute units
- **Site indexing**: Sites labeled 1 through N (1-based)
- **Boundary couplings**: Scale as √N (not uniformly bounded)

## Physical Interpretation

The constant-time transfer arises through **edge-state localization**:

1. √N-scaled impurities create quantum wells at chain boundaries
2. Transport eigenstates become >99% localized within O(1) sites of edges
3. Effective transport distance d_eff = O(1), not O(N)
4. Transfer time t* ~ π/Δ remains constant as N grows

This is **fully compatible with Lieb-Robinson bounds** because the relevant propagation distance is d_eff, not the physical chain length N.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Mereau2025,
  title={Constant-time quantum state transfer in static nearest-neighbor spin chains via edge localization},
  author={Mereau, Robert},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Contact

Robert Mereau  
University of Calgary  
Email: mereau@whitewhale.ai

## Acknowledgments

This research was conducted without external funding. Literature search and contextualization were performed using AI-assisted research tools after completion of the core theoretical analysis.
