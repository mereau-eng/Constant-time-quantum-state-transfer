#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-lean mirror-symmetric spin chain: earliest strong arrival time scanner.

- Static, nearest-neighbor model in the single-excitation subspace.
- Construction:
  (i)   dual edge impurities (leaves) at sites 3 and 5 per end with opposite signs
        w3 = +c3 * sqrt(N),  w5 = -c5 * sqrt(N)
  (ii)  short edge taper on first L bonds with Krawtchouk-like factors
  (iii) weak parity-alternating micro-rungs: (2 <-> N-1) = +eps, (4 <-> N-3) = -eps
- Launch at |1>, measure receiver-patch fidelity on {N, N-1} plus end leaves.
- t* = earliest local maximum after the first crossing of a fixed threshold.

Outputs:
  - CSV with N, t_peak, F_peak, detection_mode
  - (optional) PDF figure of scaling with linear & power-law fits.

Dependencies: numpy, matplotlib (optional for plotting), argparse
"""

from __future__ import annotations
import argparse
import math
from typing import Dict, List, Tuple

import numpy as np

# matplotlib is optional; import lazily only if saving a figure
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# --------------------- Graph construction ---------------------
def path_with_local_taper(N: int, base: float = 1.0, taper_len: int = 4, gamma: float = 1.15) -> Dict[Tuple[int,int], float]:
    """Nearest-neighbor path couplings with a short Krawtchouk-like edge taper."""
    edges: Dict[Tuple[int,int], float] = {}
    # uniform backbone
    for i in range(1, N):
        edges[(i, i+1)] = base
    # short taper (mirror-applied)
    if taper_len > 0:
        denom = math.sqrt(1*(N-1))
        for n in range(1, taper_len+1):
            factor = gamma * math.sqrt(n*(N-n)) / denom
            edges[(n, n+1)] = base * factor
            i = N-n
            edges[(i, i+1)] = base * factor
    return edges


def add_impurity_leaves(edges: Dict[Tuple[int,int], float], N: int,
                        sites: List[int], w_dict: Dict[int, float]) -> Tuple[Dict[Tuple[int,int], float], int]:
    """Attach degree-1 'leaf' impurities at given sites on both ends (mirror). Return (edges, last_node_index)."""
    next_node = N + 1
    for s in sites:
        w = w_dict[s]
        # left end leaf
        edges[(next_node, s)] = w
        next_node += 1
        # right end (mirror site)
        sm = N - s + 1
        edges[(next_node, sm)] = w
        next_node += 1
    return edges, next_node - 1


def add_micro_rungs(edges: Dict[Tuple[int,int], float], pairs: List[Tuple[int,int]], eps_list: List[float]) -> Dict[Tuple[int,int], float]:
    """Add weak parity-alternating rungs near the ends."""
    for k, (i, j) in enumerate(pairs):
        edges[(i, j)] = eps_list[k % len(eps_list)]
    return edges


def build_adj(num_nodes: int, edges: Dict[Tuple[int,int], float]) -> np.ndarray:
    """Build symmetric adjacency/coupling matrix H."""
    A = np.zeros((num_nodes, num_nodes), dtype=float)
    for (i, j), w in edges.items():
        if i == j:
            continue
        A[i-1, j-1] = A[j-1, i-1] = w
    return A


def build_timelean_tuned(N: int,
                         c3: float = 0.72, c5: float = 0.44, eps: float = 0.14,
                         taper_len: int = 4, gamma: float = 1.15) -> np.ndarray:
    """
    Build the tuned 'time-lean' static Hamiltonian H (in units ħ=1).
    Couplings are expressed in the same energy units as the bulk J=1 backbone.

    NOTE on normalization and causality:
      We report times in absolute units ħ/J0 with J0=1 (bulk bond scale).
      If you want a strict 'bounded-coupling' comparison with max|J|=1 across N,
      rescale H by its max(|H_ij|) before time evolution; times will stretch accordingly.
    """
    edges = path_with_local_taper(N, taper_len=taper_len, gamma=gamma)
    w3 =  +c3 * math.sqrt(N)
    w5 =  -c5 * math.sqrt(N)  # opposite sign
    edges, last = add_impurity_leaves(edges, N, [3, 5], {3: w3, 5: w5})
    edges = add_micro_rungs(edges, [(2, N-1), (4, N-3)], [ +eps, -eps ])
    H = build_adj(last, edges)
    return H


# --------------------- Fidelity & detection ---------------------
def receiver_patch_indices(N: int, total_nodes: int) -> np.ndarray:
    """Patch = {N, N-1} on the backbone + all leaves on the right end (indices >= N). Zero-based indices."""
    right_leaves = list(range(N, total_nodes))
    patch = list(range(N-2, N)) + right_leaves
    return np.array(patch, dtype=int)


def earliest_strong_peak(H: np.ndarray, N_chain: int,
                         t_max: float, num_pts: int = 4000,
                         threshold: float = 0.60) -> Tuple[float, float, str]:
    """
    Exact matrix exponential via diagonalization:
      - Launch at |1>
      - Compute F_patch(t) on a uniform grid
      - Find first crossing of 'threshold', then the first local maximum thereafter.
    Returns: (t_peak, F_peak, mode)
    """
    n = H.shape[0]
    lam, V = np.linalg.eigh(H)                        # H = V diag(lam) V^T
    psi0 = np.zeros(n, dtype=complex); psi0[0] = 1.0
    c0 = V.T @ psi0                                   # modal amplitudes at t=0
    patch = receiver_patch_indices(N_chain, n)

    ts = np.linspace(0.0, t_max, num_pts)
    # Evaluate F(t) (vectorized-ish loop for clarity)
    F = np.empty_like(ts)
    for k, t in enumerate(ts):
        phase = np.exp(-1j * lam * t)
        psi_t = V @ (phase * c0)
        F[k] = float(np.sum(np.abs(psi_t[patch])**2))

    # First threshold crossing
    above = np.where(F >= threshold)[0]
    if len(above) == 0:
        imax = int(np.argmax(F))
        return float(ts[imax]), float(F[imax]), "global_max"

    i0 = int(above[0])
    # Find first local maximum after crossing:
    dF = np.diff(F)
    imax = i0
    end = min(len(F)-2, i0 + max(30, len(F)//8))
    for k in range(i0+1, end):
        if dF[k-1] > 0 and dF[k] <= 0:
            imax = k
            break
    if imax == i0:
        # Fallback: argmax in a window after crossing
        window = F[i0:end+1]
        imax = i0 + int(np.argmax(window))

    # Quadratic refinement around imax (if neighbors exist)
    if 1 <= imax < len(F)-1:
        x1, x2, x3 = ts[imax-1], ts[imax], ts[imax+1]
        y1, y2, y3 = F[imax-1], F[imax], F[imax+1]
        denom = (x1-x2)*(x1-x3)*(x2-x3)
        if abs(denom) > 1e-18:
            A = (x3*(y2-y1) + x2*(y1-y3) + x1*(y3-y2)) / denom
            B = (x3**2*(y1-y2) + x2**2*(y3-y1) + x1**2*(y2-y3)) / denom
            # vertex of parabola y = A x^2 + B x + C
            if A != 0:
                t_ref = -B / (2*A)
                if x1 <= t_ref <= x3:
                    # evaluate refined peak height
                    F_ref = float(A*t_ref*t_ref + B*t_ref + (y2 - A*x2*x2 - B*x2))
                    return float(t_ref), F_ref, "threshold_localmax_refined"

    return float(ts[imax]), float(F[imax]), "threshold_localmax"


# --------------------- Utilities ---------------------
def default_time_window(N: int, margin: float = 60.0) -> float:
    """
    A safe window based on the observed linear trend: ~0.74 N + 8.4, plus a margin.
    You can reduce 'margin' to speed up scans.
    """
    return 0.74 * N + 8.4 + margin


def calculate_spectral_gap(H: np.ndarray) -> float:
    """Calculates the minimal gap between eigenvalues in the transport band."""
    lam, V = np.linalg.eigh(H)
    # A simplified proxy for the transport band: eigenvalues near the center of the spectrum
    center_band_indices = np.where(np.abs(lam) < 0.5)[0]
    if len(center_band_indices) < 2:
        return np.min(np.diff(np.sort(lam))) if len(lam) > 1 else 0.0
    
    band_lambdas = np.sort(lam[center_band_indices])
    gaps = np.diff(band_lambdas)
    return np.min(gaps) if len(gaps) > 0 else 0.0


def scan_sizes(Ns: List[int],
               threshold: float = 0.60,
               taper_len: int = 4,
               gamma: float = 1.15,
               c3: float = 0.72,
               c5: float = 0.44,
               eps: float = 0.14,
               num_pts: int = 4000,
               margin: float = 60.0,
               bounded_maxJ: bool = False) -> np.ndarray:
    """
    Compute earliest strong arrival for each N in Ns.
    If bounded_maxJ=True, rescale each H so max|H_ij|=1 (times will stretch accordingly).
    Returns: array of rows [N, t_peak, F_peak, mode]
    """
    rows = []
    for N in Ns:
        H = build_timelean_tuned(N, c3=c3, c5=c5, eps=eps, taper_len=taper_len, gamma=gamma)
        if bounded_maxJ:
            maxJ = np.max(np.abs(H))
            if maxJ > 0:
                H = H / maxJ  # keeps Lieb–Robinson speed constant across N
        t_max = default_time_window(N, margin=margin)
        tpk, Fpk, mode = earliest_strong_peak(H, N, t_max, num_pts=num_pts, threshold=threshold)
        gap = calculate_spectral_gap(H)
        rows.append((N, tpk, Fpk, mode, gap))
    return np.array(rows, dtype=object)


def fit_trends(Ns: np.ndarray, Ts: np.ndarray) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    """Return (a,b) for linear t≈aN+b and (A,alpha) for power-law t≈A N^alpha."""
    # Linear
    A_mat = np.vstack([Ns, np.ones_like(Ns)]).T
    (a_lin, b_lin), *_ = np.linalg.lstsq(A_mat, Ts, rcond=None)
    # Power law
    xlog = np.log(Ns); ylog = np.log(Ts)
    (alpha, logA), *_ = np.linalg.lstsq(np.vstack([xlog, np.ones_like(xlog)]).T, ylog, rcond=None)
    A_pl = math.exp(logA)
    return (a_lin, b_lin), (A_pl, alpha)


def save_scaling_figure(path: str, Ns: np.ndarray, Ts: np.ndarray, Gaps: np.ndarray,
                        lin: Tuple[float,float], pl: Tuple[float,float]):
    if plt is None:
        return
    a_lin, b_lin = lin
    A_pl, alpha = pl
    N_grid = np.linspace(min(Ns), max(Ns), 400)
    t_lin = a_lin * N_grid + b_lin
    t_pow = A_pl * (N_grid ** alpha)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("Performance and Spectral Analysis of Time-Lean Networks")

    # Axis 1: Arrival Time
    ax1.plot(Ns, Ts, 'o', label="Observed $t_\\star$")
    ax1.plot(N_grid, t_lin, '-', label=f"Linear Fit: $t_\\star\\approx{a_lin:.3f}N+{b_lin:.2f}$")
    ax1.plot(N_grid, t_pow, '--', label=f"Power Fit: $t_\\star\\approx{A_pl:.2f}N^{{{alpha:.2f}}}$")
    ax1.set_ylabel(r"Earliest strong arrival $t_\star$")
    ax1.set_title("Arrival-Time Scaling (absolute units, $\\hbar/J_0$)")
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.legend(frameon=False, fontsize=8, loc='upper left')

    # Axis 2: Spectral Gap
    ax2.plot(Ns, Gaps, 'o-', color='purple', label="Minimal Spectral Gap $\\Delta_{min}$")
    ax2.set_xlabel("Chain length $N$")
    ax2.set_ylabel("Spectral Gap $\\Delta_{min}$")
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.legend(frameon=False, fontsize=8, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, format="pdf", bbox_inches="tight")


# --------------------- CLI ---------------------
def main():
    p = argparse.ArgumentParser(description="Scan earliest strong arrival times for the tuned static chain.")
    p.add_argument("--Ns", type=int, nargs="+",
                   default=[21, 41, 61, 81, 101, 121, 141, 161, 181, 201],
                   help="Chain lengths to evaluate.")
    p.add_argument("--threshold", type=float, default=0.60, help="Threshold for earliest-peak detection.")
    p.add_argument("--num-pts", type=int, default=4000, help="Time samples per trace.")
    p.add_argument("--margin", type=float, default=60.0, help="Extra time added to the linear window for safety.")
    p.add_argument("--taper-len", type=int, default=4, help="Edge taper length (bonds).")
    p.add_argument("--gamma", type=float, default=1.15, help="Edge taper scale factor.")
    p.add_argument("--c3", type=float, default=0.72, help="Leaf coupling coefficient at site 3 (w3=c3*sqrt(N)).")
    p.add_argument("--c5", type=float, default=0.44, help="Leaf coupling coefficient at site 5 with opposite sign (w5=-c5*sqrt(N)).")
    p.add_argument("--eps", type=float, default=0.14, help="Micro-rung magnitude (rungs alternate ±eps).")
    p.add_argument("--bounded-maxJ", action="store_true",
                   help="Rescale H so max|H_ij|=1 for each N (keeps LR speed constant across N).")
    p.add_argument("--csv", type=str, default="timelean_arrival_summary.csv", help="Output CSV path.")
    p.add_argument("--save-fig", action="store_true", help="If set, save scaling figure PDF.")
    p.add_argument("--figure", type=str, default="Fig3_scaling_regen.pdf", help="Figure PDF path (if --save-fig).")
    args = p.parse_args()

    print("\n--- Unit and Normalization Confirmation ---")
    print("Times are reported in absolute units of ħ/J0, with J0=1 (bulk bond scale).")
    print("No N-dependent normalization is applied to time, consistent with manuscript.")
    print("-------------------------------------------\n")

    rows = scan_sizes(
        args.Ns, threshold=args.threshold, taper_len=args.taper_len, gamma=args.gamma,
        c3=args.c3, c5=args.c5, eps=args.eps, num_pts=args.num_pts,
        margin=args.margin, bounded_maxJ=args.bounded_maxJ
    )

    # Save CSV
    # rows: [N, t_peak, F_peak, mode, gap]
    Ns = np.array([int(r[0]) for r in rows], dtype=int)
    Ts = np.array([float(r[1]) for r in rows], dtype=float)
    Fs = np.array([float(r[2]) for r in rows], dtype=float)
    modes = [str(r[3]) for r in rows]
    Gaps = np.array([float(r[4]) for r in rows], dtype=float)

    import csv
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "t_peak", "F_peak", "mode", "spectral_gap",
                    "threshold", "num_pts", "margin",
                    "taper_len", "gamma", "c3", "c5", "eps", "bounded_maxJ"])
        for i in range(len(Ns)):
            w.writerow([Ns[i], f"{Ts[i]:.6f}", f"{Fs[i]:.6f}", modes[i], f"{Gaps[i]:.6f}",
                        args.threshold, args.num_pts, args.margin,
                        args.taper_len, args.gamma, args.c3, args.c5, args.eps, int(args.bounded_maxJ)])

    # Fit & figure
    lin, pl = fit_trends(Ns.astype(float), Ts)
    print(f"[Linear fit]   t* ≈ {lin[0]:.4f} N + {lin[1]:.4f}")
    print(f"[Power-law]    t* ≈ {pl[0]:.4f} N^{pl[1]:.4f}")
    if args.save_fig and plt is not None:
        save_scaling_figure(args.figure, Ns.astype(float), Ts, Gaps, lin, pl)
        print(f"[Saved figure] {args.figure}")
    print(f"[Saved CSV]    {args.csv}")


if __name__ == "__main__":
    main()
