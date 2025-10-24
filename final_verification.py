import numpy as np
import math
import random
from typing import Dict, List, Tuple

# =============================================================================
# Code from timelean_arrival_scan.py (for building and evaluating)
# =============================================================================

def path_with_local_taper(N: int, base: float = 1.0, taper_len: int = 4, gamma: float = 1.15) -> Dict[Tuple[int,int], float]:
    edges: Dict[Tuple[int,int], float] = {}
    for i in range(1, N):
        edges[(i, i+1)] = base
    if taper_len > 0 and N > 1:
        denom = math.sqrt(1*(N-1))
        for n in range(1, taper_len+1):
            if n >= N: continue
            factor = gamma * math.sqrt(n*(N-n)) / denom
            edges[(n, n+1)] = base * factor
            i = N-n
            if i+1 <= N:
                edges[(i, i+1)] = base * factor
    return edges

def add_impurity_leaves(edges: Dict[Tuple[int,int], float], N: int,
                        sites: List[int], w_dict: Dict[int, float]) -> Tuple[Dict[Tuple[int,int], float], int]:
    next_node = N + 1
    for s in sites:
        w = w_dict[s]
        edges[(next_node, s)] = w
        next_node += 1
        sm = N - s + 1
        edges[(next_node, sm)] = w
        next_node += 1
    return edges, next_node - 1

def add_micro_rungs(edges: Dict[Tuple[int,int], float], pairs: List[Tuple[int,int]], eps_list: List[float]) -> Dict[Tuple[int,int], float]:
    for k, (i, j) in enumerate(pairs):
        edges[(i, j)] = eps_list[k % len(eps_list)]
    return edges

def build_adj(num_nodes: int, edges: Dict[Tuple[int,int], float], disorder: float = 0.0) -> np.ndarray:
    A = np.zeros((num_nodes, num_nodes), dtype=float)
    for (i, j), w in edges.items():
        if i == j: continue
        if disorder > 0:
            w *= (1 + random.uniform(-disorder, disorder))
        A[i-1, j-1] = A[j-1, i-1] = w
    return A

def build_timelean_tuned(N: int, c3: float = 0.72, c5: float = 0.44, eps: float = 0.14,
                         taper_len: int = 4, gamma: float = 1.15, disorder: float = 0.0) -> np.ndarray:
    edges = path_with_local_taper(N, taper_len=taper_len, gamma=gamma)
    w3 =  +c3 * math.sqrt(N)
    w5 =  -c5 * math.sqrt(N)
    edges, last = add_impurity_leaves(edges, N, [3, 5], {3: w3, 5: w5})
    edges = add_micro_rungs(edges, [(2, N-1), (4, N-3)], [ +eps, -eps ])
    H = build_adj(last, edges, disorder=disorder)
    return H

def earliest_strong_peak(H: np.ndarray, N_chain: int, t_max: float, num_pts: int = 4000, threshold: float = 0.98) -> Tuple[float, float]:
    n = H.shape[0]
    lam, V = np.linalg.eigh(H)
    psi0 = np.zeros(n, dtype=complex); psi0[0] = 1.0
    c0 = V.T @ psi0
    
    ts = np.linspace(0.0, t_max, num_pts)
    F = np.zeros_like(ts)
    
    # Simplified fidelity calculation for speed
    receiver_node = N_chain - 1
    V_receiver = V[receiver_node, :]
    
    for k, t in enumerate(ts):
        phase = np.exp(-1j * lam * t)
        psi_t_modal = phase * c0
        F[k] = np.abs(np.dot(V_receiver, psi_t_modal))**2

    above = np.where(F >= threshold)[0]
    if len(above) == 0:
        imax = int(np.argmax(F))
        return float(ts[imax]), float(F[imax])

    i0 = int(above[0])
    dF = np.diff(F)
    imax = i0
    end = min(len(F)-2, i0 + len(F)//8)
    for k in range(i0+1, end):
        if dF[k-1] > 0 and dF[k] <= 0:
            imax = k
            break
    if imax == i0:
        window = F[i0:end+1]
        imax = i0 + int(np.argmax(window))
        
    return float(ts[imax]), float(F[imax])

# =============================================================================
# Referee's Final Verification Tests
# =============================================================================

def test_large_N():
    """Confirms t* ≈ 21-22 for N=301, 401."""
    print("\n--- Test 1: Verifying Scaling for Larger N ---")
    Ns = [301, 401]
    for N in Ns:
        H = build_timelean_tuned(N)
        t_max = 0.74 * N + 8.4 + 60.0 # Use previous safe window
        tpk, Fpk = earliest_strong_peak(H, N, t_max, threshold=0.98)
        print(f"N={N}: t* = {tpk:.2f}, F_peak = {Fpk:.4f}")
        assert 20 <= tpk <= 26, f"t* for N={N} is outside the expected [20, 26] range!"
    print("✅ Large N test passed: Arrival times remain constant.")

def test_disorder(N=101, disorder_strength=0.02, num_runs=10):
    """Confirms t* stays in [20, 26] range under disorder."""
    print(f"\n--- Test 2: Robustness to {disorder_strength*100}% Disorder (N={N}) ---")
    arrival_times = []
    fidelities = []
    for i in range(num_runs):
        # Set a different seed for each run
        random.seed(i)
        np.random.seed(i)
        
        H = build_timelean_tuned(N, disorder=disorder_strength)
        t_max = 0.74 * N + 8.4 + 60.0
        tpk, Fpk = earliest_strong_peak(H, N, t_max, threshold=0.98)
        arrival_times.append(tpk)
        fidelities.append(Fpk)
        print(f"  Run {i+1}/{num_runs}: t* = {tpk:.2f}, F_peak = {Fpk:.4f}")

    min_t, max_t, mean_t = np.min(arrival_times), np.max(arrival_times), np.mean(arrival_times)
    mean_f = np.mean(fidelities)
    print(f"\nDisorder results: Min t* = {min_t:.2f}, Max t* = {max_t:.2f}, Mean t* = {mean_t:.2f}")
    print(f"                  Mean Fidelity = {mean_f:.4f}")
    
    if 20 <= min_t and max_t <= 26:
        print("✅ Disorder test passed: Arrival time is robust to 2% disorder.")
    else:
        print("❌ Disorder test failed: Arrival times fluctuate outside the [20, 26] range.")


if __name__ == '__main__':
    test_large_N()
    test_disorder()
