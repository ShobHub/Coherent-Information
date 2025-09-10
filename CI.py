'''
## 1. What the code is for

The goal is to estimate the coherent information I_c(p) of your code when qubits go through a bit-flip/phase-flip channel with probability $p$.

* This tells you how much quantum information the code can still carry.
* The script does this by Monte Carlo sampling: it simulates many random error patterns and counts how syndromes and logical errors appear.

---

## 2. Key objects

* **$H_Z$** → matrix that describes all **Z stabilizer checks** (rows = checks, columns = qubits).
* **$l_Z$** → vector that describes the **logical Z operator** (which qubits it touches).
* **$m_0$** → reference error pattern (you can set it to zeros if you like).
* **$k$** → number of logical qubits (for your Möbius strip, it’s $k=1$).

---

## 3. The sampling step

For each simulated error pattern:

1. **Draw random errors**: Each qubit flips with probability $p$.
   → This gives a binary vector $m$.
2. **Compute the syndrome**: Multiply $H_Z m \bmod 2$.
   → This tells you which stabilizers detect an error.
3. **Check logical error**: Compare $m - m_0$ with $l_Z$.
   → If they overlap oddly, then the error flips the logical qubit (so $x_X=1$); otherwise $x_X=0$.
4. **Store the pair $(\sigma_Z, x_X)$**.
   → That’s the data we need to estimate conditional probabilities.

---

## 4. Estimating entropy

## 5. Coherent information formula

## 6. Sweeping over error probability $p$
'''

import numpy as np
from collections import Counter
from typing import Tuple, Iterable, Dict, List

# -------------------------------
# Entropy estimators
# -------------------------------

def plug_in_entropy(prob: np.ndarray) -> float:
    p = prob[prob > 0.0]
    return 0.0 if p.size == 0 else float(-(p * np.log2(p)).sum())

def miller_madow_correction(num_bins_observed: int, num_samples: int) -> float:
    if num_samples <= 0 or num_bins_observed <= 1:
        return 0.0
    return (num_bins_observed - 1) / (2.0 * num_samples * np.log(2))

def histogram_probs(counts: Dict, N: int) -> np.ndarray:
    return np.array([c / N for c in counts.values()], dtype=float)

def conditional_entropy_from_joint(counts_joint: Dict[Tuple, int], N: int, key_left_len: int = 1) -> float:
    """
    Compute H(A|B) from joint counts over tuples (A,B).
    Here A is first key_left_len entries of the tuple.
    """
    cond_groups: Dict[Tuple, Counter] = {}
    for key, c in counts_joint.items():
        A = key[:key_left_len]
        B = key[key_left_len:]
        cond_groups.setdefault(B, Counter())
        cond_groups[B][A] += c

    H = 0.0
    mm_bias = 0.0
    for B, cnts in cond_groups.items():
        Nb = sum(cnts.values())
        if Nb == 0:
            continue
        pb = Nb / N
        probs = histogram_probs(cnts, Nb)
        Hb = plug_in_entropy(probs)
        H += pb * Hb
        mm_bias += pb * miller_madow_correction(len(cnts), Nb)
    return H + mm_bias

# -------------------------------
# Möbius/cylinder strip (d=2)
# -------------------------------

def build_mobius_code(L: int, w: int) -> Tuple[np.ndarray, np.ndarray, int]:
    assert L >= 2 and w >= 2

    nH = L * (w - 1)  # horizontal edges
    nV = L * w  # vertical edges
    n = nH + nV

    def idxH(y, x):
        return y * L + (x % L)

    def idxV(y, x):
        return nH + y * L + (x % L)

    # Z-check matrix
    rows = []
    for y in range(w - 1):
        for x in range(L):
            row = np.zeros(n, dtype=np.uint8)
            row[idxH(y, x)] ^= 1
            row[idxH(y, x - 1)] ^= 1
            row[idxV(y, x)] ^= 1
            row[idxV(y + 1, x)] ^= 1
            rows.append(row)
    H_Z = np.array(rows, dtype=np.uint8)

    # Logical Z along vertical edges in second row
    l_Z = np.zeros(n, dtype=np.uint8)
    y0 = w//2
    for x in range(L):
        l_Z[idxV(y0, x)] = 1

    return H_Z, l_Z, n

def build_mobius_code_dual(L: int, w: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build H_X (plaquette checks) and l_X (vertical logical X) for a d=2 strip
    (periodic in x, open in y) with horizontal edges = L*(w-1), vertical edges = L*w.
    """
    nH = L * (w - 1)  # horizontal edges
    nV = L * w  # vertical edges
    n = nH + nV

    def idxH(y, x):
        return y * L + x  # horizontal edges indexed by y (row), x (column)

    def idxV(y, x):
        return nH + y * L + x  # vertical edges indexed by y (row), x (column)

    rows = []

    for y in range(w):
        for x in range(L):
            row = np.zeros(n, dtype=np.uint8)

            if y == 0:  # top row
                row[idxH(x, 0) if L == 1 else x] = 1  # top horizontal
                row[idxV(y, x)] = 1  # left vertical
                row[idxV(y, (x + 1) % L)] = 1  # right vertical
            elif y == w - 1:  # bottom row
                row[idxH(y - 1, x)] = 1  # bottom horizontal
                row[idxV(y, x)] = 1  # left vertical
                row[idxV(y, (x + 1) % L)] = 1  # right vertical
            else:  # middle row
                row[idxH(y - 1, x)] = 1  # top horizontal
                row[idxH(y, x)] = 1  # bottom horizontal
                row[idxV(y, x)] = 1  # left vertical
                row[idxV(y, (x + 1) % L)] = 1  # right vertical

            rows.append(row)

    H_X = np.array(rows, dtype=np.uint8)

    # Logical X along first vertical column
    l_X = np.zeros(n, dtype=np.uint8)
    for y in range(w):
        l_X[idxV(y, 0)] = 1

    return H_X, l_X, n

class CoherentInfoMC:
    """
    Estimate I_c(p) for CSS codes under i.i.d. X-errors (bit-flip) or full symmetric BSC.
    - If bitflip_only=True: I_c = k - H(X_L | Σ_Z).
    - If bitflip_only=False: symmetric BSC -> I_c = k - 2 H(X_L|Σ_Z)
      (approximation for independent X/Z errors).
    """
    def __init__(self, H_full: np.ndarray, l_op: np.ndarray, k: int = 1,
                 use_destabilisers: bool = True, dual: bool = False,
                 bitflip_only: bool = False, L: int | None = None, w: int | None = None):
        assert H_full.dtype == np.uint8 and l_op.dtype == np.uint8
        self.H_full = H_full
        self.l_op = l_op
        self.n = l_op.shape[0]
        self.k = int(k)
        self.bitflip_only = bool(bitflip_only)

        # Store geometry if given
        self.L = L
        self.w = w

        # ---------------------------
        # All rows are independent: skip row reduction
        # ---------------------------
        self.H_ind = self.H_full.copy()
        self.row_keep = np.arange(self.H_full.shape[0], dtype=int)
        self.r = self.H_full.shape[0]
        self.pivot_cols = list(range(self.r))

        # ---------------------------
        # Build destabiliser matrix R if requested
        # ---------------------------
        if use_destabilisers:
            if dual:
                # Optional: implement build_plaquette_destabilisers() if needed
                self.R = self.build_plaquette_destabilisers()
            else:
                self.R = self.build_vertex_destabilisers()

    def build_vertex_destabilisers(self) -> np.ndarray:
        n = self.n
        r = self.H_ind.shape[0]
        R_vertex = np.zeros((n, r), dtype=np.uint8)

        L, w, nH = self.L, self.w, self.L * (self.w - 1)

        # Vertical edge index
        def idxV(y, x):
            return nH + y * L + (x % L)

        # Logical X y-range
        logical_indices = np.where(self.l_op == 1)[0]
        logical_ys = (logical_indices - nH) // L
        y_min, y_max = logical_ys.min(), logical_ys.max()
        logical_edges = set(logical_indices)

        for j in range(r):
            y_j, x_j = divmod(j, L)
            dest = np.zeros(n, dtype=np.uint8)

            # Determine path direction
            if y_j < y_min:
                # Top → vertex
                path_ys = range(0, y_j + 1)
            elif y_j > y_max:
                # Bottom → vertex
                path_ys = range(w - 1, y_j, -1)
            else:
                # Vertex overlaps logical X → go bottom but avoid logical X
                path_ys = [y for y in range(w - 1, y_j - 1, -1) if idxV(y, x_j) not in logical_edges]

            # Fill path, skipping logical edges
            for y in path_ys:
                edge = idxV(y, x_j)
                if edge not in logical_edges:
                    dest[edge] = 1

            R_vertex[:, j] = dest

        rows, cols = np.where(R_vertex == 1)
        for r, c in zip(rows, cols):
            print(f"1 at row {r}, col {c}")

        return R_vertex

    def m0_from_sigma(self, sigma_reduced: np.ndarray) -> np.ndarray:
        """
        Compute the canonical error consistent with measured independent stabilizer syndrome.
        """
        return ((self.R @ sigma_reduced) % 2).astype(np.uint8)

    def sample_trial(self, p: float, rng: np.random.Generator) -> Tuple[Tuple[int, ...], int]:
        """
        Sample a random error m with probability p, compute its syndrome sigma,
        reconstruct canonical error m0 from independent stabilizers,
        and compute residual logical operator value x.
        """
        # Random error string
        m = (rng.random(self.n) < p).astype(np.uint8)

        # Full syndrome
        sigma_full = (self.H_full @ m) % 2

        # Independent stabilizer syndrome
        sigma_ind = sigma_full[self.row_keep]

        # Canonical error consistent with measured syndrome
        m0 = self.m0_from_sigma(sigma_ind)

        # Residual error
        diff = (m + m0) % 2  # addition mod 2 = XOR

        # Logical operator measurement
        x = int((diff @ self.l_op) % 2)

        return tuple(sigma_ind.tolist()), x

    def estimate_H_X_given_S(self, p: float, num_samples: int, seed: int | None = None) -> float:
        rng = np.random.default_rng(seed)
        joint = Counter()
        for _ in range(num_samples):
            sigma_ind, x = self.sample_trial(p, rng)
            joint[(x,) + sigma_ind] += 1
        print(joint)
        return conditional_entropy_from_joint(joint, num_samples, key_left_len=1)

    def coherent_info(self, p: float, num_samples: int, seed: int | None = None) -> float:
        """
        Return I_c for this noise model:
          - if bitflip_only: I_c = k - H(X_L | Σ_Z)
          - else: symmetric BSC approx: I_c = k - 2 H(X_L | Σ_Z)
        """
        H_XgS = self.estimate_H_X_given_S(p, num_samples, seed)
        if self.bitflip_only:
            return float(self.k) - 1.0 * H_XgS
        else:
            # print('$')
            return float(self.k) - 2.0 * H_XgS

def sweep_p_Ic(code: CoherentInfoMC, p_values: Iterable[float], num_samples: int, seed: int | None = None):
    p_list = np.array(list(p_values), dtype=float)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=len(p_list))
    Ic = np.zeros_like(p_list)
    for i, (p, s) in enumerate(zip(p_list, seeds)):
        Ic[i] = code.coherent_info(p, num_samples=num_samples, seed=int(s))
    return p_list, Ic

# -------------------------------
# Run & plot multiple sizes
# -------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Choose several sizes (roughly increasing distance ~ w) to see finite-size crossings
    sizes = [(5, 5, 5, 5)]
    k = 1
    # p grid (bit-flip/BSC); near threshold ~0.11 so cover 0.02..0.18
    p_grid = np.linspace(0.0, 0.25, 15)
    num_samples = 1000  # per p; increase for smoother monotone curves
    seed = 42

    plt.figure()
    legends = []
    for (L, w, l, W) in sizes:
        H_Z, l_Z, n = build_mobius_code(L, w)
        print(H_Z,H_Z.shape,l_Z,n)
        code = CoherentInfoMC(H_Z, l_Z, k=1, use_destabilisers=True, dual=False, L=L, w=w)
        p_list, Ic_z = sweep_p_Ic(code, p_grid, num_samples=num_samples, seed=seed)

        # Dual picture (Z-errors detected by X-plaquettes)
        H_X, l_X, n = build_mobius_code_dual(L, w)
        print(H_X,H_X.shape,l_X,n)
        code = CoherentInfoMC(H_X, l_X, k=1, use_destabilisers=True, dual=True)
        p_list, Ic_x = sweep_p_Ic(code, p_grid, num_samples=num_samples, seed=seed)

        print(p_list, Ic_z)
        plt.plot(p_list, Ic_z, marker='o')
        legends.append(f"L={l}, w={W}")

    plt.axhline(0.0, linestyle='--', linewidth=1)
    plt.xlabel("Physical error rate p (bit-flip)")
    plt.ylabel("Coherent information $I_c$")
    plt.title(f"$I_c(p)$ on d=2 strip (samples={num_samples} per p)")
    plt.grid(True, linestyle=':')
    plt.legend(legends, title="Sizes", loc="best")
    plt.tight_layout()
    plt.show()

'''
[0.         0.01785714 0.03571429 0.05357143 0.07142857 0.08928571
 0.10714286 0.125      0.14285714 0.16071429 0.17857143 0.19642857
 0.21428571 0.23214286 0.25      ] 
 
 [ 1.          0.93679273  0.78942731  0.59458185  0.37652388  0.15790775
 -0.05161777 -0.24257971 -0.41024489 -0.55223954 -0.6689805  -0.76114525
 -0.83309622 -0.88654776 -0.92567439]
 
 [ 1.          0.99074624  0.92747075  0.77687587  0.54444582  0.2646066
 -0.02439979 -0.28826754 -0.50798989 -0.67643478 -0.79565897 -0.87551452
 -0.92485012 -0.95377226 -0.96937676]
 
 [1.         0.99982294 0.99825186 0.99656882 0.9968508  0.99824935
 0.99933    0.99975712 0.9999242  0.99996952 0.99998585 0.99999401
 0.99999456 0.99999673 0.99999619]
 '''
