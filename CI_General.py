import time
import math
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from collections import defaultdict


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

import numpy as np
from numpy.typing import NDArray

def finite_field_gauss_jordan_elimination(
    mat: NDArray[np.int_],
    p: int
) -> NDArray[np.int_]:
    """
    Reduced row-echelon form (RREF) of mat over GF(p).

    Args:
        mat: integer matrix (not mutated; a copy is made)
        p: prime modulus

    Returns:
        RREF(mat) modulo p as a new array.
    """
    if not is_prime(p):
        raise ValueError(f"Modulus p={p} is not prime; "
                         "Gauss-Jordan over GF(p) requires a prime p.")
    p = int(p)

    A = mat.copy() % p
    n_rows, n_cols = A.shape
    r = 0  # current pivot row

    for c in range(n_cols):
        if r >= n_rows:
            break

        # Find a pivot in or below row r
        pivot_candidates = np.nonzero(A[r:, c])[0]
        if pivot_candidates.size == 0:
            continue
        piv = r + pivot_candidates[0]

        # Swap pivot row up if needed
        if piv != r:
            A[[r, piv]] = A[[piv, r]]

        # Normalize pivot to 1
        inv_piv = pow(int(A[r, c]), -1, p)
        A[r] = (A[r] * inv_piv) % p

        # Eliminate this column in all other rows
        for rr in range(n_rows):
            if rr != r and A[rr, c] != 0:
                A[rr] = (A[rr] - A[rr, c] * A[r]) % p

        r += 1

    return A

# -------------------------------
# Möbius code construction
# -------------------------------
def build_moebius_code_vertex(L: int, w: int, d: int = 2, verbose: bool = False):
    """
    Args:
        L: length (number of vertices along the length)
        w: width  (number of vertices along the width)
        d: qudit dimension (not used here, kept for interface symmetry)
        verbose: if True, prints basic info about the matrix

    Returns:
        H_Z: (L*(w-1) x (L*(w-1) + L*w)) int16 Z-check matrix with entries in {0, ±1}.
        l_Z: the logical Z operator
        vertex_destab: vertex_destabilizers - ((w-1)*L) x (L*(w-1) + L*w) int16 matrix.
    """

    # Validity checks
    if L < 3 or w < 3:
        raise ValueError("Length and width must be at least 3.")
    if L % 2 == 0:
        raise ValueError("Length must be odd for the Moebius code.")
    if w % 2 == 0:
        raise ValueError("Width must be odd for the Moebius code.")
    if d % 2 != 0:
        raise ValueError("Dimension d must be even for the Moebius code.")

    nH = L * (w - 1)  # number of horizontal edges
    nV = L * w  # number of vertical edges
    n = nH + nV  # total number of edges

    # Horizontal edges: y in [0, w-2], x in [0, L-1]
    def idxH(y, x):
        # x % L is redundant here but keeps the periodic structure explicit
        return (y * L) + (x % L)

    # Vertical edges: y in [0, w-1], x in [0, L-1]
    def idxV(y, x):
        return nH + (y * L) + (x % L)

    rows = []
    for y in range(w - 1):
        for x in range(L):
            # start with zeros, we’ll toggle 4 edges
            row = np.zeros(n, dtype=np.int16)

            # Right horizontal edge at (y, x)
            h_right = idxH(y, x)

            # Left horizontal edge:
            # - for x > 0: same row, previous x
            # - for x = 0: Möbius twist → row (w-2 - y), x = L-1
            if x == 0:
                h_left = idxH((w - 2) - y, L - 1)
            else:
                h_left = idxH(y, x - 1)

            # Vertical edges above and below
            v_top = idxV(y, x)
            v_bottom = idxV(y + 1, x)

            # Toggle the four edges mod 2 → entries become 0 or 1
            for e in (h_right, h_left, v_top, v_bottom):
                # XOR with 1: 0→1, 1→0; stays in {0,1}
                row[e] ^= 1

            rows.append(row)

    # Stack all rows
    H_Z = np.array(rows, dtype=np.int16)
    # Impose the ±1 sign pattern: we can just flip all odd rows:
    H_Z[1::2] *= -1

    if verbose:
        print(f"H_Z shape: {H_Z.shape}")
        print(f"Non-zero per row (should all be 4):",
              np.unique(np.count_nonzero(H_Z, axis=1)))

    # Logical Z operator
    l_Z = np.zeros(n, dtype=np.int16)
    y0 = w // 2  # central horizontal row
    exp = d // 2  # integer d/2
    for x in range(L):
        l_Z[idxV(y0, x)] = exp

    # Vertex_destabilizers
    rows = []
    mid = (w - 1) // 2
    for y in range(w - 1):
        for x in range(L):
            row = np.zeros(n, dtype=np.int16)
            if y < mid:
                # upper half: sum over y' from 0 to y
                for y_prime in range(y + 1):
                    if (x + y_prime) % 2 == 0:
                        row[idxV(y_prime, x)] = 1
                    else:
                        row[idxV(y_prime, x)] = -1
            else:
                # lower half: sum over y' from y+1 to w-1
                for y_prime in range(y + 1, w):
                    if (x + y_prime) % 2 == 0:
                        row[idxV(y_prime, x)] = -1
                    else:
                        row[idxV(y_prime, x)] = 1
            rows.append(row)
    vertex_destab = np.array(rows, dtype=np.int16)

    if verbose:
        print(f"vertex_destab shape: {vertex_destab.shape}")
        print("Nonzero per row:", np.unique(np.count_nonzero(vertex_destab, axis=1)))

    return H_Z, l_Z, n, vertex_destab

def build_moebius_code_plaquette(L: int, w: int, d: int = 2) -> NDArray:
    """
    Standalone version of build_moebius_code_plaquette that reproduces the
    same H_X plaquette-check matrix as in the MoebiusCode class.

    Args:
        L: length of the strip (number of vertices along the length)
        w: width of the strip (number of vertices along the width)
        d: qudit dimension (not used here, kept for interface symmetry)

    Returns:
        H_X: (L * w) x (L*(w-1) + L*w) int16 X-check (plaquette) matrix
    """

    if L < 3 or w < 3:
        raise ValueError("Length and width must be at least 3.")
    if L % 2 == 0:
        raise ValueError("Length must be odd for the Moebius code.")
    if w % 2 == 0:
        raise ValueError("Width must be odd for the Moebius code.")
    if d % 2 != 0:
        raise ValueError("Dimension d must be even for the Moebius code.")

    # Number of edges
    nH = L * (w - 1)   # horizontal edges
    nV = L * w         # vertical edges
    n  = nH + nV       # total edges

    # Indexing helpers (matching the class methods)
    def idxH(y: int, x: int) -> int:
        # horizontal edges: y in [0, w-2], x in [0, L-1]
        return y * L + (x % L)

    def idxH_twist(y: int, x: int) -> int:
        # inverted_index_h(y, x) = (w-2 - y)*L + x
        return (w - 2 - y) * L + (x % L)

    def idxV(y: int, x: int) -> int:
        # vertical edges: y in [0, w-1], x in [0, L-1]
        return nH + y * L + (x % L)

    rows = []

    # Loop over plaquettes labelled by (y, x)
    for y in range(w):
        for x in range(L):
            row = np.zeros(n, dtype=np.int16)

            if y == 0:
                # ---------- Top row plaquettes ----------
                if (x + 1) % L != 0:
                    # Non-twisted boundary plaquettes on top row
                    if x % 2 == 0:
                        row[idxH(0, x)]      =  1
                        row[idxV(0, x)]      = -1
                        row[idxV(0, x + 1)]  = -1
                    else:
                        row[idxH(0, x)]      = -1
                        row[idxV(0, x)]      =  1
                        row[idxV(0, x + 1)]  =  1
                else:
                    # Twisted plaquette connecting top and bottom-right corner
                    row[idxH(w - 2, L - 1)]       =  1
                    row[idxV(0, 0)]               = -1
                    row[idxV(w - 1, L - 1)]       = -1

            elif 0 < y < (w - 1):
                # ---------- Central bulk plaquettes ----------
                if (x + 1) % L != 0:
                    # Non-twisted central plaquettes
                    if (x + y) % 2 == 1:
                        row[idxH(y - 1, x)]    = -1
                        row[idxH(y,     x)]    = -1
                        row[idxV(y,     x)]    =  1
                        row[idxV(y,     x + 1)] = 1
                    else:
                        row[idxH(y - 1, x)]    =  1
                        row[idxH(y,     x)]    =  1
                        row[idxV(y,     x)]    = -1
                        row[idxV(y,     x + 1)] = -1
                else:
                    # Twisted boundary plaquettes (right edge, y in (0, w-1))
                    if (x + y) % 2 == 1:
                        row[idxH_twist(y,     x)] = -1
                        row[idxH_twist(y - 1, x)] = -1
                        row[idxV(y, 0)]           =  1
                        row[idxV(w - y - 1, x)]  =  1
                    else:
                        row[idxH_twist(y,     x)] =  1
                        row[idxH_twist(y - 1, x)] =  1
                        row[idxV(y, 0)]           = -1
                        row[idxV(w - y - 1, x)]  = -1

            else:
                # ---------- Bottom row plaquettes (y == w-1) ----------
                if (x + 1) % L != 0:
                    # Non-twisted bottom plaquettes
                    if x % 2 == 0:
                        row[idxH(w - 2, x)]     =  1
                        row[idxV(w - 1, x)]     = -1
                        row[idxV(w - 1, x + 1)] = -1
                    else:
                        row[idxH(w - 2, x)]     = -1
                        row[idxV(w - 1, x)]     =  1
                        row[idxV(w - 1, x + 1)] =  1
                else:
                    # Twisted plaquette connecting bottom-left and top-right
                    row[idxH(0, L - 1)]         =  1
                    row[idxV(w - 1, 0)]         = -1
                    row[idxV(0, L - 1)]         = -1

            rows.append(row)

    H_X = np.array(rows, dtype=np.int16)
    H_X = -H_X
    # Swap rows 2 and 8 (0-based indexing) - (3,3)
    if L == 3 and w == 3:
        H_X[[2, 8]] = H_X[[8, 2]]

    # -------------------------------------------------
    # Logical X: along the first column of vertical edges
    # X_logical = X_1 ⊗ X_2^{-1} ⊗ ... ⊗ X_width
    # i.e. pattern (-1, +1, -1, +1, ...) on idxV(y, 0)
    # -------------------------------------------------
    l_X = np.zeros(n, dtype=np.int16)
    for y in range(w):
        if y % 2 == 0:
            l_X[idxV(y, 0)] = -1
        else:
            l_X[idxV(y, 0)] = 1

    # Match the class behaviour: multiply by p = d/2 if it's an odd prime (≠ 2)
    p_candidate = d // 2
    if d % 2 == 0 and is_prime(int(p_candidate)) and p_candidate != 2:
        l_X *= p_candidate

    return H_X, l_X

# -------------------------------
# Face Destabilisers
# -------------------------------
def build_plaquette_destabilizers_qubit(L: int, w: int) -> NDArray[np.int16]:
    """
    Reproduces the same qubit plaquette destabilizer matrix as in the MoebiusCode class.
    Args:
        L: length of the strip (number of vertices along the length)
        w: width of the strip (number of vertices along the width)

    Returns:
        plaquette_destab_qubit: ((L*w - 1) x (L*(w-1) + L*w)) int16 matrix
                                (first row removed, as in the class)
    """
    if L < 3 or w < 3:
        raise ValueError("Length and width must be at least 3.")
    if L % 2 == 0:
        raise ValueError("Length must be odd for the Moebius code.")
    if w % 2 == 0:
        raise ValueError("Width must be odd for the Moebius code.")

    # Number of edges
    nH = L * (w - 1)   # horizontal edges
    nV = L * w         # vertical edges
    n  = nH + nV       # total edges

    def idxH(y: int, x: int) -> int:
        # horizontal edges: y in [0, w-2], x in [0, L-1]
        return y * L + (x % L)

    def idxV(y: int, x: int) -> int:
        # vertical edges: y in [0, w-1], x in [0, L-1]
        return nH + y * L + (x % L)

    rows = []
    for y in range(w):
        for x in range(L):
            row = np.zeros(n, dtype=np.int16)

            if (x + 1) != L:
                # "Left" plaquettes (not crossing the twisted boundary)
                # Vertical segment from (0,1) up to (0,x)
                for x_prime in range(1, x + 1):
                    row[idxV(0, x_prime)] = 1
                # Horizontal segment stacking up to row y-1 at column x
                for y_prime in range(y):
                    row[idxH(y_prime, x)] = 1
            else:
                # Right boundary plaquettes (those at x = L-1)
                for x_prime in range(1, x):
                    row[idxV(0, x_prime)] = 1
                for y_prime in range(w - 1 - y):
                    row[idxH(y_prime, x - 1)] = 1
                row[idxV(w - 1 - y, x)] = 1

            rows.append(row)

    plaquette_destab_qubit = np.array(rows, dtype=np.int16)
    # As in the class: drop the first row (bottom-left plaquette)
    plaquette_destab_qubit = np.delete(plaquette_destab_qubit, 0, axis=0)
    return plaquette_destab_qubit

def finite_field_right_pseudoinverse(
    mat: NDArray[np.int_],
    p: int) -> NDArray[np.int_]:
    """
    Return a right pseudoinverse of mat over GF(p).

    If A = mat is n x m with n < m and rank(A) = n, returns a m x n matrix B
    such that A @ B = I_n (mod p).

    Warning: tailored to the Moebius code use-case; not a fully general method.
    """
    n, m = mat.shape
    if n >= m:
        raise ValueError("The number of rows must be smaller than the number of columns.")

    A = mat % p
    aug = np.hstack((A, np.eye(n, dtype=int) % p))  # [A | I_n]

    # Your existing Gauss-Jordan over GF(p)
    rref = finite_field_gauss_jordan_elimination(aug, p)

    if not np.array_equal(rref[:, :n] % p, np.eye(n, dtype=int) % p):
        raise ValueError("Rank(A) must equal the number of rows (full row rank).")

    # Right part gives an n x (m - n) block; we want m x n with zeros padded
    right_block = rref[:, m:]  # shape: (n, n)
    pseudo = np.vstack((right_block, np.zeros((m - n, n), dtype=int)))
    return pseudo % p

def build_plaquette_destabilizers_mod_p(
    H_X: NDArray[np.int_],
    p: int
) -> NDArray[np.int_]:
    """
    Build plaquette destabilizers over GF(p) from H_X using the right pseudoinverse.

    Returns a matrix of shape (num_edges, num_plaquettes) (i.e. transposed
    compared to the pseudoinverse), matching the class convention
    plaquette_destab_mod_p = plaquette_destab_qupit.T.
    """
    pseudo_inv = finite_field_right_pseudoinverse(H_X, p)
    return pseudo_inv.T  # shape: (num_plaquettes, num_edges) → transpose as needed


def build_plaquette_destabilizers_type_p(
    H_X: NDArray[np.int_],
    p: int
) -> NDArray[np.int_]:
    """
    Build the plaquette destabilizers of type p:
        S_j^X[p]   ~   2 * (mod-p destabilizers)

    This matches build_plaquette_destabilizers_type_p in the class.
    """
    mod_p_destab = build_plaquette_destabilizers_mod_p(H_X, p)
    return (2 * mod_p_destab) % (2 * p)

# -------------------------------
# Conditional entropy
# -------------------------------
LOG = lambda z: math.log(z, 2)

class StreamingConditionalEntropy:
    def __init__(self, alpha=0.0):
        self.N = 0
        self.N_y = defaultdict(int)
        self.N_xy = defaultdict(int)
        self.h_y = defaultdict(float)
        self.H = 0.0
        self.alpha = alpha
        self.unique_x_per_y = defaultdict(set)

    def recompute_hy(self, y, K=None):
        Ny = self.N_y[y]
        if Ny == 0 and self.alpha == 0:
            return 0.0
        if K is None:
            K = len(self.unique_x_per_y[y])
        denom = Ny + self.alpha * K
        hy = 0.0
        for x in self.unique_x_per_y[y]:
            num = self.N_xy[(y, x)] + self.alpha
            p = num / denom
            hy -= p * LOG(p)
        return hy

    def add_sample(self, x, y, K=None):
        N_old = self.N
        self.N += 1
        Ny_old = self.N_y[y]
        hy_old = self.h_y.get(y, 0.0)

        self.N_y[y] += 1
        self.N_xy[(y, x)] += 1
        self.unique_x_per_y[y].add(x)

        hy_new = self.recompute_hy(y, K=K)
        self.h_y[y] = hy_new

        if N_old == 0:
            self.H = hy_new
        else:
            self.H = (N_old / self.N) * self.H \
                   + (self.N_y[y] / self.N) * hy_new \
                   - (Ny_old / self.N) * hy_old

        return self.H

# -------------------------------
# Poisson-modified noise sampler
# -------------------------------
def sample_poisson_mod_noise(batch_size, n, p, d, rng=None):
    """
    Sample a batch of Poisson-mod-d noise configurations.

    • If d = 2:     pure bit-flip noise P(1)=p, P(0)=1-p.

    • If d > 2: We take a Poisson distribution with mean λ = 2 * p over n ∈ ℕ,
    fold it modulo d:
        P(k) = sum_{m=0}^∞ Poiss(k + m d; λ),
    and then renormalize so that sum_k P(k) = 1. Each qudit error
    is then drawn from this discrete distribution over {0, ..., d-1}.
    Args:
        batch_size: number of samples in the batch
        n: number of qudits per sample
        p: noise parameter (here λ = 2 * p)
        d: qudit dimension
        rng: optional numpy.random.Generator
    Returns:
        samples: (batch_size, n) array of uint8 in {0, ..., d-1}
    """
    if rng is None:
        rng = np.random.default_rng()

    # -----------------------------------------
    # Case 1: d = 2  → standard bit-flip noise
    # -----------------------------------------
    # if d == 2:
    #     probs = np.array([1 - p, p], dtype=float)
    #     samples = rng.choice([0, 1], size=(batch_size, n), p=probs)
    #     return samples.astype(np.uint8)

    # -----------------------------------------
    # Case 2: d > 2  → Poisson-mod-d noise
    # -----------------------------------------
    lam = 2.0 * p  # mean of the underlying Poisson

    # Fold Poisson over Z_d
    probs = np.zeros(d, dtype=float)

    # Truncation for the Poisson tail: around λ + 10√λ + 20 is very safe
    max_n = int(lam + 10.0 * np.sqrt(lam + 1e-12) + 20)

    ns = np.arange(max_n + 1, dtype=int)
    pmf = np.exp(-lam) * (lam ** ns) / scipy.special.factorial(ns)

    # Fold modulo d
    for idx, p_n in enumerate(pmf):
        probs[idx % d] += p_n

    # Renormalize in case of truncation error
    probs /= probs.sum()

    # Vectorized sampling over the batch using the same single-qudit distribution
    values = np.arange(d, dtype=np.uint8)
    samples = rng.choice(values, size=(batch_size, n), p=probs)

    return samples.astype(np.uint8)

# -------------------------------
# Coherent information
# -------------------------------
class CoherentInfoMC:
    """
    Monte Carlo estimator of coherent information for the Möbius code,
    using the *full* decoder:

      • Vertex side: uses vertex_destab to build e_cand and extract χ_X
      • Plaquette side: uses plaquette_destab_type_two / type_p to build
        e_cand and extract χ_Z

    Coherent information:
        I_c = 1 - H(χ_X | σ_Z) - H(χ_Z | σ_X)
    """

    def __init__(
        self,
        H_Z_full: np.ndarray,
        H_X_full: np.ndarray,
        logical_Z: np.ndarray,
        logical_X: np.ndarray,
        vertex_destab: np.ndarray,
        plaquette_destab_type_two: np.ndarray | None,
        plaquette_destab_type_p: np.ndarray | None,
        n: int,
        L: int | None = None,
        w: int | None = None,
        d: int = 2,
    ):
        # Store stabilizers
        self.h_z = H_Z_full.astype(np.int16, copy=True)   # vertex checks
        self.h_x = H_X_full.astype(np.int16, copy=True)   # plaquette checks

        # Logicals
        self.logical_Z = logical_Z.astype(np.int16, copy=True)
        self.logical_X = logical_X.astype(np.int16, copy=True)

        # Destabilizers
        self.vertex_destab = vertex_destab.astype(np.int16, copy=True)
        self.plaq_destab_2 = (
            plaquette_destab_type_two.astype(np.int16, copy=True)
            if plaquette_destab_type_two is not None
            else None
        )
        self.plaq_destab_p = (
            plaquette_destab_type_p.astype(np.int16, copy=True)
            if plaquette_destab_type_p is not None
            else None
        )

        # Basic geometry
        self.n = int(n)   # number of edges
        self.L = L
        self.w = w
        self.d = int(d)

        # Keep all stabilizer rows by default
        self.row_keep_Z = np.arange(self.h_z.shape[0], dtype=int)
        self.row_keep_X = np.arange(self.h_x.shape[0], dtype=int)

        # Prime p = d/2, if applicable
        p_candidate = self.d // 2
        if (self.d % 2 == 0) and is_prime(int(p_candidate)) and p_candidate != 2:
            self.p = int(p_candidate)  # p > 2, odd prime
        else:
            self.p = None  # covers d=2 and non-(2p prime) dimensions

    # ------------- helpers -------------

    def _encode_syndrome_keys(self, sigma_ind: np.ndarray) -> np.ndarray:
        """
        Encode each syndrome row as an integer in base-d, to use as key y.
        sigma_ind: shape (batch_size, r)
        """
        batch_size, r = sigma_ind.shape
        keys = np.zeros(batch_size, dtype=np.uint64)
        if r == 0:
            return keys
        base = 1
        for col in reversed(range(r)):
            keys += sigma_ind[:, col].astype(np.uint64) * base
            base *= self.d
        return keys

    # ------------- vertex decoder (Z side) -------------

    def get_vertex_candidate_error(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Given a vertex syndrome σ_Z, build a candidate error e_cand that
        reproduces σ_Z and commutes with logical Z, using vertex_destab.

        syndrome: shape (num_vertex_checks,)
        returns e_cand: shape (n,)
        """
        d = self.d
        candidate = np.zeros(self.n, dtype=np.int16)
        # Multiply each destabilizer row by the (integer) syndrome entry
        for j, s_j in enumerate(syndrome):
            if s_j % d == 0:
                continue
            candidate = (candidate + (s_j % d) * self.vertex_destab[j]) % d
        return candidate

    def _logical_bit_from_error_diff_Z(self, error_diff: np.ndarray) -> int:
        """
        Compute χ_X from error_diff on the vertex side:
            χ_X = (error_diff · logical_Z) / p   (mod 2)
        For d = 2, treat p = 1 and work mod 2 directly.
        """
        d = self.d
        if self.p is not None:
            val = int(error_diff @ self.logical_Z % d)
            return (val // self.p) % 2
        else:
            # qubit (or general) fallback: work mod 2
            val = int(error_diff @ self.logical_Z % 2)
            return val

    def estimate_H_Z_given_S(
        self,
        p: float,
        num_samples: int,
        batch_size: int = 10**5,
        seed: int | None = None,
    ) -> float:
        """
        Estimate H(χ_X | σ_Z) with full decoding on the vertex side.
        """
        rng = np.random.default_rng(seed)
        est = StreamingConditionalEntropy()

        B = max(1, num_samples // batch_size)

        for _ in range(B):
            # Sample errors (batch_size x n)
            m = sample_poisson_mod_noise(batch_size, self.n, p, d=self.d, rng=rng)

            # Compute vertex syndromes σ_Z = H_Z e
            sigma_full = (m @ self.h_z.T) % self.d
            sigma_ind = sigma_full[:, self.row_keep_Z]

            # Encode σ_Z to keys y
            sigma_keys = self._encode_syndrome_keys(sigma_ind)

            for i in range(batch_size):
                error = m[i]
                syndrome = sigma_full[i]

                # Decoder: candidate error from vertex destabilizers
                candidate = self.get_vertex_candidate_error(syndrome)
                error_diff = (error - candidate) % self.d

                chi_x = self._logical_bit_from_error_diff_Z(error_diff)
                est.add_sample(x=int(chi_x), y=int(sigma_keys[i]))

        return est.H  # bits

    # ------------- plaquette decoder (X side) -------------

    def get_plaquette_candidate_error(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Full plaquette decoder:

        • If self.p is None or no type-p destabilizers: qubit-only:
              e_cand = sum_j σ_j (mod 2) * destab_2[j]   (with j>0 convention)

        • If self.p is defined and type-p destabilizers are given:
              e_cand = e_type2 + e_type_p, as in the MoebiusCode class:
                - split syndrome into mod-2 plus p-part
                - use plaquette_destab_type_two and plaquette_destab_type_p
        """
        d = self.d
        num_plaquettes = self.h_x.shape[0]
        candidate = np.zeros(self.n, dtype=np.int16)

        # --- Pure qubit / no p-part case ---
        if self.p is None or self.plaq_destab_p is None:
            if self.plaq_destab_2 is None:
                return candidate  # no information; return zero

            syndrome_mod_two = syndrome % 2
            # follow same convention: plaquette 0 has no destab_two
            for idx in range(num_plaquettes):
                if idx == 0:
                    continue
                s2 = syndrome_mod_two[idx]
                if s2 == 0:
                    continue
                destab_two = self.plaq_destab_2[idx - 1]
                candidate = (candidate + s2 * destab_two) % d
            return candidate

        # --- Full d = 2p, p odd prime case ---
        p = self.p
        syndrome = syndrome % d
        syndrome_mod_two = syndrome % 2

        # Auxiliary mod-p part:
        inv_2_mod_p = pow(2, -1, p)
        syndrome_mod_p_aux = ((syndrome - syndrome_mod_two * p) * inv_2_mod_p) % p

        # Optional plaquette constraint check (same as in original class)
        if np.sum(syndrome_mod_two[1:]) % 2 != syndrome_mod_two[0]:
            # You can raise here if you want strict validity
            # raise ValueError("Invalid plaquette syndrome (constraint violated).")
            pass

        candidate_type_two = np.zeros(self.n, dtype=np.int16)
        candidate_type_p = np.zeros(self.n, dtype=np.int16)

        for idx in range(num_plaquettes):
            # type-two destabilizers: first plaquette is dependent → zero
            if idx != 0:
                destab_two = self.plaq_destab_2[idx - 1]
            else:
                destab_two = 0

            destab_p = self.plaq_destab_p[idx]

            s2 = syndrome_mod_two[idx]
            sp = syndrome_mod_p_aux[idx]

            if s2 != 0:
                candidate_type_two = (candidate_type_two + s2 * destab_two) % d
            if sp != 0:
                candidate_type_p = (candidate_type_p + sp * destab_p) % d

        candidate = (candidate_type_two + candidate_type_p) % d
        return candidate

    def _logical_bit_from_error_diff_X(self, error_diff: np.ndarray) -> int:
        """
        Compute χ_Z from error_diff on the plaquette side:
            χ_Z = (error_diff · logical_X) / p   (mod 2)
        For d = 2, treat p = 1 and work mod 2 directly.
        """
        d = self.d
        if self.p is not None:
            val = int(error_diff @ self.logical_X % d)
            return (val // self.p) % 2
        else:
            val = int(error_diff @ self.logical_X % 2)
            return val

    def estimate_H_X_given_S(
        self,
        p: float,
        num_samples: int,
        batch_size: int = 10**5,
        seed: int | None = None,
    ) -> float:
        """
        Estimate H(χ_Z | σ_X) with full decoding on the plaquette side.
        """
        rng = np.random.default_rng(seed)
        est = StreamingConditionalEntropy()

        B = max(1, num_samples // batch_size)

        for _ in range(B):
            # Sample errors (batch_size x n)
            m = sample_poisson_mod_noise(batch_size, self.n, p, d=self.d, rng=rng)

            # Compute plaquette syndromes σ_X = H_X e
            sigma_full = (m @ self.h_x.T) % self.d
            sigma_ind = sigma_full[:, self.row_keep_X]

            # Encode σ_X to keys y
            sigma_keys = self._encode_syndrome_keys(sigma_ind)

            for i in range(batch_size):
                error = m[i]
                syndrome = sigma_full[i]

                candidate = self.get_plaquette_candidate_error(syndrome)
                error_diff = (error - candidate) % self.d

                chi_z = self._logical_bit_from_error_diff_X(error_diff)
                est.add_sample(x=int(chi_z), y=int(sigma_keys[i]))

        return est.H  # bits

    # ------------- coherent information -------------

    def coherent_info_full(
        self,
        p: float,
        num_samples: int = 10**6,
        seed: int | None = None,
        batch_size: int = 10**5,
    ) -> float:
        """
        Compute I_c = 1 - H(χ_X | σ_Z) - H(χ_Z | σ_X).
        """
        H_ZgS = self.estimate_H_Z_given_S(
            p, num_samples=num_samples, batch_size=batch_size, seed=seed
        )
        H_XgS = self.estimate_H_X_given_S(
            p, num_samples=num_samples, batch_size=batch_size, seed=seed
        )
        return 1.0 - H_ZgS - H_XgS

# -------------------------------
# Sweep p
# -------------------------------
def sweep_p_Ic_full(code: CoherentInfoMC, p_values, num_samples=10**6, seed=None):
    p_list = np.asarray(p_values, dtype=float)
    rng = np.random.default_rng(seed)
    # Generate independent seeds for each p
    seeds = rng.integers(0, 2**31 - 1, size=len(p_list), dtype=np.int64)
    Ic = np.zeros_like(p_list, dtype=float)
    for i, (p, s) in enumerate(zip(p_list, seeds)):
        st = time.time()
        Ic[i] = code.coherent_info_full(p, num_samples=num_samples, seed=int(s))
        print(p,time.time()-st)
    return p_list, Ic


# -------------------------------
# Main run & plot
# -------------------------------
if __name__ == "__main__":

    sizes = [(3, 3)]
    d_values = [(2, 10**6), (6, 10**7)]
    p_grid = np.linspace(0.0, 0.25, 15)
    seed = 50

    plt.figure()
    legends = []
    for (d, num_samples) in d_values:
        print(d)
        for (L, w) in sizes:
            print(L,w)
            '''
            # --- Build stabilizer matrices + logical operators ---
            H_Z, l_Z, n, D_Z = build_moebius_code_vertex(L, w, d)
            H_X, l_X = build_moebius_code_plaquette(L, w, d)

            # --- Construct coherent-info engine ---
            code = CoherentInfoMC(H_Z_full=H_Z, H_X_full=H_X, logical_Z=l_Z, logical_X=l_X, n=n, L=L, w=w, d=d)
            '''

            H_Z, l_Z, n, D_Z = build_moebius_code_vertex(L, w, d)
            H_X, l_X = build_moebius_code_plaquette(L, w, d)
            plaq_destab_2 = build_plaquette_destabilizers_qubit(L, w)

            p_candidate = d // 2
            if (d % 2 == 0) and is_prime(int(p_candidate)) and p_candidate != 2:
                plaq_destab_p = build_plaquette_destabilizers_type_p(H_X, p_candidate)
            else:
                plaq_destab_p = None

            code = CoherentInfoMC(
                H_Z_full=H_Z,
                H_X_full=H_X,
                logical_Z=l_Z,
                logical_X=l_X,
                vertex_destab=D_Z,
                plaquette_destab_type_two=plaq_destab_2,
                plaquette_destab_type_p=plaq_destab_p,
                n=n,
                L=L,
                w=w,
                d=d,
            )

            # --- Sweep over physical noise ---
            p_list, Ic = sweep_p_Ic_full(code, p_values=p_grid, num_samples=num_samples, seed=seed)

            print(f"d={d} → I_c:", Ic)
            plt.plot(p_list, Ic, marker='o')
            legends.append(f"L={L}, w={w}, d={d}")

    # --- Plot formatting ---
    plt.axhline(0.0, linestyle='--', linewidth=1, color="black")
    plt.xlabel("Physical error rate p (Poisson-modified)")
    plt.ylabel("Coherent information $I_c$")
    plt.title("$I_c(p)$ Möbius code for different qudit dimensions")
    plt.grid(True, linestyle=':')
    plt.legend(legends, title="Parameters", loc="best")
    plt.tight_layout()
    plt.show()


# (3,3) d= 6, single p 25 sec

'''
class CoherentInfoMC:
    def __init__(self, H_full: np.ndarray, l_op: np.ndarray, R_vertex, k: int = 1, L: int | None = None, w: int | None = None, d: int = 6):
        assert H_full.dtype == np.int16 and l_op.dtype == np.int16
        self.H_full = H_full
        self.l_op = l_op
        self.n = l_op.shape[0]
        # self.k = int(k)
        self.R = R_vertex
        self.L = L
        self.w = w
        self.d = d
        self.H_ind = self.H_full.copy()
        self.row_keep = np.arange(self.H_full.shape[0], dtype=int)
        self.r = self.H_full.shape[0]

    def estimate_H_Z_given_S(self, p, num_samples, batch_size=10**5, seed=None):
        rng = np.random.default_rng(seed)
        est = StreamingConditionalEntropy()

        for _ in range(max(1, num_samples // batch_size)):
            m = sample_poisson_mod_noise(batch_size, self.n, p, d=self.d, rng=rng)
            sigma_full = (m @ self.H_full.T) % self.d
            sigma_ind = sigma_full[:, self.row_keep]

            # Safe sigma_keys computation
            sigma_keys = np.zeros(batch_size, dtype=np.uint64)
            if sigma_ind.shape[1] > 0:
                base = 1
                for col in reversed(range(sigma_ind.shape[1])):
                    sigma_keys += sigma_ind[:, col].astype(np.uint64) * base
                    base *= self.d

            x = (m @ self.l_op) % self.d
            for sx, y in zip(x, sigma_keys):
                est.add_sample(int(sx), int(y))
        return est.H

    def plaquette_candidate_error(
            syndrome: NDArray,
            d: int,
            p: int,
            plaquette_destab_type_two: NDArray,
            plaquette_destab_type_p: NDArray,) -> NDArray:
        """
        Args:
            syndrome: plaquette syndrome vector (length = num_plaquette_checks)
            d: total qudit dimension (d = 2p)
            p: odd prime (p = d/2)
            plaquette_destab_type_two: destabilizers for S^X[2], shape (num_plaquettes - 1, num_edges)
            plaquette_destab_type_p:   destabilizers for S^X[p], shape (num_plaquettes, num_edges)

        Returns:
            candidate: candidate error vector of length num_edges.
        """
        if d != 2 * p:
            raise ValueError("This routine assumes d = 2p.")

        syndrome = syndrome % d
        syndrome_mod_two = syndrome % 2

        # p-part aux vector as in the class
        inv_2_mod_p = pow(2, -1, p)
        syndrome_mod_p_aux = ((syndrome - syndrome_mod_two * p) * inv_2_mod_p) % p

        # Plaquette constraint (same as in class)
        if np.sum(syndrome_mod_two[1:]) % 2 != syndrome_mod_two[0]:
            raise ValueError("Invalid syndrome: plaquette constraint not satisfied.")

        num_plaquettes = plaquette_destab_type_p.shape[0]
        num_edges = plaquette_destab_type_p.shape[1]

        candidate_type_two = np.zeros(num_edges, dtype=np.int16)
        candidate_type_p = np.zeros(num_edges, dtype=np.int16)

        for idx in range(num_plaquettes):
            # type-two destabilizers: first plaquette is dependent → zero
            if idx != 0:
                destab_two = plaquette_destab_type_two[idx - 1, :]
            else:
                destab_two = 0

            destab_p = plaquette_destab_type_p[idx, :]

            candidate_type_two = (candidate_type_two +
                                  syndrome_mod_two[idx] * destab_two) % d
            candidate_type_p = (candidate_type_p +
                                syndrome_mod_p_aux[idx] * destab_p) % d

        return (candidate_type_two + candidate_type_p) % d

    def estimate_H_X_given_S(
            self,
            num_samples: int,
            p: float,
            batch_size: int = 10000,
            seed: int | None = None
    ) -> float:

        rng = np.random.default_rng(seed)
        est = StreamingConditionalEntropy()

        n = self.num_edges
        d = self.d

        # Number of batches
        B = max(1, num_samples // batch_size)

        for _ in range(B):

            # ---- SAMPLE PHYSICAL NOISE ----
            m = sample_poisson_mod_noise(batch_size, n, p, d, rng=rng)
            # m has shape (batch_size, n)

            # ---- COMPUTE ALL SYNDROMES AT ONCE ----
            syndromes = (m @ self.h_x.T) % d  # shape (batch_size, num_plaquettes)

            # ---- PROCESS EACH SAMPLE ----
            for i in range(batch_size):
                error = m[i]
                syndrome = syndromes[i]

                # Candidate error from stabilizer solving
                candidate = self.get_plaquette_candidate_error(syndrome)

                # Effective logical diffference
                logical_val = (error - candidate) @ self.logical_x % d
                chi = int(logical_val / self.p)  # logical bit 0 or 1

                # Hash syndrome into a key
                synd_key = "_".join(map(str, syndrome))

                est.add_sample(x=chi, y=synd_key)

        return est.H

    def coherent_info_full(self, p: float, num_samples: int = 10**6, seed: int | None = None) -> float:
        H_ZgS = self.estimate_H_Z_given_S(p, num_samples=num_samples, seed=seed)
        #H_XgS = self.estimate_H_X_given_S(p, num_samples=num_samples, seed=seed)
        return (1 - 2*H_ZgS) # - H_XgS)
'''