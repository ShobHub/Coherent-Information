import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

# ------------------------------------------------------------
# Möbius code construction (faster H_Z build)
# ------------------------------------------------------------
def build_moebius_code_vertex(L: int, w: int, d: int = 2):
    """
    Returns:
        H_Z: int8 matrix with entries in {0, +1, -1}  (signs irrelevant mod 2, but kept)
        l_Z: int8 vector with entries in {0,1} for d=2 (since exp=d/2=1)
        n: number of edges
    """
    if L < 3 or w < 3:
        raise ValueError("Length and width must be at least 3.")
    if L % 2 == 0:
        raise ValueError("Length must be odd for the Moebius code.")
    if w % 2 == 0:
        raise ValueError("Width must be odd for the Moebius code.")
    if d % 2 != 0:
        raise ValueError("Dimension d must be even for the Moebius code.")

    nH = L * (w - 1)
    nV = L * w
    n = nH + nV
    r_z = L * (w - 1)

    def idxH(y, x):  # y in [0,w-2]
        return y * L + (x % L)

    def idxV(y, x):  # y in [0,w-1]
        return nH + y * L + (x % L)

    # Preallocate and fill rows directly (only 4 nonzeros per row)
    H_Z = np.zeros((r_z, n), dtype=np.int8)

    row_idx = 0
    for y in range(w - 1):
        for x in range(L):
            h_right = idxH(y, x)
            if x == 0:
                h_left = idxH((w - 2) - y, L - 1)
            else:
                h_left = idxH(y, x - 1)

            v_top = idxV(y, x)
            v_bottom = idxV(y + 1, x)

            # For d=2, values mod 2 only depend on support; we keep ±1 pattern as before.
            H_Z[row_idx, h_right] = 1
            H_Z[row_idx, h_left] = 1
            H_Z[row_idx, v_top] = 1
            H_Z[row_idx, v_bottom] = 1
            row_idx += 1

    # Impose your alternating sign pattern (optional for d=2)
    H_Z[1::2, :] *= -1

    # Logical Z: central vertical row; for d=2 exp=1
    l_Z = np.zeros(n, dtype=np.int8)
    y0 = w // 2
    exp = (d // 2) & 1  # for d=2, exp=1
    for x in range(L):
        l_Z[idxV(y0, x)] = exp

    return H_Z, l_Z, n

# ------------------------------------------------------------
# Noise model: wrapped-FT Eq(18) specialized to d=2
#   P0 = 1/2(1+e^{-alpha}), P1 = 1/2(1-e^{-alpha})
# ------------------------------------------------------------
def noise_probs_wrapped_d2(alpha: float) -> tuple[float, float]:
    ea = np.exp(-alpha)
    P0 = 0.5 * (1.0 + ea)
    P1 = 0.5 * (1.0 - ea)
    # safe clip
    P0 = float(np.clip(P0, 0.0, 1.0))
    P1 = float(np.clip(P1, 0.0, 1.0))
    return P0, P1

def alpha_to_p(alpha: float) -> float:
    return noise_probs_wrapped_d2(alpha)[1]

def h2_binary(p1: float) -> float:
    """Binary entropy in bits (NOT the noise model)."""
    if p1 <= 0.0 or p1 >= 1.0:
        return 0.0
    return -(p1 * np.log2(p1) + (1.0 - p1) * np.log2(1.0 - p1))

# ------------------------------------------------------------
# CSR adjacency (fast)
# ------------------------------------------------------------
def build_adjacency_csr(H_Z: np.ndarray):
    """
    Build:
      - check_ptr: int32, shape (r_z+1,)
      - check_edges: int32, shape (nnz,)
      - edge_checks: int32, shape (n,2) with -1 for missing
    using nonzero pattern of H_Z.
    """
    r_z, n = H_Z.shape
    rows, cols = np.nonzero(H_Z)

    # Sort by row for CSR
    order = np.argsort(rows, kind="mergesort")
    rows_s = rows[order].astype(np.int32)
    cols_s = cols[order].astype(np.int32)

    check_ptr = np.zeros(r_z + 1, dtype=np.int32)
    np.add.at(check_ptr, rows_s + 1, 1)
    check_ptr = np.cumsum(check_ptr)

    check_edges = cols_s  # length nnz

    # Each edge ideally touches 2 checks (bulk); store up to 2
    edge_checks = -np.ones((n, 2), dtype=np.int32)
    fill = np.zeros(n, dtype=np.int8)
    for i, e in zip(rows.astype(np.int32), cols.astype(np.int32)):
        j = fill[e]
        if j < 2:
            edge_checks[e, j] = i
            fill[e] += 1

    return check_ptr, check_edges, edge_checks

# ------------------------------------------------------------
# Fast Bernoulli sampling for m_start using P1(alpha)
# ------------------------------------------------------------
def sample_x_error_vector_u8(n: int, P1: float, rng: np.random.Generator) -> np.ndarray:
    """Return uint8 vector in {0,1}^n with P(1)=P1."""
    return (rng.random(n) < P1).astype(np.uint8)

# ------------------------------------------------------------
# Logical bit extraction (d=2): x_X = <l_Z, delta_m> mod 2
# ------------------------------------------------------------
# Precompute lZ support indices once (O(n))
def precompute_lZ_support(l_Z: np.ndarray) -> np.ndarray:
    """
    Returns indices where l_Z is odd (==1 mod 2). For d=2, these are the edges in logical Z support.
    """
    return np.flatnonzero(l_Z & 1).astype(np.int32)

# Fast logical bit extraction using only support indices (O(|lZ|) ~ O(L))
def logical_bit_xX_from_delta_idx(delta_m_u8: np.ndarray, lZ_idx: np.ndarray) -> int:
    """
    Compute x_X = <l_Z, delta_m> mod 2, but only over the support of l_Z.
    delta_m_u8: uint8 array in {0,1}^n
    lZ_idx: int32 array of indices where l_Z==1 mod 2
    """
    # Sum only the support bits and take parity
    return int(delta_m_u8[lZ_idx].sum() & 1)

# ------------------------------------------------------------
# For sanity check - calculate syndrome for any error
# ------------------------------------------------------------
def syndrome_Z_u8(H_Z: np.ndarray, m_u8: np.ndarray) -> np.ndarray:
    H2 = (H_Z != 0).astype(np.int8)
    v = H2.astype(np.int16) @ m_u8.astype(np.int16)
    return (v & 1).astype(np.uint8)

# ------------------------------------------------------------
# Fixed reference error, m_0 ??
# ------------------------------------------------------------


# ------------------------------------------------------------
# Worm kernel (pure Python, optimized)
# ------------------------------------------------------------
def worm_closed_loop_update_fast(
    m: np.ndarray,                 # uint8 vector
    check_ptr: np.ndarray,         # int32
    check_edges: np.ndarray,       # int32
    edge_checks: np.ndarray,       # int32 (n,2)
    rng: np.random.Generator,
    r01: float,                    # P1/P0
    r10: float,                    # P0/P1
    max_steps: int = 200000,
) -> bool:
    """
    One worm update (d=2): walk on check graph, flip one edge each accepted step,
    stop when head returns to tail. Preserves syndrome sector.
    """
    # local bindings for speed
    integers = rng.integers
    rand = rng.random
    m_arr = m
    cptr = check_ptr
    cedges = check_edges
    echecks = edge_checks

    r_z = cptr.shape[0] - 1
    tail = int(integers(r_z))
    head = tail

    flipped = []  # store edges we actually flipped (accepted moves)

    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            # ABORT: undo partial worm so syndrome is unchanged
            for e in flipped:
                m_arr[e] ^= 1
            return False

        start = cptr[head]
        end = cptr[head + 1]
        deg = end - start
        e = int(cedges[start + int(integers(deg))])

        c0 = echecks[e, 0]
        c1 = echecks[e, 1]
        if c0 < 0 or c1 < 0:
            # boundary edge (not handled here) -> reject
            continue

        next_head = int(c0 if c1 == head else c1)

        bit = int(m_arr[e] & 1)

        # Inline Metropolis (min(1, ratio))
        if bit == 0:
            # 0 -> 1
            if r01 >= 1.0 or rand() < r01:
                m_arr[e] ^= 1
                flipped.append(e)
                head = next_head
                if head == tail:
                    return True
        else:
            # 1 -> 0
            if r10 >= 1.0 or rand() < r10:
                m_arr[e] ^= 1
                flipped.append(e)
                head = next_head
                if head == tail:
                    return True

# ------------------------------------------------------------
# Main estimators (Hx and Icoh)
# ------------------------------------------------------------
def estimate_Hx_given_syndrome_fast(
    H_Z: np.ndarray,
    lZ_idx: np.ndarray,               # int8
    m_start: np.ndarray,           # uint8
    check_ptr: np.ndarray,
    check_edges: np.ndarray,
    edge_checks: np.ndarray,
    rng: np.random.Generator,
    r01: float,
    r10: float,
    burn_in_worms: int,
    N_log: int,
    worms_per_sample: int,
    DEBUG: bool,
) -> float:

    m_cur = m_start.copy()

    count = 0
    # m_start_frozen = m_start.copy()
    # sig0 = syndrome_Z_u8(H_Z, m_start)

    # burn-in
    for _ in range(burn_in_worms):
        ok = worm_closed_loop_update_fast(m_cur, check_ptr, check_edges, edge_checks, rng, r01, r10)
        if not ok:
            count += 1

    # check after burn-in
    # if DEBUG:
    #     sigB = syndrome_Z_u8(H_Z, m_cur)
    #     if not np.array_equal(sig0, sigB):
    #         raise RuntimeError("Syndrome changed during burn-in.")

    n1 = 0
    for _ in range(N_log):
        # m_before = m_cur.copy()
        for _ in range(worms_per_sample):
            ok = worm_closed_loop_update_fast(m_cur, check_ptr, check_edges, edge_checks, rng, r01, r10)
            if not ok:
                count += 1

        # if DEBUG:
        #     sig_before = syndrome_Z_u8(H_Z, m_before)
        #     sig_after = syndrome_Z_u8(H_Z, m_cur)
        #     if np.any(sig_before ^ sig_after):
        #         raise RuntimeError("Single worm step batch changed syndrome.")

        delta = (m_cur ^ m_start)
        n1 += logical_bit_xX_from_delta_idx(delta, lZ_idx)

    if count!=0: print(count)

    # Sanity check - worms are syndrome preserving - can go inside N_log loop also
    # if DEBUG:
    #     sig1 = syndrome_Z_u8(H_Z, m_cur)
    #     if not np.array_equal(sig0, sig1):
    #         raise RuntimeError("Worm did not preserve syndrome sector.")
    # assert np.array_equal(m_start, m_start_frozen), "m_start was mutated!"

    p1 = n1 / float(N_log)
    return h2_binary(p1)

def estimate_Icoh_vs_alpha_d2_fast(
    L: int,
    w: int,
    alphas: np.ndarray,
    seed: int,
    N_syn: int,
    N_log: int,
    burn_in_worms: int,
    worms_per_sample: int,
):

    rng = np.random.default_rng(seed)

    H_Z, l_Z, n = build_moebius_code_vertex(L=L, w=w, d=2)
    check_ptr, check_edges, edge_checks = build_adjacency_csr(H_Z)

    # Precompute logical-Z support once
    lZ_idx = precompute_lZ_support(l_Z)

    ps = np.array([alpha_to_p(float(a)) for a in alphas], dtype=float)
    Icoh = np.zeros_like(alphas, dtype=float)
    Hx = np.zeros_like(alphas, dtype=float)

    for ia, alpha in enumerate(alphas):
        alpha = float(alpha)
        P0, P1 = noise_probs_wrapped_d2(alpha)

        # precompute ratios once per alpha
        # (handle edge cases)
        if P0 == 0.0:
            r01 = 1.0
        else:
            r01 = P1 / P0
        if P1 == 0.0:
            r10 = 1.0
        else:
            r10 = P0 / P1

        H_accum = 0.0
        for _ in range(N_syn):
            m_start = sample_x_error_vector_u8(n, P1, rng)
            H_accum += estimate_Hx_given_syndrome_fast(
                H_Z = H_Z,          # just for sanity check
                lZ_idx=lZ_idx,
                m_start=m_start,
                check_ptr=check_ptr,
                check_edges=check_edges,
                edge_checks=edge_checks,
                rng=rng,
                r01=r01,
                r10=r10,
                burn_in_worms=burn_in_worms,
                N_log=N_log,
                worms_per_sample=worms_per_sample,
                DEBUG=True,   # False
            )

        H_mean = H_accum / float(N_syn)
        Hx[ia] = H_mean
        Icoh[ia] = 1.0 - 2.0 * H_mean

    return ps, Icoh, Hx

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    alphas = np.linspace(0.02, 1.0, 20)

    mc_params = {
        # (3, 3): dict(seed=100, N_syn=100, N_log=300,  burn_in=50,  worms_per_sample=2),
        # (5, 5): dict(seed=200, N_syn=150, N_log=350,  burn_in=55,  worms_per_sample=3),
        (7, 7): dict(seed=300, N_syn=150, N_log=350,  burn_in=55,  worms_per_sample=3),
        # (9, 9): dict(seed=400, N_syn=150, N_log=350,  burn_in=55,  worms_per_sample=3),
        # (11,11): dict(seed=500, N_syn=200, N_log=400, burn_in=60, worms_per_sample=5),
        # (15, 15): dict(seed=600, N_syn=200, N_log=400, burn_in=60, worms_per_sample=5)
    }

    for (L, w), pars in mc_params.items():
        st = time.time()

        ps_out, Icoh_out, Hx_out = estimate_Icoh_vs_alpha_d2_fast(
            L=L, w=w,
            alphas=alphas,
            seed=pars["seed"],
            N_syn=pars["N_syn"],
            N_log=pars["N_log"],
            burn_in_worms=pars["burn_in"],
            worms_per_sample=pars["worms_per_sample"]
        )

        dt = time.time() - st
        print(f"(L,w)=({L},{w}) time={dt:.2f}s")

        # Save
        # pickle.dump([ps_out, Icoh_out, Hx_out],open(f"../d2_data/CI_Worm_d2_Eq18_L={L}_w={w}.pkl", "wb"))

        label = f"(L,w)=({L},{w}), Nsyn={pars['N_syn']}, Nlog={pars['N_log']}"
        plt.plot(ps_out, Icoh_out, label=label)
        plt.scatter(ps_out, Icoh_out, s=12)

    plt.axhline(0.0, color="k", linewidth=1)
    plt.xlabel(r"$p = P(1) = \frac{1-e^{-\alpha}}{2}$")
    plt.ylabel(r"$I_{\rm coh} = 1 - 2H(x_X|\sigma_Z)$")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    
'''
for (L, w), pars in mc_params.items():
    ps_out, Icoh_out, _ = pickle.load(open('CI_Worm_d2_Eq18_L=%i_w=%i.pkl' % (L, w), 'rb'))
    label = f"(L,w)=({L},{w}), Nsyn={pars['N_syn']}, Nlog={pars['N_log']}"
    plt.plot(ps_out, Icoh_out, label=label)
    plt.scatter(ps_out, Icoh_out, s=12)

plt.axhline(0.0, color="k", linewidth=1)
plt.axhline(0., color="k", linewidth=1)
plt.xlabel(r"$p = P(1) = \frac{1-e^{-\alpha}}{2}$")
plt.ylabel(r"$I_{\rm coh} = 1 - 2H(x_X|\sigma_Z)$")
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()
'''
