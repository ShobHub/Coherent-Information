from __future__ import annotations

import json
import math
import os
import platform
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

INT_DTYPE = np.int32


# =========================
# Finite-field linear algebra
# =========================

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    r = int(math.sqrt(n))
    for i in range(2, r + 1):
        if n % i == 0:
            return False
    return True


def finite_field_gauss_jordan_elimination(mat: np.ndarray, p: int) -> np.ndarray:
    if not is_prime(int(p)):
        raise ValueError(f"Modulus p={p} is not prime")
    out = np.array(mat, dtype=np.int64, copy=True) % p
    n_rows, n_cols = out.shape
    row = 0
    for col in range(n_cols):
        if row >= n_rows:
            break
        pivot_row = -1
        for r in range(row, n_rows):
            if out[r, col] % p != 0:
                pivot_row = r
                break
        if pivot_row < 0:
            continue
        if pivot_row != row:
            out[[row, pivot_row]] = out[[pivot_row, row]]
        inv = pow(int(out[row, col]), -1, int(p))
        out[row, :] = (out[row, :] * inv) % p
        for r in range(n_rows):
            if r != row and out[r, col] % p != 0:
                out[r, :] = (out[r, :] - out[r, col] * out[row, :]) % p
        row += 1
    return out.astype(INT_DTYPE)


# =========================
# Error model
# =========================

@nb.njit(cache=True)
def _probabilities_lindblad_kernel(d: int, gamma_t: float) -> np.ndarray:
    probs = np.zeros(d, dtype=np.float64)
    for n in range(d):
        total_real = 0.0
        total_imag = 0.0
        for l in range(d):
            amp = math.exp(-4.0 * gamma_t * (math.sin(l * math.pi / d) ** 2)) / d
            angle = 2.0 * math.pi * l * n / d
            total_real += amp * math.cos(angle)
            total_imag += amp * math.sin(angle)
        probs[n] = total_real
    s = probs.sum()
    if s <= 0.0:
        probs[:] = 1.0 / d
    else:
        probs /= s
    return probs


def modular_probability_from_probs(probs: np.ndarray, p: int, m_mod_2: int, m_mod_p: int) -> float:
    mm2 = m_mod_2 % 2
    mmp = m_mod_p % p
    m = (p * mm2 + ((1 - p) // 2) * 2 * mmp) % (2 * p)
    return probs[m]


def sample_error_from_cdf(cdf: np.ndarray, num_subsys: int, out: np.ndarray) -> None:
    for i in range(num_subsys):
        u = np.random.random()
        idx = np.searchsorted(cdf, u, side='left')
        out[i] = idx


class ErrorModelLindbladTwoOddPrime:
    def __init__(self, num_subsys: int, d: int, gamma_t: float):
        p = d // 2
        if d % 2 != 0 or p == 2 or not is_prime(p):
            raise ValueError("d must be 2*p with p odd prime")
        self.num_subsys = int(num_subsys)
        self.d = int(d)
        self.p = int(p)
        self.gamma_t = float(gamma_t)
        self.probs = _probabilities_lindblad_kernel(self.d, self.gamma_t)
        self.cdf = np.cumsum(self.probs)
        self.cdf[-1] = 1.0

    def generate_random_error(self) -> np.ndarray:
        out = np.empty(self.num_subsys, dtype=INT_DTYPE)
        sample_error_from_cdf(self.cdf, self.num_subsys, out)
        return out


# =========================
# Moebius code
# =========================

class MoebiusCode:
    def __init__(self, length: int, width: int, d: int = 2):
        if length < 3 or width < 3:
            raise ValueError("Length and width must be at least 3")
        if length % 2 == 0 or width % 2 == 0:
            raise ValueError("Length and width must be odd")
        if d % 2 != 0:
            raise ValueError("Dimension d must be even")
        self._length = int(length)
        self._width = int(width)
        self._d = int(d)
        self.compute_and_set_code_properties()

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    @property
    def d(self):
        return self._d

    def index_h(self, y: int, x: int) -> int:
        return y * self.length + x

    def inverted_index_h(self, y: int, x: int) -> int:
        return (self.width - 2 - y) * self.length + x

    def index_v(self, y: int, x: int) -> int:
        return self.length * (self.width - 1) + y * self.length + x

    def compute_and_set_code_properties(self):
        self.num_h_edges = self.length * (self.width - 1)
        self.num_v_edges = self.length * self.width
        self.num_edges = self.num_h_edges + self.num_v_edges
        self.num_vertex_checks = self.length * (self.width - 1)
        self.num_plaquette_checks = self.length * self.width
        self.h_z = self.build_moebius_code_vertex()
        self.h_x = self.build_moebius_code_plaquette()
        self.h_z_qubit = self.h_z % 2
        self.h_x_qubit = np.delete(self.h_x, 0, axis=0) % 2
        self.logical_z = self.get_logical_z()
        self.logical_x = self.get_logical_x()
        self.vertex_destab = self.build_vertex_destabilizers()
        self.vertex_destab_qubit = self.vertex_destab % 2
        self.plaquette_destab_qubit = self.build_plaquette_destabilizers_qubit()

    def build_moebius_code_vertex(self) -> np.ndarray:
        rows = []
        for y in range(self.width - 1):
            for x in range(self.length):
                row = np.zeros(self.num_edges, dtype=INT_DTYPE)
                sgn = 1 if (x + y) % 2 == 0 else -1
                if x != 0:
                    row[self.index_h(y, x)] = sgn
                    row[self.index_h(y, x - 1)] = sgn
                    row[self.index_v(y, x)] = sgn
                    row[self.index_v(y + 1, x)] = sgn
                else:
                    row[self.index_h(y, 0)] = sgn
                    row[self.inverted_index_h(y, self.length - 1)] = sgn
                    row[self.index_v(y, 0)] = sgn
                    row[self.index_v(y + 1, 0)] = sgn
                rows.append(row)
        return np.asarray(rows, dtype=INT_DTYPE)

    def build_moebius_code_plaquette(self) -> np.ndarray:
        rows = []
        for y in range(self.width):
            for x in range(self.length):
                row = np.zeros(self.num_edges, dtype=INT_DTYPE)
                if y == 0:
                    if (x + 1) % self.length != 0:
                        if x % 2 == 0:
                            row[self.index_h(0, x)] = 1
                            row[self.index_v(0, x)] = -1
                            row[self.index_v(0, x + 1)] = -1
                        else:
                            row[self.index_h(0, x)] = -1
                            row[self.index_v(0, x)] = 1
                            row[self.index_v(0, x + 1)] = 1
                    else:
                        row[self.index_h(self.width - 2, self.length - 1)] = 1
                        row[self.index_v(0, 0)] = -1
                        row[self.index_v(self.width - 1, self.length - 1)] = -1
                elif y < self.width - 1:
                    if (x + 1) % self.length != 0:
                        if (x + y) % 2 == 1:
                            row[self.index_h(y - 1, x)] = -1
                            row[self.index_h(y, x)] = -1
                            row[self.index_v(y, x)] = 1
                            row[self.index_v(y, x + 1)] = 1
                        else:
                            row[self.index_h(y - 1, x)] = 1
                            row[self.index_h(y, x)] = 1
                            row[self.index_v(y, x)] = -1
                            row[self.index_v(y, x + 1)] = -1
                    else:
                        if (x + y) % 2 == 1:
                            row[self.inverted_index_h(y, x)] = -1
                            row[self.inverted_index_h(y - 1, x)] = -1
                            row[self.index_v(y, 0)] = 1
                            row[self.index_v(self.width - y - 1, x)] = 1
                        else:
                            row[self.inverted_index_h(y, x)] = 1
                            row[self.inverted_index_h(y - 1, x)] = 1
                            row[self.index_v(y, 0)] = -1
                            row[self.index_v(self.width - y - 1, x)] = -1
                else:
                    if (x + 1) % self.length != 0:
                        if x % 2 == 0:
                            row[self.index_h(self.width - 2, x)] = 1
                            row[self.index_v(self.width - 1, x)] = -1
                            row[self.index_v(self.width - 1, x + 1)] = -1
                        else:
                            row[self.index_h(self.width - 2, x)] = -1
                            row[self.index_v(self.width - 1, x)] = 1
                            row[self.index_v(self.width - 1, x + 1)] = 1
                    else:
                        row[self.index_h(0, self.length - 1)] = 1
                        row[self.index_v(self.width - 1, 0)] = -1
                        row[self.index_v(0, self.length - 1)] = -1
                rows.append(row)
        return np.asarray(rows, dtype=INT_DTYPE)

    def get_logical_z(self) -> np.ndarray:
        logical_z = np.zeros(self.num_edges, dtype=INT_DTYPE)
        y0 = self.width // 2
        for x in range(self.length):
            logical_z[self.index_v(y0, x)] = self.d // 2
        return logical_z

    def get_logical_x(self) -> np.ndarray:
        logical_x = np.zeros(self.num_edges, dtype=INT_DTYPE)
        for y in range(self.width):
            logical_x[self.index_v(y, 0)] = -1 if y % 2 == 0 else 1
        return (self.d // 2) * logical_x.astype(INT_DTYPE)

    def build_vertex_destabilizers(self) -> np.ndarray:
        rows = []
        for y in range(self.width - 1):
            for x in range(self.length):
                row = np.zeros(self.num_edges, dtype=INT_DTYPE)
                if y < (self.width - 1) / 2:
                    for y_prime in range(y + 1):
                        row[self.index_v(y_prime, x)] = 1 if (x + y_prime) % 2 == 0 else -1
                else:
                    for y_prime in range(y + 1, self.width):
                        row[self.index_v(y_prime, x)] = -1 if (x + y_prime) % 2 == 0 else 1
                rows.append(row)
        return np.asarray(rows, dtype=INT_DTYPE)

    def build_plaquette_destabilizers_qubit(self) -> np.ndarray:
        rows = []
        for y in range(self.width):
            for x in range(self.length):
                row = np.zeros(self.num_edges, dtype=INT_DTYPE)
                if (x + 1) != self.length:
                    for xp in range(1, x + 1):
                        row[self.index_v(0, xp)] = 1
                    for yp in range(y):
                        row[self.index_h(yp, x)] = 1
                else:
                    for xp in range(1, x):
                        row[self.index_v(0, xp)] = 1
                    for yp in range(self.width - 1 - y):
                        row[self.index_h(yp, x - 1)] = 1
                    row[self.index_v(self.width - 1 - y, x)] = 1
                rows.append(row)
        arr = np.asarray(rows, dtype=INT_DTYPE)
        return np.delete(arr, 0, axis=0)


class MoebiusCodeTwoOddPrime(MoebiusCode):
    def __init__(self, length: int, width: int, d: int = 6):
        p = d // 2
        if d % 2 != 0 or p == 2 or not is_prime(p):
            raise ValueError("d must be 2 * p with p odd prime")
        self.p = int(p)
        self.inverse_two_mod_p = pow(2, -1, self.p)
        super().__init__(length, width, d)

    def compute_and_set_code_properties(self):
        super().compute_and_set_code_properties()
        self.h_x_mod_2 = self.h_x % 2
        self.h_x_mod_p = self.h_x % self.p
        self.h_z_mod_2 = self.h_z % 2
        self.h_z_mod_p = self.h_z % self.p
        self.plaquette_destab_type_two = self.p * self.plaquette_destab_qubit
        self.plaquette_destab_mod_p = self.build_plaquette_destabilizers_mod_p()
        self.plaquette_destab_type_p = 2 * self.plaquette_destab_mod_p
        self.vertex_edge_lookup = build_edge_lookup(self.h_z_mod_p)
        self.plaquette_edge_lookup = build_edge_lookup(self.h_x_mod_p)
        self.vertex_stab_edges = build_stab_edges(self.h_z_mod_p)
        self.plaquette_stab_edges = build_stab_edges(self.h_x_mod_p)

    @staticmethod
    def finite_field_right_pseudoinverse(mat: np.ndarray, p: int) -> np.ndarray:
        if mat.shape[0] >= mat.shape[1]:
            raise ValueError("The number of rows should be smaller than the number of columns")
        n, m = mat.shape
        aug = np.hstack((mat % p, np.eye(n, dtype=np.int64) % p))
        rref = finite_field_gauss_jordan_elimination(aug, p)
        if not np.array_equal(rref[:, :n], np.eye(n, dtype=INT_DTYPE)):
            raise ValueError("The rank of the matrix must be equal to the number of rows")
        pseudo_inv = np.vstack((rref[:, m:], np.zeros((m - n, n), dtype=INT_DTYPE)))
        return pseudo_inv.astype(INT_DTYPE)

    def build_plaquette_destabilizers_mod_p(self) -> np.ndarray:
        return self.finite_field_right_pseudoinverse(self.h_x, self.p).T.astype(INT_DTYPE)

    def get_vertex_syndrome(self, error: np.ndarray) -> np.ndarray:
        return (self.h_z @ error) % self.d

    def get_plaquette_syndrome(self, error: np.ndarray) -> np.ndarray:
        return (self.h_x @ error) % self.d

    def get_vertex_candidate_error(self, syndrome: np.ndarray) -> np.ndarray:
        return (syndrome % self.d) @ self.vertex_destab % self.d

    def get_plaquette_candidate_error(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = syndrome % self.d
        syndrome_mod_two = syndrome % 2
        syndrome_mod_p_aux = ((syndrome - syndrome_mod_two * self.p) * self.inverse_two_mod_p) % self.p
        candidate_type_two = (np.delete(syndrome_mod_two, 0) @ self.plaquette_destab_type_two) % self.d
        candidate_type_p = (syndrome_mod_p_aux @ self.plaquette_destab_type_p) % self.d
        return (candidate_type_two + candidate_type_p) % self.d

    def compute_vertex_syndrome_chi_z(self, error: np.ndarray) -> np.ndarray:
        syndrome = self.get_vertex_syndrome(error)
        candidate_error = self.get_vertex_candidate_error(syndrome)
        error_diff = error - candidate_error
        chi_z = int((error_diff @ self.logical_z.T) % self.d)
        return np.append(syndrome, np.int16(chi_z))

    def compute_plaquette_syndrome_chi_x(self, error: np.ndarray) -> np.ndarray:
        syndrome = self.get_plaquette_syndrome(error)
        candidate_error = self.get_plaquette_candidate_error(syndrome)
        error_diff = error - candidate_error
        chi_x = int((error_diff @ self.logical_x.T) % self.d)
        return np.append(syndrome, np.int16(chi_x))


# =========================
# Lookup builders for numba kernels
# =========================

def build_edge_lookup(stab_mat_mod_p: np.ndarray) -> np.ndarray:
    num_stabs, num_edges = stab_mat_mod_p.shape
    lookup = np.full((num_edges, 2), -1, dtype=INT_DTYPE)
    for e in range(num_edges):
        idx = 0
        for s in range(num_stabs):
            if stab_mat_mod_p[s, e] != 0:
                lookup[e, idx] = s
                idx += 1
                if idx == 2:
                    break
    return lookup


def build_stab_edges(stab_mat_mod_p: np.ndarray) -> np.ndarray:
    num_stabs, num_edges = stab_mat_mod_p.shape
    out = np.full((num_stabs, 4), -1, dtype=INT_DTYPE)
    for s in range(num_stabs):
        idx = 0
        for e in range(num_edges):
            if stab_mat_mod_p[s, e] != 0:
                out[s, idx] = e
                idx += 1
                if idx == 4:
                    break
    return out


# =========================
# Numba worm kernels
# =========================

def choose_candidate_stab(edge_lookup: np.ndarray, edge: int, stab_bool: int) -> int:
    s0 = edge_lookup[edge, 0]
    s1 = edge_lookup[edge, 1]
    if s1 == -1 or stab_bool == 1:
        return s0
    return s1


def single_move_probability_numba(edge: int, power: int, error_mod_2: np.ndarray, error_mod_p: np.ndarray,
                                  stab_bool: int, h_error_mod_p: np.ndarray, edge_lookup: np.ndarray,
                                  stab_edges: np.ndarray, p: int, probs: np.ndarray) -> Tuple[float, float]:
    if edge < 0:
        return -1.0, -1.0
    candidate_stab_label = choose_candidate_stab(edge_lookup, edge, stab_bool)
    edges_candidate_stab = stab_edges[candidate_stab_label]
    candidate_stab = h_error_mod_p[candidate_stab_label]
    total_new = 1.0
    total_old = 1.0
    for j in range(4):
        edge_ch = edges_candidate_stab[j]
        if edge_ch < 0:
            continue
        is_edge = 1 if edge_ch == edge else 0
        error_at_edge = (is_edge + error_mod_2[edge_ch]) % 2
        new_error_mod_p = (error_mod_p[edge_ch] + power * candidate_stab[edge_ch]) % p
        total_new *= modular_probability_from_probs(probs, p, error_at_edge, new_error_mod_p)
        total_old *= modular_probability_from_probs(probs, p, int(error_mod_2[edge_ch]), int(error_mod_p[edge_ch]))
    return total_new, total_old


def apply_move_numba(edge: int, power: int, stab_bool: int, h_error_mod_p: np.ndarray,
                     edge_lookup: np.ndarray, p: int,
                     error_mod_2: np.ndarray, error_mod_p: np.ndarray,
                     new_error_mod_2: np.ndarray, new_error_mod_p: np.ndarray) -> None:
    candidate_stab_label = choose_candidate_stab(edge_lookup, edge, stab_bool)
    candidate_stab = h_error_mod_p[candidate_stab_label]
    n = error_mod_2.shape[0]
    for i in range(n):
        new_error_mod_2[i] = error_mod_2[i]
        new_error_mod_p[i] = error_mod_p[i]
    new_error_mod_2[edge] = (new_error_mod_2[edge] + 1) % 2
    for i in range(n):
        new_error_mod_p[i] = (new_error_mod_p[i] + power * candidate_stab[i]) % p


def binary_entropy_from_chi(chi_vec: np.ndarray, success_vec: np.ndarray) -> float:
    num_success = 0
    total = 0
    for i in range(chi_vec.shape[0]):
        if success_vec[i] != 0:
            num_success += 1
            total += chi_vec[i]
    if num_success == 0:
        return 0.0
    p1 = total / num_success
    p0 = 1.0 - p1
    eps = np.finfo(np.float64).tiny
    p0s = min(max(p0, eps), 1.0)
    p1s = min(max(p1, eps), 1.0)
    return -(p0 * math.log(p0s) + p1 * math.log(p1s)) / math.log(2.0)


def p1_from_chi(chi_vec: np.ndarray, success_vec: np.ndarray) -> float:
    num_success = 0
    total = 0
    for i in range(chi_vec.shape[0]):
        if success_vec[i] != 0:
            num_success += 1
            total += chi_vec[i]
    if num_success == 0:
        return 0.0
    return total / num_success


def run_worm_numba(initial_error_mod_2: np.ndarray, initial_error_mod_p: np.ndarray,
                   h_error_mod_p: np.ndarray, h_mod_p: np.ndarray,
                   edge_lookup_error: np.ndarray, edge_lookup_mod: np.ndarray,
                   stab_edges_error: np.ndarray, stab_edges_mod: np.ndarray,
                   probs: np.ndarray, p: int, d: int,
                   burn_in_steps: int, max_worm_steps: int, num_stabs: int,
                   logical_vec: np.ndarray,
                   candidate_destab_vertex: np.ndarray,
                   plaquette_destab_type_two: np.ndarray,
                   plaquette_destab_type_p: np.ndarray,
                   inverse_two_mod_p: int,
                   syndrome_mode: int) -> Tuple[np.ndarray, np.ndarray, int, int, int, int, int]:
    n = initial_error_mod_2.shape[0]
    error_mod_2 = initial_error_mod_2.copy()
    error_mod_p = initial_error_mod_p.copy()
    new_error_mod_2 = np.empty_like(error_mod_2)
    new_error_mod_p = np.empty_like(error_mod_p)

    head = np.random.randint(0, num_stabs)
    tail = head
    boundary = 0
    worm_success = 0
    accepted_moves = 0
    attempted_moves = 0

    for _ in range(max_worm_steps):
        if worm_success == 1:
            continue
        head_edges = stab_edges_mod[head]
        if head_edges[3] == -1:
            rand_idx = np.random.randint(0, 3)
        else:
            rand_idx = np.random.randint(0, 4)
        edge = head_edges[rand_idx]
        power = np.random.randint(0, p)
        stab_bool = np.random.randint(0, 2)

        prob_move, prob_old = single_move_probability_numba(
            edge, power, error_mod_2, error_mod_p, stab_bool,
            h_error_mod_p, edge_lookup_error, stab_edges_error, p, probs
        )
        if prob_move < 0.0 or prob_old <= 0.0:
            attempted_moves += 1
            continue
        acceptance_prob = prob_move / prob_old
        if acceptance_prob > 1.0:
            acceptance_prob = 1.0
        if np.random.random() > acceptance_prob:
            attempted_moves += 1
            continue

        apply_move_numba(edge, power, stab_bool, h_error_mod_p, edge_lookup_error, p,
                         error_mod_2, error_mod_p, new_error_mod_2, new_error_mod_p)
        tmp = error_mod_2
        error_mod_2 = new_error_mod_2
        new_error_mod_2 = tmp
        tmp = error_mod_p
        error_mod_p = new_error_mod_p
        new_error_mod_p = tmp

        s0 = edge_lookup_mod[edge, 0]
        s1 = edge_lookup_mod[edge, 1]
        tmp_head = s1 if s0 == head else s0

        individual_success = ((tmp_head == tail) and (boundary == 0)) or ((tmp_head == -1) and (boundary == 1))
        worm_success = 1 if (individual_success and (attempted_moves + 1 > burn_in_steps)) else 0
        reset_head_and_tail = 1 if (individual_success and (attempted_moves + 1 <= burn_in_steps)) else 0

        if reset_head_and_tail == 1:
            head = np.random.randint(0, num_stabs)
            tail = head
            boundary = 0
        else:
            set_head_to_tail = 1 if ((tmp_head == -1) and (boundary == 0)) else 0
            head = tail if set_head_to_tail == 1 else tmp_head
            boundary = 1 if ((tmp_head == -1) or (boundary == 1)) else 0
        accepted_moves += 1
        attempted_moves += 1

    full_error = ((p * error_mod_2.astype(np.int64) + (1 - p) * error_mod_p.astype(np.int64)) % d).astype(np.int32)

    if syndrome_mode == 0:
        syndrome = np.zeros(h_mod_p.shape[0], dtype=np.int32)
        for i in range(h_mod_p.shape[0]):
            s = 0
            for j in range(n):
                s += h_mod_p[i, j] * full_error[j]
            syndrome[i] = s % d
        syndrome_mod_two = syndrome % 2
        syndrome_mod_p_aux = ((syndrome - syndrome_mod_two * p) * inverse_two_mod_p) % p
        candidate = np.zeros(n, dtype=INT_DTYPE)
        # delete first syndrome bit for type-two destabilizers
        for i in range(plaquette_destab_type_two.shape[0]):
            coeff = syndrome_mod_two[i + 1]
            if coeff != 0:
                for e in range(n):
                    candidate[e] = (candidate[e] + coeff * plaquette_destab_type_two[i, e]) % d
        for i in range(plaquette_destab_type_p.shape[0]):
            coeff = syndrome_mod_p_aux[i]
            if coeff != 0:
                for e in range(n):
                    candidate[e] = (candidate[e] + coeff * plaquette_destab_type_p[i, e]) % d
    else:
        syndrome = np.zeros(h_mod_p.shape[0], dtype=np.int32)
        for i in range(h_mod_p.shape[0]):
            s = 0
            for j in range(n):
                s += h_mod_p[i, j] * full_error[j]
            syndrome[i] = s % d
        candidate = np.zeros(n, dtype=np.int32)
        for j in range(syndrome.shape[0]):
            coeff = syndrome[j]
            if coeff != 0:
                for e in range(n):
                    candidate[e] = (candidate[e] + coeff * candidate_destab_vertex[j, e]) % d

    diff = full_error - candidate
    dotv = 0
    for i in range(n):
        dotv += diff[i] * logical_vec[i]
    chi_full = int(dotv % d)
    chi = 1 if chi_full == p else 0
    return error_mod_2, error_mod_p, worm_success, accepted_moves, attempted_moves, chi_full, chi


# =========================
# High-level DS driver
# =========================


def _seed_numpy(seed: int) -> None:
    np.random.seed(int(seed) % (2**32 - 1))


def run_worm_moebius_ds(gamma_t: float, syndrome_id: str, moebius_setup: Dict, worm_setup: Dict,
                        keys_setup: Dict) -> Dict[str, np.ndarray]:
    length = moebius_setup["length"]
    width = moebius_setup["width"]
    p = moebius_setup["p"]
    d = 2 * p
    num_samples = worm_setup["num_samples"]
    num_worms = worm_setup["num_worms"]
    burn_in_steps = worm_setup["burn_in_steps"]
    max_worm_steps = worm_setup["max_worm_steps"]
    worm_master_seed = keys_setup["worm_master_seed"]
    error_master_seed = keys_setup["error_master_seed"]

    code = MoebiusCodeTwoOddPrime(length=length, width=width, d=d)
    err_model = ErrorModelLindbladTwoOddPrime(code.num_edges, d=d, gamma_t=gamma_t)

    if syndrome_id == "plaquette":
        num_stabs = code.num_plaquette_checks
        h_error_mod_p = code.h_z_mod_p.astype(INT_DTYPE)
        h_mod_p = code.h_x_mod_p.astype(INT_DTYPE)
        edge_lookup_error = code.vertex_edge_lookup
        edge_lookup_mod = code.plaquette_edge_lookup
        stab_edges_error = code.vertex_stab_edges
        stab_edges_mod = code.plaquette_stab_edges
        logical_vec = code.logical_x.astype(INT_DTYPE)
        syndrome_mode = 0
    elif syndrome_id == "vertex":
        num_stabs = code.num_vertex_checks
        h_error_mod_p = code.h_x_mod_p.astype(INT_DTYPE)
        h_mod_p = code.h_z_mod_p.astype(INT_DTYPE)
        edge_lookup_error = code.plaquette_edge_lookup
        edge_lookup_mod = code.vertex_edge_lookup
        stab_edges_error = code.plaquette_stab_edges
        stab_edges_mod = code.vertex_stab_edges
        logical_vec = code.logical_z.astype(INT_DTYPE)
        syndrome_mode = 1
    else:
        raise ValueError("syndrome_id must be 'plaquette' or 'vertex'")

    chi = np.zeros((num_samples, num_worms), dtype=INT_DTYPE)
    worm_success = np.zeros((num_samples, num_worms), dtype=INT_DTYPE)
    accepted_moves = np.zeros((num_samples, num_worms), dtype=INT_DTYPE)
    attempted_moves = np.zeros((num_samples, num_worms), dtype=INT_DTYPE)

    for s in range(num_samples):
        _seed_numpy(error_master_seed + s)
        initial_error = err_model.generate_random_error().astype(INT_DTYPE)
        initial_error_mod_2 = (initial_error % 2).astype(INT_DTYPE)
        initial_error_mod_p = (initial_error % p).astype(INT_DTYPE)
        for w in range(num_worms):
            _seed_numpy(worm_master_seed + s * num_worms + w)
            _, _, ws, am, tm, _, c = run_worm_numba(
                initial_error_mod_2, initial_error_mod_p,
                h_error_mod_p, h_mod_p,
                edge_lookup_error, edge_lookup_mod, stab_edges_error, stab_edges_mod,
                err_model.probs, p, d,
                burn_in_steps, max_worm_steps, num_stabs,
                logical_vec,
                code.vertex_destab.astype(INT_DTYPE),
                code.plaquette_destab_type_two.astype(INT_DTYPE),
                code.plaquette_destab_type_p.astype(INT_DTYPE),
                code.inverse_two_mod_p,
                syndrome_mode,
            )
            worm_success[s, w] = ws
            accepted_moves[s, w] = am
            attempted_moves[s, w] = tm
            chi[s, w] = c

    return {
        "chi": chi,
        "worm_success": worm_success,
        "accepted_moves": accepted_moves,
        "attempted_moves": attempted_moves,
    }


def worm_ds_conditional_entropy(gamma_t: float, syndrome_id: str, moebius_setup: Dict,
                                worm_setup: Dict, keys_setup: Dict) -> float:
    state = run_worm_moebius_ds(gamma_t, syndrome_id, moebius_setup, worm_setup, keys_setup)
    vals = np.empty(state["chi"].shape[0], dtype=np.float64)
    for i in range(vals.shape[0]):
        vals[i] = binary_entropy_from_chi(state["chi"][i], state["worm_success"][i])
    return float(vals.mean())


def worm_ds_coherent_information(gamma_t: float, moebius_setup: Dict, worm_setup: Dict,
                                 plaquette_keys_setup: Dict, vertex_keys_setup: Dict) -> Tuple[float, float, float]:
    plaquette_ce = worm_ds_conditional_entropy(gamma_t, "plaquette", moebius_setup, worm_setup, plaquette_keys_setup)
    vertex_ce = worm_ds_conditional_entropy(gamma_t, "vertex", moebius_setup, worm_setup, vertex_keys_setup)
    return 1.0 - plaquette_ce - vertex_ce, plaquette_ce, vertex_ce


def worm_ds_logical_error_rate(gamma_t: float, syndrome_id: str, moebius_setup: Dict,
                               worm_setup: Dict, keys_setup: Dict) -> float:
    state = run_worm_moebius_ds(gamma_t, syndrome_id, moebius_setup, worm_setup, keys_setup)
    vals = np.empty(state["chi"].shape[0], dtype=np.float64)
    for i in range(vals.shape[0]):
        vals[i] = p1_from_chi(state["chi"][i], state["worm_success"][i])
    return float(vals.mean())


def worm_ds_logical_error_rates(gamma_t: float, moebius_setup: Dict, worm_setup: Dict,
                                plaquette_keys_setup: Dict, vertex_keys_setup: Dict) -> Tuple[float, float]:
    p_plaq = worm_ds_logical_error_rate(gamma_t, "plaquette", moebius_setup, worm_setup, plaquette_keys_setup)
    p_vert = worm_ds_logical_error_rate(gamma_t, "vertex", moebius_setup, worm_setup, vertex_keys_setup)
    return p_plaq, p_vert


# =========================
# Plotting / script helpers
# =========================

matplotlib.rcParams['mathtext.fontset'] = 'cm'
tex_rc_params = {
    'backend': 'ps',
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'serif',
}


def plot_logical_error_rate(result: Dict, save: bool = False):
    length = result["moebius_setup"]["length"]
    width = result["moebius_setup"]["width"]
    p = result["moebius_setup"]["p"]
    gamma_array = result["gamma_t"]
    plaquette_array = result["plaquette_logical_error_rate"]
    vertex_array = result["vertex_logical_error_rate"]
    with plt.rc_context(tex_rc_params):
        _, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.scatter(gamma_array, plaquette_array, label="plaquette")
        ax.scatter(gamma_array, vertex_array, label="vertex")
        ax.set_xlabel("$\\gamma t$")
        ax.set_ylabel("logical error rate")
        ax.grid()
        ax.legend()
        ax.set_title(f"$L={length},\\, w={width},\\, p={p}$")
        filename = f"logical_error_rate_moebius_length_{length}_width_{width}_p_{p}"
        if save:
            plt.savefig(filename + ".svg", bbox_inches='tight', transparent=True, pad_inches=0)
            plt.savefig(filename + ".pdf", bbox_inches='tight', transparent=True, pad_inches=0)
            plt.savefig(filename + ".png", bbox_inches='tight', transparent=True, pad_inches=0)
        plt.show()


def plot_coherent_information(result: Dict, save: bool = False):
    length = result["moebius_setup"]["length"]
    width = result["moebius_setup"]["width"]
    p = result["moebius_setup"]["p"]
    gamma_array = result["gamma_t"]
    coherent_info_array = result["coherent_information"]
    with plt.rc_context(tex_rc_params):
        _, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.scatter(gamma_array, coherent_info_array)
        ax.set_xlabel("$\\gamma t$")
        ax.set_ylabel("$I_{\\mathrm{coh}}$")
        ax.grid()
        ax.set_title(f"$L= {length}, \\, w = {width}, \\, p={p}$")
        if save:
            filename = f"coherent_information_moebius_length_{length}_width_{width}_p_{p}"
            plt.savefig(filename + ".svg", bbox_inches='tight', transparent=True, pad_inches=0)
            plt.savefig(filename + ".pdf", bbox_inches='tight', transparent=True, pad_inches=0)
            plt.savefig(filename + ".png", bbox_inches='tight', transparent=True, pad_inches=0)
        plt.show()


def run_worm_simulation(gamma_list: list, moebius_setup: Dict, worm_setup: Dict,
                        compute_coherent_information: bool = False) -> Dict:
    num_gamma = len(gamma_list)
    result = {
        "gamma_t": gamma_list,
        "worm_setup": worm_setup,
        "moebius_setup": moebius_setup,
        "plaquette_worm_master_seed": np.random.randint(0, 1_000_000, num_gamma).tolist(),
        "plaquette_error_master_seed": np.random.randint(0, 1_000_000, num_gamma).tolist(),
        "vertex_worm_master_seed": np.random.randint(0, 1_000_000, num_gamma).tolist(),
        "vertex_error_master_seed": np.random.randint(0, 1_000_000, num_gamma).tolist(),
    }
    try:
        with open('/sys/devices/virtual/dmi/id/product_name') as f:
            result["machine_id"] = f.read().strip()
        with open('/sys/devices/virtual/dmi/id/sys_vendor') as f:
            result["vendor"] = f.read().strip()
    except FileNotFoundError:
        result["machine_id"] = platform.machine()
        result["vendor"] = "Apple" if platform.system() == "Darwin" else "unknown"
    result["number_of_available_cpus"] = os.cpu_count()
    result["number_of_used_cpus"] = os.cpu_count()
    result["plaquette_logical_error_rate"] = []
    result["vertex_logical_error_rate"] = []
    if compute_coherent_information:
        result["plaquette_conditional_entropy"] = []
        result["vertex_conditional_entropy"] = []
        result["coherent_information"] = []

    start = time.time()
    for index, gamma_t in enumerate(gamma_list):
        print(f"Index: {index}")
        print(f"Gamma: {gamma_t}")
        plaquette_keys_setup = {
            "worm_master_seed": result["plaquette_worm_master_seed"][index],
            "error_master_seed": result["plaquette_error_master_seed"][index],
        }
        vertex_keys_setup = {
            "worm_master_seed": result["vertex_worm_master_seed"][index],
            "error_master_seed": result["vertex_error_master_seed"][index],
        }
        p_plaq, p_vert = worm_ds_logical_error_rates(gamma_t, moebius_setup, worm_setup, plaquette_keys_setup, vertex_keys_setup)
        print(f"Plaquette logical error rate: {p_plaq}")
        print(f"Vertex logical error rate: {p_vert}")
        result["plaquette_logical_error_rate"].append(float(p_plaq))
        result["vertex_logical_error_rate"].append(float(p_vert))
        if compute_coherent_information:
            ci, ce_p, ce_v = worm_ds_coherent_information(gamma_t, moebius_setup, worm_setup, plaquette_keys_setup, vertex_keys_setup)
            result["coherent_information"].append(float(ci))
            result["plaquette_conditional_entropy"].append(float(ce_p))
            result["vertex_conditional_entropy"].append(float(ce_v))

    result["computation_time"] = time.time() - start
    result["time_unit"] = "sec"
    return result


def save_json(data: Dict, filename: str) -> None:
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=2)
