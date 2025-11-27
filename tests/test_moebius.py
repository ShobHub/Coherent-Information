# Tests for the generation of the Moebius code

import pytest
from coherentinfo.moebius import MoebiusCode
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray
from coherentinfo.linalg import finite_field_matrix_rank

@pytest.fixture
def moebius_code_example() -> List[Tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """Provides example Moebius code matrices for testing."""
    examples = []

    # Example 1: length=5, width=3
    moebius_code_1 = MoebiusCode(length=5, width=3, d=2 * 17)
    examples.append((moebius_code_1))

    # Example 2: length=7, width=9
    moebius_code_2 = MoebiusCode(length=7, width=9, d=2 * 7)
    examples.append((moebius_code_2))

    # Example 3: length=11, width=21
    moebius_code_3 = MoebiusCode(length=11, width=21, d=2 * 31)
    examples.append((moebius_code_3))

    # Example 4: length=11, width=27
    moebius_code_4 = MoebiusCode(length=17, width=27, d=2 * 3)
    examples.append((moebius_code_4))

    # Example 5: length=5, width=45
    moebius_code_5 = MoebiusCode(length=5, width=45, d=2 * 107)
    examples.append((moebius_code_5))

    # Example 6: length=5, width=45
    moebius_code_6 = MoebiusCode(length=7, width=5, d=2 * 3)
    examples.append((moebius_code_6))

    return examples

def test_vertex_shapes(moebius_code_example) -> None:
    """Tests that the generated matrices have the correct shapes."""
    # The fixture provides example (h_z, l_z, num_edges) tuples.
    for idx, moebius_code in enumerate(moebius_code_example):
        # derive expected sizes from returned matrices
        exp_num_edges = moebius_code.num_edges
        exp_num_vertex_checks = moebius_code.num_vertex_checks
        exp_num_plaquette_checks = moebius_code.num_plaquette_checks

        h_z = moebius_code.h_z
        l_z = moebius_code.logical_z
        h_x = moebius_code.h_x

        assert h_z.shape == (exp_num_vertex_checks, exp_num_edges), \
            f"Fixture example #{idx} produced unexpected h_z shape"
        assert l_z.shape == (exp_num_edges,), \
            f"Fixture example #{idx} produced unexpected l_z shape"
        assert h_x.shape == (exp_num_plaquette_checks, exp_num_edges), \
            f"Fixture example #{idx} produced unexpected h_x shape"

def test_commutation(moebius_code_example) -> None:
    for idx, moebius_code in enumerate(moebius_code_example):
        h_z = moebius_code.h_z
        h_x = moebius_code.h_x
        assert np.count_nonzero(h_x @ h_z.T) == 0, \
            f"The stabilizers do not commute for example #{idx}."

def test_logical_stab_commutation(moebius_code_example) -> None:
    """Tests that the logical operators commute with the stabilizers."""
    for idx, moebius_code in enumerate(moebius_code_example):
        h_x = moebius_code.h_x
        logical_z = moebius_code.logical_z
        d = moebius_code.d
        commutation_z = np.count_nonzero(np.mod(h_x @ logical_z, d))
        assert commutation_z == 0, \
            f"Logical Z operator does not commute with stabilizers \n" \
            f"in example #{idx}"
        h_z = moebius_code.h_z
        logical_x = moebius_code.logical_x
        commutation_x = np.count_nonzero(np.mod(h_z @ logical_x, d))
        assert commutation_x == 0, \
            f"Logical X operator does not commute with stabilizers in example #{idx}"

def test_logical_commutation(moebius_code_example) -> None:
    """Tests that the logical X and Z operators anticommute."""
    for idx, (moebius_code) in enumerate(moebius_code_example):
        logical_x = moebius_code.logical_x
        logical_z = moebius_code.logical_z
        d = moebius_code.d
        commutation = np.mod(logical_x @ logical_z, d)
        assert commutation == np.int16(d / 2), \
            f"Logical X and Z operators do not anticommute in example #{idx}"

def test_invalid_parameters() -> None:
    """Tests that invalid parameters raise ValueError."""
    invalid_params = [
        (2, 3),  # length too small
        (3, 2),  # width too small
        (4, 3),  # length not odd
        (3, 4),  # width not odd
    ]
    for length, width in invalid_params:
        with pytest.raises(ValueError):
            MoebiusCode(length=length, width=width, d=2)

def test_hz() -> None:
    """Tests specific known values of the h_z matrix for a small 
    Moebius code."""
    length = 5
    width = 3
    moebius_code = MoebiusCode(length=length, width=width, d=2)
    h_z = moebius_code.h_z

    # Manually constructed expected h_z matrix for length=5, width=3
    expected_h_z = np.zeros([10, 25], dtype=np.int16)
    expected_h_z[0, [0, 9, 10, 15]] = 1
    expected_h_z[1, [0, 1, 11, 16]] = -1
    expected_h_z[2, [1, 2, 12, 17]] = 1
    expected_h_z[3, [2, 3, 13, 18]] = -1
    expected_h_z[4, [3, 4, 14, 19]] = 1
    expected_h_z[5, [5, 4, 15, 20]] = -1
    expected_h_z[6, [5, 6, 16, 21]] = 1
    expected_h_z[7, [6, 7, 17, 22]] = -1
    expected_h_z[8, [7, 8, 18, 23]] = 1
    expected_h_z[9, [8, 9, 19, 24]] = -1

    assert np.all(h_z == expected_h_z) == True, \
            f"The H_Z matrix does not match the expected one."

def test_hx() -> None:
    """Tests specific known values of the h_x matrix for a small Moebius code."""
    length = 5
    width = 3
    moebius_code = MoebiusCode(length=length, width=width, d=2)
    h_x = moebius_code.h_x

    expected_h_x = np.zeros([moebius_code.num_plaquette_checks,
               moebius_code.num_edges], dtype=np.int16)

    # Plaquette 0
    expected_h_x[0, 0] = 1
    expected_h_x[0, 10] = -1
    expected_h_x[0, 11] = -1

    # Plaquette 1
    expected_h_x[1, 1] = -1
    expected_h_x[1, 11] = 1
    expected_h_x[1, 12] = 1

    # Plaquette 2
    expected_h_x[2, 2] = 1
    expected_h_x[2, 12] = -1
    expected_h_x[2, 13] = -1

    # Plaquette 3
    expected_h_x[3, 3] = -1
    expected_h_x[3, 13] = 1
    expected_h_x[3, 14] = 1

    # Plaquette 4 (twisted)
    expected_h_x[4, 24] = -1
    expected_h_x[4, 9] = 1
    expected_h_x[4, 10] = -1

    # Plaquette 5
    expected_h_x[5, 0] = -1
    expected_h_x[5, 5] = -1
    expected_h_x[5, 15] = 1
    expected_h_x[5, 16] = 1

    # Plaquette 6
    expected_h_x[6, 1] = 1
    expected_h_x[6, 6] = 1
    expected_h_x[6, 16] = -1
    expected_h_x[6, 17] = -1

    # Plaquette 7
    expected_h_x[7, 2] = -1
    expected_h_x[7, 7] = -1
    expected_h_x[7, 17] = 1
    expected_h_x[7, 18] = 1

    # Plaquette 8
    expected_h_x[8, 3] = 1
    expected_h_x[8, 8] = 1
    expected_h_x[8, 18] = -1
    expected_h_x[8, 19] = -1

    # Plaquette 9 (twisted)
    expected_h_x[9, 4] = -1
    expected_h_x[9, 9] = -1
    expected_h_x[9, 19] = 1
    expected_h_x[9, 15] = 1

    # Plaquette 10
    expected_h_x[10, 5] = 1
    expected_h_x[10, 20] = -1
    expected_h_x[10, 21] = -1

    # Plaquette 11
    expected_h_x[11, 6] = -1
    expected_h_x[11, 21] = 1
    expected_h_x[11, 22] = 1

    # Plaquette 12
    expected_h_x[12, 7] = 1
    expected_h_x[12, 22] = -1
    expected_h_x[12, 23] = -1

    # Plaquette 13
    expected_h_x[13, 8] = -1
    expected_h_x[13, 23] = 1
    expected_h_x[13, 24] = 1

    # Plaquette 14 (twisted)
    expected_h_x[14, 4] = 1
    expected_h_x[14, 14] = -1
    expected_h_x[14, 20] = -1

    assert np.all(h_x == expected_h_x) == True, \
            f"The H_X matrix does not match the expected one."
    
def test_plaquette_constraint(moebius_code_example) -> None:
    """Tests that the product of all the plaquette stabilizers to the power 
    of d/2 is identity."""

    for idx, moebius_code in enumerate(moebius_code_example):
        h_x = moebius_code.h_x
        d = moebius_code.d
        sum_of_rows = np.mod(np.int16(d / 2) * np.sum(h_x, axis=0), d)
        assert np.count_nonzero(sum_of_rows) == 0, \
                f"The H_X matrix does not satisfy the plaquette \n " \
                f"constraint in example #{idx}."

def test_vertex_destabilizers(moebius_code_example) -> None:
    """Tests that the vertex destabilizers (X-type) anticommute only with the 
    corresponding vertex destabilizer (Z-type) and commute with the logical
     Z operator """
    
    for idx, moebius_code in enumerate(moebius_code_example):
        h_z = moebius_code.h_z
        vertex_destab = moebius_code.vertex_destab 
        id_mat = np.identity(moebius_code.num_vertex_checks)
        logical_z = moebius_code.logical_z
        res_h_z = np.count_nonzero(h_z @ vertex_destab.T - id_mat)
        res_logical_com = np.count_nonzero(logical_z @ vertex_destab.T)
        assert res_h_z == 0, \
                f"The vertex destabilizers and checks do not have the \n" \
                f"correct commutation relation in example #{idx}."
        assert res_logical_com == 0, \
                f"The vertex destabilizers do not commute with the \n" \
                f"logical Z in example #{idx}."

def test_plaquette_destabilizer_qubit(moebius_code_example) -> None:
    """Tests that the plaquette destabilizers for the qubit case (Z-type)
    anticommute only with the corresponding plaquette stabilizer (X-type)
    and commute with the logical X operator"""

    for idx, moebius_code in enumerate(moebius_code_example):
        h_x_qubit = moebius_code.h_x_qubit 
        plaquette_destab_qubit = moebius_code.plaquette_destab_qubit
        id_mat = np.identity(moebius_code.num_plaquette_checks - 1)
        logical_x = moebius_code.logical_x
        res_h_x_com = \
            np.count_nonzero((h_x_qubit @ plaquette_destab_qubit.T % 2) - 
                             id_mat) 
        
        res_logical_com = \
            np.count_nonzero(logical_x @ plaquette_destab_qubit.T % 2) 
        assert res_h_x_com == 0, \
                f"The plaquette destabilizers and checks in the qubit \n" \
                f"case do not have the correct commutation relation \n" \
                f"in example #{idx}."
        assert res_logical_com == 0, \
                f"The plaquette destabilizers in the qubit case do not \n" \
                f"commute with the logical X in example #{idx}."


def test_q_not_odd_prime() -> None:
    """Tests that invalid dimension raise ValueError when trying to build
    plaquette destabilizers."""
    invalid_dims = [2 * 4, 2 * 15, 2 * 2, 2 * 99]

    for dim in invalid_dims:
        moebius_code = MoebiusCode(length=7, width=5, d=dim)
        mat = moebius_code.plaquette_destab_type_two
        assert mat is None, \
                f"The plaquette destabilizers are not None as \n" \
                f"expected for qudit dimension #{dim}"

        
def test_plaquette_destabilizers_type_two(moebius_code_example) -> None:
    """Tests that the plaquette destabilizers (Z-type) of type two
    anticommute only with the corresponding plaquette stabilizer (X-type)
    and commute with the logical X operator"""

    for idx, moebius_code in enumerate(moebius_code_example):
        p = moebius_code.p
        h_x_eff = p * np.delete(moebius_code.h_x, 0, axis=0)
        plaquette_destab = moebius_code.plaquette_destab_type_two
        id_mat = np.identity(moebius_code.num_plaquette_checks - 1)
        res_h_x_com = \
            np.count_nonzero((h_x_eff @ plaquette_destab.T % 2) - id_mat)
        assert res_h_x_com == 0, \
                f"The plaquette destabilizers and checks do not have \n" \
                f"the correct commutation relation in example #{idx}." 
         

def test_rank_moebius(moebius_code_example) -> None:
    """Tests that H_X matrix mod d has full rank over the finite field
    F_q with p = d / 2"""
    for idx, moebius_code in enumerate(moebius_code_example):
        p = moebius_code.p
        h_x = moebius_code.h_x
        num_plaquette_checks = moebius_code.num_plaquette_checks 
        rank_h_x = finite_field_matrix_rank(h_x % moebius_code.d, p)
        assert rank_h_x == num_plaquette_checks, \
                f"The matrix H_X is not full rank over the finite field \n" \
                f"p = d / 2 in example #{idx}."
        
def test_plaquette_destabilizers_mod_p(moebius_code_example):
    """Tests that the plaquette destabilizers (Z-type) mod p (p = d / 2)
    anticommute only with the corresponding plaquette stabilizers (X-type) 
    mod p and commute with the logical X. Note that the commutation with the 
    logical X needs to be evaluate mod 2 * p. """
    
    for idx, moebius_code in enumerate(moebius_code_example):
        p = moebius_code.p
        h_x_mod_p = moebius_code.h_x_mod_p
        plaquette_destab_mod_p = moebius_code.plaquette_destab_mod_p
        id_mat = np.identity(moebius_code.num_plaquette_checks)
        res_h_x_com = \
            np.count_nonzero((h_x_mod_p @ plaquette_destab_mod_p.T % p) - id_mat)
        assert res_h_x_com == 0, \
                f"The plaquette destabilizers and checks mod p do not have \n" \
                f"the correct commutation relation in example #{idx}." 
        res_logical_com = \
            moebius_code.logical_x @ (2 * plaquette_destab_mod_p.T) % (2 * p)
        assert np.count_nonzero(res_logical_com) == 0, \
                f"The plaquette destabilizers of type p and logical\n" \
                f"do not have the correct commutation relation in \n" \
                f"example #{idx}."





                
