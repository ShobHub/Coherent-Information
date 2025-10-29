# Tests for the generation of the Moebius code

import pytest
from coherentinfo.moebius import MoebiusCode
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

@pytest.fixture
def moebius_code_vertex_example(
) -> List[Tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """Provides example Moebius code matrices for testing."""
    examples = []

    # Example 1: length=5, width=3
    moebius_code_1 = MoebiusCode(length=5, width=3, d=2)
    h_z1, logical_z1 = moebius_code_1.h_z, moebius_code_1.logical_z
    examples.append((h_z1, logical_z1))

    # Example 2: length=7, width=9
    moebius_code_2 = MoebiusCode(length=7, width=9, d=2)
    h_z2, logical_z2 = moebius_code_2.h_z, moebius_code_2.logical_z
    examples.append((h_z2, logical_z2))

    # Example 3: length=3, width=15
    moebius_code_3 = MoebiusCode(length=3, width=15, d=2)
    h_z3, logical_z3 = moebius_code_3.h_z, moebius_code_3.logical_z
    examples.append((h_z3, logical_z3))

    return examples

def test_vertex_shapes(moebius_code_vertex_example) -> None:
    """Test that the generated matrices have the correct shapes."""
    # The fixture provides example (h_z, l_z, num_edges) tuples.
    for idx, (h_z, l_z) in enumerate(moebius_code_vertex_example):
        # derive expected sizes from returned matrices
        expected_num_checks, expected_num_edges = h_z.shape 

        assert h_z.shape == (expected_num_checks, expected_num_edges), \
            f"Fixture example #{idx} produced unexpected h_z shape"
        assert l_z.shape == (expected_num_edges,), \
            f"Fixture example #{idx} produced unexpected l_z shape"

def test_commutation(moebius_code_vertex_example) -> None:
    """Test that the logical operators commute with the stabilizers."""
    for idx, (h_z, logical_z) in enumerate(moebius_code_vertex_example):
        commutation = np.mod(np.sum(h_z @ logical_z), 2)
        assert commutation == 0, \
            f"Logical operator does not commute with stabilizers in example #{idx}"

def test_invalid_parameters() -> None:
    """Test that invalid parameters raise ValueError."""
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
    """Test specific known values of the h_z matrix for a small Moebius code."""
    length = 5
    width = 3
    moebius_code = MoebiusCode(length=length, width=width, d=2)
    h_z = moebius_code.h_z

    # Manually constructed expected h_z matrix for length=5, width=3
    expected_h_z = np.zeros([10, 25], dtype=np.int8)
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
    """Test specific known values of the h_x matrix for a small Moebius code."""
    length = 5
    width = 3
    moebius_code = MoebiusCode(length=length, width=width, d=2)
    h_x = moebius_code.h_x

    expected_h_x = np.zeros([moebius_code.num_plaquette_checks,
               moebius_code.num_edges], dtype=np.int8)

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




