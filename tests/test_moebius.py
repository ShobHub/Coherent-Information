# Tests for the generation of the Moebius code

import pytest
from coherentinfo.moebius import build_moebius_code_vertex
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

@pytest.fixture
def moebius_code_vertex_example(
) -> List[Tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """Provides example Moebius code matrices for testing."""
    examples = []

    # Example 1: length=5, width=3
    h_z1, l_z1 = build_moebius_code_vertex(5, 3)
    examples.append((h_z1, l_z1))

    # Example 2: length=7, width=9
    h_z2, l_z2 = build_moebius_code_vertex(7, 9)
    examples.append((h_z2, l_z2))

    # Example 3: length=3, width=15
    h_z3, l_z3 = build_moebius_code_vertex(3, 15)
    examples.append((h_z3, l_z3))

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
    for idx, (h_z, l_z) in enumerate(moebius_code_vertex_example):
        commutation = np.mod(np.sum(h_z @ l_z), 2)
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
            build_moebius_code_vertex(length, width)

def test_hz() -> None:
    """Test specific known values of the h_z matrix for a small Moebius code."""
    length = 5
    width = 3
    h_z, _ = build_moebius_code_vertex(length, width)

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

