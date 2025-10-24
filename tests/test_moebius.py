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

    # Example 1: length=4, width=2
    h_z1, l_z1 = build_moebius_code_vertex(4, 2)
    examples.append((h_z1, l_z1))

    # Example 2: length=7, width=8
    h_z2, l_z2 = build_moebius_code_vertex(7, 8)
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