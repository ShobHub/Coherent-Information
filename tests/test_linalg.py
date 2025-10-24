# Tests for the linear algebra utilities

import pytest
from coherentinfo.linalg import (
    finite_field_gauss_jordan_elimination,
    finite_field_matrix_rank,
    gauss_jordan_elimination,
    matrix_rank,
    is_prime
)
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

def test_is_prime() -> None:
    """Test the is_prime function."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    non_primes = [0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]

    for p in primes:
        assert is_prime(p), f"{p} should be prime"

    for n in non_primes:
        assert not is_prime(n), f"{n} should not be prime"

@pytest.fixture
def example_integer_matrices() -> List[Tuple[NDArray[np.int_], int, NDArray[np.int_]]]:
    """Provides example matrices and moduli for testing."""
    examples = []

    # Example 1
    mat1 = np.array([[2, 1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0]])
    p1 = 3
    result1 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]])
    rank1 = 2
    
    examples.append((mat1, p1, result1, rank1))

    # Example 2
    mat2 = np.array([[1, 2, 3],
                     [4, 0, 3],
                     [2, 1, 4]])
    p2 = 5
    result2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rank2 = 3
    examples.append((mat2, p2, result2, rank2))

    # Example 3
    mat3 = np.array([[0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 0]])
    p3 = 2
    result3 = np.array([[1, 0, 1], 
                        [0, 1, 1], 
                        [0, 0, 0]])
    rank3 = 2
    examples.append((mat3, p3, result3, rank3))

    return examples


def test_finite_field_gauss_jordan_elimination(
        example_integer_matrices
) -> None:

    for idx, (mat, p, result, _) in enumerate(example_integer_matrices):
        mat_new = finite_field_gauss_jordan_elimination(mat, p)
        assert np.all(mat_new == result) == True, \
            f"Matrices #{idx} do not match"

def test_finite_field_rank(
        example_integer_matrices
) -> None:

    for idx, (mat, p, _, rank) in enumerate(example_integer_matrices):
        rank_new = finite_field_matrix_rank(mat, p)
        assert rank_new == rank, \
            f"Computed rank of matrix #{idx} do not match the expected rank"
        
    
@pytest.fixture
def example_real_matrices() -> List[Tuple[NDArray[np.int_], int, NDArray[np.int_]]]:
    """Provides example matrices and moduli for testing."""
    examples = []

    # Example 1
    mat1 = np.array([[2.0, 1.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0, 0.0],
                     [0, 1.0, 0.0, 0.0]])
    result1 = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]])
    # Here we just give it given the result above.
    rank1 = 2
    
    examples.append((mat1, result1, rank1))

    # Example 2
    mat2 = np.array([[1.0, 2.0, 3.0],
                     [4.0, 0.0, 3.0],
                     [2.0, 1.0, 4.0]])
    result2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # Here we rely on numpy
    rank2 = np.linalg.matrix_rank(mat2)
    examples.append((mat2, result2, rank2))

    # Example 3
    mat3 = np.array([[0.0, 1.0, 1.0],
                     [1.0, 0.0, 1.0],
                     [1.0, 1.0, 0.0]])
    result3 = np.array([[1.0, 0.0, 0.0], 
                        [0.0, 1.0, 0.0], 
                        [0.0, 0.0, 1.0]])
    # Here we rely on numpy
    rank3 = np.linalg.matrix_rank(mat3)
    examples.append((mat3, result3, rank3))

    return examples

def test_gauss_jordan_elimination(
        example_real_matrices
) -> None:

    for idx, (mat, result, _) in enumerate(example_real_matrices):
        mat_new = gauss_jordan_elimination(mat)
        assert np.all(mat_new == result) == True, \
            f"Matrices #{idx} do not match"

def test_rank(
        example_real_matrices
) -> None:

    for idx, (mat, _, rank) in enumerate(example_real_matrices):
        rank_new = matrix_rank(mat)
        assert rank_new == rank, \
            f"Computed rank of matrix #{idx} do not match the expected rank"
