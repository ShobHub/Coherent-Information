# Tests for the linear algebra utilities

import pytest
from coherentinfo.linalg import (
    finite_field_gauss_jordan_elimination,
    finite_field_matrix_rank,
    finite_field_inverse,
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
def example_finite_field_matrices() -> List[Tuple[NDArray[np.int_], int, NDArray[np.int_]]]:
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

@pytest.fixture
def example_invertible_finite_field_matrices() -> List[Tuple[NDArray[np.int_], int]]:
    """Provides example invertible matrices and moduli for testing."""
    examples = []

    # Example 1
    mat1 = np.array([[1, 2],
                     [3, 4]])
    p1 = 5

    examples.append((mat1, p1))

    # Example 2
    mat2 = np.array([[2, 3, 1],
                     [1, 0, 4],
                     [4, 2, 5]])
    p2 = 7
    examples.append((mat2, p2))

    # Example 3
    mat3 = np.array([[1, 1, 1],
                     [0, 1, 2],
                     [1, 0, 1]])
    p3 = 3
    examples.append((mat3, p3))

    return examples

@pytest.fixture()
def example_full_rank_finite_field_matrices() -> List[Tuple[NDArray[np.int_], int]]:
    """Provides example full rank matrices and moduli for testing"""

    examples = []

    # Example 1
    mat1 = np.array([[1, 4, 1, 5, 2], 
                    [0, 3, 6, 2, 5], 
                    [1, 1, 3, 0, 1], 
                    [2, 2, 1, 0, 6]])
    p1 = 7

    examples.append((mat1, p1))

    mat2 = np.array([[1, 0, 1, 2, 2, 0, 0], 
                     [0, 2, 0, 1, 1, 0, 0], 
                     [1, 1, 0, 0, 1, 1, 1]])
    p2 = 3

    examples.append((mat2, p2))

    mat3 = np.array([[1, 0, 1, 2, 2, 0, 0, 6, 9], 
                     [0, 2, 0, 1, 1, 0, 0, 0, 8], 
                     [1, 1, 0, 0, 1, 1, 1, 5, 5], 
                     [1, 1, 0, 1, 1, 3, 1, 5, 5], 
                     [1, 0, 0, 1, 1, 3, 4, 6, 5]])
    p3 = 11

    examples.append((mat3, p3))

    mat4 = np.array([[1, 0, 1, 2, 4, 0, 0, 0, 0, 1], 
                    [0, 2, 0, 1, 1, 0, 0, 0, 2, 4], 
                    [1, 1, 0, 0, 1, 1, 1, 2, 2, 2], 
                    [1, 3, 0, 1, 1, 0, 1, 2, 2, 0], 
                    [1, 0, 0, 1, 1, 0, 1, 0, 2, 1], 
                    [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]])
    p4 = 5

    examples.append((mat4, p4))

    return examples


def test_finite_field_gauss_jordan_elimination(
        example_finite_field_matrices
) -> None:

    for idx, (mat, p, result, _) in enumerate(example_finite_field_matrices):
        mat_new = finite_field_gauss_jordan_elimination(mat, p)
        assert np.all(mat_new == result) == True, \
            f"Matrices #{idx} do not match"

def test_finite_field_rank(
        example_finite_field_matrices
) -> None:

    for idx, (mat, p, _, rank) in enumerate(example_finite_field_matrices):
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

def test_not_invertible_finite_field_matrices() -> None:
    """Test that non-invertible matrices raise errors."""
    mat1 = np.array([[2, 4],
                     [1, 2]])
    p1 = 3  # Modulus where the matrix is not invertible

    with pytest.raises(ValueError):
        finite_field_inverse(mat1, p1)

    mat2 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    p2 = 5  # Modulus where the matrix is not invertible

    with pytest.raises(ValueError):
        finite_field_inverse(mat2, p2)

def test_finite_field_inverse(
        example_invertible_finite_field_matrices
) -> None:
    """Test the finite field matrix inversion."""
    for idx, (mat, p) in enumerate(example_invertible_finite_field_matrices):
        inv_mat = finite_field_inverse(mat, p)
        identity = (mat @ inv_mat) % p
        expected_identity = np.eye(mat.shape[0], dtype=int) % p
        assert np.all(identity == expected_identity), \
            f"Inverse computation failed for matrix #{idx}"
    



    
