# Module for the linear algebra over finite fields

import numpy as np
from typing import Tuple
from numpy.typing import NDArray
import math

def is_prime(n: int) -> bool:
    """Check if a number is prime using trial division.

    This simple implementation is fine for small/moderate primes. For very
    large moduli you may want a faster probabilistic test (Miller-Rabin).
    """
    
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def finite_field_gauss_jordan_elimination(
        mat: NDArray[np.int_],
        p: int | np.int_
) -> NDArray[np.int_]:
    """Return the reduced row-echelon form (RREF) of ``mat`` over GF(p).

    This performs Gauss-Jordan elimination: each pivot is normalized to 1
    and the pivot column is eliminated in all other rows. Over a finite 
    field the resulting RREF is unique. Note that if the matrix is 
    full-rank, the RREF will be the identity matrix.

    Args:
        mat: Numpy array representing the matrix. The array is copied
            and not mutated.
        p: Prime modulus (must be prime so inverses exist for non-zero elems).

    Returns:
        A new numpy array containing the RREF of ``mat`` modulo ``p``.
    """
    if not is_prime(p):
        raise ValueError(f"Modulus p={p} is not prime; "
                         f"gauss_jordan_elimination requires a prime p.")

    p = int(p)
    num_rows, num_cols = mat.shape
    # Converts the matrix to mod p
    mat = mat.copy() % p
    row = 0
    for col in range(num_cols):
        if row >= num_rows:
            break
        pivot_rows = np.where(mat[row:, col] != 0)[0]
        if len(pivot_rows) == 0:
            continue
        pivot_row = pivot_rows[0] + row
        if pivot_row != row:
            mat[[row, pivot_row]] = mat[[pivot_row, row]]
        inv_pivot = pow(int(mat[row, col]), -1, p)
        mat[row] = (mat[row] * inv_pivot) % p
        for r in range(num_rows):
            if r != row and mat[r, col] != 0:
                mat[r] = (mat[r] - mat[r, col] * mat[row]) % p
        row += 1
    return mat

def finite_field_matrix_rank(
        mat: NDArray[np.int_],
        p: int       
) -> int:
    """Return the rank of ``mat`` over GF(p).

    Args:
        mat: Numpy array representing the matrix. The array is copied
            and not mutated.
        p: Prime modulus (must be prime so inverses exist for non-zero elems).

    Returns:
        The rank of the matrix.
    """

    new_mat = finite_field_gauss_jordan_elimination(mat, p)
    
    rank = mat.shape[0]
    for row in reversed(range(mat.shape[0])):
        if np.count_nonzero(new_mat[row, :]) != 0:
            break
        else:
            rank -= 1
    return int(rank)

def finite_field_inverse(
        mat: NDArray[np.int_],
        p: int
) -> NDArray[np.int_]:
    """Return the inverse of ``mat`` over GF(p).

    Args:
        mat: Numpy array representing the matrix. 
        p: Prime modulus (must be prime so inverses exist for non-zero elems).

    Returns:
        A new numpy array containing the inverse of ``mat`` modulo ``p``.

    Raises:
        ValueError: If the matrix is not invertible.
    """

    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix must be square to compute inverse.")

    n = mat.shape[0]
    # Create an augmented matrix [mat | I]
    augmented_mat = np.hstack((mat % p, np.eye(n, dtype=np.int_) % p))
    rref_mat = finite_field_gauss_jordan_elimination(augmented_mat, p)
    # Check if left side is identity
    if not np.array_equal(rref_mat[:, :n], np.eye(n, dtype=np.int_)):
        raise ValueError("Matrix is not invertible over GF(p).")
    inverse_mat = rref_mat[:, n:] % p
    return inverse_mat

def finite_field_pseudoinverse(
        mat: NDArray[np.int_],
        p: int
) -> NDArray[np.int_]:
    """Return the pseudoinverse of ``mat`` over GF(p). If A = mat
    is a n x m matrix with n < m and rank(A) = n the function 
    returns a m x n matrix B such that A B = I_{n x n}.

    Args:
        mat: Numpy array representing the matrix. 
        p: Prime modulus (must be prime so inverses exist for non-zero elems).

    Returns:
        A new numpy array containing the pseudoinverse of ``mat`` modulo ``p``.

    Raises:
        ValueError: If the rank of the matrix is not equal to the number of
        rows.
    """

    if mat.shape[0] >= mat.shape[1]:
        raise ValueError(f"The number of rows should be smaller than the"
                         f"number of columns.")
    n, m = mat.shape
    
    augmented_mat = np.hstack((mat % p, np.eye(n, dtype=np.int_) % p))
    rref_mat = finite_field_gauss_jordan_elimination(augmented_mat, p)
    if not np.array_equal(rref_mat[:, :n], np.eye(n, dtype=np.int_)):
        raise ValueError(f"The rank of the matrix must be equal to the "
                         f"number of rows.")
    
    pseudo_inv_mat = \
        np.vstack((rref_mat[:, m:], np.zeros([m - n, n], dtype=np.int_)))
        
    
    return pseudo_inv_mat
    

    



    
    


def gauss_jordan_elimination(
        mat: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the reduced row-echelon form (RREF) of ``mat`` over the 
    real numbers.

    This performs Gauss-Jordan elimination: each pivot is normalized to 1
    and the pivot column is eliminated in all other rows. 
    The resulting RREF is unique. Note that if the matrix is 
    full-rank, the RREF will be the identity matrix.

    Args:
        mat: Numpy array representing the matrix. The array is copied
            and not mutated.

    Returns:
        A new numpy array containing the RREF of ``mat``.
    """

    num_rows, num_cols = mat.shape
    mat = mat.copy()
    row = 0
    for col in range(num_cols):
        if row >= num_rows:
            break
        pivot_rows = np.where(mat[row:, col] != 0)[0]
        if len(pivot_rows) == 0:
            continue
        pivot_row = pivot_rows[0] + row
        if pivot_row != row:
            mat[[row, pivot_row]] = mat[[pivot_row, row]]
        inv_pivot = 1 / mat[row, col]
        mat[row] = (mat[row] * inv_pivot)
        for r in range(num_rows):
            if r != row and mat[r, col] != 0:
                mat[r] = (mat[r] - mat[r, col] * mat[row])
        row += 1
    return mat

def matrix_rank(
        mat: NDArray[np.float64]  
) -> int:
    """Return the rank of ``mat`` over the real numbers.

    Args:
        mat: Numpy array representing the matrix. The array is copied
            and not mutated.

    Returns:
        The rank of the matrix.
    """

    new_mat = gauss_jordan_elimination(mat)
    # We compute the rank by summing the diagonal elements after 
    # Gauss-Jordan elimination
    rank = np.sum(np.diag(new_mat))
    return int(rank)








