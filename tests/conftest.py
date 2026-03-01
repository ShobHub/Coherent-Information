# Test configuration

import pytest
from coherentinfo.moebius_two_odd_prime import MoebiusCodeTwoOddPrime
from coherentinfo.moebius_qubit import MoebiusCodeQubit
from typing import List, Tuple
from jax.typing import ArrayLike
import jax

@pytest.fixture
def moebius_code_example() -> List[Tuple[ArrayLike, ArrayLike]]:
    """Provides example Moebius code matrices for testing."""
    examples = []

    # Example 0: length=5, width=3
    moebius_code_0 = MoebiusCodeTwoOddPrime(length=5, width=3, d=2 * 17)
    examples.append((moebius_code_0))

    # Example 1: length=7, width=9
    moebius_code_1 = MoebiusCodeTwoOddPrime(length=7, width=9, d=2 * 7)
    examples.append((moebius_code_1))

    # Example 2: length=11, width=21
    moebius_code_2 = MoebiusCodeTwoOddPrime(length=11, width=21, d=2 * 31)
    examples.append((moebius_code_2))

    # Example 3: length=17, width=27
    moebius_code_3 = MoebiusCodeTwoOddPrime(length=17, width=27, d=2 * 3)
    examples.append((moebius_code_3))

    # Example 4: length=5, width=45
    moebius_code_4 = MoebiusCodeTwoOddPrime(length=5, width=45, d=2 * 107)
    examples.append((moebius_code_4))

    # Example 5: length=5, width=45
    moebius_code_5 = MoebiusCodeTwoOddPrime(length=7, width=5, d=2 * 3)
    examples.append((moebius_code_5))

    return examples

@pytest.fixture
def moebius_code_qubit_example() -> List[Tuple[ArrayLike, ArrayLike]]:
    """Provides example Moebius code matrices for testing."""
    examples = []

    # Example 0: length=5, width=3
    moebius_code_0 = MoebiusCodeQubit(length=5, width=3)
    examples.append((moebius_code_0))

    # Example 1: length=7, width=9
    moebius_code_1 = MoebiusCodeQubit(length=7, width=9)
    examples.append((moebius_code_1))

    # Example 2: length=11, width=21
    moebius_code_2 = MoebiusCodeQubit(length=11, width=21)
    examples.append((moebius_code_2))

    # Example 3: length=17, width=27
    moebius_code_3 = MoebiusCodeQubit(length=17, width=27)
    examples.append((moebius_code_3))

    # Example 4: length=5, width=45
    moebius_code_4 = MoebiusCodeQubit(length=5, width=45)
    examples.append((moebius_code_4))

    # Example 5: length=5, width=45
    moebius_code_5 = MoebiusCodeQubit(length=7, width=5)
    examples.append((moebius_code_5))

    return examples