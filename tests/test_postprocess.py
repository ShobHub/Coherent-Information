# Tests for the postprocess module

import pytest 
import jax.numpy as jnp 
from coherentinfo.postprocess import aggregate_data_jax
from jax import Array
from typing import Tuple
import jax

@pytest.fixture 
def syndrome_chi_examples() -> Tuple[Array, Array, Array]:
    examples = []
    results_1 = jnp.array(
        [[0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 1, 0, 0], 
        [0, 1, 0, 0, 0, 0, 3], 
        [0, 0, 0, 0, 5, 0, 3],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]]
    )

    # padded with -1 and ordered in ascending order from top to bottom
    # as jnp.unique does
    unique_syndromes_1 = jnp.array(
        [[0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 5, 0], 
        [0, 1, 0, 0, 0, 0], 
        [0, 1, 0, 0, 1, 0], 
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1]]
    )

    # padded with zeros
    counts_1 = jnp.array(
        [[3, 0], 
        [0, 1], 
        [1, 1], 
        [2, 0], 
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1]]
    )

    examples.append((results_1, unique_syndromes_1, counts_1))

    return examples 


def test_aggregate_data_jax(syndrome_chi_examples) -> None:
    for idx, (results, unique_syndromes, counts) in enumerate(
        syndrome_chi_examples
    ):
        aggregate_data_jit = jax.jit(
        aggregate_data_jax, static_argnums=(1,))
        vertex_pads = -1 * jnp.ones(results.shape[1] - 1) 

        num_samples, _ = results.shape 

        my_unique_syndromes, my_counts = aggregate_data_jit(
            results,
            num_samples,  # Passed as the static (Python int) argument
            vertex_pads
        )

        assert jnp.all(my_unique_syndromes == unique_syndromes) == True, \
            f"The unique syndromes in example #{idx} do not match"
        
        assert jnp.all(my_counts == my_counts) == True, \
            f"The counts in example #{idx} do not match"





