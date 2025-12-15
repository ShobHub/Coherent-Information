# Contains functions to postprocess the sampled data 
# and compute the coherent information.

import jax
import jax.numpy as jnp
from jax import Array, jit
from typing import Tuple


def aggregate_data(result: Array) -> Tuple[Array, Array]:
    """ It aggregates results by syndrome and computes their
    counts.

    Args:
        result: array whose rows represent the syndromes and the 
            corresponding chi = 0, 1. The chi is the last element
            in each row, while the rest of the row represents
            the syndrome. 
    """

    num_data, m = result.shape

    # 1. Isolate the key (first m-1 elements) and the flag (last element)
    syndrome_rows = result[:, :-1] # Shape: num_data x (m-1)
    flags = result[:, -1].astype(jnp.int16) # Shape: num_data, (0 or 1)

    # 2. Find unique prefixes and assign a unique index to each original row
    # 'indices' will map each original row to a unique index ID 
    # (0 to num_unique-1)
    unique_syndromes, indices = jnp.unique(
        syndrome_rows, 
        axis=0, 
        return_inverse=True
    )
    
    num_unique = unique_syndromes.shape[0] # number of unique prefixes

    # 3. Use JAX's bincount to aggregate counts based on the unique index.
    # We use the unique index (indices) as the 'vector' to group by.
    # The 'weights' are the flags (0 or 1).
    # The output will have shape (num_unique,), where the value at index i is the 
    # sum of flags for all rows belonging to the unique prefix i.
    # Since the flags are 0 or 1, this sum gives the count of '1's.
    counts_of_ones = jnp.bincount(
        indices,            # The group ID for each row (0 to num_unique-1)
        weights=flags,      # The value to sum (1 for flag=1, 0 for flag=0)
        length=num_unique   # Ensures output size is num_unique (number of unique prefixes)
    )
    
    # The total count of rows for each unique prefix is simply the bincount 
    # of the indices themselves (weights=None means weights=1).
    total_counts = jnp.bincount(
        indices, 
        length=num_unique
    )
    
    # 4. Calculate counts for 0s and combine
    # Count of 0s = Total Count - Count of 1s
    counts_of_zeros = total_counts - counts_of_ones

    sampled_freqs_zero_and_one = \
        jnp.stack([counts_of_zeros, counts_of_ones], axis=1) / num_data

    return unique_syndromes, sampled_freqs_zero_and_one

def compute_conditional_entropy_term(
    probs_zero_and_one: Array
) -> Array:
    """Computes the entropy term associated with a syndrome in 
    the expression for the conditional entropy 
    
    Args:
        probs_zero_and_one: probabilities of having a certain syndrome
            with chi either zero or one. Typically these are estimated
            from sample frequencies. The array must have length 2.
    
    Returns:
        -prob(syndrome, 0) log_2 prob(0| syndrome) 
            -prob(syndrome, 1) log_2 prob(1| syndrome) 


    """
    prob_syndrome = jnp.sum(probs_zero_and_one)
    cond_prob_zero = probs_zero_and_one[0] / prob_syndrome 
    cond_prob_one = probs_zero_and_one[1] / prob_syndrome
    entropy_term_a = -jax.scipy.special.xlogy(
        probs_zero_and_one[0], 
        cond_prob_zero
    )
    entropy_term_b = -jax.scipy.special.xlogy(
        probs_zero_and_one[1], 
        cond_prob_one
    )
    return (entropy_term_a + entropy_term_b) / jnp.log(2)

def conditional_entropy(
    probs_zero_and_one: Array
) -> Array:
    """Computes the conditional entropy associated with the probabilities
    of observing a certain syndrome and chi either 0 or 1
    
    Args:
        probs_zero_and_one: array of probabilities of having the observed 
            syndromes with chi either zero or one. Typically these are 
            estimated from sample frequencies.
    
    Returns:
        sum_{syndrome} (-prob(syndrome, 0) log_2 prob(0| syndrome) 
            -prob(syndrome, 1) log_2 prob(1| syndrome) )
    """

    compute_conditional_entropy_term_jit = jax.jit(compute_conditional_entropy_term)
    entropy_terms = jax.vmap(compute_conditional_entropy_term_jit)(probs_zero_and_one)
    return jnp.sum(entropy_terms)







