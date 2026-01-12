# Contains functions to postprocess the sampled data 
# and compute the coherent information.

import jax
import jax.numpy as jnp
from jax import Array, jit
from typing import Tuple


def aggregate_data(result: Array) -> Tuple[Array, Array]:
    """ It aggregates results by syndrome and computes their
    counts. Not compatible with jax.jit, despite the use of JAX. 

    Args:
        result: array whose rows represent the syndromes and the 
            corresponding chi = 0, 1. The chi is the last element
            in each row, while the rest of the row represents
            the syndrome. 
        
    Returns:
        Observed syndromes and probabilities of observing 
        a logical error or not.
    """

    num_data, _ = result.shape

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

def aggregate_data_jax(
    result: Array,
    max_unique_size: int,
    pads: Array
) -> Tuple[Array, Array]:
    """
    Aggregates results by syndrome. Maps the logical error flag (p) 
    to 1 for proper probability distribution.

    Args:
        result: array whose rows represent the syndromes and the 
            corresponding chi = 0, 1. The chi is the last element
            in each row, while the rest of the row represents
            the syndrome.
        max_unique_size: fixed number of rows of the output array.
            When jit compiled it must be treated as static argument.
            This is usually equal to the number of samples.
        pads: array of pads to ensure the size is fixed.
    
        Returns:
            Observed syndromes and counts of observing 
            a logical error or not. Both results are padded 
            with dummy values to ensure that the size of the output
            is known at compile time for jax compatiability.
    """
    num_data, _ = result.shape

    # 1. Isolate syndrome and the flag
    syndrome_rows = result[:, :-1] 
    
    # FIX: Map the last column to a binary flag.
    # If the value is p (or anything > 0), it becomes 1.
    # If the value is 0, it stays 0.
    # Note this may be a bit risky, as it would not catch an error
    # of an incorrect commutation. A better way would be to explicitly
    # pass p, but I leave it for now, as the error should be catched
    # by other tests. 
    flags = (result[:, -1] > 0).astype(jnp.int16) 

    # 2. Find unique syndromes. Note that this function gives the unique
    # syndromes in ascending order from top to bottom as if the rows
    # identify a number. You have to keep this in mind when designing
    # tests. 
    unique_syndromes, indices = jnp.unique(
        syndrome_rows, 
        axis=0, 
        return_inverse=True,
        size=max_unique_size, 
        fill_value=pads
    )
    
    # 3. Aggregate binary counts
    # counts_of_ones will now correctly be the NUMBER of times p occurred
    counts_of_ones = jnp.bincount(
        indices, 
        weights=flags, 
        length=max_unique_size
    )
    
    total_counts = jnp.bincount(
        indices, 
        length=max_unique_size
    )
    
    # 4. Calculate counts for 0s (No logical error)
    # This is now guaranteed to be >= 0 because flags are binary
    counts_of_zeros = total_counts - counts_of_ones

    # 5. Normalize and Mask
    is_real_data = total_counts > 0
    raw_counts = jnp.stack([counts_of_zeros, counts_of_ones], axis=1)

    # Ensure padding slots are zeroed out. 
    # Note that we return the counts because it is easier later
    # to run multiple batches in a for loop for example.
    sampled_counts_zero_and_one = jnp.where(
        is_real_data[:, None], 
        raw_counts, 
        0
    )

    return unique_syndromes, sampled_counts_zero_and_one

def update_aggregated_data_jax(
    current_syndromes: jax.Array,
    current_counts: jax.Array,
    new_results: jax.Array,
    max_unique_size: int,
    pads: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    
    # 1. Process New Batch
    new_syndromes = new_results[:, :-1]
    new_flags = (new_results[:, -1] > 0).astype(jnp.int32)
    
    # 2. Unique in Batch (Generic unpacking to avoid ValueError)
    res_batch = jnp.unique(new_syndromes, axis=0, size=max_unique_size, fill_value=pads, return_inverse=True)
    batch_uniques, batch_indices = res_batch[0], res_batch[1]
    
    batch_ones = jnp.bincount(batch_indices, weights=new_flags, length=max_unique_size)
    batch_totals = jnp.bincount(batch_indices, length=max_unique_size)
    batch_counts = jnp.stack([batch_totals - batch_ones, batch_ones], axis=1)

    # 3. Combine
    combined_syndromes = jnp.concatenate([current_syndromes, batch_uniques], axis=0)
    combined_counts = jnp.concatenate([current_counts, batch_counts], axis=0)
    
    # 4. Re-Collapse
    res_final = jnp.unique(combined_syndromes, axis=0, size=max_unique_size, fill_value=pads, return_inverse=True)
    final_syndromes, final_indices = res_final[0], res_final[1]

    updated_zeros = jnp.bincount(final_indices, weights=combined_counts[:, 0], length=max_unique_size)
    updated_ones = jnp.bincount(final_indices, weights=combined_counts[:, 1], length=max_unique_size)
    raw_counts = jnp.stack([updated_zeros, updated_ones], axis=1)

    # 5. RE-SORTING LOGIC: Move pads to the end
    # We create a sorting key: 1 for real syndromes, 0 for pads.
    # By sorting in descending order, real data comes first.
    is_not_pad = jnp.any(final_syndromes != pads, axis=-1)
    
    # argsort with a stable sort to keep syndromes grouped
    # We use -is_not_pad because argsort is ascending; this puts True (1) at the start
    sort_indices = jnp.argsort(~is_not_pad) 
    
    sorted_syndromes = final_syndromes[sort_indices]
    sorted_counts = raw_counts[sort_indices]

    # Final wipe: ensure the rows that are pads have 0 counts
    # (Just in case index 0 collected 'count' logic before sorting)
    final_valid_mask = jnp.any(sorted_syndromes != pads, axis=-1)
    clean_counts = jnp.where(final_valid_mask[:, None], sorted_counts, 0)

    return sorted_syndromes, clean_counts

def compute_conditional_entropy_term(
    probs_zero_and_one: Array
) -> Array:
    """Computes the entropy term associated with a syndrome in 
    the expression for the conditional entropy. It is written 
    in a JAX compatible way
    
    Args:
        probs_zero_and_one: probabilities of having a certain syndrome
            with chi either zero or one. Typically these are estimated
            from sample frequencies. The array must have length 2.
    
    Returns:
        -prob(syndrome, 0) log_2 prob(0| syndrome) 
            -prob(syndrome, 1) log_2 prob(1| syndrome) 


    """
    pred = jnp.any(probs_zero_and_one == 0)
    def false_func(x):
        prob_syndrome = jnp.sum(x)

        cond_prob_zero = x[0] / prob_syndrome 
        cond_prob_one = x[1] / prob_syndrome
        entropy_term_a = -jax.scipy.special.xlogy(
            x[0], 
            cond_prob_zero
        )
        entropy_term_b = -jax.scipy.special.xlogy(
            x[1], 
            cond_prob_one
        )
        return (entropy_term_a + entropy_term_b) / jnp.log(2)
    
    def true_func(x):
        return 0.0
    
    value = jax.lax.cond(pred, true_func, 
                         false_func, probs_zero_and_one)
    return value

def miller_madow_conditional_entropy(
    probs_zero_and_one: Array,
    num_samples: int
) -> Array:
    """Computes the conditional entropy associated with the probabilities
    of observing a certain syndrome and chi either 0 or 1. It uses the 
    Miller-Madow estimator by adding the corresponding correction.
    
    Args:
        probs_zero_and_one: array of probabilities of having the observed 
            syndromes with chi either zero or one. Typically these are 
            estimated from sample frequencies.
        num_samples: number of samples needed for the Miller-Madow correction
    
    Returns:
        sum_{syndrome} (-prob(syndrome, 0) log_2 prob(0| syndrome) 
            -prob(syndrome, 1) log_2 prob(1| syndrome) )
    """
    num_syndrome = jnp.shape(probs_zero_and_one)[0]
    num_syndrome_chi = jnp.count_nonzero(probs_zero_and_one)
    compute_conditional_entropy_term_jit = \
        jax.jit(compute_conditional_entropy_term)
    entropy_terms = \
        jax.vmap(compute_conditional_entropy_term_jit)(probs_zero_and_one)
    # Double check correctness of the Miller Madow correction for conditional 
    # entropy
    mm_correction = (num_syndrome_chi - num_syndrome) / (2 * num_samples)
    return jnp.sum(entropy_terms) + mm_correction







