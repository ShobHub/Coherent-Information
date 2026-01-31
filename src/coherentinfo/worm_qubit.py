import jax.numpy as jnp
import jax
from jax import Array
from typing import Tuple

def run_worm_plaquette_qubit(
    syndrome: Array,
    base_key: Array,
    h_x: Array,
    num_plaquette: int,
    num_edges: int,
    p_error: float,
    max_worms: int,
    max_steps_per_worm: int
) -> Tuple[Array, Array]:
    """Jax implementation of the worm algorithm for the plaquette 
    checks in the qubit case. The implementation should work 
    for any code whose checks are either weight 3 or 4 and 
    for which the syndromes always come in pairs. Note that 
    the number of checks need not to be independent. Note that
    this implementation assumes the same error model on all edges.

    Args:
        syndrome: the vector syndrome
        base_key: the key that will be subsequently split to generate 
            the random numbers
        h_x: the check matrix, which should be given as binary matrix
        num_plaquette: the number of plaquette checks to be considered,
            i.e., the number of rows of h_x. This is passed separately
            because when jit compiling it should be treated as static
            argument, while jax cannot treat h_x as static
        num_edges: the number of edges (qubit), i.e., the number of columns
            of h_x. Similarly to num_plaquette it is provided explicitly
            since it needs to be known at compile time if one wants to 
            use jax.jit
        p_error: error probability
        max_worms: maximum number of worms, which is usually the max number
            of syndrome pairs, i.e., (num_plaquette - 1) / 2
        max_steps_per_worm: maximum step for each worm, which should be set 
            to guarantee that each worm succeeds in clearing a pair of 
            syndromes

    Returns:
        A tuple with two jax arrays. The first is a new error with the 
        same initial syndrome. The second is the final syndrome after 
        the worm algorithm run, which should be all zeros if the 
        worm algorithm is successful.
    """

    def random_edge_boundary(key):
        return jax.random.randint(key, 1, 0, 3)

    def random_edge_bulk(key):
        return jax.random.randint(key, 1, 0, 4)

    def worm_step(carry_worm_step, x):
        # Get the current state
        error, syndrome, head, tail, continue_worm, key = carry_worm_step
        # Split the key into a new key that will get passed at
        # the end and a subkey that will be used in this step
        key, subkey = jax.random.split(key)
        # Etract the locations, i.e., plaquette stabilizers
        # with nonzero syndrome. We pad -1 to make it compatible
        # with jax.jit.

        # Note that the following approach works only in the case where
        # the error is the same for every edge. It would have to be modified
        # if that is not the case.

        def reject(args):
            error, syndrome, head, continue_worm = args
            return error, syndrome, head, continue_worm

        def accept(args):
            error, syndrome, head, continue_worm = args

            syndrome_locations = jnp.nonzero(
                syndrome, size=num_plaquette, fill_value=-1)[0]
            # We extract the edges that take part in the stabilizer
            # identified by head. These are in total either 3 or 4
            # dependeing on whether the stabilizer is at the rough boundary
            # or not. Again we pad with -1 in case it is 3 to make the code
            # compatible with jax.jit

            head_edges = jnp.nonzero(h_x[head], size=4, fill_value=-1)[0]
            # We sample a random integer between either 3 or 4 possible
            # values depening on whether head stabilizer has 3 or 4 edges
            random_integer = jax.lax.cond(
                head_edges[-1] == -1, 
                random_edge_boundary, 
                random_edge_bulk, 
                subkey
            )

            # We select the candidate edge based on the random integer 
            # we sample
            candidate_edge = head_edges[random_integer]

            # Note that now for the vertex case we would need to check whether
            # we are at the boundary or not while this is not necessary 
            # for the plaquette as syndromes will always be 
            # annihilated in pairs

            # We know that the new candidate head is associated with the
            # candidate_edge which has vertices the current head and the
            # new one. The following identifies the candidate_head in
            # a way compatible with JAX
            candidate_head_vec = jnp.nonzero(
                h_x[:, candidate_edge], size=num_edges, fill_value=-1)[0]

            def candidate_head_is_first(candidate_head_vec):
                return candidate_head_vec[0]

            def candidate_head_is_second(candidate_head_vec):
                return candidate_head_vec[1]

            candidate_head = jax.lax.cond(
                candidate_head_vec[1] == head,
                candidate_head_is_first,
                candidate_head_is_second,
                candidate_head_vec,
            )

            error = error.at[candidate_edge].set(
                jnp.mod(error[candidate_edge] + 1, 2))

            head = candidate_head
            success = jnp.logical_and(
                jnp.any(syndrome_locations == head), head != tail)

            def worm_success(args):
                syndrome, head, continue_worm = args
                continue_worm = False
                # jax.debug.print("syndrome 1: {}", syndrome)
                error_syndrome = jnp.mod(h_x @ error, 2)
                syndrome = jnp.mod(syndrome + error_syndrome, 2)
                # jax.debug.print("syndrome 2: {}", syndrome)

                return syndrome, head, continue_worm

            def worm_fail(args):
                syndrome, head, continue_worm = args
                return syndrome, head, continue_worm

            syndrome, head, continue_worm = jax.lax.cond(
                success, 
                worm_success, 
                worm_fail, 
                (syndrome, head, continue_worm)
            )

            return error, syndrome, head, continue_worm

        random_number = jax.random.uniform(subkey)

        condition = jnp.logical_and(
            random_number < p_error, continue_worm)
        condition = jnp.logical_and(condition, head != -1)

        error, syndrome, head, continue_worm = jax.lax.cond(
            condition, accept, reject, (error, syndrome, head, continue_worm))

        return (error, syndrome, head, tail, continue_worm, key), None

    def run_worm(carry_worm, x):
        error, syndrome, key = carry_worm

        syndrome_locations = jnp.nonzero(
            syndrome, size=num_plaquette, fill_value=-1)[0]
        # jax.debug.print("syndrome locations: {}", syndrome_locations)
        # I do not think there is any problem in starting always from
        # the first non-zero syndrome.
        head = syndrome_locations[0]
        # jax.debug.print("head: {}", head)
        tail = head.copy()

        continue_worm = True
        new_error = jnp.zeros(num_edges, dtype=jnp.int32)
        initial_carry_worm_step = (
            new_error, syndrome, head, tail, continue_worm, key)

        new_error, syndrome, head, tail, continue_worm, key = \
            jax.lax.scan(worm_step, initial_carry_worm_step,
                         jnp.arange(max_steps_per_worm))[0]

        def update_error(args):
            error, new_error = args
            return jnp.mod(error + new_error, 2)

        def do_not_update_error(args):
            error, new_error = args
            return error

        # This ensures that the error is not updated if the
        # worm did not run successfully, i.e., if continue_worm 
        # is still set to True
        error = jax.lax.cond(
            continue_worm,
            do_not_update_error,
            update_error,
            (error, new_error)
        )

        return (error, syndrome, key), None

    error = jnp.zeros(num_edges, dtype=jnp.int32)
    initial_carry_worm = (error, syndrome, base_key)

    error, syndrome, _ = jax.lax.scan(
        run_worm, initial_carry_worm, jnp.arange(max_worms))[0]

    return error, syndrome