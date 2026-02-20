# Module with functions for the implementation of the worm algorithm
# for qudits with d = 2 * p and p odd prime

from jax.typing import ArrayLike
import jax.numpy as jnp
from typing import Tuple, List
from coherentinfo.errormodel import ErrorModelLindbladTwoOddPrime
import jax


def stab_labels(
    edge: int,
    stab_mat: ArrayLike
) -> ArrayLike:
    """Returns the labels of the two stabilizers 
    associated with the edge
    
    Args:
        edge: index of the edge
        stab_matrix: the stabilizer matrix of interest
    
    Returns:
        An array of length two with the labels.
    
    Warning
        The function shall be used only with stabilizer matrices
        where the edge is involved in at most two checks.
    """
    return jnp.nonzero(stab_mat[:, edge], size=2)[0]

def stabilizer_edges(
    index: int,
    stab_mat: ArrayLike
) -> ArrayLike:
    """Returns edges associated with a stabilizer.
    
    Args:
        index: index of the stabilizer
        stab_matrix: the stabilizer matrix of interest.
    
    Returns:
        An array of length 4 with the labels. If the plaquette
        is at the boundary, i.e., the number of edges involved is
        3, the last element is set to -1.
    """
    return jnp.nonzero(stab_mat[index], size=4, fill_value=-1)[0]

def move_error(
    edge: int,
    power: int,
    head: int,
    h_mod_p: ArrayLike,
    p: int,
) -> ArrayLike:
    """ Returns the error associated with a local move.

    Args:
        edge: the edge where the mod_2 error is placed
        power: the power of the mod_p stabilizer error
        head: the current head position.
        h_mod_p: the stabilizer matrix mod p
        p: the odd prime

    Returns:
        An array with two rows, where the first row is the 
        error associated with the mod_2 move, i.e., zero everywhere 
        and 1 at edge, while the second row the mod_p error with 
        the corresponding power        
    """

    incident_stabs = stab_labels(edge, h_mod_p)

    head_is_first = incident_stabs[0] == head 

    candidate_head = jax.lax.cond(
        head_is_first, 
        lambda : incident_stabs[1], 
        lambda : incident_stabs[0]
    )

    candidate_stab = h_mod_p[candidate_head]
    error_mod_p = jnp.mod(power * candidate_stab, p)
    # This is a simple trick to define a zero array of the 
    # desired size, without using jnp.zeros, which would require
    # to get the actual value of the shape, or passing it explicitly.
    error_mod_2 = (0 * error_mod_p).at[edge].set(1)
    return jnp.vstack((error_mod_2, error_mod_p))






def single_move_probability(
    edge: int,
    power: int,
    error: ArrayLike,
    head: int,
    h_mod_p: ArrayLike,
    p: int,
    error_model: ErrorModelLindbladTwoOddPrime,
) -> Tuple[ArrayLike]:
    """Gives the probability of a move
    
    Args:
        edge: the edge where mod_2 error is placed
        power: the power of the mod_p error associated with
            the stabilizer adjacent to edge, that is not head
        error: the current error, where the first row is the error mod 2
            and the second the error mod p
        head: the current head position, needed to identify the 
            stabilizer where the mod_p error should be placed
        h_mod_p: the stabilizer matrix mod p
        p: the odd prime
        error_model: the error model to be used to obtain the probabilities
    
    Returns:
        The probability of a move and the corresponding probability associated
        with the initial error. Their ratio gives the acceptance probability
    """
    error_mod_2 = error[0, :]
    error_mod_p = error[1, :] 

    def edge_negative():
        return -1.0, -1.0

    def edge_non_negative():
        # This gives the two stabilizers incident at edge, but 
        # we know that one of them must be head, so we take the 
        # other one
        incident_stabs = stab_labels(edge, h_mod_p)

        head_is_first = incident_stabs[0] == head 

        candidate_head = jax.lax.cond(
            head_is_first, 
            lambda : incident_stabs[1], 
            lambda : incident_stabs[0]
        )

        candidate_stab = h_mod_p[candidate_head]

        edges_candidate_head = stabilizer_edges(candidate_head, h_mod_p)

        new_error_mod_p = jnp.mod(error_mod_p + power * candidate_stab, p)

        def fun_prob(edge_ch):
            def fun_negative(edge_ch):
                return 1.0, 1.0

            def fun_non_negative(edge_ch):
                is_edge = jnp.where(edge_ch == edge, 1, 0)
                error_at_edge = jnp.mod(is_edge + error_mod_2[edge_ch], 2)
                prob_move = error_model.get_modular_probability(
                    error_at_edge, new_error_mod_p[edge_ch]
                )
                prob_move_initial_error = error_model.get_modular_probability(
                    error_mod_2[edge_ch], error_mod_p[edge_ch]
                )
                return prob_move, prob_move_initial_error

            is_negative = edge_ch < 0

            prob_move, prob_move_initial_error = jax.lax.cond(
                is_negative, fun_negative, fun_non_negative, edge_ch
                )

            return prob_move, prob_move_initial_error

        probs_move, prob_move_initial_error = \
            jax.vmap(fun_prob)(edges_candidate_head)
        return jnp.prod(probs_move), jnp.prod(prob_move_initial_error) 
    
    is_edge_negative = edge < 0

    total_prob_move, total_prob_move_initial_error = jax.lax.cond(
        is_edge_negative, 
        edge_negative, 
        edge_non_negative
    )

    return total_prob_move, total_prob_move_initial_error

def all_move_probabilities(
    error_mod_2: ArrayLike,
    error_mod_p: ArrayLike,
    head: int,
    h_mod_p: ArrayLike,
    p: int,
    error_model: ErrorModelLindbladTwoOddPrime,
) -> Tuple[ArrayLike, ArrayLike]:
    """Returns the probability of all possible moves from head.
     This should be used if you want to already weigh the probabilities in 
    advance so that you always accept. However, it comes at the price that you
     always need to compute all of them. """

    head_edges = stabilizer_edges(head, h_mod_p) 

    powers = jnp.arange(p)

    def get_single_prob(edge, power):        

        def edge_negative():
            return -1.0

        def edge_non_negative():
            # This gives the two plaquettes incident at edge, but 
            # we know that one of them must be head, so we take the 
            # other one
            incident_plaquettes = stab_labels(edge, h_mod_p)

            head_is_first = incident_plaquettes[0] == head 

            candidate_head = jax.lax.cond(
                head_is_first, 
                lambda : incident_plaquettes[1], 
                lambda : incident_plaquettes[0]
            )

            candidate_plaquette = h_mod_p[candidate_head]

            edges_candidate_head = stabilizer_edges(candidate_head, h_mod_p)
            power_plaquette_mod_p = jnp.mod(power * candidate_plaquette, p)

            new_error_mod_p = jnp.mod(error_mod_p + power_plaquette_mod_p, p)

            def fun_prob(edge_ch):
                def fun_negative(edge_ch):
                    return 1.0

                def fun_non_negative(edge_ch):
                    # index = args[0]
                    is_edge = jnp.where(edge_ch == edge, 1, 0)
                    error_at_edge = jnp.mod(is_edge + error_mod_2[edge_ch], 2)
                    prob = error_model.get_modular_probability(
                        error_at_edge, new_error_mod_p[edge_ch]
                    )
                    return prob

                is_negative = edge_ch < 0

                prob = jax.lax.cond(is_negative, fun_negative,
                                    fun_non_negative, edge_ch)

                return prob

            total_prob = jnp.prod(jax.vmap(fun_prob)(edges_candidate_head))
            return total_prob
        
        is_edge_negative = edge < 0

        total_prob = jax.lax.cond(
            is_edge_negative, 
            edge_negative, 
            edge_non_negative
        )

        return total_prob
    
    vmapped_get_single_prob = jax.vmap(
        jax.vmap(
            get_single_prob, in_axes=(None, 0)
        ), in_axes=(0, None)
    )

    all_probs = vmapped_get_single_prob(head_edges, powers) 

    return all_probs, head_edges
    


    

    



    



