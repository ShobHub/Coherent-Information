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

def plaquette_move_probabilities(
    head: int,
    h_x_mod_p: ArrayLike,
    p: int,
    error_model: ErrorModelLindbladTwoOddPrime,
) -> Tuple[ArrayLike, ArrayLike]:
    """Returns the probability of all possible moves from head. """

    head_edges = stabilizer_edges(head, h_x_mod_p) 

    powers = jnp.arange(p)

    def get_single_prob(edge, power):
        

        def edge_negative():
            return -1.0

        def edge_non_negative():
            # This gives the two plaquettes incident at edge, but 
            # we know that one of them must be head, so we take the 
            # other one
            incident_plaquettes = stab_labels(edge, h_x_mod_p)

            head_is_first = incident_plaquettes[0] == head 

            candidate_head = jax.lax.cond(
                head_is_first, 
                lambda : incident_plaquettes[1], 
                lambda : incident_plaquettes[0]
            )

            candidate_plaquette = h_x_mod_p[candidate_head]

            indices = jnp.nonzero(candidate_plaquette, size=4, fill_value=-1)
            power_plaquette_mod_p = jnp.mod(power * candidate_plaquette, p)

            def fun_prob(index):
                def fun_negative(args):
                    return 1.0

                def fun_non_negative(args):
                    index = args[0]
                    is_edge = jnp.where(index == edge, 1, 0)
                    prob = error_model.get_modular_probability(
                        is_edge, power_plaquette_mod_p[index]
                    )
                    return prob

                is_negative = index[0] < 0

                prob = jax.lax.cond(is_negative, fun_negative,
                                    fun_non_negative, index)

                return prob

            total_prob = jnp.prod(jax.vmap(fun_prob)(indices))
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
    


    

    



    



