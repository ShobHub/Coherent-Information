# Module with functions for the implementation of the worm algorithm
# for qudits with d = 2 * p and p odd prime


from typing import Tuple, Dict, Callable
from coherentinfo.errormodel import ErrorModelLindbladTwoOddPrime
from coherentinfo.moebius_two_odd_prime import MoebiusCodeTwoOddPrime
from coherentinfo.dtypes import INT_DTYPE
import os
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.typing import ArrayLike
import jax.numpy as jnp
from functools import partial


def stab_labels(
    edge: int,
    stab_mat: ArrayLike
) -> ArrayLike:
    """Returns the labels of the two stabilizers 
    associated with the edge.
    
    Args:
        edge (int): Index of the edge.
        stab_matrix (ArrayLike): The stabilizer matrix of interest.
    
    Returns:
        ArrayLike: An array of length two with the labels.
    
    Warning:
        The function shall be used only with stabilizer matrices
        where the edge is involved in at most two checks.
    """
    return jnp.nonzero(stab_mat[:, edge], size=2, fill_value=-1)[0]

def stabilizer_edges(
    index: int,
    stab_mat: ArrayLike
) -> ArrayLike:
    """Returns edges associated with a stabilizer.
    
    Args:
        index (int): Index of the stabilizer.
        stab_matrix (ArrayLike): The stabilizer matrix of interest.
    
    Returns:
        ArrayLike: An array of length 4 with the labels. If the plaquette
        is at the boundary, i.e., the number of edges involved is
        3, the last element is set to -1.
    """
    return jnp.nonzero(stab_mat[index], size=4, fill_value=-1)[0]

def move_error(
    edge: int,
    power: int,
    stab_bool: bool,
    h_mod_p: ArrayLike,
    p: int,
) -> ArrayLike:
    """ Returns the error associated with a local move.

    Args:
        edge (int): The edge where the mod_2 error is placed
        power (int): The power of the mod_p stabilizer error
        stab_bool (bool): A boolean variable that decides whether
            the first (True) or second (False) incident stabilizer should be 
            picked. If the edge is incident to a single-stabilizer
            then it does not matter. 
        h_mod_p (ArrayLike): The stabilizer matrix mod p.
        p (int): The odd prime.

    Returns:
        ArrayLike: An array with two rows, where the first row is the 
        error associated with the mod_2 move, i.e., zero everywhere 
        and 1 at edge, while the second row the mod_p error with 
        the corresponding power.       
    """

    incident_stabs = stab_labels(edge, h_mod_p)

    # head_is_first = incident_stabs[0] == head 

    # candidate_head = jax.lax.cond(
    #     head_is_first, 
    #     lambda : incident_stabs[1], 
    #     lambda : incident_stabs[0]
    # )

    stab_bool_tot = jnp.logical_or(stab_bool, incident_stabs[-1] == -1)
    candidate_stab_label = incident_stabs[
        jnp.where(stab_bool_tot, 0, 1)
    ]

    candidate_stab = h_mod_p[candidate_stab_label, :]
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
    stab_bool: bool,
    h_mod_p: ArrayLike,
    p: int,
    error_model: ErrorModelLindbladTwoOddPrime,
) -> Tuple:
    """Gives the probability of a move.
    
    Args:
        edge: The edge where mod_2 error is placed.
        power: The power of the mod_p error associated with
            the stabilizer adjacent to edge, that is not head.
        error: The current error, where the first row is the error mod 2
            and the second the error mod p.
        stab_bool: A boolean variable that decides whether
            the first (True) or second (False) incident stabilizer should be 
            picked. If the edge is incident to a single-stabilizer
            then it does not matter. 
        h_mod_p: The stabilizer matrix mod p.
        p: the odd prime.
        error_model: The error model to be used to obtain the probabilities.
    
    Returns:
        Tuple: The probability of a move and the corresponding probability 
        associated with the initial error. Their ratio gives the acceptance 
        probability.
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

        stab_bool_tot = jnp.logical_or(stab_bool, incident_stabs[-1] == -1)
        candidate_stab_label = incident_stabs[
            jnp.where(stab_bool_tot, 0, 1)
        ]

        edges_candidate_stab = stabilizer_edges(candidate_stab_label, h_mod_p)

        candidate_stab = h_mod_p[candidate_stab_label, :]

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
            jax.vmap(fun_prob)(edges_candidate_stab)
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
) -> Tuple:
    """Returns the probability of all possible moves from head.
     This should be used if you want to already weigh the probabilities in 
    advance so that you always accept. However, it comes at the price that you
     always need to compute all of them. 
     
    Args:
        error_mod_2 (ArrayLike): The error mod 2.
        error_mod_p (ArrayLike): The error mod p.
        head (int): The head position
        h_mod_p: The stabilizer matrix mod p.
        p: the odd prime.
        error_model: The error model to be used to obtain the probabilities.
    
    Returns:
        Tuple: It contains the move probabilities, as well as the edges 
            associated with the head.


     """

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

def random_edge_boundary(key: ArrayLike) -> int:
    """Generates a random integer between 0 and 2 (inclusive) using JAX's 
     random number generator.

    Args:
        key (ArrayLike): A random seed key for JAX's random number generator.
            Usually generated using jax.random.PRNGKey() or 
            jax.random.split().

    Returns:
        int: A uniformly distributed random integer in [0, 1, 2].
    """
    
    return jax.random.randint(key, 1, 0, 3)


def random_edge_bulk(key: ArrayLike) -> int:
    """Generates a random integer between 0 and 3 (inclusive) using JAX's 
     random number generator.

    Args:
        key (ArrayLike): A random seed key for JAX's random number generator.
            Usually generated using jax.random.PRNGKey() or 
            jax.random.split().

    Returns:
        int: A uniformly distributed random integer in [0, 1, 2, 3].
    """
    return jax.random.randint(key, 1, 0, 4)

def worm_step(
    worm_state: Dict,
    x: Dict | None,
    h_error_mod_p: ArrayLike,
    h_mod_p: ArrayLike,
    error_model: ErrorModelLindbladTwoOddPrime,
) -> Tuple:
    """ Implements a single step of the "split" worm algorithm which is 
    suited for the Moebius code for qudits and d = 2 * p p odd prime. As the 
    function, is thought to be used in jax.lax.cond the first two arguments
    are the current state, i.e., the carry, and x, i.e., a slice of the 
    output (see jax.lax.scan documentation)

    Args:
        worm_state (Dict): A dictionary with the following keys:
            worm_error (ArrayLike): A JAX array with two rows, where the first
                row is the error mod 2 and the second the error mod p.
            boundary (bool): a boolean that marks if a boundary was ever hit
                by the current worm or not. If true then the current worm 
                needs to end at a boundary.
            accepted_moves (int): Total number of accepted moves.
            attempted_moves (int): Number of attempted moves.
            burn_in_step (int): Minimum number of worm steps after which
                the worm is declared successful.
            num_stabs (int): An integer that is the number of rows in 
                h_mod_p.
            head (int): The final position of the head.
            tail (int): The final position of the tail. If the worm is 
                successful and does not hit a boundary then head should be 
                equal to tail, while this does not necessarily hold 
                if a boundary was not hit.
            key (ArrayLike): The last key used for random number generation.
        x (Dict | None): A slice of the worm state, from previous iterations. 
            Usually needed only if you want to keep track of how the 
            worm_state evolves during the scan
        h_error_mod_p (ArrayLike): The stabilizers mod p that are used to
            generate p errors that give no syndrome
        h_mod_p (ArrayLike): The stabilizers mod p from which the mod p
            syndrome can be obtained. Note that if h_error_mod_p is 
            h_z_mod_p, then h_mod_p is h_x_mod_p (and vice versa). The 
            function is set up so that both cases are handled.
        error_model (ErrorModelLindbladTwoOddPrime): The error model used
                to compute the necessary probabilities.
        

    Returns:
        Tuple: A tuple with the new worm_state and None, since we do not 
        implement keeping track of the state during the scan.
    """

    def do_not_attempt_step(worm_state):
        return worm_state

    def attempt_step(worm_state):
        # We unpack only the variables that we need
        worm_error = worm_state["worm_error"]
        head = worm_state["head"]
        tail = worm_state["tail"]
        # The stabilizers of the same kind as the error
        
        p = error_model.p
        key = worm_state["key"]

        key, subkey = jax.random.split(key)

        head_edges = stabilizer_edges(head, h_mod_p)
        # Select a random edge among those available
        random_integer = jax.lax.cond(
            head_edges[-1] == -1,
            random_edge_boundary,
            random_edge_bulk,
            subkey
        )
        edge = head_edges[random_integer][0]
        # We need to keep splitting the key any time we use it
        key, subkey = jax.random.split(key)
        # Select a random power of the error stabilizers
        power = jax.random.randint(subkey, 1, 0, p)[0]
        key, subkey = jax.random.split(key)
        # Any edge has 1 or 2 stabilizers connected to it
        # so we select randomly one of the 2. If it is only 1
        # then obviously that one will be selected (this happens for
        # boundary vertex stabilizers)
        stab_bool = jax.random.randint(subkey, 1, 0, 2)[0] == True
        key, subkey = jax.random.split(key)

        # We add these three keys for later convenience. They will be removed
        # at the end since the function needs to return a worm state
        # with the same keys
        worm_state["edge"] = edge
        worm_state["power"] = power
        worm_state["stab_bool"] = stab_bool

        # We compute the probability of performing the move, as well
        # as the one associated with the initial error
        prob_move, prob_move_worm_error = single_move_probability(
            edge,
            power,
            worm_error,
            stab_bool,
            h_error_mod_p,
            p,
            error_model
        )

        # Their ratios is used to determine the acceptance probability
        acceptance_prob = jnp.min(
            jnp.array([1.0, prob_move / prob_move_worm_error]))
        # This is to handle the case in which we have 3 edges which is 
        # marked with a -1, and so should always be rejected. 
        # In that case, the acceptance probability is set to 0
        acceptance_prob = jax.lax.cond(
            prob_move < 0, lambda: 0.0, lambda: acceptance_prob)

        def reject(state):
            # In this case, the only thing that gets updated is the
            # counter of attempted moves
            state["attempted_moves"] += 1
            return state

        def accept(state):
            # We unpack only the variables that are needed
            worm_error = state["worm_error"]
            head = state["head"]
            p = error_model.p
            edge = state["edge"]
            power = state["power"]
            stab_bool = state["stab_bool"]
            num_stabs = state["num_stabs"]
            burn_in_steps = state["burn_in_steps"]
            # Only now we compute the proposed move and the new worm
            proposed_move = move_error(
                edge, power, stab_bool, h_error_mod_p, p
            )
            new_worm_error_mod_2 = jnp.mod(
                proposed_move[0, :] + worm_error[0, :], 2
            )
            new_worm_error_mod_p = jnp.mod(
                proposed_move[1, :] + worm_error[1, :], p
            )
            new_worm_error = jnp.vstack(
                (new_worm_error_mod_2, new_worm_error_mod_p), dtype=INT_DTYPE)
            # We now update the new parameters
            new_state = {}
            new_state["worm_error"] = new_worm_error
            # The new head is the stabilizer label associated with the
            # selected edge that that is not the previous head. Note that
            # if there is only one stabilizer, i.e., we are at the boundary
            # then it means that the worm succeded and in that case
            # new_head will be -1
            incident_stab = stab_labels(edge, h_mod_p)
            # jax.debug.print("jax.debug.print(x) -> {x}", x=incident_stab)
            # jax.debug.print("Head {x}", x=head)
            tmp_head = jax.lax.cond(
                incident_stab[0] == head, 
                lambda x: x[1], 
                lambda x: x[0], 
                incident_stab
                )
            # jax.debug.print("New head {x}", x=tmp_head)
            # jax.debug.print("Incident stab {x}", x=incident_stab)

            # We note that in the current implementation, if the worm hits 
            # a boundary during its path, it is forced to end at a boundary. 
            # Whether the boundary was hit or not is marked by 
            # state["boundary"]. This is obviously only important for 
            # vertex checks. Thus, the worm is successful if either we never 
            # found a boundary and we hit the tail again, or we found a 
            # boundary before and we find a boundary again, which we find out 
            # if the temporary head tmp_head ends up being -1
            individual_worm_success = jnp.logical_or(
                jnp.logical_and(tmp_head == tail, state["boundary"] == False),
                jnp.logical_and(tmp_head == -1, state["boundary"] == True)
            )
            worm_success = jnp.logical_and(
                individual_worm_success,
                state["attempted_moves"] + 1 > burn_in_steps
            )

            # This is the condition that we hit the tail or the boundary,
            # but we did not reach the necessary steps. In this case,
            # head and tail are re-initialized randomly and so is
            # the boolean boundary
            reset_head_and_tail = jnp.logical_and(
                individual_worm_success, 
                state["attempted_moves"] + 1 <= burn_in_steps
            )
            new_state["worm_success"] = worm_success
            new_state["num_stabs"] = num_stabs
            new_state["burn_in_steps"] = burn_in_steps

            # Now if we hit a boundary for the first time, the head is 
            # set back to tail. 
            set_head_to_tail = jnp.logical_and(
                tmp_head == -1, 
                state["boundary"] == False
            )
            # Notice that the second time a boundary is hit the head 
            # should end up being -1
            new_state["head"] = jax.lax.cond(
                set_head_to_tail, 
                lambda: state["tail"], 
                lambda: tmp_head
            )
            # In addition, we mark whether we hit a boundary or not.
            # If we hit it before it stays True of course
            new_state["boundary"] = jnp.logical_or(
                tmp_head == -1, 
                state["boundary"]
            )
            new_state["tail"] = state["tail"]
            new_state["key"] = state["key"]

            def fun_reset(args):
                new_state, state = args
                num_stabs = new_state["num_stabs"]
                new_base_key = new_state["key"]
                new_base_key, subkey = jax.random.split(new_base_key)
                new_initial_head = jax.random.randint(subkey, 1, 0, num_stabs)[0]
                new_state["head"] = new_initial_head
                new_state["tail"] = new_initial_head
                new_state["boundary"] = False
                new_state["key"] = new_base_key
                return new_state
                

            def fun_do_not_reset(args):
                new_state, state = args
                set_head_to_tail = jnp.logical_and(
                    tmp_head == -1, 
                    state["boundary"] == False
                )
                # Notice that the second time a boundary is hit the head 
                # should end up being -1
                new_state["head"] = jax.lax.cond(
                    set_head_to_tail, 
                    lambda: state["tail"], 
                    lambda: tmp_head
                )
                # In addition, we mark whether we hit a boundary or not.
                # If we hit it before it stays True of course
                new_state["boundary"] = jnp.logical_or(
                    tmp_head == -1, 
                    state["boundary"]
                )
                new_state["tail"] = state["tail"]
                new_state["key"] = state["key"]
                return new_state
            
            new_state = jax.lax.cond(reset_head_and_tail, fun_reset, fun_do_not_reset, (new_state, state))

            new_state["accepted_moves"] = state["accepted_moves"] + 1
            new_state["attempted_moves"] = state["attempted_moves"] + 1
            new_state["edge"] = edge
            new_state["power"] = power
            new_state["stab_bool"] = stab_bool
            return new_state

        # A random number between 0 and 1 to determing if the move is accepted
        # or not
        acceptance_random_number = jax.random.uniform(subkey)

        # The acceptance condition
        accept_condition = acceptance_random_number <= acceptance_prob

        accept_condition = accept_condition

        # Important to update the key!
        worm_state["key"] = key
        new_worm_state = \
            jax.lax.cond(
                accept_condition,
                accept,
                reject,
                worm_state
            )

        new_worm_state

        new_worm_state.pop("edge", None)
        new_worm_state.pop("power", None)
        new_worm_state.pop("stab_bool", None)
        return new_worm_state

    # A worm step is attempted only if worm_state["worm_success"]
    # is False, i.e., the worm has not succeded yet.
    new_worm_state = jax.lax.cond(
        worm_state["worm_success"] == False,
        attempt_step,
        do_not_attempt_step,
        worm_state,
    )

    return new_worm_state, None

# @jax.jit(static_argnames=['error_model', "max_worm_steps"])
def run_worm(
    worm_error: ArrayLike,
    base_key: ArrayLike,
    h_error_mod_p: ArrayLike,
    h_mod_p: ArrayLike,
    error_model: ErrorModelLindbladTwoOddPrime,
    compute_full_chi: Callable,
    num_stabs: int,
    burn_in_steps: int,
    max_worm_steps: int,
) -> Dict:
    """ Implements the "split" worm algorithm which is 
    suited for the Moebius code for qudits and d = 2 * p p odd prime. 

    Args:
        worm_error (ArrayLike): A JAX array with two rows, where the first
                row is the error mod 2 and the second the error mod p
        h_error_mod_p (ArrayLike): The stabilizers mod p that are used to
            generate p errors that give no syndrome
        h_mod_p (ArrayLike): The stabilizers mod p from which the mod p
            syndrome can be obtained. Note that if h_error_mod_p is 
            h_z_mod_p, then h_mod_p is h_x_mod_p (and vice versa). The 
            function is set up so that both cases are handled.
        error_model (ErrorModelLindbladTwoOddPrime): The error model used
                to compute the necessary probabilities.
        compute_chi (Callable): A function that computes the full logical bit
            i.e., it is equal to 0 if the logical bit is 0 and p if the 
            logical bit is p
        num_stabs (int): An integer that is the number of rows in 
            h_mod_p, which is needs to be explicitly passed for JAX 
            compatibility
        burn_in_steps (int): Minimum number of worm steps after which
            the worm is declared successful
        max_worm_steps (int): The maximum number of worm steps
        

    Returns (Dict):
        The worm state at the end of the algorithm represented as a 
        dictionary with keys:
            worm_error (ArrayLike): A JAX array with two rows, where the first
                row is the error mod 2 and the second the error mod p.
            boundary (bool): A boolean that marks if a boundary was ever hit
                by the current worm or not. If true then the current worm 
                needs to end at a boundary.
            accepted_moves (int): Total number of accepted moves.
            attempted_moves (int): Number of attempted moves.
            burn_in_steps (int): Minimum number of worm steps after which
                the worm is declared successful.
            num_stabs (int): An integer that is the number of rows in 
                h_mod_p.
            head (int): The final position of the head.
            tail (int): The final position of the tail. If the worm is 
                successful and does not hit a boundary then head should be 
                equal to tail, while this does not necessarily hold 
                if a boundary was not hit.
            key (ArrayLike): The last key used for random number generation.
            full_chi (int): An integer that identifies if the final error
                gives or not a logical error. It is essentially the result 
                of the commutation relation of new_error - candidate_error
                with the logical. For d = 2 * p with p odd prime this should 
                be either 0 or p. It is still stored for sanity check
            chi (int): A bit, stored as integer, that is in one to one 
                correspondance with full_chi, and thus is 0 if the error 
                causes no logical error, while 1 otherwise.
    """
    initial_worm_state = {}
    initial_worm_state["worm_error"] = worm_error
    p = error_model.p
    d = error_model.d
    # The base key will be split several times inside the function
    base_key, subkey = jax.random.split(base_key)
    initial_head = jax.random.randint(subkey, 1, 0, num_stabs)[0]
    initial_worm_state["boundary"] = False
    initial_worm_state["worm_success"] = False
    # initial_worm_state["h_error_mod_p"] = h_z_mod_p
    # initial_worm_state["h_mod_p"] = h_x_mod_p
    initial_worm_state["accepted_moves"] = 0
    initial_worm_state["attempted_moves"] = 0
    initial_worm_state["burn_in_steps"] = burn_in_steps
    initial_worm_state["num_stabs"] = num_stabs
    initial_worm_state["head"] = initial_head
    initial_worm_state["tail"] = initial_head
    initial_worm_state["key"] = base_key
    
    worm_step_partial = partial(
        worm_step, 
        h_error_mod_p=h_error_mod_p, 
        h_mod_p=h_mod_p, 
        error_model=error_model
        )
    new_worm_state = initial_worm_state.copy()

    new_worm_state, _ = jax.lax.scan(
        worm_step_partial, 
        initial_worm_state, 
        jnp.arange(max_worm_steps)
        )
    
    worm_error_mod_2 = new_worm_state["worm_error"][0, :]
    worm_error_mod_p = new_worm_state["worm_error"][1, :]
    # The following converts from mod 2 and mod p to mod 2 * p
    full_worm_error = jnp.mod(
        p * worm_error_mod_2 + (1 - p) * worm_error_mod_p, 
        d
    )
    new_worm_state["full_chi"] = compute_full_chi(full_worm_error)[-1]
    new_worm_state["chi"] = jax.lax.cond(
        new_worm_state["full_chi"] == p,
        lambda: 1, 
        lambda: 0
    )
    
    return new_worm_state




    
    




    
    
    
    


    

    



    



