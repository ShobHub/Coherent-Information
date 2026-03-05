# Module with functions for the implementation of the worm algorithm
# for qudits with d = 2 * p and p odd prime


from typing import Tuple, Dict, Callable
from coherentinfo.errormodel import ErrorModelLindbladTwoOddPrime
from coherentinfo.moebius_two_odd_prime import MoebiusCodeTwoOddPrime
from coherentinfo.dtypes import INT_DTYPE
import os
import jax
N_CPUS = os.cpu_count()
N_USED_CPUS = N_CPUS
jax.config.update('jax_num_cpu_devices', N_USED_CPUS)
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.typing import ArrayLike
import jax.numpy as jnp
from functools import partial


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
    return jnp.nonzero(stab_mat[:, edge], size=2, fill_value=-1)[0]

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
    stab_bool: bool,
    h_mod_p: ArrayLike,
    p: int,
) -> ArrayLike:
    """ Returns the error associated with a local move.

    Args:
        edge: the edge where the mod_2 error is placed
        power: the power of the mod_p stabilizer error
        stab_bool: a boolean variable that decides whether
            the first (True) or second (False) incident stabilizer should be 
            picked. If the edge is incident to a single-stabilizer
            then it does not matter. 
        h_mod_p: the stabilizer matrix mod p
        p: the odd prime

    Returns:
        An array with two rows, where the first row is the 
        error associated with the mod_2 move, i.e., zero everywhere 
        and 1 at edge, while the second row the mod_p error with 
        the corresponding power        
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
) -> Tuple[ArrayLike]:
    """Gives the probability of a move
    
    Args:
        edge: the edge where mod_2 error is placed
        power: the power of the mod_p error associated with
            the stabilizer adjacent to edge, that is not head
        error: the current error, where the first row is the error mod 2
            and the second the error mod p
        stab_bool: a boolean variable that decides whether
            the first (True) or second (False) incident stabilizer should be 
            picked. If the edge is incident to a single-stabilizer
            then it does not matter. 
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

def random_edge_boundary(key):
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


def random_edge_bulk(key):
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
) -> Tuple[Dict, Dict]:
    """ Implements a single step of the "split" worm algorithm which is 
    suited for the Moebius code for qudits and d = 2 * p p odd prime. As the 
    function, is thought to be used in jax.lax.cond the first two arguments
    are the current state, i.e., the carry, and x, i.e., a slice of the 
    output (see jax.lax.scan documentation)

    Args:
        worm_state (Dict): a dictionary with the following keys:
            worm_error (ArrayLike): a JAX array with two rows, where the first
                row is the error mod 2 and the second the error mod p
            head (int): the label of the current head of the worm
            tail (int): the label of the tail of the worm (which stays fixed)
            worm_success (bool): a boolean that marks whether the worm has 
                succeded or not. If it succeeds, it skips all remaining
                attempts
            accepted_moves (int): counter of accepted moves
            attempted_moves (int): counter of attempted moves
            key (ArrayLike): the key used for random number generation, 
                which will be split inside the function
        x (Dict | None): A slice of the worm state, from previous iterations. 
            Usually needed only if you want to keep track of how the 
            worm_state evolves during the scan
        h_error_mod_p (ArrayLike): the stabilizers mod p that are used to
            generate p errors that give no syndrome
        h_mod_p (ArrayLike): the stabilizers mod p from which the mod p
            syndrome can be obtained. Note that if h_error_mod_p is 
            h_z_mod_p, then h_mod_p is h_x_mod_p (and vice versa). The 
            function is set up so that both cases are handled.
        error_model (ErrorModelLindbladTwoOddPrime): The error model used
                to compute the necessary probabilities.
        

    Returns:
        A tuple with the new worm_state and None, since we do not implement
        keeping track of the state during the scan.
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

        # We add these three keys for later convenience. They will be removes
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
            # Only now we compute the proposed move and the new worm
            proposed_move = move_error(
                edge, power, stab_bool, h_error_mod_p, p)
            new_worm_error_mod_2 = jnp.mod(
                proposed_move[0, :] + worm_error[0, :], 2)
            new_worm_error_mod_p = jnp.mod(
                proposed_move[1, :] + worm_error[1, :], p)
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
            new_state["worm_success"] = jnp.logical_or(
                jnp.logical_and(tmp_head == tail, state["boundary"] == False),
                jnp.logical_and(tmp_head == -1, state["boundary"] == True)
            )

            # Now if we hit a boundary for the first time, the head is 
            # set back to tail. 
            set_head_to_tail = jnp.logical_and(
                tmp_head == -1, 
                state["boundary"] == False
            )
            # Notice that the second time you hit a boundary the head 
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
            new_state["accepted_moves"] = state["accepted_moves"] + 1
            new_state["attempted_moves"] = state["attempted_moves"] + 1
            new_state["key"] = state["key"]
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
    max_worm_steps: int,
) -> Dict:
    """ Implements the "split" worm algorithm which is 
    suited for the Moebius code for qudits and d = 2 * p p odd prime. 

    Args:
        worm_error (ArrayLike): a JAX array with two rows, where the first
                row is the error mod 2 and the second the error mod p
        h_error_mod_p (ArrayLike): the stabilizers mod p that are used to
            generate p errors that give no syndrome
        h_mod_p (ArrayLike): the stabilizers mod p from which the mod p
            syndrome can be obtained. Note that if h_error_mod_p is 
            h_z_mod_p, then h_mod_p is h_x_mod_p (and vice versa). The 
            function is set up so that both cases are handled.
        error_model (ErrorModelLindbladTwoOddPrime): The error model used
                to compute the necessary probabilities.
        compute_chi (Callable): a function that computes the full logical bit
            i.e., it is equal to 0 if the logical bit is 0 and p if the 
            logical bit is p
        num_stab (int): an integer that is the number of rows in 
            h_mod_p, which is needs to be explicitly passed for JAX 
            compatibility
        max_worm_steps (int): the maximum number of worm steps
        

    Returns (Dict):
        The new worm_state at the end of the split worm algorithm, as a 
        dictionary with the same entries of initial_worm_state, as well
        as:
            worm_error (ArrayLike): a JAX array with two rows, where the first
                row is the error mod 2 and the second the error mod p
            key (ArrayLike): the last key used for random number generation

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

def run_worm_moebius(
    syndrome_id: str, 
    moebius_setup: Dict,
    worm_setup : Dict, 
    gamma_t: ArrayLike,
) -> Dict:
    length = moebius_setup["length"]
    width = moebius_setup["length"]
    p = moebius_setup["p"]
    num_samples = worm_setup["num_samples"]
    num_worms = worm_setup["num_worms"]
    max_worm_steps = worm_setup["max_worm_steps"]
    worm_master_seed = worm_setup["worm_master_seed"]
    error_master_seed = worm_setup["error_master_seed"]

    
    d = 2 * p
    moebius_code = MoebiusCodeTwoOddPrime(length=length, width=width, d=d)
    em_lindblad = ErrorModelLindbladTwoOddPrime(
        moebius_code.num_edges, d=d, gamma_t=gamma_t
    )

    if syndrome_id == "plaquette":
        num_stabs, h_error_mod_p, h_mod_p = (
            moebius_code.num_plaquette_checks,
            moebius_code.h_z_mod_p,
            moebius_code.h_x_mod_p,
        )
        compute_chi = moebius_code.compute_plaquette_syndrome_chi_x
    elif syndrome_id == 'vertex':
        num_stabs, h_error_mod_p, h_mod_p = (
            moebius_code.num_vertex_checks,
            moebius_code.h_x_mod_p,
            moebius_code.h_z_mod_p,
        )
        compute_chi = moebius_code.compute_vertex_syndrome_chi_z
    else:
        raise ValueError("The syndrome id must be either plaquette or vertex")

    master_worm_key = jax.random.PRNGKey(worm_master_seed)

    worm_keys = jax.random.split(
        master_worm_key, num=num_samples * num_worms).reshape(num_samples, num_worms, 2)

    error_master_key = jax.random.PRNGKey(error_master_seed)
    error_keys = jax.random.split(error_master_key, num_samples)


    def generate_initial_worm_errors(
        key: ArrayLike,
        error_model: ErrorModelLindbladTwoOddPrime
    ) -> ArrayLike:
        initial_error = error_model.generate_random_error(key)
        initial_error_mod_2 = jnp.mod(initial_error, 2)
        initial_error_mod_p = jnp.mod(initial_error, p)
        initial_worm_error = jnp.vstack((initial_error_mod_2, initial_error_mod_p))
        return initial_worm_error


    initial_worm_errors = jax.vmap(
        generate_initial_worm_errors, in_axes=(0, None))(error_keys, em_lindblad)
    
    # Sharding the arrays
    devices = jax.devices()  # Assuming this returns 16 devices
    mesh = Mesh(devices, ('batch',))

    # sharding_for_keys = NamedSharding(mesh,  PartitionSpec('batch', None))
    # worm_keys_sharded = jax.device_put(worm_keys, sharding_for_keys)

    # 2. Define sharding: Split the 0th axis across 'batch', leave others whole
    sharding_for_error = NamedSharding(mesh,  PartitionSpec('batch', None, None))
    initial_worm_errors_sharded = jax.device_put(
        initial_worm_errors, sharding_for_error)
    
    initial_worm_state = {}
    # worm_error = jnp.vstack(
    #     (initial_error_mod_2, initial_error_mod_p))
    # initial_worm_state["worm_success"] = False
    # # The plan is to use this as a marker when we hit a boundary
    # # which is important only for vertex checks for the Moebius Code.
    # # Essentially this is turn to True whenever we hit a boundary edge and at 
    # # that point the head is set again to tail and the success condition becomes
    # # finding another (or the same) boundary.
    # initial_worm_state["boundary"] = False
    # initial_worm_state["accepted_moves"] = 0
    # initial_worm_state["attempted_moves"] = 0

    run_worm_partial = partial(
        run_worm,
        h_error_mod_p=h_error_mod_p,
        h_mod_p=h_mod_p,
        error_model=em_lindblad,
        compute_full_chi=compute_chi,
        num_stabs=num_stabs,
        max_worm_steps=max_worm_steps
    )

    # First over keys
    run_worm_vmap = jax.vmap(run_worm_partial, in_axes=(None, 0))
    # Then over initial errors
    run_worm_vmap = jax.vmap(run_worm_vmap, in_axes=(0, 0))
    run_worm_jit = jax.jit(run_worm_vmap)
    new_worm_state = run_worm_jit(initial_worm_errors_sharded, worm_keys)

    return new_worm_state

def worm_conditional_entropy(
    gamma_t: ArrayLike,
    syndrome_id: str, 
    moebius_setup: Dict,
    worm_setup : Dict, 
)-> ArrayLike:
    
    new_worm_state = run_worm_moebius(
        syndrome_id=syndrome_id,
        moebius_setup=moebius_setup,
        worm_setup=worm_setup,
        gamma_t=gamma_t
    )

    def get_binary_entropy(chi_vec, success_vec):
        # Number of successful worms
        num_success = jnp.sum(success_vec)
        # Sets simply to zero failed attempts so that they are not counted
        chi_vec_marked = jnp.where(success_vec, chi_vec, 0)
        p1 = jnp.mean(chi_vec_marked) / num_success
        p0 = 1 - p1
        binary_entropy = -jax.scipy.special.xlogy(p0, p0) / jnp.log(2)
        binary_entropy += -jax.scipy.special.xlogy(p1, p1) / jnp.log(2)
        return binary_entropy


    binary_entropies = jax.vmap(get_binary_entropy)(new_worm_state["chi"], new_worm_state["worm_success"])
    cond_entropy = jnp.mean(binary_entropies)
    return cond_entropy
    




    
    
    
    


    

    



    



