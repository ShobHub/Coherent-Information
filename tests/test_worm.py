# It contains tests for the functions used in the worm
# algorithm

import pytest
from coherentinfo.errormodel import ErrorModelLindbladTwoOddPrime
from coherentinfo.worm import (stab_labels, 
                               stabilizer_edges,
                               move_error,
                               single_move_probability,
                               all_move_probabilities,
                               run_worm)
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from jax.typing import ArrayLike
import jax
from coherentinfo.dtypes import INT_DTYPE



def test_stab_labels(moebius_code_example) -> None:
    """Tests that the function that given an edge returns the 
    stabilizers is correct."""
    num_examples = len(moebius_code_example)
    idx = np.random.randint(num_examples)
    moebius_code = moebius_code_example[idx]
    num_edges = moebius_code.num_edges
    h_x_mod_p = moebius_code.h_x_mod_p
    random_edge = np.random.randint(num_edges)
    labels = stab_labels(random_edge, h_x_mod_p)
    cond = (h_x_mod_p[labels[0], random_edge] != 0 and 
            h_x_mod_p[labels[1], random_edge] != 0)
    assert cond == True, \
        f"The stabilizers do not include the edge in example #{idx}"

def test_stab_edges(moebius_code_example):
    """Tests that the function that given a stabilizer returns its
    edges is correct."""
    stab = 7
    num_examples = len(moebius_code_example)
    idx = np.random.randint(num_examples)
    moebius_code = moebius_code_example[idx]
    h_x_mod_p = moebius_code.h_x_mod_p
    stab_edges = stabilizer_edges(stab, h_x_mod_p)
    labels = jnp.where(stab_edges != -1)        
    stab_edges_nonzero = stab_edges[labels]
    cond = jnp.all(h_x_mod_p[stab, stab_edges_nonzero] != 0)
    assert cond == True, \
        f"The edges do not belong to the stabilizer in example #{idx}"

def test_move_error(moebius_code_example):
    """Tests the function that returns an allowed local error"""
    num_examples = len(moebius_code_example)
    idx = np.random.randint(num_examples)
    moebius_code = moebius_code_example[idx]
    p = moebius_code.p
    head = np.random.randint(moebius_code.num_plaquette_checks)
    h_x_mod_p = moebius_code.h_x_mod_p
    head_edges = stabilizer_edges(head, h_x_mod_p)
    edge = head_edges[np.random.randint(4)]
    power = np.random.randint(p)
    stab_bool = np.random.randint(0, 2) == 0
    jit_move_error = jax.jit(move_error)
    error = jit_move_error(edge, power, stab_bool, h_x_mod_p, p)

    incident_stabs = stab_labels(edge, h_x_mod_p)
    stab_bool_tot = jnp.logical_or(stab_bool, incident_stabs[-1] == -1)
    candidate_stab_label = incident_stabs[
        jnp.where(stab_bool_tot, 0, 1)
    ]

    # edges_candidate_stab = stabilizer_edges(candidate_stab_label, h_x_mod_p)

    candidate_stab = h_x_mod_p[candidate_stab_label, :]

    candidate_stab_power = jnp.mod(power * candidate_stab, p)
    error_mod_2_test = jnp.zeros(moebius_code.num_edges).at[edge].set(1)
    error_test = jnp.vstack((error_mod_2_test, candidate_stab_power))
    cond = jnp.all(error == error_test)
    assert cond, \
        f"The error associated with a move is not correct in example #{idx}"

def test_single_move_probability(moebius_code_example):
    """Tests that the function to compute the single move probabilities
    is correct """

    num_examples = len(moebius_code_example)
    idx = np.random.randint(num_examples)
    moebius_code = moebius_code_example[idx]
    gamma_t = 0.1
    em_lindblad = ErrorModelLindbladTwoOddPrime(
        moebius_code.num_edges, d=moebius_code.d, gamma_t=gamma_t
    )
    p = moebius_code.p
    head = np.random.randint(moebius_code.num_plaquette_checks)
    h_x_mod_p = moebius_code.h_x_mod_p
    base_key = jax.random.PRNGKey(42)
    error_mod_d = em_lindblad.generate_random_error(base_key)
    error = jnp.vstack([
        jnp.mod(error_mod_d, 2), jnp.mod(error_mod_d, p)]
        )
    head_edges = stabilizer_edges(head, h_x_mod_p)
    edge = head_edges[np.random.randint(4)]
    power = np.random.randint(p)
    stab_bool = np.random.randint(0, 2) == 0

    jit_single_move_probability = jax.jit(
        single_move_probability, static_argnums=(5, 6)
    )

    prob_move, prob_move_error = jit_single_move_probability(
        edge,
        power,
        error,
        stab_bool,
        h_x_mod_p,
        p,
        em_lindblad
    )

    if edge < 0:
        prob_move_test = -1.0
        prob_move_error_test = -1.0
    else:
        incident_stabs = stab_labels(edge, h_x_mod_p)

        stab_bool_tot = jnp.logical_or(stab_bool, incident_stabs[-1] == -1)
        candidate_stab_label = incident_stabs[
            jnp.where(stab_bool_tot, 0, 1)
        ]

        edges_candidate_stab = stabilizer_edges(candidate_stab_label, h_x_mod_p)

        candidate_stab = h_x_mod_p[candidate_stab_label, :]



        new_error_mod_p = jnp.mod(error[1, :] + power * candidate_stab, p)

        prob_move_test = 1.0
        prob_move_error_test = 1.0

        for edge_ch in edges_candidate_stab:
            if edge_ch != -1:
                if edge_ch == edge:
                    prob_move_test = prob_move_test * \
                        em_lindblad.get_modular_probability(
                            jnp.mod(1 + error[0, edge_ch], 2),
                            new_error_mod_p[edge_ch]
                            )
                else:
                    prob_move_test = prob_move_test * \
                        em_lindblad.get_modular_probability(
                            error[0, edge_ch],
                            new_error_mod_p[edge_ch]
                            )

                
                prob_move_error_test = prob_move_error_test * \
                    em_lindblad.get_modular_probability(
                        error[0, edge_ch],
                        error[1, edge_ch]
                        ) 
    epsilon = 1e-8
    assert jnp.abs(prob_move - prob_move_test) < epsilon, \
        f"Move probabilities do not match in example #{idx}!"
    assert jnp.abs(prob_move_error - prob_move_error_test) < epsilon, \
        f"Initial move error probabilities do not match in example #{idx}!"


def test_all_plaquette_moves_probability(moebius_code_example):
    """Tests that the probabilities of all possible moves are
    correct. """
    num_examples = len(moebius_code_example)
    idx = np.random.randint(num_examples)
    moebius_code = moebius_code_example[idx]
    gamma_t = 0.1
    em_lindblad = ErrorModelLindbladTwoOddPrime(
        moebius_code.num_edges, d=moebius_code.d, gamma_t=gamma_t
    )
    head = np.random.randint(moebius_code.num_plaquette_checks)
    h_x_mod_p = moebius_code.h_x_mod_p
    # This then also tests jit compilation
    error_mod_2 = jnp.zeros(moebius_code.num_edges, dtype=jnp.int16)
    error_mod_p = error_mod_2
    func = jax.jit(all_move_probabilities, static_argnums=(4, 5))
    move_probs = func(error_mod_2, error_mod_p, 
                      head, h_x_mod_p, moebius_code.p, em_lindblad)

    move_probs_test = np.zeros([4, moebius_code.p], dtype=np.float32)

    head_edges = stabilizer_edges(head, h_x_mod_p)
    for edge_index in range(4):
        for power in range(moebius_code.p):
            test_edge = head_edges[edge_index]
            incident_plaquettes = stab_labels(test_edge, h_x_mod_p)
            head_is_first = incident_plaquettes[0] == head 

            candidate_head = jax.lax.cond(
                head_is_first, 
                lambda : incident_plaquettes[1], 
                lambda : incident_plaquettes[0]
            )
            candidate_head_edges = stabilizer_edges(candidate_head, h_x_mod_p)
            candidate_head_stab = jnp.mod(
                power * h_x_mod_p[candidate_head, :], moebius_code.p
            )

            prob = 1.0
            for edge in candidate_head_edges:
                if edge != -1:
                    if edge == test_edge:
                        prob = prob * em_lindblad.get_modular_probability(
                            1, candidate_head_stab[edge]
                            )
                    else:
                        prob = prob * em_lindblad.get_modular_probability(
                            0, candidate_head_stab[edge]
                            )
                    move_probs_test[edge_index, power] = prob
                    
    
    if head_edges [-1] == -1:
        move_probs_test[3, :] = -1.0
    
    # Due to the behaviour on GPU the assertion has to be implemented 
    # in this way for the test to pass on GPU as well

    try:
        np.testing.assert_allclose(
            np.array(move_probs[0]),
            move_probs_test,
            rtol=1e-5,
            atol=1e-6
        )
    except AssertionError as e:
        raise AssertionError(
            f"The move probabilities are not correct in example #{idx}\n{e}"
        )
    

def test_run_worm_plaquette(moebius_code_example):
    """Test for an elementary run of the split worm algorithm 
    for plaquette checks"""

    num_examples = len(moebius_code_example)
    idx = np.random.randint(num_examples)
    moebius_code = moebius_code_example[idx]
    p = moebius_code.p
    gamma_t = 0.3
    em_lindblad = ErrorModelLindbladTwoOddPrime(
        moebius_code.num_edges, d=moebius_code.d, gamma_t=gamma_t
    )
    h_x_mod_2 = moebius_code.h_x_mod_2
    h_x_mod_p = moebius_code.h_x_mod_p
    h_z_mod_p = moebius_code.h_z_mod_p

    error_key = jax.random.PRNGKey(np.random.randint(1000))
    initial_error = em_lindblad.generate_random_error(error_key)
    initial_error_mod_2 = jnp.mod(initial_error, 2)
    initial_error_mod_p = jnp.mod(initial_error, p)
    # Here we consider the full syndrome including the plaquette
    # we usually remove because of the constraint as this simplified the
    # coding of the worm algorithm. In fact, in this the syndromes will
    # always be annihilated in pairs, and the total number of syndromes is
    # always even as one can check numerically.
    syndrome = moebius_code.get_plaquette_syndrome(initial_error)
    syndrome_mod_2 = jnp.mod(syndrome, 2)
    syndrome_mod_p = jnp.mod(syndrome, p)

    burn_in_steps = 0
    max_steps = 10000
    base_key = jax.random.PRNGKey(87)
    initial_worm_state = {}
    # Note specifying INT_DTYPE here is important because it 
    worm_error = jnp.vstack(
        (initial_error_mod_2, initial_error_mod_p), dtype=INT_DTYPE)
    # head = np.random.randint(moebius_code.num_plaquette_checks)
    # initial_worm_state["head"] = head
    # initial_worm_state["tail"] = head
    # initial_worm_state["boundary"] = False
    # initial_worm_state["worm_success"] = False
    # # initial_worm_state["h_error_mod_p"] = h_z_mod_p
    # # initial_worm_state["h_mod_p"] = h_x_mod_p
    # initial_worm_state["accepted_moves"] = 0
    # initial_worm_state["attempted_moves"] = 0

    new_worm_state = run_worm(
        worm_error,
        base_key,
        h_z_mod_p,
        h_x_mod_p,
        em_lindblad,
        moebius_code.compute_plaquette_syndrome_chi_x,
        moebius_code.num_plaquette_checks,
        burn_in_steps,
        max_steps
    )
    
    if new_worm_state["worm_success"] == False:
        pytest.skip()

    new_syndrome_mod_2 = jnp.mod(
        h_x_mod_2 @ new_worm_state["worm_error"][0, :], 
        2
    )

    cond = jnp.all(jnp.mod(new_syndrome_mod_2 - syndrome_mod_2, 2) == 0)
    assert cond == True, \
        f"The syndromes mod 2 do not match in example #{idx}"
    
    new_syndrome_mod_p = jnp.mod(
        h_x_mod_p @ new_worm_state["worm_error"][1, :], 
        p
    )
    
    cond = jnp.all(jnp.mod(new_syndrome_mod_p - syndrome_mod_p, p) == 0)
    assert cond == True, \
        f"The syndromes mod p do not match in example #{idx}"
    

def test_run_worm_vertex(moebius_code_example):
    """Test for an elementary run of the split worm algorithm for vertex
    checks"""

    num_examples = len(moebius_code_example)
    idx = np.random.randint(num_examples)
    moebius_code = moebius_code_example[idx]
    p = moebius_code.p
    gamma_t = 0.3
    em_lindblad = ErrorModelLindbladTwoOddPrime(
        moebius_code.num_edges, d=moebius_code.d, gamma_t=gamma_t
    )
    h_z_mod_2 = moebius_code.h_z_mod_2
    h_x_mod_p = moebius_code.h_x_mod_p
    h_z_mod_p = moebius_code.h_z_mod_p

    error_key_seed = np.random.randint(1000) # 231
    # print("Error key seed = {}".format(error_key_seed))
    error_key = jax.random.PRNGKey(error_key_seed)
    initial_error = em_lindblad.generate_random_error(error_key)
    initial_error_mod_2 = jnp.mod(initial_error, 2)
    initial_error_mod_p = jnp.mod(initial_error, p)
    # Here we consider the full syndrome including the plaquette
    # we usually remove because of the constraint as this simplified the
    # coding of the worm algorithm. In fact, in this the syndromes will
    # always be annihilated in pairs, and the total number of syndromes is
    # always even as one can check numerically.
    syndrome = moebius_code.get_vertex_syndrome(initial_error)
    syndrome_mod_2 = jnp.mod(syndrome, 2)
    syndrome_mod_p = jnp.mod(syndrome, p)

    max_steps = 10000
    burn_in_steps = 5000
    base_key = jax.random.PRNGKey(7)
    initial_worm_state = {}
    # Note specifying INT_DTYPE here is important because it 
    worm_error = jnp.vstack(
        (initial_error_mod_2, initial_error_mod_p), dtype=INT_DTYPE)
    # head = np.random.randint(moebius_code.num_vertex_checks)
    # initial_worm_state["head"] = head
    # initial_worm_state["tail"] = head
    initial_worm_state["boundary"] = False
    initial_worm_state["worm_success"] = False
    # initial_worm_state["h_error_mod_p"] = h_z_mod_p
    # initial_worm_state["h_mod_p"] = h_x_mod_p
    initial_worm_state["accepted_moves"] = 0
    initial_worm_state["attempted_moves"] = 0

    new_worm_state = run_worm(
        worm_error,
        base_key,
        h_x_mod_p,
        h_z_mod_p,
        em_lindblad,
        moebius_code.compute_vertex_syndrome_chi_z,
        moebius_code.num_vertex_checks,
        burn_in_steps,
        max_steps
    )

    print("Head = {}".format(new_worm_state["head"]))
    print("Tail = {}".format(new_worm_state["tail"]))
    print("Success = {}".format(new_worm_state["worm_success"]))
    
    if new_worm_state["worm_success"] == False:
        pytest.skip()

    new_syndrome_mod_2 = jnp.mod(
        h_z_mod_2 @ new_worm_state["worm_error"][0, :], 
        2
    )

    cond = jnp.all(jnp.mod(new_syndrome_mod_2 - syndrome_mod_2, 2) == 0)
    assert cond == True, \
        f"The syndromes mod 2 do not match in example #{idx}"
    
    new_syndrome_mod_p = jnp.mod(
        h_z_mod_p @ new_worm_state["worm_error"][1, :], 
        p
    )
    
    cond = jnp.all(jnp.mod(new_syndrome_mod_p - syndrome_mod_p, p) == 0)
    assert cond == True, \
        f"The syndromes mod p do not match in example #{idx}"
    
    










        

