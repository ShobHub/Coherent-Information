# It contains tests for the functions used in the worm
# algorithm

import pytest
from coherentinfo.moebius_two_odd_prime import MoebiusCodeTwoOddPrime
from coherentinfo.errormodel import ErrorModelLindbladTwoOddPrime
from coherentinfo.worm import (stab_labels, 
                               stabilizer_edges, 
                               plaquette_move_probabilities)
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from jax.typing import ArrayLike
from test_moebius import moebius_code_example
import jax


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

def test_plaquette_moves_probability(moebius_code_example):
    num_examples = len(moebius_code_example)
    idx = np.random.randint(num_examples)
    moebius_code = moebius_code_example[idx]
    gamma_t = 0.1
    em_lindblad = ErrorModelLindbladTwoOddPrime(
        moebius_code.num_edges, d=moebius_code.d, gamma_t=gamma_t
    )
    head = 3
    h_x_mod_p = moebius_code.h_x_mod_p
    # This then also tests jit compilation
    func = jax.jit(plaquette_move_probabilities, static_argnums=(2, 3))
    move_probs = func(head, h_x_mod_p, moebius_code.p, em_lindblad)

    move_probs_test = np.zeros([4, moebius_code.p])

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
    
    cond = jnp.all(move_probs[0] == move_probs_test)
    assert cond == True, \
        f"The move probabilities are not correct in example #{idx}"







        

