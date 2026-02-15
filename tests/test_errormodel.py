# Tests for the error model

import pytest
from coherentinfo.errormodel import (
    ErrorModelBernoulli,
    ErrorModelPoisson, 
    ErrorModelLindblad,
    ErrorModelLindbladTwoOddPrime,
)
import numpy as np
import jax.numpy as jnp
from numpy.typing import NDArray
import jax
jax.config.update("jax_enable_x64", True)

def test_instantiate_errormodel():
    """Tests that we can instantiate an error model"""

    try:
        em_poisson = ErrorModelPoisson(5, 6, 0.01)
    except Exception as e:
        pytest.fail(f"Error model instantiation failed with exception: {e}")

def test_inverse_sampling_poisson():
    """Tests that the inverse sampling method for Poisson 
    gives the correct distribution. """

    num_samples = int(1e6)
    d = 6
    gamma = 0.01
    em_poisson = ErrorModelPoisson(num_samples, d, gamma)

    probs = em_poisson.get_probabilities()
    seed = 90
    key = jax.random.PRNGKey(seed)
    error_string = em_poisson.generate_random_error(key)

    sampled_frequencies = np.array([
        np.argwhere(error_string == x).shape[0] / num_samples
        for x in range(d)
    ])

    epsilon = 1e-3 
    delta = np.sqrt(np.sum((probs - sampled_frequencies)**2))
    assert delta < epsilon, \
        f"Sampled probabilities do not match the expected ones."

def test_lindblad_symmetry():
    """Tests the the Lindblad error model gives symmetry and normalized
    probabilities """
    d = 10
    gamma = 0.1
    em_lindblad = ErrorModelLindblad(1, d, gamma)
    index = 1
    eps = 1e-6
    delta = np.abs(em_lindblad.probs[index] - em_lindblad.probs[d - index])  
    assert delta < eps, \
        f"The probabilities are not symmetric"
    index = 3
    eps = 1e-6
    delta = np.abs(em_lindblad.probs[index] - em_lindblad.probs[d - index])  
    assert delta < eps, \
        f"The probabilities are not symmetric"
    
def test_inverse_sampling_lindblad():
    """Tests that the inverse sampling method for Lindblad
    gives the correct distribution. """

    num_samples = int(1e6)
    d = 6
    gamma = 0.01
    em_lindblad = ErrorModelLindblad(num_samples, d, gamma)

    probs = em_lindblad.get_probabilities()
    seed = 90
    key = jax.random.PRNGKey(seed)
    error_string = em_lindblad.generate_random_error(key)

    sampled_frequencies = np.array(
        [np.argwhere(error_string == x).shape[0] / num_samples
        for x in range(d)]
    )

    epsilon = 1e-3 
    delta = np.sqrt(np.sum((probs - sampled_frequencies)**2))
    assert delta < epsilon, \
        f"Sampled probabilities do not match the expected ones."


def test_inverse_sampling_bernoulli():
    """Tests that the inverse sampling method for Poisson with JAX
    gives the correct distribution. """

    num_samples = int(1e6)
    em_bernoulli = ErrorModelBernoulli(num_samples, 2, 0.01)

    probs = em_bernoulli.get_probabilities()
    seed = 90
    key = jax.random.PRNGKey(seed)
    error_string = em_bernoulli.generate_random_error(key)

    sampled_frequencies = np.array([
        np.argwhere(error_string == x).shape[0] / num_samples
        for x in range(2)
    ])

    epsilon = 1e-3 
    delta = np.sqrt(np.sum((probs - sampled_frequencies)**2))
    assert delta < epsilon, \
        f"Sampled probabilities do not match the expected ones."
    
def test_modular_probability():
    """Tests that the probility computes using m mod d and those
    computed using m mod 2 and m mod p match, with d = 2 * p and p odd prime. 
    This is guaranteed from the one-to-one correspondance between m mod d and 
    m mod 2 and m mod p, which follows from the Chinese remainder theorem."""
    d = 6
    gamma = 0.1
    em_lindblad = ErrorModelLindbladTwoOddPrime(1, d, gamma)
    m_mod_2_vec = jnp.array([0, 1, 0, 1, 0, 1])
    m_mod_p_vec = jnp.array([0, 1, 2, 0, 1, 2])
    prob_m_vec = em_lindblad.probs
    prob_m_mod_vec = jax.vmap(em_lindblad.get_modular_probability)(
        m_mod_2_vec, 
        m_mod_p_vec
    )

    assert jnp.all(prob_m_vec == prob_m_mod_vec) == True, \
        f"The probability computed with the normal method \n" \
        f"and the modular method do not match."


    


