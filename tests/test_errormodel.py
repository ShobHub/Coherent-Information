# Tests for the error model

import pytest
from coherentinfo.errormodel import (
    ErrorModelBernoulli,
    ErrorModelPoisson, 
    ErrorModelLindblad
)
import numpy as np
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
    


