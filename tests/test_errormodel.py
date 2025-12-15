# Tests for the error model

import pytest
from coherentinfo.errormodel import (
    ErrorModelBernoulli, 
    ErrorModelBernoulliJax,
    ErrorModelPoisson, 
    ErrorModelPoissonJax,
)
import numpy as np
from numpy.typing import NDArray
import jax

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
    em_poisson = ErrorModelPoisson(num_samples, 6, 0.01)

    probs = em_poisson.get_probabilities()
    error_string = em_poisson.generate_random_error()

    sampled_frequencies = np.array([
        np.argwhere(error_string == x).shape[0] / num_samples
        for x in range(d)
    ])

    epsilon = 1e-3 
    delta = np.sqrt(np.sum((probs - sampled_frequencies)**2))
    assert delta < epsilon, \
        f"Sampled probabilities do not match the expected ones."
    
def test_inverse_sampling_poisson_jax():
    """Tests that the inverse sampling method for Poisson with JAX
    gives the correct distribution. """

    num_samples = int(1e6)
    d = 6
    em_poisson = ErrorModelPoissonJax(num_samples, 6, 0.01)

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
    
def test_inverse_sampling_bernoulli():
    """Tests that the inverse sampling method for Bernoulli gives 
    the correct distribution. """

    num_samples = int(1e6)
    d = 2
    em_bernoulli = ErrorModelPoisson(num_samples, 2, 0.01)

    probs = em_bernoulli.get_probabilities()
    error_string = em_bernoulli.generate_random_error()

    sampled_frequencies = np.array([
        np.argwhere(error_string == x).shape[0] / num_samples
        for x in range(d)
    ])

    epsilon = 1e-3 
    delta = np.sqrt(np.sum((probs - sampled_frequencies)**2))
    assert delta < epsilon, \
        f"Sampled probabilities do not match the expected ones."

def test_inverse_sampling_bernoulli_jax():
    """Tests that the inverse sampling method for Poisson with JAX
    gives the correct distribution. """

    num_samples = int(1e6)
    em_bernoulli_jax = ErrorModelBernoulliJax(num_samples, 2, 0.01)

    probs = em_bernoulli_jax.get_probabilities()
    seed = 90
    key = jax.random.PRNGKey(seed)
    error_string = em_bernoulli_jax.generate_random_error(key)

    sampled_frequencies = np.array([
        np.argwhere(error_string == x).shape[0] / num_samples
        for x in range(2)
    ])

    epsilon = 1e-3 
    delta = np.sqrt(np.sum((probs - sampled_frequencies)**2))
    assert delta < epsilon, \
        f"Sampled probabilities do not match the expected ones."
    


