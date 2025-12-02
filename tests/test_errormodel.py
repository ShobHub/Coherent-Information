# Tests for the error model

import pytest
from coherentinfo.errormodel import ErrorModelPoisson
import numpy as np
from numpy.typing import NDArray

def test_instantiate_errormodel():
    """Tests that we can instantiate an error model"""

    try:
        em_poisson = ErrorModelPoisson(5, 6, 0.01)
    except Exception as e:
        pytest.fail(f"Error model instantiation failed with exception: {e}")

def test_inverse_sampling():
    """Tests that the inverse sampling method gives the correct
    distribution. """

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
    


