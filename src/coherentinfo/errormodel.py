# Contains error models to generate random error strings

import numpy as np 
from numpy.typing import NDArray
import jax
from jax import Array
from jax.typing import ArrayLike

from abc import ABC, abstractmethod
import scipy
import jax.numpy as jnp 


class ErrorModel(ABC):
    """Abstract class for error models."""

    def __init__(self, num_subsys: int, d: int):
        """ Init method for the abstract class. 

        Args:
            num_subsys: Length of the error vector to be generated
            d: Max value of the elements of the vector plus one.
        """

        self.num_subsys = num_subsys
        self.d = d
    
    @abstractmethod
    def get_probabilities(self) -> ArrayLike:
        """Computes all the probabilities of having the possible single
        qudit errors and stores them in a numpy array."""
        pass

    @abstractmethod
    def generate_random_error(self) -> ArrayLike:
        """Generates an error vector according to the corresponding
        probability distribution"""
        pass 

# class ErrorModelLindblad(ErrorModel):
#     """Class for Lindbladian error model."""

#     def __init__(self, num_qudits: int, d: int, gamma_t: float):
#         super().__init__(num_qudits, d)
#         self.gamma_t = gamma_t 
    
#     def get_probabilities(self) -> NDArray:
#         probs = np.zeros([self.d], dtype=float)
#         probs[0] = 1.0 
#         return probs 
    
#     def generate_random_error(self) -> NDArray[int]:
#         return np.zeros([self.num_qudits], dtype=int)

class ErrorModelBernoulli(ErrorModel):
    """Class for Qubit error model."""

    def __init__(self, num_subsys: int, d: int=2, p_error: float=0.1):
        super().__init__(num_subsys, d)
        if d != 2:
            raise ValueError("d must be equal to 2 for this error model.")
        self.p_error = p_error
        self.probs = self.get_probabilities()
        # The following cumulative sum is used to create from 
        # our distribution starting from the uniform distribution 
        # using the Inverse Transform Sampling method
        self.cdf = np.cumsum(self.probs)

    def get_probabilities(self) -> NDArray:
        """Computes the probabilities of having or not a single-qubit error
        
        Returns:
            An array with the probabilities of single-qubit error.
        """
        
        return np.array([1 - self.p_error, self.p_error])

    def generate_random_error(self) -> NDArray[int]:
        """Generates a random error string where each element is distributed 
        according to a Poisson distribution. 
        
        Returns:
            An array of integers representing the sampled error.

        """
        uniform_error = np.random.rand(self.num_subsys)
        error = np.searchsorted(self.cdf, uniform_error, side='left')
        return error
    
class ErrorModelBernoulliJax(ErrorModel):
    """Class for Qubit error model using Jax."""

    def __init__(self, num_subsys: int, d: int=2, p_error: float=0.1):
        super().__init__(num_subsys, d)
        if d != 2:
            raise ValueError("d must be equal to 2 for this error model.")
        self.p_error = p_error
        self.probs = self.get_probabilities()
        # The following cumulative sum is used to create from 
        # our distribution starting from the uniform distribution 
        # using the Inverse Transform Sampling method
        self.cdf = jnp.cumsum(self.probs)

    def get_probabilities(self) -> NDArray:
        """Computes the probabilities of having or not a single-qubit error
        
        Returns:
            An array with the probabilities of single-qubit error.
        """
        
        return jnp.array([1 - self.p_error, self.p_error])
    
    def generate_random_error(self, key) -> Array:
        """Generates a random error string where each element is distributed 
        according to a Bernoulli distribution.
        
        Returns:
            An array of integers representing the sampled errors.

        """
        uniform_error = jax.random.uniform(key, self.num_subsys)
        error = jnp.searchsorted(self.cdf, uniform_error, side='left')
        return error


class ErrorModelPoisson(ErrorModel):
    """Class for Poissonian error model."""

    def __init__(self, num_subsys: int, d: int, gamma_t: float):
        super().__init__(num_subsys, d)
        self.gamma_t = gamma_t
        self.probs = self.get_probabilities()
        # The following cumulative sum is used to create from 
        # our distribution starting from the uniform distribution 
        # using the Inverse Transform Sampling method
        self.cdf = np.cumsum(self.probs)

    def get_probabilities(self) -> NDArray:
        """Computes the probabilities of having the d possible
        single-qudit errors
        
        Returns:
            An array with the probabilities of single-qudit error.
        """
        def prob_dist(n: NDArray[int]) -> NDArray[float]:
            prob = np.exp(-2 * self.gamma_t) * (2 * self.gamma_t)**n / \
                scipy.special.factorial(n)
            return prob
        return prob_dist(np.arange(self.d))

    def generate_random_error(self) -> NDArray[int]:
        """Generates a random error string where each element is distributed 
        according to a Poisson distribution. 
        
        Returns:
            An array of integers representing the sampled error.

        """
        uniform_error = np.random.rand(self.num_subsys)
        error = np.searchsorted(self.cdf, uniform_error, side='left')
        return error

class ErrorModelPoissonJax(ErrorModel):
    """Class for Poissonian error model using Jax."""

    def __init__(self, num_subsys: int, d: int, gamma_t: float):
        super().__init__(num_subsys, d)
        self.gamma_t = gamma_t
        self.probs = self.get_probabilities()
        # The following cumulative sum is used to create from 
        # our distribution starting from the uniform distribution 
        # using the Inverse Transform Sampling method
        self.cdf = jnp.cumsum(self.probs)

    def get_probabilities(self) -> Array:
        """Computes the probabilities of having the d possible
        single-qudit errors
        
        Returns:
            An array with the probabilities of single-qudit error.
        """
        def prob_dist(n: Array) -> Array:
            prob = jnp.exp(-2 * self.gamma_t) * (2 * self.gamma_t)**n / \
                jax.scipy.special.factorial(n)
            return prob
        return prob_dist(jnp.arange(self.d))

    def generate_random_error(self, key) -> Array:
        """Generates a random error string where each element is distributed 
        according to a Poisson distribution. 
        
        Returns:
            An array of integers representing the sampled error.

        """
        uniform_error = jax.random.uniform(key, self.num_subsys)
        error = jnp.searchsorted(self.cdf, uniform_error, side='left')
        return error


    


    
