# Contains error models to generate random error strings

import numpy as np 
from numpy.typing import NDArray
import jax
from jax import Array
from jax.typing import ArrayLike
from coherentinfo.linalg import is_prime
from coherentinfo.dtypes import INT_DTYPE

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
        self.cdf = jnp.cumsum(self.probs)

    def get_probabilities(self) -> Array:
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
        return error.astype(INT_DTYPE)
    
class ErrorModelLindblad(ErrorModel):
    """Class for generalized Pauli-Lindblad error model"""

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
        single-qudit errors distributed according to the 
        generalized Pauli-Lindblad error model
        
        Returns:
            An array with the probabilities of single-qudit error.
        """
        def prob_dist(n: Array) -> Array:
            def func(l: Array):
                term = jnp.exp(
                    -4 * self.gamma_t * jnp.sin(l * jnp.pi / self.d)**2
                )
                term *= jnp.exp(2 * jnp.pi * 1j * l * n / self.d) 
                return term / self.d
            l_vec = jnp.arange(0, self.d)
            prob = jnp.sum(jax.vmap(func)(l_vec), axis=0)
            return jnp.real(prob)
        return prob_dist(jnp.arange(0, self.d))

    def get_probability(self, m: int):
        """Computes the probability of having error m with m=0, 1, ..., d-1
        
        Args:
            m: integer that defines the power of the error
        
        Returns:
            The probability of having error labeled by m
        """

        return self.probs[m]
            
    def generate_random_error(self, key) -> Array:
        """Generates a random error string where each element is distributed 
        according to a Poisson distribution. 
        
        Returns:
            An array of integers representing the sampled error.

        """
        uniform_error = jax.random.uniform(key, self.num_subsys)
        error = jnp.searchsorted(self.cdf, uniform_error, side='left')
        return error.astype(INT_DTYPE)

class ErrorModelLindbladTwoOddPrime(ErrorModelLindblad):
    """Class for generalized Pauli-Lindblad error model 
    specialized to the case of d = 2 * p with p odd prime"""

    def __init__(self, num_subsys: int, d: int, gamma_t: float):
        cond = is_prime(jnp.int32(d / 2)) and jnp.int32(d / 2) != 2
        if not cond:
            raise ValueError("d must be 2 * p with p odd prime")
        super().__init__(num_subsys, d, gamma_t)
        self.p = jnp.int32(self.d / 2)
    
    def get_modular_probability(self, m_mod_2: int, m_mod_p: int):
        m_mod_2 = jnp.mod(m_mod_2, 2)
        m_mod_p = jnp.mod(m_mod_p, self.p)
        m = jnp.int32(self.p * m_mod_2 + (1 - self.p) / 2 * 2 * m_mod_p)        
        prob = self.probs[m]
        return prob





class ErrorModelPoisson(ErrorModel):
    """Class for Poissonian error model.
    TO DO. Fix the non-periodicity of the distribution.
    """


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
        return error.astype(INT_DTYPE)


    


    
