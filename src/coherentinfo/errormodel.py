# Contains error models to generate random error strings

import numpy as np 
from numpy.typing import NDArray

from abc import ABC, abstractmethod
import scipy


class ErrorModel(ABC):
    """Abstract class for error models."""

    def __init__(self, num_qudits: int, d: int):
        """ Init method for the abstract class. 

        Args:
            num_qudits: Length of the error vector to be generated
            d: Max value of the elements of the vector plus one.
        """

        self.num_qudits = num_qudits
        self.d = d
    
    @abstractmethod
    def get_probabilities(self) -> NDArray:
        """Computes all the probabilities of having the possible single
        qudit errors and stores them in a numpy array."""
        pass

    @abstractmethod
    def generate_random_error(self) -> NDArray:
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

class ErrorModelPoisson(ErrorModel):
    """Class for Poissonian error model."""

    def __init__(self, num_qudits: int, d: int, gamma_t: float):
        super().__init__(num_qudits, d)
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
        error = np.zeros([self.num_qudits], dtype=int)
        def get_single_error() -> int:
            u = np.random.rand()
            outcome_index = np.searchsorted(self.cdf, u, side='left')
            return outcome_index

        for edge in range(self.num_qudits):
            single_error = get_single_error()
            error[edge] = single_error
        return error



    


    
