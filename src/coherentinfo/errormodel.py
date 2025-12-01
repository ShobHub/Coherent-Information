# Contains error models to generate random error strings

import numpy as np 
from numpy.typing import NDArray

def generate_error(
    num_edges: int,
    d: int,
) -> NDArray:
    """ Generates a random error with elements

    Args:
        num_edges: Length of the vector
        d: 
    """
