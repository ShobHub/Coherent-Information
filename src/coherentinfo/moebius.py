# Module for the generation of the moebius code

import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from functools import partial

class MoebiusCode():
    """ Class representing the Moebius code.
    """
    pass
      

def index_h(y: int, x: int, length: int, width: int) -> int:
    """ Gives the index of a horizontal edge.

    Args:
        y: y coordinate
        x: x coordinate
        length: Length of the Moebius strip (number of vertices 
            along the length)
        width: Width of the Moebius strip (number of vertices 
            along the width)
     
     Returns:
        Index associated with the edge.
     """
    
    if x >= length or y >= (width - 1):
        raise ValueError("Coordinates out of bounds.")
    return y * length +  x

def inverted_index_h(y: int, x: int, length: int, width: int) -> int:
    """ Gives the inverted index of a horizontal edge. Useful for the twisted
    boundaries.

    Args:
        y: y coordinate
        x: x coordinate
        length: Length of the Moebius strip (number of vertices 
            along the length)
        width: Width of the Moebius strip (number of vertices 
            along the width)
     
     Returns:
        Index associated with the edge.
     """
    
    if x >= length or y >= (width - 1):
        raise ValueError("Coordinates out of bounds.")
    return (width - 2 - y) * length +  (x % length)

def index_v(y: int, x: int, length: int, width: int) -> int:
    """ Gives the index of a vertical edge.

    Args:
        y: y coordinate
        x: x coordinate
        length: Length of the Moebius strip (number of vertices 
            along the length)
        width: Width of the Moebius strip (number of vertices 
            along the width)
     
     Returns:
        Index associated with the edge.
     """
    # if x >= length or y >= width:
    #     raise ValueError("Coordinates out of bounds.")
    num_h_edges = length * (width - 1)  # horizontal edges
    return num_h_edges + y * length + (x % length)

def build_moebius_code_vertex(
        length: int, 
        width: int
        ) -> Tuple[NDArray, NDArray]:
    """ Generates the Moebius code vertex checks and logical operators.
    This implementation is only for the d=2 case.

    Args:
        length: Length of the Moebius strip (number of vertices 
            along the length)
        width: Width of the Moebius strip (number of vertices 
            along the width)
     
     Returns:
        h_z: The Z-check matrix
        l_z: The logical Z operator
     """
    if length < 3 or width < 3:
        raise ValueError("Length and width must be at least 3.")
    
    if length % 2 == 0:
        raise ValueError("Length must be odd for the Moebius code.")
    
    if width % 2 == 0:
        raise ValueError("Width must be odd for the Moebius code")

    num_h_edges = length * (width - 1)  # horizontal edges
    num_v_edges = length * width  # vertical edges
    num_edges = num_h_edges + num_v_edges

    idx_h = partial(index_h, length=length, width=width)
    inv_idx_h = partial(inverted_index_h, length=length, width=width)
    idx_v = partial(index_v, length=length, width=width)

    # Z-check matrix
    rows = []
    # Note the -1 are irrelevant, but it is better to keep them
    # for clarity and make sure that the coordinates are
    for y in range(width - 1):
        for x in range(length):
            row = np.zeros(num_edges, dtype=np.int8)
            if x != 0:
                if (x + y) % 2 == 0:
                    row[idx_h(y, x)] = 1
                    row[idx_h(y, x - 1)] = 1
                    row[idx_v(y, x)] = 1
                    row[idx_v(y + 1, x)] = 1
                    rows.append(row)
                else:
                    row[idx_h(y, x)] = -1
                    row[idx_h(y, x - 1)] = -1
                    row[idx_v(y, x)] = -1
                    row[idx_v(y + 1, x)] = -1
                    rows.append(row)
            else:
                if (x + y) % 2 == 0:
                    row[idx_h(y, 0)] = 1
                    row[inv_idx_h(y, length - 1)] = 1
                    row[idx_v(y, 0)] = 1
                    row[idx_v(y + 1, 0)] = 1
                    rows.append(row)
                else:
                    row[idx_h(y, 0)] = -1
                    row[inv_idx_h(y, length - 1)] = -1
                    row[idx_v(y, 0)] = -1
                    row[idx_v(y + 1, 0)] = -1
                    rows.append(row)



    h_z = np.array(rows, dtype=np.int8)

    # Logical Z along vertical edges in second row
    l_z = np.zeros(num_edges, dtype=np.int8)
    y0 = width // 2
    for x in range(length):
        l_z[idx_v(y0, x)] = 1

    return h_z, l_z