# Module for the generation of the moebius code

import numpy as np
from typing import Tuple
from numpy.typing import NDArray

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
    assert length >= 2 and width >= 2

    num_h_edges = length * (width - 1)  # horizontal edges
    num_v_edges = length * width  # vertical edges
    num_edges = num_h_edges + num_v_edges

    def idx_h(y, x):
        return y * length + (x % length)

    def idx_v(y, x):
        return num_h_edges + y * length + (x % length)

    # Z-check matrix
    rows = []
    for y in range(width - 1):
        for x in range(length):
            row = np.zeros(num_edges, dtype=np.uint8)
            row[idx_h(y, x)] = 1
            row[idx_h(y, x - 1)] = 1
            row[idx_v(y, x)] = 1
            row[idx_v(y + 1, x)] = 1
            rows.append(row)
    h_z = np.array(rows, dtype=np.uint8)

    # Logical Z along vertical edges in second row
    l_z = np.zeros(num_edges, dtype=np.uint8)
    y0 = width // 2
    for x in range(length):
        l_z[idx_v(y0, x)] = 1

    return h_z, l_z