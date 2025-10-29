# Module for the generation of the moebius code

import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from functools import partial

class MoebiusCode:
    """ Class representing the Moebius code.
    """

    def __init__(
            self,
            length: int, 
            width: int,
            d: int      
    ):
        """ Initializes the Moebius code.

        Args:
            length: Length of the Moebius strip (number of vertices 
                along the length)
            width: Width of the Moebius strip (number of vertices 
                along the width)
            d: qudit dimension
        
        Returns:
            Index associated with the edge.
        """
        
        
        if length < 3 or width < 3:
            raise ValueError("Length and width must be at least 3.")
    
        if length % 2 == 0:
            raise ValueError("Length must be odd for the Moebius code.")
        
        if width % 2 == 0:
            raise ValueError("Width must be odd for the Moebius code")
        
        if d % 2 != 0:
            raise ValueError("Dimension d must be even for the Moebius code")
        
        self.length = length 
        self.width = width 
        self.d = d
        self.num_h_edges = length * (width - 1)  # horizontal edges
        self.num_v_edges = length * width  # vertical edges
        self.num_edges = self.num_h_edges + self.num_v_edges
        self.num_vertex_checks = (length * (width - 1))
        self.num_plaquette_checks = length * width
        # Note that the plaquette checks are not independent, 
        # but we will write all of them anyway to test the implementation
        self.h_z = self.build_moebius_code_vertex()
        self.logical_z = self.get_logical_z()
      

    def index_h(self, y: int, x: int) -> int:
        """ Gives the index of a horizontal edge.

        Args:
            y: y coordinate
            x: x coordinate
        
        Returns:
            Index associated with the edge.
        """
        
        if x >= self.length or y >= (self.width - 1):
            raise ValueError("Coordinates out of bounds.")
        return y * self.length +  x

    def inverted_index_h(self, y: int, x: int) -> int:
        """ Gives the inverted index of a horizontal edge. Useful for the twisted
        boundaries.

        Args:
            y: y coordinate
            x: x coordinate
        
        Returns:
            Index associated with the edge.
        """

        if x >= self.length or y >= (self.width - 1):
            raise ValueError("Coordinates out of bounds.")
        return (self.width - 2 - y) * self.length +  x

    def index_v(self, y: int, x: int) -> int:
        """ Gives the index of a vertical edge.

        Args:
            y: y coordinate
            x: x coordinate
        
        Returns:
            Index associated with the edge.
        """
        if x >= self.length or y >= self.width:
            raise ValueError("Coordinates out of bounds.")
        num_h_edges = self.length * (self.width - 1)  # horizontal edges
        return num_h_edges + y * self.length + x

    def build_moebius_code_vertex(
            self
    ) -> NDArray:
        """ Generates the Moebius code vertex checks.
        
        Returns:
            h_z: The Z-check matrix
        """

        # Z-check matrix
        rows = []
        # Note the -1s are irrelevant, but it is better to keep them
        # for clarity and make sure that the coordinates are
        for y in range(self.width - 1):
            for x in range(self.length):
                row = np.zeros(self.num_edges, dtype=np.int8)
                if x != 0:
                    if (x + y) % 2 == 0:
                        row[self.index_h(y, x)] = 1
                        row[self.index_h(y, x - 1)] = 1
                        row[self.index_v(y, x)] = 1
                        row[self.index_v(y + 1, x)] = 1
                        rows.append(row)
                    else:
                        row[self.index_h(y, x)] = -1
                        row[self.index_h(y, x - 1)] = -1
                        row[self.index_v(y, x)] = -1
                        row[self.index_v(y + 1, x)] = -1
                        rows.append(row)
                else:
                    if (x + y) % 2 == 0:
                        row[self.index_h(y, 0)] = 1
                        row[self.inverted_index_h(y, self.length - 1)] = 1
                        row[self.index_v(y, 0)] = 1
                        row[self.index_v(y + 1, 0)] = 1
                        rows.append(row)
                    else:
                        row[self.index_h(y, 0)] = -1
                        row[self.inverted_index_h(y, self.length - 1)] = -1
                        row[self.index_v(y, 0)] = -1
                        row[self.index_v(y + 1, 0)] = -1
                        rows.append(row)

        h_z = np.array(rows, dtype=np.int8)

        return h_z
    
    def get_logical_z(
            self
    ) -> NDArray:
        """ Returns the logical Z operator.

        Returns:
            l_z: The logical Z operator
        """

        # Logical Z along vertical edges in second row
        logical_z = np.zeros(self.num_edges, dtype=np.int8)
        y0 = self.width // 2
        for x in range(self.length):
            logical_z[self.index_v(y0, x)] = 1
        return logical_z

    def build_moebius_code_plaquette(
            self
    ) -> NDArray:
        """ Generates the Moebius code plaquette checks.
        
        Returns:
            h_x: The X-check matrix
        """

        # Z-check matrix
        rows = []
        # Note that differently than the vertex checks, here
        # the -1 are relevant for the twisted boundarys
        for y in range(self.width):
            for x in range(self.length):
                row = np.zeros(self.num_edges, dtype=np.int8)
                if y == 0:
                    if (x + 1) % self.length != 0:
                        if x % 2 == 0:
                            row[self.index_h(0, x)] = 1
                            row[self.index_v(0, x)] = -1 
                            row[self.index_v(0, x + 1)] = -1
                        else:
                            row[self.index_h(0, x)] = -1
                            row[self.index_v(0, x)] = 1 
                            row[self.index_v(0, x + 1)] = 1
                    else:
                        row[self.index_h(self.width - 2, self.length - 1)] = +1
                        row[self.index_v(0, 0)] = -1 
                        row[self.index_v(self.width - 1, self.length - 1)] = -1

                elif y > 0 and y < (self.width - 1):
                    if (x + 1) % self.length != 0:
                        # This is for the central plaquettes, excluding top 
                        # and bottom and the twisted boundary.
                        # It passes a basic check with width=3 and length=5
                        if (x + y) % 2 == 1:
                            row[self.index_h(y - 1, x)] = -1
                            row[self.index_h(y, x)] = -1
                            row[self.index_v(y, x)] = 1
                            row[self.index_v(y, x + 1)] = 1
                        else:
                            row[self.index_h(y - 1, x)] = 1
                            row[self.index_h(y, x)] = 1
                            row[self.index_v(y, x)] = -1
                            row[self.index_v(y, x + 1)] = -1
                    else:
                        # This is for the twisted boundary plaquettes
                        if (x + y) % 2 == 1:
                            row[self.inverted_index_h(y, x)] = -1
                            row[self.inverted_index_h(y - 1, x)] = -1
                            row[self.index_v(y, 0)] = 1 
                            row[self.index_v(self.width - y - 1, x)] = 1 
                        if (x + y) % 2 == 0:
                            row[self.inverted_index_h(y, x)] = 1
                            row[self.inverted_index_h(y - 1, x)] = 1
                            row[self.index_v(y, 0)] = -1 
                            row[self.index_v(self.width - y -1, x)] = -1           
                else:
                    pass
                rows.append(row)
        
        h_x = np.array(rows, dtype=np.int8)

        return h_x



    def build_vertex_destabilizers(
            self
    ):
        pass

    def build_plaquette_destabilizers(
            self
    ):
        pass