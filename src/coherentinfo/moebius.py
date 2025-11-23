# Module for the generation of the moebius code

import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from functools import partial
from coherentinfo.linalg import (is_prime,
                                 finite_field_pseudoinverse, 
                                 finite_field_inverse)


class MoebiusCode:
    """ Class representing the Moebius code.
    """

    def __init__(self, length: int, width: int, d: int = 2):
        """ Initializes the Moebius code.

        Args:
            length: Length of the Moebius strip (number of vertices 
                along the length)
            width: Width of the Moebius strip (number of vertices 
                along the width)
            d: qudit dimension. It is relevant for the logical operators.
        
        Returns:
            Index associated with the edge.
        """
        
        
        if length < 3 or width < 3:
            raise ValueError("Length and width must be at least 3.")
    
        if length % 2 == 0:
            raise ValueError("Length must be odd for the Moebius code.")
        
        if width % 2 == 0:
            raise ValueError("Width must be odd for the Moebius code.")
        
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
        self.h_x = self.build_moebius_code_plaquette()

        # We directly store also the qubit version of the H_X and H_Z matrices
        # with binary entries. In this case, for the H_X matrix we can 
        # directly pop one of the rows since we know that for qubits 
        # the product of the plaquette stabilizers is the identity which 
        # translates to the sum of the rows of H_X being zero mod 2. 
        # We conventionally pop the first row, which corresponds to the 
        # bottom left plaquette

        self.h_z_qubit = self.h_z % 2
        self.h_x_qubit = np.delete(self.h_x, 0, axis=0) % 2

        # Logicals
        self.logical_z = self.get_logical_z()
        self.logical_x = self.get_logical_x()

        # Destabilizers
        self.vertex_destab = self.build_vertex_destabilizers()
        self.plaquette_destab_qubit = self.build_plaquette_destabilizers_qubit()
        self._plaquette_destab_type_two = None
    
    @property
    def plaquette_destab_type_two(self):
        if self._plaquette_destab_type_two is not None:
            return self._plaquette_destab_type_two
        
        result = self.build_plaquette_destabilizers_type_two()

        if result is None:
            print(f"INFO: The attribute 'plaquette_destab_type_two' \n" \
                  f"is NOT set because d={self.d} is not 2 * p where p \n" \
                  f"is an odd prime.")
        return result
        

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
        """ Gives the inverted index of a horizontal edge. Useful for the 
        twisted boundaries.

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
                row = np.zeros(self.num_edges, dtype=np.int16)
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

        h_z = np.array(rows, dtype=np.int16)

        return h_z

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
                row = np.zeros(self.num_edges, dtype=np.int16)
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
                        row[self.index_h(self.width - 2, self.length - 1)] = 1
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
                    if (x + 1) % self.length != 0:
                        if x % 2 == 0:
                            row[self.index_h(self.width - 2, x)] = 1
                            row[self.index_v(self.width - 1, x)] = -1 
                            row[self.index_v(self.width - 1, x + 1)] = -1
                        else:
                            row[self.index_h(self.width - 2, x)] = -1
                            row[self.index_v(self.width - 1, x)] = 1 
                            row[self.index_v(self.width - 1, x + 1)] = 1
                    else:
                        row[self.index_h(0, self.length - 1)] = 1
                        row[self.index_v(self.width - 1, 0)] = -1 
                        row[self.index_v(0, self.length - 1)] = -1

                    
                rows.append(row)
        
        h_x = np.array(rows, dtype=np.int16)

        return h_x
    
    def get_logical_z(
            self
    ) -> NDArray:
        """ Returns the logical Z operator. This is defined as in the notes
        Z_logical = Z_1^{d/2} \tensor ... \tensor Z_length^{d/2}
        along the central line of vertical edges. 

        Returns:
            l_z: The logical Z operator
        """

        # Logical Z along vertical edges in second row
        logical_z = np.zeros(self.num_edges, dtype=np.int16)
        y0 = self.width // 2
        for x in range(self.length):
            logical_z[self.index_v(y0, x)] = np.int16(self.d / 2)
        return logical_z
    
    def get_logical_x(
            self
    ) -> NDArray:
        """ Returns the logical X operator. This is defined as in the notes
        X_logical = X_1 \tensor X_2^{-1} \tensor ... \tensor X_width
        along the first columns of vertical qudits. 

        Returns:
            l_x: The logical X operator
        """

        # Logical X along horizontal edges in second row
        logical_x = np.zeros(self.num_edges, dtype=np.int16)
        for y in range(self.width):
            if y % 2 == 0:
                logical_x[self.index_v(y, 0)] = -1
            else:
                logical_x[self.index_v(y, 0)] = 1
        p = np.int16(self.d / 2) 
        return p * logical_x
    

    def build_vertex_destabilizers(
            self
    ) -> NDArray:
        """ Returns the vertex destabilizers. Remember that the vertex 
        destabilizers are associated with X-type errors. 

        Returns:
            vertex_destab: The matrix of the vertex destabilizers 
        """

        rows = []
        
        for y in range(self.width - 1):
            for x in range(self.length):
                row = np.zeros(self.num_edges, dtype=np.int16)
                if y < (self.width - 1) / 2:
                    for y_prime in range(y + 1):
                        if (x + y_prime) % 2 == 0:
                            row[self.index_v(y_prime, x)] = 1
                        else:
                            row[self.index_v(y_prime, x)] = -1
                else:
                    for y_prime in range(y + 1, self.width):
                        if (x + y_prime) % 2 == 0:
                            row[self.index_v(y_prime, x)] = -1
                        else:
                            row[self.index_v(y_prime, x)] = 1
                rows.append(row)


        vertex_destab = np.array(rows, dtype=np.int16)

        return vertex_destab



    def build_plaquette_destabilizers_qubit(
            self
    ) -> NDArray:
        """ Returns the plaquette destabilizers assuming qubits as fundamental 
        system on the edges. Remember that the vertex destabilizers are 
        associated with Z-type errors. The more general case of qudits 
        with d = 2 q and q a prime can be also obtained from the basic qubit 
        case. 

        Returns:
            plaquette_destab_qubit: The matrix of the qubit plaquette 
                destabilizers

        """

        rows = []
        for y in range(0, self.width):
            for x in range(0, self.length):
                row = np.zeros(self.num_edges, dtype=np.int16)
                if (x + 1) != self.length:
                    for x_prime in range(1, x + 1):
                        row[self.index_v(0, x_prime)] = 1
                    for y_prime in range(y):
                        row[self.index_h(y_prime, x)] = 1
                else:
                    for x_prime in range(1, x):
                        row[self.index_v(0, x_prime)] = 1
                    for y_prime in range(self.width - 1 - y):
                        row[self.index_h(y_prime, x - 1)] = 1
                    row[self.index_v(self. width - 1 - y, x)] = 1

                rows.append(row)

        plaquette_destab_qubit = np.array(rows, dtype=np.int16)
        plaquette_destab_qubit = np.delete(plaquette_destab_qubit, 0, axis=0)

        return plaquette_destab_qubit
    
    def build_plaquette_destabilizers_type_two(
            self
    ) -> NDArray:
        """ Returns the plaquette destabilizers associated with the 
        stabilizers S_j^X[2] assuming  qudits with d = 2 p and p odd prime. 
        These can be simply obtained from the qubit case.

        Returns:
            The matrix of the plaquette destabilizers of type two.
        """

        p = np.int16(self.d / 2)

        if not is_prime(p) or p % 2 == 0:
            return None
        else:
            return self.plaquette_destab_qubit * p
    
    def build_pre_plaquette_destabilizers_type_p(
            self
    ) -> Tuple[NDArray, NDArray]:
        """ Returns the pre-plaquette destabilizers associated with the 
        stabilizers S_j^X[p] assuming  qudits with d = 2 p and p odd prime.
        The pre-plaquette destabilizers are used in the construction of 
        the proper destabilizers. They have the property that they do 
        not commute with the corresponding destabilizer, but they might 
        also not commute with other ones. Additionally, they are constructed
        in such a way that they are independent. This is a necessary condition
        for the later construction of the proper plaquette destabilizers
        of type p. 

        Returns:
            The matrix of the pre-plaquette destabilizers of type p.
        """

        p = np.int16(self.d / 2)

        h_x_tilde = 2 * self.h_x.copy() % p
        # We first remove the columns associated with edges that define
        # the logical X
        logical_x_edges = [self.index_v(y, 0) for y in range(self.width)]
        h_x_tilde = np.delete(h_x_tilde, logical_x_edges, axis=-1)
        destab_tilde = finite_field_pseudoinverse(h_x_tilde, p).T
        # We now need to add zero columns corresponding to the edges that
        # define the logical X
        destab = destab_tilde.copy()
        for y in range(self.width):
            destab = np.insert(destab, self.index_v(y, 0), 0, axis=-1)
        # destab = np.hstack((destab, np.zeros([destab.shape[0], 1], dtype=np.int_)))

        return h_x_tilde, destab_tilde, destab
        


    
    def build_plaquette_destabilizers_type_p(
            self
    ) -> NDArray:
        """ Returns the plaquette destabilizers associated with the 
        stabilizers S_j^X[p] assuming  qudits with d = 2 p and p odd prime. 

        Returns:
            The matrix of the plaquette destabilizers of type p.
        """


        p = np.int16(self.d / 2)

        if not is_prime(p) or p % 2 == 0:
            return None
        else:
            v_mat = np.vstack(self.h_x % p, self.logical_x % p)
            w_mat = (v_mat.T @ finite_field_inverse(v_mat @ v_mat.T)).T % p

            return w_mat[:-1] 

                





        

            


        
        

        

