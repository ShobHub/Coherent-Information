# Module for the generation of the moebius code

import numpy as np
from typing import Tuple, Dict
from numpy.typing import NDArray
from coherentinfo.linalg import (is_prime,
                                 finite_field_gauss_jordan_elimination)
from coherentinfo.errormodel import ErrorModel
import scipy 



class MoebiusCode:
    """ Class representing the Moebius code.
    """

    def __init__(self, length: int, width: int, d: int = 2):
        """ Initializes the Moebius code. The strategy is to store
        all the necessary data, since they would need to be called
        repeatedly if we use this for computing the coherent 
        information. 

        Args:
            length: Length of the Moebius strip (number of vertices 
                along the length)
            width: Width of the Moebius strip (number of vertices 
                along the width)
            d: qudit dimension. It is relevant for the logical operators.
        """
        
        
        if length < 3 or width < 3:
            raise ValueError("Length and width must be at least 3.")
    
        if length % 2 == 0:
            raise ValueError("Length must be odd for the Moebius code.")
        
        if width % 2 == 0:
            raise ValueError("Width must be odd for the Moebius code.")
        
        if d % 2 != 0:
            raise ValueError("Dimension d must be even for the Moebius code")
        
        self._length = length 
        self._width = width 
        self._d = d
        self.compute_and_set_code_properties()
        
    
    def compute_and_set_code_properties(self):
        """Helper method to run any time a parameter changes"""
        if is_prime(np.int16(self.d / 2)) and np.int16(self.d / 2) != 2:
            self.p = np.int16(self.d / 2)
        else:
            self.p = None
        self.num_h_edges = self.length * (self.width - 1)  # horizontal edges
        self.num_v_edges = self.length * self.width  # vertical edges
        self.num_edges = self.num_h_edges + self.num_v_edges
        self.num_vertex_checks = (self.length * (self.width - 1))
        self.num_plaquette_checks = self.length * self.width
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
        self.h_x_mod_p = self.h_x % self.p if self.p is not None else None
        self.plaquette_destab_type_two = \
            self.p * self.plaquette_destab_qubit if self.p is not None else None
        self.plaquette_destab_mod_p = \
            self.build_plaquette_destabilizers_mod_p() \
                if self.p is not None else None
        self.plaquette_destab_type_p = 2 * self.plaquette_destab_mod_p \
            if self.p is not None else None

    @property
    def length(self):
        return self._length
    
    @property
    def width(self):
        return self._width
    
    @property
    def d(self):
        return self._d
    
    @length.setter
    def length(self, length):
        self._length = length
        self.compute_and_set_code_properties()
    
    @width.setter
    def width(self, width):
        self._width = width
        self.compute_and_set_code_properties()
    
    @d.setter
    def d(self, d):
        self._d = d
        self.compute_and_set_code_properties()
    

    
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
        # Note here we 
        if self.p is not None:
            return self.p * logical_x
        else:
            return logical_x
    

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
    
    # It is convenient when we study qudits to be able to call
    # the method above in the following way.
    build_plaquette_destabilizers_mod_two = \
        build_plaquette_destabilizers_qubit
    
    # def build_plaquette_destabilizers_type_two(
    #         self
    # ) -> NDArray:
    #     """ Returns the plaquette destabilizers associated with the 
    #     stabilizers S_j^X[2] assuming  qudits with d = 2 p and p odd prime. 
    #     These can be simply obtained from the qubit case.

    #     Returns:
    #         The matrix of the plaquette destabilizers of type two.
    #     """

    #     if self.p is not None:
    #         return self.p * self.plaquette_destab_qubit
    #     else:
    #         None
    
    @staticmethod    
    def finite_field_right_pseudoinverse(
        mat: NDArray[np.int_],
        p: int
    ) -> NDArray[np.int_]:
        """Return a right pseudoinverse of ``mat`` over GF(p). If A = mat
        is a n x m matrix with n < m and rank(A) = n the function 
        returns a m x n matrix B such that A B = I_{n x n}. 
        Watch out. The method seems to work for the matrices involved in the 
        Moebius code, but it does not work in general. A more general and 
        correct method should be implemented based on the singular value
        decomposition.  

        Args:
            mat: Numpy array representing the matrix. 
            p: Prime modulus (must be prime so inverses exist for non-zero elems).

        Returns:
            A new numpy array containing the pseudoinverse of ``mat`` modulo ``p``.

        Raises:
            ValueError: If the rank of the matrix is not equal to the number of
            rows.
        """

        if mat.shape[0] >= mat.shape[1]:
            raise ValueError(f"The number of rows should be smaller than the"
                            f"number of columns.")
        n, m = mat.shape
        
        augmented_mat = np.hstack((mat % p, np.eye(n, dtype=np.int_) % p))
        rref_mat = finite_field_gauss_jordan_elimination(augmented_mat, p)
        if not np.array_equal(rref_mat[:, :n], np.eye(n, dtype=np.int_)):
            raise ValueError(f"The rank of the matrix must be equal to the "
                            f"number of rows.")
        
        pseudo_inv_mat = \
            np.vstack((rref_mat[:, m:], np.zeros([m - n, n], dtype=np.int_)))
            
        
        return pseudo_inv_mat


    def build_plaquette_destabilizers_mod_p(
        self
    ) -> Tuple[NDArray, NDArray] | None:
        """ Returns the destabilizers assuming qupit with p 
        odd prime on the edges. If p is not odd prime it returns 
        None

        Returns:
            The matrix of the pre-plaquette destabilizers of type p.
        """

        if self.p is not None:
            plaquette_destab_qupit = \
                self.finite_field_right_pseudoinverse(self.h_x, 
                                                      self.p)
            return plaquette_destab_qupit.T
        else:
            return None

    def build_plaquette_destabilizers_type_p(
        self
    ) -> NDArray:
        """ Returns the plaquette destabilizers associated with the 
        stabilizers S_j^X[p] assuming  qudits with d = 2 p and p odd prime. 

        Returns:
            The matrix of the plaquette destabilizers of type p.
        """

        if self.p is not None:
            return 2 * self.build_plaquette_destabilizers_mod_p()
        else:
            return None
    
    def get_vertex_syndrome(
        self,
        error: NDArray
    ) -> NDArray:
        """Computes the vertex syndrome (Z-type) associated with 
        and error """

        return self.h_z @ error.T % self.d
    
    def get_plaquette_syndrome(
        self,
        error: NDArray
    ) -> NDArray:
        """Computes the plaquette syndrome (X-type) associated with 
        and error """

        return self.h_x @ error.T % self.d
    
    def get_vertex_candidate_error(
        self,
        syndrome: NDArray
    ) -> NDArray:
        """ Given a valid vertex syndrome it returns the candidate 
        error vector, that generates the same syndrome and commutes 
        with the logical Z.   
        """
        syndrome = syndrome % (self.d)
        candidate = np.zeros(self.num_edges, dtype=np.int16)
        for index in range(self.num_vertex_checks):
            destab = self.vertex_destab[index, :]
            candidate = (candidate + syndrome[index] * destab) % self.d 
        return candidate % self.d
    
    def get_plaquette_candidate_error(
        self,
        syndrome: NDArray
    ) -> NDArray:
        """ Given a valid plaquette syndrome it returns the candidate 
        error vector, that generate the same syndrome and commutes 
        with the logical X.   
        """
        
        syndrome = syndrome % self.d
        syndrome_mod_two = syndrome % 2
        # Note the following is not the syndrome mod p, but it is the vector
        # needed to correctly account for the syndrome mod p
        syndrome_mod_p_aux = \
            (syndrome - syndrome_mod_two * self.p) * \
                pow(2, -1, int(self.p)) % self.p 

        if np.sum(syndrome_mod_two[1:]) % 2 != syndrome_mod_two[0]:
            raise ValueError(f"The syndrome is not valid as it does not"
                             f"satisfy the plaquette constraint")
        
        candidate_type_two = np.zeros(self.num_edges, dtype=np.int16)
        candidate_type_p = np.zeros(self.num_edges, dtype=np.int16)
        for index in range(self.num_plaquette_checks):
            if index != 0:
                destab_type_two = self.plaquette_destab_type_two[index - 1, :]
            else:
                destab_type_two = np.zeros(self.num_edges, dtype=np.int16)
            destab_type_p = self.plaquette_destab_type_p[index, :]
            candidate_type_two = (candidate_type_two + \
                syndrome_mod_two[index] * destab_type_two) % self.d 
            candidate_type_p = (candidate_type_p + \
                syndrome_mod_p_aux[index] * destab_type_p) % self.d

        return (candidate_type_two + candidate_type_p) % self.d

    def compute_vertex_syndrome_chi_probabilities(
        self,
        num_samples: int,
        error_model: ErrorModel
    ) -> Dict:
        """Computes the probability of observing a certain vertex syndrome
        and a corresponding logical chi_x by sampling many errors 
        
        Args:
            num_samples: number of samples used to evaluate the entropy
            errormodel: an error model to generate error samples
        
        Returns:
            Dictionary with syndrome and chi probabilities
        """

        result = {}

        for _ in range(num_samples):
            error = error_model.generate_random_error()
            syndrome = self.get_vertex_syndrome(error)
            candidate_error = self.get_vertex_candidate_error(syndrome)
            error_diff = error - candidate_error 
            res_logical_com_diff = error_diff @ self.logical_z.T % self.d
            chi = int(res_logical_com_diff / self.p)
            syndrome_chi_id = "_".join(map(str, syndrome))
            if syndrome_chi_id in result.keys():
                result[syndrome_chi_id][chi] += 1 / num_samples
            else:
                result[syndrome_chi_id] = [0.0, 0.0]
                result[syndrome_chi_id][chi] = 1 / num_samples
        return result
    
    def compute_vertex_conditional_entropy(
        self,
        num_samples: int,
        error_model: ErrorModel
    ) -> float:
        """Computes the vertex entropy H(chi_x| sigma_z) 
        
        Args:
            num_samples: number of samples used to evaluate the entropy
            errormodel: an error model to generate error samples
        
        Returns
            The vertex conditional entropy
        """

        result = self.compute_vertex_syndrome_chi_probabilities(
            num_samples, error_model
            )
        
        entropy = 0.0
        for key in result.keys():
            prob_syndrome = sum(result[key])
            cond_prob_syndrome_chi_zero = result[key][0] / prob_syndrome
            cond_prob_syndrome_chi_one = result[key][1] / prob_syndrome
            entropy += -scipy.special.xlogy(result[key][0], 
                                           cond_prob_syndrome_chi_zero) 
            entropy += -scipy.special.xlogy(result[key][1], 
                                           cond_prob_syndrome_chi_one) 
        # Convert to base 2 entropy
        entropy = entropy / np.log(2)
        return entropy
    
                
    def compute_plaquette_syndrome_chi_probabilities(
        self,
        num_samples: int,
        error_model: ErrorModel
    ) -> Dict:
        """Computes the probability of observing a certain plaquette syndrome
        and a corresponding logical chi_z by sampling many errors 
        
        Args:
            num_samples: number of samples used to evaluate the entropy
            errormodel: an error model to generate error samples

        Returns:
            Dictionary with syndrome and chi probabilities
        """

        result = {}

        for _ in range(num_samples):
            error = error_model.generate_random_error()
            syndrome = self.get_plaquette_syndrome(error)
            candidate_error = self.get_plaquette_candidate_error(syndrome)
            error_diff = error - candidate_error 
            res_logical_com_diff = error_diff @ self.logical_x.T % self.d
            chi = int(res_logical_com_diff / self.p)
            syndrome_chi_id = "_".join(map(str, syndrome))
            if syndrome_chi_id in result.keys():
                result[syndrome_chi_id][chi] += 1 / num_samples
            else:
                result[syndrome_chi_id] = [0.0, 0.0]
                result[syndrome_chi_id][chi] = 1 / num_samples
        return result
    
    def compute_plaquette_conditional_entropy(
        self,
        num_samples: int,
        error_model: ErrorModel
    ) -> float:
        """Computes the plaquette conditional entropy H(chi_z| sigma_X) 
        
        Args:
            num_samples: number of samples used to evaluate the entropy
            errormodel: an error model to generate error samples
        
        Returns
            The plaquette conditional entropy
        """

        result = self.compute_plaquette_syndrome_chi_probabilities(
            num_samples, error_model
            )
        
        entropy = 0.0
        for key in result.keys():
            prob_syndrome = sum(result[key])
            cond_prob_syndrome_chi_zero = result[key][0] / prob_syndrome
            cond_prob_syndrome_chi_one = result[key][1] / prob_syndrome
            entropy += -scipy.special.xlogy(result[key][0], 
                                           cond_prob_syndrome_chi_zero) 
            entropy += -scipy.special.xlogy(result[key][1], 
                                           cond_prob_syndrome_chi_one) 
        # Convert to base 2 entropy
        entropy = entropy / np.log(2)
        return entropy
    
    def compute_coherent_information(
        self,
        num_samples: int,
        error_model,
    ) -> float:
        """Computes the coherent information 
        
        Args:
            num_samples: number of samples used to evaluate the entropy
            errormodel: an error model to generate error samples
        
        Returns:
            The coherent information
        """

        vertex_entropy = self.compute_vertex_conditional_entropy(
            num_samples, 
            error_model
            )
        plaquette_entropy = self.compute_plaquette_conditional_entropy(
            num_samples, 
            error_model
            )
        
        coherent_info = 1.0 - vertex_entropy - plaquette_entropy
        return coherent_info
        
    
    


        







        

            


        
        

        

