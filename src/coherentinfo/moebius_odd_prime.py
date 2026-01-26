# Module for the generation of the moebius code for d = 2 * p
# with p odd prime

from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray
import jax.numpy as jnp
import numpy as np
import jax
from coherentinfo.linalg import (is_prime,
                                 finite_field_gauss_jordan_elimination)
from coherentinfo.errormodel import ErrorModel
from coherentinfo.moebius import MoebiusCode

class MoebiusCodeOddPrime(MoebiusCode):
    """ Subclass representing the Moebius code when d = 2 * p
    and p is an odd prime.
    """

    def __init__(self, length: int, width: int, d: int = 6):
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
        if is_prime(np.int16(d / 2)) and np.int16(d / 2) != 2:
            self.p = np.int16(d / 2)
            self.inverse_two_mod_p = pow(2, -1, int(self.p))
        else:
            raise ValueError("d must be 2 * p with p odd prime.")

        super().__init__(length, width, d)
    
    def compute_and_set_code_properties(self):
        """Helper method to run any time a parameter changes"""
        # This calls the parent version
        super().compute_and_set_code_properties()
        self.h_x_mod_p = jnp.mod(self.h_x, self.p)
        self.plaquette_destab_type_two = \
            self.p * self.plaquette_destab_qubit 
        self.plaquette_destab_mod_p = \
            self.build_plaquette_destabilizers_mod_p() 
        self.plaquette_destab_type_p = 2 * self.plaquette_destab_mod_p
    
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
    ) -> Array:
        """ Returns the destabilizers assuming qupit with p 
        odd prime on the edges. If p is not odd prime it returns 
        None

        Returns:
            The matrix of the pre-plaquette destabilizers of type p.
        """

        plaquette_destab_qupit = \
            self.finite_field_right_pseudoinverse(self.h_x, 
                                                  self.p)
        return jnp.array(plaquette_destab_qupit.T)

    def build_plaquette_destabilizers_type_p(
        self
    ) -> Array:
        """ Returns the plaquette destabilizers associated with the 
        stabilizers S_j^X[p] assuming  qudits with d = 2 p and p odd prime. 

        Returns:
            The matrix of the plaquette destabilizers of type p.
        """

        return 2 * self.build_plaquette_destabilizers_mod_p()
    
    def get_vertex_syndrome(
        self,
        error: NDArray
    ) -> Array:
        """Computes the vertex syndrome (Z-type) associated with 
        error 
        
        Args:
            error: The vertex error array
        
        Returns:
            An array representing the vertex error syndrome
        """

        return jnp.mod(jnp.dot(self.h_z, error), self.d) 
    
    def get_plaquette_syndrome(
        self,
        error: NDArray
    ) -> NDArray:
        """Computes the plaquette syndrome (X-type) associated with 
        error 
        
        Args:
            error: The plaquette error array
        
        Returns:
            An array representing the plaquette error syndrome
        """

        return jnp.mod(jnp.dot(self.h_x, error), self.d) 
    
    def get_vertex_candidate_error(
        self,
        syndrome: NDArray
    ) -> Array:
        """ Given a valid vertex syndrome it returns the candidate 
        error vector, that generates the same syndrome and commutes 
        with the logical Z. 

        Args:
            syndrome: syndrome vector 

        Returns:
            Candidate X-type error that gives the syndrome and commutes
            with the logical Z.   
        """
        syndrome = jnp.mod(syndrome, self.d)
        vertex_destab_j = jnp.asarray(self.vertex_destab)
        # candidate = np.zeros(self.num_edges, dtype=np.int16)
        # for index in range(self.num_vertex_checks):
        #     destab = self.vertex_destab[index, :]
        #     candidate = (candidate + syndrome[index] * destab) % self.d 
        candidate = jnp.mod(jnp.dot(syndrome, vertex_destab_j), self.d)

        return jnp.mod(candidate, self.d)
    
    def get_plaquette_candidate_error(
        self,
        syndrome: ArrayLike
    ) -> Array:
        """ Given a valid plaquette syndrome it returns the candidate 
        error vector, that generate the same syndrome and commutes 
        with the logical X.  

        Args:
            syndrome: syndrome vector
        
        Returns:
            Candidate Z-type error that gives the syndrome and commutes
            with the logical X. 
        """
        
        syndrome = jnp.mod(syndrome, self.d)
        syndrome_mod_two = jnp.mod(syndrome, 2)
        # Note the following is not the syndrome mod p, but it is the vector
        # needed to correctly account for the syndrome mod p
        syndrome_mod_p_aux = jnp.mod(
            (syndrome - syndrome_mod_two * self.p) * self.inverse_two_mod_p, self.p
        ) 
        
        # Here I leave as a comment also an implementation that was 
        # previously used and it is not in JAX. It is more intuitive to understand 
        # what is going on and can be used to compare the JAX implemtation 
        # in case there are issues

        # if np.sum(syndrome_mod_two[1:]) % 2 != syndrome_mod_two[0]:
        #     raise ValueError(f"The syndrome is not valid as it does not"
        #                      f"satisfy the plaquette constraint")
        
        # candidate_type_two = np.zeros(self.num_edges, dtype=np.int16)
        # candidate_type_p = np.zeros(self.num_edges, dtype=np.int16)
        # for index in range(self.num_plaquette_checks):
        #     if index != 0:
        #         destab_type_two = self.plaquette_destab_type_two[index - 1, :]
        #     else:
        #         destab_type_two = np.zeros(self.num_edges, dtype=np.int16)
        #     destab_type_p = self.plaquette_destab_type_p[index, :]
        #     candidate_type_two = (candidate_type_two + \
        #         syndrome_mod_two[index] * destab_type_two) % self.d 
        #     candidate_type_p = (candidate_type_p + \
        #         syndrome_mod_p_aux[index] * destab_type_p) % self.d
        # plaquette_destab_type_two_j = jnp.asarray(
        #     jnp.delete(self.plaquette_destab_type_two, 0, axis=0)
        # )

        plaquette_destab_type_two_j = jnp.asarray(
            self.plaquette_destab_type_two
        )

        candidate_type_two = jnp.mod(
            jnp.dot(jnp.delete(syndrome_mod_two, 0), 
                    plaquette_destab_type_two_j), 
            self.d
        )

        plaquette_destab_type_p_j = jnp.asarray(self.plaquette_destab_type_p)

        candidate_type_p = jnp.mod(
            jnp.dot(syndrome_mod_p_aux, plaquette_destab_type_p_j), 
            self.d
        )

        return jnp.mod(candidate_type_two + candidate_type_p, self.d)

    def compute_vertex_syndrome_chi_z(
        self,
        error: ArrayLike
    ) -> Array:
        """Computes the vertex vector syndrome and the corresponding chi_z
        associated with an error. 
        
        Args:
            error: array of errors on each subsystems.
        
        Returns:
            An array where the first num_vertex elements are the syndromes
            and the last one is the chi_z.
        """

        syndrome = self.get_vertex_syndrome(error)
        candidate_error = self.get_vertex_candidate_error(syndrome)
        error_diff = error - candidate_error 
        res_logical_com_diff = \
            jnp.mod(jnp.dot(error_diff, self.logical_z.T), self.d)
        chi_z = jnp.int16(res_logical_com_diff)
        return jnp.append(syndrome, chi_z)
    
    def compute_batched_vertex_syndrome_chi_z(
        self,
        num_samples: int, 
        error_model: ErrorModel,
        master_key: Array
    ) -> Array:
        """Computes the vertex vector syndrome and the corresponding chi_z
        associated with many sampled error. 
        
        Args:
            num_samples: number of sampled errors
            error_model: error model that generates the errors. It needs
                to be compatible with JAX
        
        Returns:
            An array where in each row the first num_vertex elements 
            are the syndromes and the last one is the chi_z.
        """
        #master_vertex_key = jax.random.PRNGKey(48090)
        vertex_keys = jax.random.split(master_key, num_samples)

        batched_generate_random_error = \
            jax.vmap(error_model.generate_random_error)

        vertex_errors = batched_generate_random_error(vertex_keys)

        compute_vertex_syndrome_chi_z_jit = \
            jax.jit(self.compute_vertex_syndrome_chi_z)
        vertex_result = \
            jax.vmap(compute_vertex_syndrome_chi_z_jit)(vertex_errors)
        return vertex_result
    
    def compute_plaquette_syndrome_chi_x(
        self,
        error: ArrayLike
    ) -> Array:
        """Computes the plaquette vector syndrome and the corresponding chi_x
        associated with an error. 
        
        Args:
            error: array of errors on each subsystems.

        Returns:
            An array where the first num_vertex elements are the syndromes
            and the last one is the chi_x.
        """

        syndrome = self.get_plaquette_syndrome(error)
        candidate_error = self.get_plaquette_candidate_error(syndrome)
        error_diff = error - candidate_error 
        res_logical_com_diff = \
            jnp.mod(jnp.dot(error_diff, self.logical_x.T), self.d)
        chi_x = jnp.int16(res_logical_com_diff)
        return jnp.append(syndrome, chi_x)
    
    def compute_batched_plaquette_syndrome_chi_x(
        self,
        num_samples: int, 
        error_model: ErrorModel,
        master_key: Array
    ) -> Array:
        """Computes the plaquette vector syndrome and the corresponding chi_x
        associated with many sampled error. 
        
        Args:
            num_samples: number of sampled errors
            error_model: error model that generates the errors. It needs
                to be compatible with JAX
        
        Returns:
            An array where in each row the first num_plaquette - 1 
            (independent) elements are the syndromes and the last 
            one is the chi_x.
        """
        #master_plaquette_key = jax.random.PRNGKey(687090)
        plaquette_keys = jax.random.split(master_key, num_samples)

        batched_generate_random_error = jax.vmap(error_model.generate_random_error)

        plaquette_errors = batched_generate_random_error(plaquette_keys)

        compute_plaquette_syndrome_chi_x_jit = \
            jax.jit(self.compute_plaquette_syndrome_chi_x)
        plaquette_result = \
            jax.vmap(compute_plaquette_syndrome_chi_x_jit)(plaquette_errors)
        return plaquette_result