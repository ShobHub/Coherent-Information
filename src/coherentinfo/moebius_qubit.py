# Module for the generation of the moebius code for qubits

from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax
from coherentinfo.linalg import (is_prime,
                                 finite_field_gauss_jordan_elimination)
from coherentinfo.errormodel import ErrorModel
from coherentinfo.moebius import MoebiusCode
import scipy 

class MoebiusCodeQubit(MoebiusCode):
    """ Subclass representing the Moebius code for the qubit case.
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
        if d != 2:
            raise ValueError("d must be 2")

        super().__init__(length, width, d)
    
    def get_vertex_syndrome(
        self,
        error: ArrayLike
    ) -> Array:
        """Computes the vertex syndrome (Z-type) associated with 
        error 
        
        Args:
            error: The vertex error array
        
        Returns:
            An array representing the vertex error syndrome
        """

        return jnp.mod(jnp.dot(self.h_z_qubit, error), self.d)
    
    def get_plaquette_syndrome(
        self,
        error: ArrayLike
    ) -> Array:
        """Computes the plaquette syndrome (X-type) associated with 
        error 
        
        Args:
            error: The plaquette error array
        
        Returns:
            An array representing the plaquette error syndrome
        """

        return jnp.mod(jnp.dot(self.h_x_qubit, error), self.d)
    
    def get_vertex_candidate_error(
        self,
        syndrome: ArrayLike
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
        candidate = jnp.mod(jnp.dot(syndrome, vertex_destab_j), self.d)
        return candidate
    
    def get_plaquette_candidate_error(
        self,
        syndrome: ArrayLike
    ) -> Array:
        """ Given a valid plaquette syndrome it returns the candidate 
        error vector, that generates the same syndrome and commutes 
        with the logical X.  

        Args:
            syndrome: syndrome vector
        
        Returns:
            Candidate Z-type error that gives the syndrome and commutes
            with the logical X. 
        """
        
        syndrome = jnp.mod(syndrome, self.d)
        plaquette_destab_j = jnp.asarray(self.plaquette_destab_qubit)
        candidate = jnp.mod(jnp.dot(syndrome, plaquette_destab_j), self.d)
        return candidate
    
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
        # master_vertex_key = jax.random.PRNGKey(48090)
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
        # master_plaquette_key = jax.random.PRNGKey(key)
        plaquette_keys = jax.random.split(master_key, num_samples)

        batched_generate_random_error = jax.vmap(error_model.generate_random_error)

        plaquette_errors = batched_generate_random_error(plaquette_keys)

        compute_plaquette_syndrome_chi_x_jit = \
            jax.jit(self.compute_plaquette_syndrome_chi_x)
        plaquette_result = \
            jax.vmap(compute_plaquette_syndrome_chi_x_jit)(plaquette_errors)
        return plaquette_result