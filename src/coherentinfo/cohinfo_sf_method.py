# Contains functions to run the worm using the syndrome first (SF)
#  approach which computes the syndromes and 
# their frequencies and then runs the split worm algorithm on them.

from typing import Tuple, Dict, Callable
from coherentinfo.errormodel import ErrorModelLindbladTwoOddPrime
from coherentinfo.moebius_two_odd_prime import MoebiusCodeTwoOddPrime
from coherentinfo.worm import run_worm
from coherentinfo.dtypes import INT_DTYPE
import os
import jax
# N_CPUS = os.cpu_count()
# N_USED_CPUS = N_CPUS
# jax.config.update('jax_num_cpu_devices', N_USED_CPUS)
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.typing import ArrayLike
import jax.numpy as jnp
from functools import partial


def generate_syndromes(
    gamma_t: ArrayLike,
    compute_syndrome: Callable,
    moebius_setup: Dict,
    num_samples: int,
    master_error_seed: int
) -> Dict:
    """ Runs the split worm algorithm on either the vertex or the plaquettes. 
    In particular, it generates many syndromes and correspondingly many errors
    giving the same syndrome. 
    
    Args:
        gamma_t (ArrayLike): The error parameter gamma_t.
        syndrome_id (str): A string that is either 'vertex' or 'plaquette' that
            defines whether the worm should be run using vertex or plaquette
            stabilizers.
        moebius_setup (Dict): A dictionary that specifies the details of the 
            Moebius code with keys:
                length (int): Length of the Moebius.
                width (int): Widths of the Moebius.
                p (int): Odd prime that specifies the dimension of the qudit
                    as d = 2 * p. 
        worm_setup (Dict): A dictionary that speficies the parameters of 
            the worm simulation. The keys are:
                num_samples (int): Number of sampled syndromes. Note the
                    same syndrome might be sampled multiple times
                num_worms (int): Number of worms, i.e., errors with the same
                    syndrome per syndrome. Note that some worms might fail
                    so not necessarily all of them are used in the end.
                burn_in_steps (int): Minimum number of worm steps after which
                the worm is declared successful.
        keys_setup (Dict): A dictionary that stores the necessary seeds, with
            keys:
                worm_master_seed (int): The seed used to generate the worms.
                error_master_seed (int): The seed used to generate the errors.  

    Returns:
        Dict:    
    """
    length = moebius_setup["length"]
    width = moebius_setup["width"]
    p = moebius_setup["p"]

    
    d = 2 * p
    moebius_code = MoebiusCodeTwoOddPrime(length=length, width=width, d=d)
    em_lindblad = ErrorModelLindbladTwoOddPrime(
        moebius_code.num_edges, d=d, gamma_t=gamma_t
    )

    master_error_key = jax.random.PRNGKey(master_error_seed)

    error_keys = jax.random.split(master_error_key, num=num_samples)

    def generate_error_and_get_syndrome(key):
        error = em_lindblad.generate_random_error(key)
        return compute_syndrome(error)
    
    syndromes = jax.vmap(generate_error_and_get_syndrome)(error_keys)

    return syndromes

def aggregate_syndromes(syndromes: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """ It aggregates results by syndrome and computes their
    counts. Not compatible with jax.jit, despite the use of JAX. 

    Args:
        syndromes: array whose rows represent the samples syndromes 
        
    Returns:
        Observed syndromes and their counts
    """

    # 2. Find unique prefixes and assign a unique index to each original row
    # 'indices' will map each original row to a unique index ID 
    # (0 to num_unique-1)
    unique_syndromes, indices = jnp.unique(
        syndromes, 
        axis=0, 
        return_inverse=True
    )
    
    num_unique = unique_syndromes.shape[0] # number of unique prefixes

    
    # The total count of rows for each unique prefix is simply the bincount 
    # of the indices themselves (weights=None means weights=1).
    total_counts = jnp.bincount(
        indices, 
        length=num_unique
    )
    

    return unique_syndromes, total_counts

def run_worm_moebius_sf(
    gamma_t: ArrayLike,
    syndrome_id: str, 
    moebius_setup: Dict,
    num_samples: int,
    worm_setup: Dict,
    keys_setup: Dict
):
    length = moebius_setup["length"]
    width = moebius_setup["width"]
    p = moebius_setup["p"]
    num_worms = worm_setup["num_worms"]
    burn_in_steps = worm_setup["burn_in_steps"]
    max_worm_steps = worm_setup["max_worm_steps"]
    master_error_seed = keys_setup["master_error_seed"]
    master_worm_seed = keys_setup["master_worm_seed"]

    moebius_code = MoebiusCodeTwoOddPrime(
    length=length, width=width, d=2 * p)
    em_lindblad = ErrorModelLindbladTwoOddPrime(
        moebius_code.num_edges, d=2 * p, gamma_t=gamma_t
    )

    if syndrome_id == "plaquette":
        num_stabs, h_error_mod_p, h_mod_p = (
            moebius_code.num_plaquette_checks,
            moebius_code.h_z_mod_p,
            moebius_code.h_x_mod_p,
        )
        compute_syndrome = moebius_code.get_plaquette_syndrome
        compute_chi = moebius_code.compute_plaquette_syndrome_chi_x
        compute_candidate_error = moebius_code.get_plaquette_candidate_error
    elif syndrome_id == 'vertex':
        num_stabs, h_error_mod_p, h_mod_p = (
            moebius_code.num_vertex_checks,
            moebius_code.h_x_mod_p,
            moebius_code.h_z_mod_p,
        )
        compute_syndrome = moebius_code.get_vertex_syndrome
        compute_chi = moebius_code.compute_vertex_syndrome_chi_z
        compute_candidate_error = moebius_code.get_vertex_candidate_error
    else:
        raise ValueError("The syndrome id must be either plaquette or vertex")

    syndromes = generate_syndromes(
        gamma_t=gamma_t,
        compute_syndrome=compute_syndrome,
        moebius_setup=moebius_setup,
        num_samples=num_samples,
        master_error_seed=master_error_seed
    )

    unique_syndromes, syndrome_counts = aggregate_syndromes(syndromes) 

    num_unique_syndromes = unique_syndromes.shape[0]

    def generate_initial_candidate_error(
        syndrome: ArrayLike
    ) -> ArrayLike:
        candidate_error_mod_d = compute_candidate_error(syndrome)
        candidate_error_mod_2 = jnp.mod(candidate_error_mod_d, 2)
        candidate_error_mod_p = jnp.mod(candidate_error_mod_d, p)
        candidate_error = jnp.vstack((candidate_error_mod_2, candidate_error_mod_p))
        return candidate_error
    
    candidate_errors = jax.vmap(generate_initial_candidate_error)(unique_syndromes)

    devices = jax.devices()  # Assuming this returns 16 devices
    mesh = Mesh(devices, ('batch',))

    master_worm_key = jax.random.PRNGKey(master_worm_seed)

    worm_keys = jax.random.split(
        master_worm_key, num=num_unique_syndromes * num_worms
    ).reshape(num_unique_syndromes, num_worms, 2)

    # sharding_for_keys = NamedSharding(mesh,  PartitionSpec('batch', None))
    # worm_keys_sharded = jax.device_put(worm_keys, sharding_for_keys)

    run_worm_partial = partial(
        run_worm,
        h_error_mod_p=h_error_mod_p,
        h_mod_p=h_mod_p,
        error_model=em_lindblad,
        compute_full_chi=compute_chi,
        num_stabs=num_stabs,
        burn_in_steps=burn_in_steps,
        max_worm_steps=max_worm_steps
    )

    # First over keys
    run_worm_vmap = jax.vmap(run_worm_partial, in_axes=(None, 0))
    # Then over initial errors
    run_worm_vmap = jax.vmap(run_worm_vmap, in_axes=(0, 0))
    run_worm_jit = jax.jit(run_worm_vmap)
    new_worm_state = run_worm_jit(candidate_errors, worm_keys)

    return new_worm_state, unique_syndromes, syndrome_counts

    



    



def worm_sf_conditional_entropy(
    gamma_t: ArrayLike,
    syndrome_id: str, 
    moebius_setup: Dict,
    sampling_setup: Dict,
    worm_setup: Dict, 
)-> ArrayLike:
    pass
    

def worm_sf_coherent_information(
    gamma_t: ArrayLike,
    moebius_setup: Dict,
    worm_setup: Dict,
    plaquette_keys_setup: Dict,
    vertex_keys_setup
)-> Tuple:
    pass