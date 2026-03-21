# Contains functions to run the worm and compute via direct sampling (DS) errors and directly 
# computing the conditional entropies. This approach will be substituted by the one in 
# which we first sample the errors, obtain the syndromes with the corresponding
# probabilities and then run the worm on each sampled syndrome. The two approaches
# must agree, but the latter should be faster. 

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


def run_worm_moebius_ds(
    gamma_t: ArrayLike,
    syndrome_id: str, 
    moebius_setup: Dict,
    worm_setup: Dict,
    keys_setup: Dict,
    shard: bool = True
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
    num_samples = worm_setup["num_samples"]
    num_worms = worm_setup["num_worms"]
    burn_in_steps = worm_setup["burn_in_steps"]
    max_worm_steps = worm_setup["max_worm_steps"]
    worm_master_seed = keys_setup["worm_master_seed"]
    error_master_seed = keys_setup["error_master_seed"]

    
    d = 2 * p
    moebius_code = MoebiusCodeTwoOddPrime(length=length, width=width, d=d)
    em_lindblad = ErrorModelLindbladTwoOddPrime(
        moebius_code.num_edges, d=d, gamma_t=gamma_t
    )

    if syndrome_id == "plaquette":
        num_stabs, h_error_mod_p, h_mod_p = (
            moebius_code.num_plaquette_checks,
            moebius_code.h_z_mod_p,
            moebius_code.h_x_mod_p,
        )
        compute_chi = moebius_code.compute_plaquette_syndrome_chi_x
    elif syndrome_id == 'vertex':
        num_stabs, h_error_mod_p, h_mod_p = (
            moebius_code.num_vertex_checks,
            moebius_code.h_x_mod_p,
            moebius_code.h_z_mod_p,
        )
        compute_chi = moebius_code.compute_vertex_syndrome_chi_z
    else:
        raise ValueError("The syndrome id must be either plaquette or vertex")

    master_worm_key = jax.random.PRNGKey(worm_master_seed)

    worm_keys = jax.random.split(
        master_worm_key, num=num_samples * num_worms).reshape(num_samples, num_worms, 2)

    error_master_key = jax.random.PRNGKey(error_master_seed)
    error_keys = jax.random.split(error_master_key, num_samples)


    def generate_initial_worm_errors(
        key: ArrayLike,
        error_model: ErrorModelLindbladTwoOddPrime
    ) -> ArrayLike:
        initial_error = error_model.generate_random_error(key)
        initial_error_mod_2 = jnp.mod(initial_error, 2)
        initial_error_mod_p = jnp.mod(initial_error, p)
        initial_worm_error = jnp.vstack((initial_error_mod_2, initial_error_mod_p))
        return initial_worm_error


    initial_worm_errors = jax.vmap(
        generate_initial_worm_errors, in_axes=(0, None))(error_keys, em_lindblad)
    
    # Sharding the arrays
    # devices = jax.devices()  # Assuming this returns 16 devices
    # mesh = Mesh(devices, ('batch',))

    # sharding_for_keys = NamedSharding(mesh,  PartitionSpec('batch', None))
    # worm_keys_sharded = jax.device_put(worm_keys, sharding_for_keys)

    # 2. Define sharding: Split the 0th axis across 'batch', leave others whole
    # sharding_for_error = NamedSharding(mesh,  PartitionSpec('batch', None, None))
    # initial_worm_errors_sharded = jax.device_put(
    #     initial_worm_errors, sharding_for_error)
    
    # initial_worm_state = {}
    # worm_error = jnp.vstack(
    #     (initial_error_mod_2, initial_error_mod_p))
    # initial_worm_state["worm_success"] = False
    # # The plan is to use this as a marker when we hit a boundary
    # # which is important only for vertex checks for the Moebius Code.
    # # Essentially this is turn to True whenever we hit a boundary edge and at 
    # # that point the head is set again to tail and the success condition becomes
    # # finding another (or the same) boundary.
    # initial_worm_state["boundary"] = False
    # initial_worm_state["accepted_moves"] = 0
    # initial_worm_state["attempted_moves"] = 0

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
    if shard:
        devices = jax.devices()  # Assuming this returns 16 devices
        mesh = Mesh(devices, ('batch',))

        # sharding_for_keys = NamedSharding(mesh,  PartitionSpec('batch', None))
        # worm_keys_sharded = jax.device_put(worm_keys, sharding_for_keys)

        # 2. Define sharding: Split the 0th axis across 'batch', leave others whole
        sharding_for_error = NamedSharding(mesh,  PartitionSpec('batch', None, None))
        initial_worm_errors_sharded = jax.device_put(
            initial_worm_errors, sharding_for_error)
        new_worm_state = run_worm_jit(initial_worm_errors_sharded, worm_keys)
    else:
        new_worm_state = run_worm_jit(initial_worm_errors, worm_keys)


    return new_worm_state

def worm_ds_conditional_entropy(
    gamma_t: ArrayLike,
    syndrome_id: str, 
    moebius_setup: Dict,
    worm_setup: Dict, 
    keys_setup: Dict,
    shard: bool = True
)-> ArrayLike:
    
    new_worm_state = run_worm_moebius_ds(
        gamma_t=gamma_t,
        syndrome_id=syndrome_id,
        moebius_setup=moebius_setup,
        worm_setup=worm_setup,
        keys_setup=keys_setup,
        shard=shard
    )

    def get_binary_entropy(chi_vec, success_vec):
        # Number of successful worms
        num_success = jnp.sum(success_vec)
        # Sets simply to zero failed attempts so that they are not counted
        chi_vec_marked = jnp.where(success_vec, chi_vec, 0)
        p1 = jnp.sum(chi_vec_marked) / num_success
        p0 = 1 - p1
        # This is a trick needed for JAX not to evaluate log(0)
        # and give nan
        eps = jnp.finfo(jnp.float32).tiny
        p0_safe = jnp.clip(p0, eps, 1.0)
        p1_safe = jnp.clip(p1, eps, 1.0)
        binary_entropy = -jax.scipy.special.xlogy(p0, p0_safe) / jnp.log(2)
        binary_entropy += -jax.scipy.special.xlogy(p1, p1_safe) / jnp.log(2)
        return binary_entropy


    binary_entropies = jax.vmap(get_binary_entropy)(
        new_worm_state["chi"], new_worm_state["worm_success"]
    )
    cond_entropy = jnp.mean(binary_entropies)
    return cond_entropy

def worm_ds_coherent_information(
    gamma_t: ArrayLike,
    moebius_setup: Dict,
    worm_setup: Dict,
    plaquette_keys_setup: Dict,
    vertex_keys_setup,
    shard: bool = True
)-> Tuple:
    
    plaquette_conditional_entropy = worm_ds_conditional_entropy(
        gamma_t=gamma_t,
        syndrome_id="plaquette",
        moebius_setup=moebius_setup,
        worm_setup=worm_setup,
        keys_setup=plaquette_keys_setup,
        shard=shard
    )

    vertex_conditional_entropy = worm_ds_conditional_entropy(
        gamma_t=gamma_t,
        syndrome_id="vertex",
        moebius_setup=moebius_setup,
        worm_setup=worm_setup,
        keys_setup=vertex_keys_setup,
        shard=shard
    )

    coherent_info = (1.0 - plaquette_conditional_entropy - 
                     vertex_conditional_entropy)
    
    return (
        coherent_info, 
        plaquette_conditional_entropy, 
        vertex_conditional_entropy
    )
