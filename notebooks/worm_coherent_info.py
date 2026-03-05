from jax.typing import ArrayLike
import jax.numpy as jnp
from functools import partial
import numpy as np
from typing import Dict
from coherentinfo.moebius_two_odd_prime import MoebiusCodeTwoOddPrime
from coherentinfo.errormodel import ErrorModelLindbladTwoOddPrime
from coherentinfo.worm import (
    worm_coherent_information
)
import time
import json
import os
import jax
N_CPUS = os.cpu_count()
N_USED_CPUS = N_CPUS
# print("Number of CPUs available: {}".format(n_cpus))
# print("Number of used CPUs: {}".format(n_used_cpus))
jax.config.update('jax_num_cpu_devices', N_USED_CPUS)
# # jax.config.update("jax_log_compiles", True)
# # Devices assumed by JAX
# print(jax.devices())
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

tex_rc_params = {
    'backend': 'ps',
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'serif',
}

def plot_coherent_information(
    result: Dict,
    save: bool=False
):
    length = result["moebius_setup"]["length"]
    width = result["moebius_setup"]["width"]
    p = result["moebius_setup"]["p"]
    gamma_array = result["gamma_t"]
    coherent_info_array = result["coherent_information"]
    with plt.rc_context(tex_rc_params):
        _, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.scatter(gamma_array, coherent_info_array, color="midnightblue")
        ax.set_xlabel("$\\gamma t$")
        x_ticks = [0.1 + 0.1 * x for x in range(6)]
        ax.set_xticks(x_ticks)
        x_ticks_labels = ['$0.1$', '$0.2$', '$0.3$', '$0.4$', '$0.5$', '$0.6$']
        ax.set_xticklabels(x_ticks_labels)
        y_ticks = [-1.0 + 0.2 * y for y in range(11)]
        ax.set_yticks(y_ticks)
        y_ticks_labels = ['$-1.0$', '$-0.8$', '$-0.6$', '$-0.4$', '$-0.2$', '$0.0$',
                          '$0.2$', '$0.4$', '$0.6$', '$0.8$', '$1.0$']
        ax.set_yticklabels(y_ticks_labels)
        ax.set_ylabel("$I_{\\mathrm{coh}}$")
        ax.grid()
        ax.set_title(f"$L= {length}, \\, w = {width}, \\, p={p}$")
        filename = ("coherent_information_moebius_length_" +
            str(length) +
            "_width_" +
            str(width) +
            "_p_" +
            str(p) + ".json"
            )
        if save:
            plt.savefig(filename + ".svg", bbox_inches='tight', 
                        transparent=True, pad_inches=0)
            plt.savefig(filename + ".pdf", bbox_inches='tight', 
                        transparent=True, pad_inches=0)
            plt.savefig(filename + ".png", bbox_inches='tight', 
                        transparent=True, pad_inches=0)
        plt.show()

def run_worm_simulation(
    moebius_setup: Dict,
    worm_setup: Dict
):
    num_gamma = 25
    gamma_min = 0.08
    gamma_max = 0.6
    gamma_t_list = (np.linspace(gamma_min, gamma_max, num_gamma)).tolist()

    result = {}
    result["gamma_t"] = gamma_t_list

    result["worm_setup"] = worm_setup
    result["moebius_setup"] = moebius_setup

    max_integer = 1_000_000
    result["plaquette_worm_master_seed"] = (
        np.random.randint(0, max_integer, num_gamma)).tolist()
    result["plaquette_error_master_seed"] = (
        np.random.randint(0, max_integer, num_gamma)).tolist()

    result["vertex_worm_master_seed"] = (
        np.random.randint(0, max_integer, num_gamma)).tolist()
    result["vertex_error_master_seed"] = (
        np.random.randint(0, max_integer, num_gamma)).tolist()

    # This is to save on which machine the results are obtained
    with open('/sys/devices/virtual/dmi/id/product_name') as f:
        result["machine_id"] = f.read()

    with open('/sys/devices/virtual/dmi/id/sys_vendor') as f:
        result["vendor"] = f.read()

    result["number_of_available_cpus"] = N_CPUS
    result["number_of_used_cpus"] = N_USED_CPUS

    result["plaquette_conditional_entropy"] = []
    result["vertex_conditional_entropy"] = []
    result["coherent_information"] = []

    start = time.time()
    for index in range(num_gamma):
        print(f"Gamma: {result["gamma_t"][index]}")
        plaquette_keys_setup = {}
        plaquette_keys_setup["worm_master_seed"] = \
            result["plaquette_worm_master_seed"][index]
        plaquette_keys_setup["error_master_seed"] = \
            result["plaquette_error_master_seed"][index]

        vertex_keys_setup = {}
        vertex_keys_setup["worm_master_seed"] = \
            result["plaquette_worm_master_seed"][index]
        vertex_keys_setup["error_master_seed"] = \
            result["plaquette_error_master_seed"][index]

        coherent_information, plaquette_ce, vertex_ce = \
            worm_coherent_information(
                gamma_t=result["gamma_t"][index],
                moebius_setup=moebius_setup,
                worm_setup=worm_setup,
                plaquette_keys_setup=plaquette_keys_setup,
                vertex_keys_setup=vertex_keys_setup
            )
        
        print(f"Coherent Information: {coherent_information}")
        elapsed_time = time.time() - start
        print(f"Elapsed time: {elapsed_time}")

        result["plaquette_conditional_entropy"].append(plaquette_ce.tolist())
        result["vertex_conditional_entropy"].append(vertex_ce.tolist())
        result["coherent_information"].append(coherent_information.tolist())
    end = time.time()

    computation_time = end - start

    result["computation_time"] = computation_time
    result["time_unit"] = "sec"

    return result



def main():
    moebius_setup = {"length": 9, "width": 9, "p": 3}

    worm_setup = {}
    worm_setup["num_samples"] = 5 * N_CPUS
    worm_setup["num_worms"] = 200
    worm_setup["burn_in_steps"] = 4000
    worm_setup["max_worm_steps"] = 7000

    result = run_worm_simulation(moebius_setup, worm_setup)

    save = True

    filename = ("coherent_information_moebius_length_" +
                str(moebius_setup["length"]) +
                "_width_" +
                str(moebius_setup["width"]) +
                "_p_" +
                str(moebius_setup["p"]) + ".json"
                )

    if save:
        with open(filename, "w") as fp:
            json.dump(result, fp)


    plot_coherent_information(result, True)


if __name__ == '__main__':
    main()

