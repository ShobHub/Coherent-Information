import numpy as np
from coherentinfo_numba.core import run_worm_simulation, save_json, plot_logical_error_rate


def main():
    num_gamma = 20
    gamma_min = 0.05
    gamma_max = 0.7
    gamma_list = np.linspace(gamma_min, gamma_max, num_gamma).tolist()

    moebius_setup = {"length": 3, "width": 3, "p": 3}
    print(f"Moebius setup = {moebius_setup}")

    worm_setup = {}
    worm_setup["num_samples"] = 64 * 8
    worm_setup["num_worms"] = 50
    burn_in_const = 500
    max_worm_const = 300
    scaling = "quadratic"
    worm_setup["burn_in_const"] = burn_in_const
    worm_setup["max_worm_const"] = max_worm_const
    if scaling == "quadratic":
        worm_setup["burn_in_steps"] = moebius_setup["length"] ** 2 * burn_in_const
        worm_setup["max_worm_steps"] = worm_setup["burn_in_steps"] + moebius_setup["length"] ** 2 * max_worm_const
    elif scaling == "linear":
        worm_setup["burn_in_steps"] = moebius_setup["length"] * burn_in_const
        worm_setup["max_worm_steps"] = worm_setup["burn_in_steps"] + moebius_setup["length"] * max_worm_const
    else:
        raise ValueError("scaling must be 'quadratic' or 'linear'")

    result = run_worm_simulation(gamma_list, moebius_setup, worm_setup, compute_coherent_information=False)
    filename = (
        f"logical_error_rate_moebius_length_{moebius_setup['length']}"
        f"_width_{moebius_setup['width']}_p_{moebius_setup['p']}"
        f"_{scaling}_gamma_min_{gamma_min}_gamma_max_{gamma_max}_num_gamma_{num_gamma}.json"
    )
    save_json(result, filename)
    plot_logical_error_rate(result, save=True)


if __name__ == '__main__':
    main()
