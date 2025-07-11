"""
nuts.py - customize pymc's NUTS initialization

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Optional

import pymc as pm
from pymc.variational.callbacks import CheckParametersConvergence
from pymc.step_methods.hmc import quadpotential


def init_nuts(
    model: pm.Model,
    init: str = "advi+adapt_diag",
    n_init: int = 100_000,
    chains: int = 4,
    rel_tolerance: float = 0.001,
    abs_tolerance: float = 0.001,
    learning_rate: float = 1e-3,
    obj_n_mc: int = 5,
    start: Optional[dict] = None,
    nuts_kwargs: dict = None,
    seed: int = 1234,
    verbose: bool = False,
) -> tuple[list, pm.NUTS]:
    """Custom NUTS initialization.

    :param model: Model to initialize
    :type model: pm.Model
    :param init: Initialization strategy, defaults to "advi+adapt_diag"
    :type init: str, optional
    :param n_init: Number of initialization iterations, defaults to 100_000
    :type n_init: int, optional
    :param chains: Number of independent Markov chains, defaults to 4
    :type chains: int, optional
    :param rel_tolerance: VI relative convergence threshold, defaults to 0.001
    :type rel_tolerance: float, optional
    :param abs_tolerance: VI absolute convergence threshold, defaults to 0.001
    :type abs_tolerance: float, optional
    :param learning_rate: VI learning rate, defaults to 1e-3
    :type learning_rate: float, optional
    :param obj_n_mc: Number of Monte Carlo gradient samples, defaults to 5
    :type obj_n_mc: int, optional
    :param start: Starting point, defaults to None
    :type start: Optional[dict], optional
    :param nuts_kwargs: Additional keyword arguments passed to :class:`pm.NUTS`, defaults to None
    :type nuts_kwargs: dict, optional
    :param seed: Random seed, defaults to 1234
    :type seed: int, optional
    :param verbose: Verbose output, defaults to False
    :type verbose: bool, optional
    :return: Initial point and step method
    :rtype: tuple[list, pm.NUTS]
    """
    initial_points = None
    step = None
    allowed_init = ["advi+adapt_diag"]
    if init not in allowed_init:
        return initial_points, step

    with model:
        if verbose:
            print(f"Initializing NUTS using custom {init} strategy")
        callbacks = [
            CheckParametersConvergence(tolerance=rel_tolerance, diff="relative"),
            CheckParametersConvergence(tolerance=abs_tolerance, diff="absolute"),
        ]

        if init == "advi+adapt_diag":
            approx = pm.fit(
                random_seed=seed,
                n=n_init,
                method="advi",
                model=model,
                start=start,
                callbacks=callbacks,
                progressbar=verbose,
                obj_n_mc=obj_n_mc,
                obj_optimizer=pm.adagrad_window(learning_rate=learning_rate, n_win=10),
            )
            approx_sample = approx.sample(
                draws=chains,
                random_seed=seed,
                return_inferencedata=False,
            )
            initial_points = [approx_sample[i] for i in range(chains)]
            std_apoint = approx.std.eval()
            cov = std_apoint**2
            mean = approx.mean.get_value()
            n = len(cov)
            potential = quadpotential.QuadPotentialDiagAdapt(n, mean, cov, 50)

        step = pm.NUTS(potential=potential, **nuts_kwargs)

        # Filter deterministics from initial_points
        value_var_names = [var.name for var in model.value_vars]
        initial_points = [
            {k: v for k, v in initial_point.items() if k in value_var_names}
            for initial_point in initial_points
        ]

    return initial_points, step
