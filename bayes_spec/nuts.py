"""
nuts.py - customize pymc's NUTS initialization

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Changelog:
Trey Wenger - March 2024
"""

import pymc as pm
from pymc.variational.callbacks import CheckParametersConvergence
from pymc.step_methods.hmc import quadpotential


def init_nuts(
    model,
    init: str = "advi+adapt_diag",
    n_init: int = 100_000,
    chains: int = 4,
    rel_tolerance: float = 0.001,
    abs_tolerance: float = 0.001,
    learning_rate: float = 1e-3,
    nuts_kwargs: dict = None,
    seed: int = 1234,
    verbose: bool = False,
):
    """
    Expands pymc's init_nuts to customize ADVI initialization.

    Inputs:
        model :: pm.Model
            Model
        init :: string
            Initialization strategy
        n_init :: integer
            Number of initialization iterations
        chains :: integer
            Number of chains
        rel_tolerance :: scalar
            Relative parameter tolerance for ADVI convergence
        abs_tolerance :: scalar
            Absolute parameter tolerance for ADVI convergence
        learning_rate :: scalar
            adagrad_widnow learning rate for ADVI
        nuts_kwargs :: dictionary
            Keyword arguments passed to pm.NUTS
            (target_accept)
        target_accept :: scalar
            Target acceptance rate
        seed :: integer
            Random seed

    Returns:
        initial_points :: list
            Starting points for each chain
        step :: pm.step_methods.NUTS
            Instantiated and initialized NUTS sampler object
    """
    initial_points = None
    step = None
    allowed_init = ["advi", "advi_map", "advi+adapt_diag", "advi+adapt_full"]
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
                callbacks=callbacks,
                progressbar=verbose,
                obj_optimizer=pm.adagrad_window(learning_rate=learning_rate),
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

        elif init == "advi+adapt_full":
            approx = pm.fit(
                random_seed=seed,
                n=n_init,
                method="advi",
                model=model,
                callbacks=callbacks,
                progressbar=verbose,
                obj_optimizer=pm.adagrad_window(learning_rate=learning_rate),
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
            potential = quadpotential.QuadPotentialFullAdapt(n, mean, cov, 10)

        elif init == "advi":
            approx = pm.fit(
                random_seed=seed,
                n=n_init,
                method="advi",
                model=model,
                callbacks=callbacks,
                progressbar=verbose,
                obj_optimizer=pm.adagrad_window(learning_rate=learning_rate),
            )
            approx_sample = approx.sample(
                draws=chains,
                random_seed=seed,
                return_inferencedata=False,
            )
            initial_points = [approx_sample[i] for i in range(chains)]
            cov = approx.std.eval() ** 2
            potential = quadpotential.QuadPotentialDiag(cov)

        elif init == "advi_map":
            start = pm.find_MAP(include_transformed=True, seed=seed)
            approx = pm.MeanField(model=model, start=start)
            pm.fit(
                random_seed=seed,
                n=n_init,
                method=pm.KLqp(approx),
                callbacks=callbacks,
                progressbar=verbose,
                obj_optimizer=pm.adagrad_window(learning_rate=learning_rate),
            )
            approx_sample = approx.sample(
                draws=chains,
                random_seed=seed,
                return_inferencedata=False,
            )
            initial_points = [approx_sample[i] for i in range(chains)]
            cov = approx.std.eval() ** 2
            potential = quadpotential.QuadPotentialDiag(cov)

        step = pm.NUTS(potential=potential, **nuts_kwargs)

        # Filter deterministics from initial_points
        value_var_names = [var.name for var in model.value_vars]
        initial_points = [
            {k: v for k, v in initial_point.items() if k in value_var_names}
            for initial_point in initial_points
        ]

    return initial_points, step
