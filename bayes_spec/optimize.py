"""
optimize.py - Fit spectra with MCMC and determine optimal number of
spectral components.

Copyright(C) 2023-2024 by
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
Trey Wenger - September 2023
Trey Wenger - June 2024 - Updates to mimic "caribou" framework
Trey Wenger - July 2024 - Add SMC sampling
"""

from typing import Type

import numpy as np

from bayes_spec.base_model import BaseModel
from bayes_spec.spec_data import SpecData


class Optimize:
    """
    Optimize class definition
    """

    def __init__(
        self,
        model_type: Type[BaseModel],
        data: dict[str, SpecData],
        max_n_clouds: int = 5,
        baseline_degree: int = 0,
        seed: int = 1234,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize a new Optimize instance.

        Inputs:
            model_type :: child of BaseModel
                Model definition
            data :: dictionary
                Spectral data sets, where the "key" defines the name of the
                dataset, and the value is a SpecData instance.
            max_n_clouds :: integer
                Maximum number of cloud components
            baseline_degree :: integer
                Degree of polynomial baseline
            seed :: integer
                Random seed
            verbose :: boolean
                Print extra info
            **kwargs :: additional arguments passed to model

        Returns: optimize_model
            optimize_model :: Optimize
                New Optimize instance
        """
        self.model_type = model_type
        self.max_n_clouds = max_n_clouds
        self.verbose = verbose
        self.n_clouds = [i for i in range(1, self.max_n_clouds + 1)]
        self.seed = seed
        self.data = data

        # Initialize models
        self.models: dict[int, Type[BaseModel]] = {}
        for n_cloud in self.n_clouds:
            self.models[n_cloud] = model_type(
                self.data,
                n_cloud,
                baseline_degree,
                seed=seed,
                verbose=self.verbose,
                **kwargs,
            )
        self.best_model = None

    def add_priors(self, *args, **kwargs):
        """
        Add priors to the models.

        Inputs:
            See model_type.add_priors

        Returns: Nothing
        """
        for n_cloud in self.n_clouds:
            self.models[n_cloud].add_priors(*args, **kwargs)

    def add_likelihood(self, *args, **kwargs):
        """
        Add likelihood to the models.

        Inputs:
            See model_type.add_likelihood

        Returns: Nothing
        """
        for n_cloud in self.n_clouds:
            self.models[n_cloud].add_likelihood(*args, **kwargs)

    def fit_all(self, **kwargs):
        """
        Fit posterior of all models using variational inference.

        Inputs:
            see model.fit

        Returns: Nothing
        """
        if self.verbose:
            print(f"Null hypothesis BIC = {self.models[1].null_bic():.3e}")

        for n_cloud in self.n_clouds:
            if self.verbose:
                print(f"Approximating n_cloud = {n_cloud} posterior...")
            self.models[n_cloud].fit(**kwargs)
            self.models[n_cloud].solve()
            if self.verbose:
                for solution in self.models[n_cloud].solutions:
                    print(
                        f"n_cloud = {n_cloud} "
                        + f"solution = {solution} "
                        + f"BIC = {self.models[n_cloud].bic(solution=solution):.3e}"
                    )
                print()

    def sample_all(self, **kwargs):
        """
        Sample posterior of all models using MCMC.

        Inputs:
            see model.sample

        Returns: Nothing
        """
        if self.verbose:
            print(f"Null hypothesis BIC = {self.models[1].null_bic():.3e}")

        for n_cloud in self.n_clouds:
            if self.verbose:
                print(f"Sampling n_cloud = {n_cloud} posterior...")
            self.models[n_cloud].sample(**kwargs)
            self.models[n_cloud].solve()
            if self.verbose:
                for solution in self.models[n_cloud].solutions:
                    print(
                        f"n_cloud = {n_cloud} "
                        + f"solution = {solution} "
                        + f"BIC = {self.models[n_cloud].bic(solution=solution):.3e}"
                    )
                print()

    def sample_smc_all(self, **kwargs):
        """
        Sample posterior of all models using SMC.

        Inputs:
            see model.sample_smc

        Returns: Nothing
        """
        if self.verbose:
            print(f"Null hypothesis BIC = {self.models[1].null_bic():.3e}")

        for n_cloud in self.n_clouds:
            if self.verbose:
                print(f"Sampling n_cloud = {n_cloud} posterior...")
            self.models[n_cloud].sample_smc(**kwargs)
            self.models[n_cloud].solve()
            if self.verbose:
                for solution in self.models[n_cloud].solutions:
                    print(
                        f"n_cloud = {n_cloud} "
                        + f"solution = {solution} "
                        + f"BIC = {self.models[n_cloud].bic(solution=solution):.3e}"
                    )
                print()

    def optimize(
        self,
        bic_threshold: float = 10.0,
        fit_kwargs={},
        sample_kwargs={},
        smc=False,
        approx=True,
    ):
        """
        Determine the optimal number of clouds by minimizing the BIC
        using MCMC, SMC, or variational inference and then sampling
        the best model using MCMC or SMC. The labeling degeneracy is solved.

        Inputs:
            bic_threshold :: scalar
                Sample the first model that is within min(BIC)+bic_threshold
            fit_kwargs :: dictionary
                Arguments passed to fit()
            sample_kwargs :: dictionary
                Arguments passed to sample()
            smc :: boolean
                If True, use SMC instead of MCMC.
            approx :: boolean
                If True, use VI for first pass, then sample best with MCMC.
                Otherwise, MCMC (slower, better) every model (don't set max_n_clouds too high!).

        Returns: Nothing
        """
        if approx:
            # fit all with VI
            self.fit_all(**fit_kwargs)
        elif smc:
            # sample with SMC
            self.sample_smc_all(**sample_kwargs)
        else:
            # sample with MCMC
            self.sample_all(**sample_kwargs)

        # get best model
        model_bics = np.array(
            [
                (
                    self.models[n_cloud].bic(solution=0)
                    if len(self.models[n_cloud].solutions) > 0
                    else np.inf
                )
                for n_cloud in self.n_clouds
            ]
        )
        best_n_clouds = self.n_clouds[
            np.where(model_bics < (np.nanmin(model_bics) + bic_threshold))[0][0]
        ]
        self.best_model = self.models[best_n_clouds]

        if approx:
            # sample best
            if self.verbose:
                print(f"Sampling best model (n_cloud = {self.best_model.n_clouds})...")
            if smc:
                self.best_model.sample_smc(**sample_kwargs)
            else:
                self.best_model.sample(**sample_kwargs)
            self.best_model.solve()
