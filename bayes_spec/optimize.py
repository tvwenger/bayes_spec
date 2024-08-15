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
"""

from typing import Type

import numpy as np

from bayes_spec.base_model import BaseModel
from bayes_spec import SpecData


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
        """Initialize a new Optimize instance.

        :param model_type: Model to optimize
        :type model_type: Type[BaseModel]
        :param data: Spectral data sets, where the "key" defines the name of the dataset
        :type data: dict[str, SpecData]
        :param max_n_clouds: Maximum number of clouds to fit, defaults to 5
        :type max_n_clouds: int, optional
        :param baseline_degree: Polynomial baseline degree, defaults to 0
        :type baseline_degree: int, optional
        :param seed: Random seed, defaults to 1234
        :type seed: int, optional
        :param verbose: Verbose output, defaults to False
        :type verbose: bool, optional
        :param `**kwargs`: Additional keyword arguments passed to :param:`model` initialization
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
        """Add priors to the models

        :param `*args`: Arguments passed to :func:`model.add_priors`
        :param `**kwargs`: Keyword arguments passed to :func:`model.add_priors`
        """
        for n_cloud in self.n_clouds:
            self.models[n_cloud].add_priors(*args, **kwargs)

    def add_likelihood(self, *args, **kwargs):
        """Add likelihood to the models

        :param `*args`: Arguments passed to :func:`model.add_likelihood`
        :param `**kwargs`: Keyword arguments passed to :func:`model.add_likelihood`
        """
        for n_cloud in self.n_clouds:
            self.models[n_cloud].add_likelihood(*args, **kwargs)

    def fit_all(self, **kwargs):
        """Fit all models using variational inference.

        :param `**kwargs`: Keyword arguments passed to :func:`model.fit`
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
        """Sample posterior distribution of all models using MCMC.

        :param `**kwargs`: Keyword arguments passed to :func:`model.sample`
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
        """Sample posterior distribution of all models using sequential Monte Carlo.

        :param `**kwargs`: Keyword arguments passed to :func:`model.sample_smc`
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
        fit_kwargs: dict = {},
        sample_kwargs: dict = {},
        smc: bool = False,
        approx: bool = True,
    ):
        """Determine optimal number of clouds by minimizing the Bayesian Information Criterion
        using MCMC, Sequntial Monte Carlo, or Variational Inference. Then, sample the best model
        using MCMC or SMC, and solve the labeling degeneracy.

        :param bic_threshold: The `best_model` is the first with BIC within `min(BIC)+bic_threshold`, defaults to 10.0
        :type bic_threshold: float, optional
        :param fit_kwargs: Keyword arguments passed to :func:`fit`, defaults to {}
        :type fit_kwargs: dict, optional
        :param sample_kwargs: Keyword arguments passed to :func:`sample`, defaults to {}
        :type sample_kwargs: dict, optional
        :param smc: If True, sample all models using SMC, defaults to False
        :type smc: bool, optional
        :param approx: If True, approximate all models using VI, defaults to True
        :type approx: bool, optional
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
                (self.models[n_cloud].bic(solution=0) if len(self.models[n_cloud].solutions) > 0 else np.inf)
                for n_cloud in self.n_clouds
            ]
        )
        best_n_clouds = self.n_clouds[np.where(model_bics < (np.nanmin(model_bics) + bic_threshold))[0][0]]
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
