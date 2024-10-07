"""
optimize.py - Fit spectra with MCMC and determine optimal number of
spectral components.

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
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

    @property
    def null_bic(self) -> float:
        """Evaluate the Bayesian Information Criterion for the null hypothesis (baseline only, no clouds)

        :return: Null hypothesis BIC
        :rtype: float
        """
        return self.models[1].null_bic()

    @property
    def bics(self) -> dict[int, float]:
        """Return the Bayesian Information Criteria for the best solution of each model.

        :return: BIC for each model, indexed by the number of clouds
        :rtype: dict[int, float]
        """
        model_bics = {0: self.null_bic}
        for n_cloud, model in self.models.items():
            best_solution_bic = np.inf
            # VI does not have solution, only single "chain"
            if model.solutions is None or len(model.solutions) == 0:
                best_solution_bic = model.bic(chain=[0])
            else:
                for solution in model.solutions:
                    solution_bic = model.bic(solution=solution)
                    if solution_bic < best_solution_bic:
                        best_solution_bic = solution_bic
            model_bics[n_cloud] = best_solution_bic
        return model_bics

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
            if self.verbose:
                bic = self.models[n_cloud].bic(chain=[0])
                print(f"n_cloud = {n_cloud} BIC = {bic:.3e}")
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
        bics = self.bics
        n_clouds = list(bics.keys())
        model_bics = list(bics.values())
        best_idx = np.where(model_bics < (np.nanmin(model_bics) + bic_threshold))[0]
        if len(best_idx) == 0:
            if self.verbose:  # pragma: no cover
                print("No good models found!")
        else:
            best_n_clouds = n_clouds[best_idx[0]]
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
