"""
optimize.py - Fit spectra with MCMC and determine optimal number of
spectral components.

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Type, Optional, Iterable

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
            # Check that model has been sampled
            if model.trace is not None:
                # VI does not have solution, only single "chain"
                if len(model.trace.posterior.chain) == 1:
                    best_solution_bic = model.bic(chain=[0])
                else:
                    for solution in model.solutions:
                        solution_bic = model.bic(solution=solution)
                        if solution_bic < best_solution_bic:
                            best_solution_bic = solution_bic
            model_bics[n_cloud] = best_solution_bic
        return model_bics

    def _check_stop(self, n_cloud: int, bic_threshold: float = 10.0):
        """Check if any of the stopping criteria are met. Stopping criteria are:
        1. Model did not converge
        2. BIC did not improve by more than bic_threshold over previous model

        :param `n_cloud`: model to check
        :type n_cloud: int
        :param bic_threshold: The `best_model` is the first with BIC within `min(BIC)+bic_threshold`, defaults to 10.0
        :type bic_threshold: float, optional

        :return: True if any stopping criteria are met
        :rtype: bool
        """
        # Check there's a trace
        if self.models[n_cloud].trace is None:  # pragma: no cover
            return True

        # Check if there are no solutions
        if len(self.models[n_cloud].trace.posterior.chain) > 1:
            if (
                self.models[n_cloud].solutions is None
                or len(self.models[n_cloud].solutions) == 0
            ):  # pragma: no cover
                return True

        # Get last non-inf BIC
        bics = self.bics
        last_bic = bics[0]
        for n in range(1, n_cloud):
            model_bic = bics[n]
            if not np.isinf(model_bic):
                last_bic = model_bic

        # Check BIC
        this_bic = bics[n_cloud]
        if this_bic + bic_threshold >= last_bic:
            return True

        # No stopping criteria are met
        return False

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

    def fit_all(
        self, start_spread: Optional[dict[str, Iterable[float]]] = None, **kwargs
    ):
        """Fit all models using variational inference.

        :param `start_spread`: Keys are parameter names and values are range, defaults to None
        :type start_spread: Optional[dict[str, Iterable[float]]], optional
        :param `**kwargs`: Keyword arguments passed to :func:`model.fit`
        """
        if self.verbose:
            print(f"Null hypothesis BIC = {self.models[1].null_bic():.3e}")
        if start_spread is not None and "start" not in kwargs:
            kwargs["start"] = {}

        for n_cloud in self.n_clouds:
            if self.verbose:
                print(f"Approximating n_cloud = {n_cloud} posterior...")

            if start_spread is not None:
                for key, value in start_spread.items():
                    kwargs["start"][key] = np.linspace(value[0], value[1], n_cloud)

            self.models[n_cloud].fit(**kwargs)
            if self.verbose:
                bic = self.models[n_cloud].bic(chain=[0])
                print(f"n_cloud = {n_cloud} BIC = {bic:.3e}")
                print()

    def sample_all(
        self,
        kl_div_threshold: float = 0.1,
        start_spread: Optional[dict[str, Iterable[float]]] = None,
        **kwargs,
    ):
        """Sample posterior distribution of all models using MCMC.

        :param `kl_div_threshold`: GMM convergence threshold
        :type kl_div_threshold: float, optional
        :param `start_spread`: Keys are parameter names and values are range, defaults to None
        :type start_spread: Optional[dict[str, Iterable[float]]], optional
        :param `**kwargs`: Keyword arguments passed to :func:`model.sample`
        """
        if self.verbose:
            print(f"Null hypothesis BIC = {self.models[1].null_bic():.3e}")
        if start_spread is not None and "init_kwargs" not in kwargs:
            kwargs["init_kwargs"] = {}
        if start_spread is not None and "start" not in kwargs["init_kwargs"]:
            kwargs["init_kwargs"]["start"] = {}

        for n_cloud in self.n_clouds:
            if self.verbose:
                print(f"Sampling n_cloud = {n_cloud} posterior...")

            if start_spread is not None:
                for key, value in start_spread.items():
                    kwargs["init_kwargs"]["start"][key] = np.linspace(
                        value[0], value[1], n_cloud
                    )

            self.models[n_cloud].sample(**kwargs)
            self.models[n_cloud].solve(kl_div_threshold=kl_div_threshold)
            if self.verbose:
                for solution in self.models[n_cloud].solutions:
                    print(
                        f"n_cloud = {n_cloud} "
                        + f"solution = {solution} "
                        + f"BIC = {self.models[n_cloud].bic(solution=solution):.3e}"
                    )
                print()

    def sample_smc_all(self, kl_div_threshold: float = 0.1, **kwargs):
        """Sample posterior distribution of all models using sequential Monte Carlo.

        :param `kl_div_threshold`: GMM convergence threshold
        :type kl_div_threshold: float, optional
        :param `**kwargs`: Keyword arguments passed to :func:`model.sample_smc`
        """
        if self.verbose:
            print(f"Null hypothesis BIC = {self.models[1].null_bic():.3e}")

        for n_cloud in self.n_clouds:
            if self.verbose:
                print(f"Sampling n_cloud = {n_cloud} posterior...")

            self.models[n_cloud].sample_smc(**kwargs)
            self.models[n_cloud].solve(kl_div_threshold=kl_div_threshold)
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
        kl_div_threshold: float = 0.1,
        fit_kwargs: Optional[dict] = None,
        sample_kwargs: Optional[dict] = None,
        start_spread: Optional[dict[str, Iterable[float]]] = None,
        smc: bool = False,
        approx: bool = True,
    ):
        """Determine optimal number of clouds by minimizing the Bayesian Information Criterion
        using MCMC, Sequntial Monte Carlo, or Variational Inference. Models are sampled in sequential
        order starting with n_clouds = 1 until the stopping criteria are met twice in succession.
        Then, if approx=True, sample the best model using MCMC or SMC and solve the labeling degeneracy.
        Stopping criteria are:
        1. Model did not converge
        2. Model has multiple solutions (excludeing VI results, which only have one chain)
        3. BIC did not improve by more than bic_threshold over previous model

        :param bic_threshold: The `best_model` is the first with BIC within `min(BIC)+bic_threshold`, defaults to 10.0
        :type bic_threshold: float, optional
        :param `kl_div_threshold`: GMM convergence threshold
        :type kl_div_threshold: float, optional
        :param fit_kwargs: Keyword arguments passed to :func:`fit`, defaults to None
        :type fit_kwargs: Optional[dict], optional
        :param sample_kwargs: Keyword arguments passed to :func:`sample`, defaults to None
        :type sample_kwargs: Optional[dict], optional
        :param `start_spread`: Keys are parameter names and values are range, defaults to None
        :type start_spread: Optional[dict[str, Iterable[float]]], optional
        :param smc: If True, sample all models using SMC, defaults to False
        :type smc: bool, optional
        :param approx: If True, approximate all models using VI, defaults to True
        :type approx: bool, optional
        """
        if self.verbose:
            print(f"Null hypothesis BIC = {self.models[1].null_bic():.3e}")
        if fit_kwargs is None:
            fit_kwargs = {}
        if sample_kwargs is None:
            sample_kwargs = {}

        if start_spread is not None and "start" not in fit_kwargs:
            fit_kwargs["start"] = {}
        if start_spread is not None and "init_kwargs" not in sample_kwargs:
            sample_kwargs["init_kwargs"] = {}
        if start_spread is not None and "start" not in sample_kwargs["init_kwargs"]:
            sample_kwargs["init_kwargs"]["start"] = {}

        stop = False
        for n_cloud in self.n_clouds:
            if approx:
                # fit with VI
                if self.verbose:
                    print(f"Approximating n_cloud = {n_cloud} posterior...")

                if start_spread is not None:
                    for key, value in start_spread.items():
                        fit_kwargs["start"][key] = np.linspace(
                            value[0], value[1], n_cloud
                        )

                self.models[n_cloud].fit(**fit_kwargs)
                if self.verbose:
                    bic = self.models[n_cloud].bic(chain=[0])
                    print(f"n_cloud = {n_cloud} BIC = {bic:.3e}")
                    print()

            else:
                if self.verbose:
                    print(f"Sampling n_cloud = {n_cloud} posterior...")
                if smc:
                    # sample with SMC
                    self.models[n_cloud].sample_smc(**sample_kwargs)
                else:
                    if start_spread is not None:
                        for key, value in start_spread.items():
                            sample_kwargs["init_kwargs"]["start"][key] = np.linspace(
                                value[0], value[1], n_cloud
                            )

                    # sample with MCMC
                    self.models[n_cloud].sample(**sample_kwargs)
                self.models[n_cloud].solve(kl_div_threshold=kl_div_threshold)
                if self.verbose:
                    for solution in self.models[n_cloud].solutions:
                        print(
                            f"n_cloud = {n_cloud} "
                            + f"solution = {solution} "
                            + f"BIC = {self.models[n_cloud].bic(solution=solution):.3e}"
                        )
                    print()

            # check stopping criteria
            if self._check_stop(n_cloud, bic_threshold=bic_threshold):
                if self.verbose:
                    print("Stopping criteria met.")
                # stop if we met criteria twice
                if stop:
                    if self.verbose:
                        print("Stopping early.")
                    break
                stop = True
            else:
                stop = False

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
                    print(
                        f"Sampling best model (n_cloud = {self.best_model.n_clouds})..."
                    )
                if smc:
                    self.best_model.sample_smc(**sample_kwargs)
                else:
                    if start_spread is not None:
                        for key, value in start_spread.items():
                            sample_kwargs["init_kwargs"]["start"][key] = np.linspace(
                                value[0], value[1], best_n_clouds
                            )

                    self.best_model.sample(**sample_kwargs)
                self.best_model.solve(kl_div_threshold=kl_div_threshold)
