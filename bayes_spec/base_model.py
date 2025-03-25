"""
base_model.py - BaseModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Iterable

import pymc as pm
from pymc.variational.callbacks import CheckParametersConvergence
from pymc.model.transform.optimization import freeze_dims_and_data

import pytensor.tensor as pt

import arviz as az
import arviz.labels as azl
import graphviz

import numpy as np
from numpy.polynomial import Polynomial

from scipy.stats import norm

from bayes_spec import SpecData
from bayes_spec.cluster_posterior import cluster_posterior
from bayes_spec.nuts import init_nuts


class BaseModel(ABC):
    """
    BaseModel defines functions and attributes common to all model definitions.
    """

    def __init__(
        self,
        data: dict[str, SpecData],
        n_clouds: int,
        baseline_degree: int = 0,
        seed: int = 1234,
        verbose: bool = False,
    ):
        """Initialize a new BaseModel

        :param data: Spectral data sets, where the "key" defines the name of the dataset
        :type data: dict[str, SpecData]
        :param n_clouds: Number of cloud components
        :type n_clouds: int
        :param baseline_degree: Polynomial baseline degree, defaults to 0
        :type baseline_degree: int, optional
        :param seed: Random seed, defaults to 1234
        :type seed: int, optional
        :param verbose: Print verbose output, defaults to False
        :type verbose: bool, optional
        """
        self.n_clouds = n_clouds
        self.baseline_degree = baseline_degree
        self.seed = seed
        self.verbose = verbose
        self.data = data

        # Initialize the model
        coords = {
            "baseline_coeff": range(self.baseline_degree + 1),
            "cloud": range(self.n_clouds),
        }
        self._n_data = 0
        for _, dataset in self.data.items():
            self._n_data += len(dataset.spectral)
        self.model = pm.Model(coords=coords)

        # Parameters used for posterior clustering
        self._cluster_features = []

        # Arviz labeller map
        self.var_name_map = {
            f"baseline_{key}_norm": r"$\beta_{\rm " + key + r"}$"
            for key in self.data.keys()
        }

        # set results and convergence checks
        self.reset_results()

    @abstractmethod
    def add_priors(self, *args, **kwargs):  # pragma: no cover
        """Must be defined in inhereted class."""
        pass

    @abstractmethod
    def add_likelihood(self, *args, **kwargs):  # pragma: no cover
        """Must be defined in inhereted class."""
        pass

    def _get_param_names(
        self, dim: Optional[str] = None, deterministics: bool = False
    ) -> Iterable[str]:
        """Get a subset of the model parameter names.

        :param dims: Select only parameters with this dimension name. If None, select dimensionless
            parameters, defaults to None
        :type dims: Optional[str], optional
        :param deterministics: If True, select deterministic parameters, otherwise free
            parameters, defaults to False
        :type deterministics: bool, optional
        :return: Parameter names matching selection
        :rtype: Iterable[str]
        """
        all_params = (
            self.model.deterministics if deterministics else self.model.free_RVs
        )
        if dim is None:
            return [
                param.name
                for param in all_params
                if param.name not in self.model.named_vars_to_dims
            ]
        return [
            param.name
            for param in all_params
            if dim in self.model.named_vars_to_dims.get(param.name, [])
        ]

    @property
    def baseline_freeRVs(self) -> Iterable[str]:
        """Get the free baseline parameter names.

        :return: Free baseline parameter names
        :rtype: Iterable[str]
        """
        return self._get_param_names("baseline_coeff", deterministics=False)

    @property
    def baseline_deterministics(self) -> Iterable[str]:
        """Get the deterministic baseline parameter names.

        :return: Deterministic baseline parameter names
        :rtype: Iterable[str]
        """
        return self._get_param_names("baseline_coeff", deterministics=True)

    @property
    def cloud_freeRVs(self) -> Iterable[str]:
        """Get the free cloud parameter names.

        :return: Free cloud parameter names
        :rtype: Iterable[str]
        """
        return self._get_param_names("cloud", deterministics=False)

    @property
    def cloud_deterministics(self) -> Iterable[str]:
        """Get the deterministic cloud parameter names.

        :return: Deterministic cloud parameter names
        :rtype: Iterable[str]
        """
        return self._get_param_names("cloud", deterministics=True)

    @property
    def hyper_freeRVs(self) -> Iterable[str]:
        """Get the free hyper parameter names.

        :return: Free hyper parameter names
        :rtype: Iterable[str]
        """
        return self._get_param_names(deterministics=False)

    @property
    def hyper_deterministics(self) -> Iterable[str]:
        """Get the deterministic hyper parameter names.

        :return: Deterministic hyper parameter names
        :rtype: Iterable[str]
        """
        return self._get_param_names(deterministics=True)

    @property
    def _n_params(self) -> int:
        """Determine the number of free model parameters.

        :return: Number of free model parameters
        :rtype: int
        """
        return np.sum([param.size.eval().prod() for param in self.model.free_RVs])

    @property
    def _get_unique_solution(self) -> int:
        """Return the unique solution index (0) if there is a unique solution, otherwise raise an exception.

        :raises ValueError: No unique solution
        :return: Unique solution index (`0`)
        :rtype: int
        """
        if not self.unique_solution:  # pragma: no cover
            raise ValueError("There is not a unique solution. Must supply solution.")
        return 0

    @property
    def unique_solution(self) -> bool:
        """Check if posterior samples suggest a unique solution.

        :raises ValueError: No solutions
        :return: True if there is a unique solution, False otherwise
        :rtype: bool
        """
        if self.solutions is None or len(self.solutions) == 0:
            raise ValueError("No solutions. Try solve()")
        return len(self.solutions) == 1

    @property
    def labeller(self) -> azl.MapLabeller:
        """Get the arviz labeller.

        :return: Arviz labeller
        :rtype: azl.MapLabeller
        """
        return azl.MapLabeller(var_name_map=self.var_name_map)

    def _validate(self) -> bool:
        """Validate the model by checking the log probability at the initial point.

        :raises ValueError: Model does not contain likelihood
        :raises ValueError: Model likelihood fails to evaluate at the initial point
        :return: True
        :rtype: bool
        """
        # check that likelihood has been added
        if len(self.model.observed_RVs) == 0:
            raise ValueError("No observed variables found! Did you add_likelihood()?")

        # check that model can be evaluated
        if not np.isfinite(self.model.logp().eval(self.model.initial_point())):
            raise ValueError(
                "Model initial point is not finite! Mis-specified model or bad priors?"
            )

        return True

    def reset_results(self):
        """Reset results and convergence checks."""
        self.approx: pm.Approximation = None
        self.trace: az.InferenceData = None
        self.solutions: list = None
        self._chains_converged: bool = None

    def graph(self) -> graphviz.sources.Source:
        """Generate visualization of the model graph. The output can be displayed in-line in a Jupyter notebook,
        or rendered with `graph().render('filename')`.

        :return: Graph visualization
        :rtype: graphviz.sources.Source
        """
        gviz = pm.model_to_graphviz(self.model)
        gviz.graph_attr["rankdir"] = "TB"
        gviz.graph_attr["splines"] = "ortho"
        gviz.graph_attr["newrank"] = "false"
        source = gviz.unflatten(stagger=3)
        return source

    def null_bic(self) -> float:
        """Evaluate the Bayesian Information Criterion for the null hypothesis (baseline only, no clouds)

        :return: Null hypothesis BIC
        :rtype: float
        """
        lnlike = 0.0
        for _, dataset in self.data.items():
            # fit polynomial baseline to un-normalized spectral data
            baseline = Polynomial.fit(
                dataset.spectral, dataset.brightness, self.baseline_degree
            )(dataset.spectral)

            # evaluate likelihood
            lnlike += norm.logpdf(
                dataset.brightness - baseline, scale=dataset.noise
            ).sum()

        n_params = len(self.data) * (self.baseline_degree + 1)
        return n_params * np.log(self._n_data) - 2.0 * lnlike

    def mean_lnlike(
        self, chain: Optional[int] = None, solution: Optional[int] = None
    ) -> float:
        """Evaluate mean log-likelihood over posterior samples.

        :param chain: Evaluate mean log-likelihood for this chain using un-clustered posterior samples.
            If `None` evaluate across all chains using clustered posterior samples, defaults to None
        :type chain: Optional[int], optional
        :param solution: Evaluate mean log-likelihood for this solution. If `None` use the unique solution if any.
            If :param:chain is not None, this parameter has no effect, defaults to None
        :type solution: Optional[int], optional
        :return: Mean log-likelihood over posterior samples
        :rtype: float
        """
        if chain is None and solution is None:
            solution = self._get_unique_solution

        # likelihood samples
        if chain is None:
            chain = self.trace[f"solution_{solution}"].chain
        lnlike = self.trace["log_likelihood"].sel(chain=chain)

        spec_dims = [f"{key}_dim_0" for key in self.data.keys()]
        spec_sum_lnlike = lnlike.sum(dim=spec_dims)
        sum_lnlike = np.sum([spec_sum_lnlike[key] for key in self.data.keys()], axis=0)
        return np.mean(sum_lnlike)

    def bic(self, chain: Optional[int] = None, solution: Optional[int] = None) -> float:
        """Calculate the Bayesian information criterion at the mean point estimate.

        :param chain: Evaluate BIC for this chain using un-clustered posterior samples. If `None` evaluate across all
            chains using clustered posterior samples, defaults to None
        :type chain: Optional[int], optional
        :param solution: Evaluate BIC for this solution. If `None` use the unique solution if any. If :param:chain
            is not None, this parameter has no effect, defaults to None
        :type solution: Optional[int], optional
        :return: Bayesian information criterion
        :rtype: float
        """
        try:
            lnlike = self.mean_lnlike(chain=chain, solution=solution)
            return self._n_params * np.log(self._n_data) - 2.0 * lnlike
        except ValueError as e:  # pragma: no cover
            print(e)
            return np.inf

    def add_baseline_priors(
        self, prior_baseline_coeffs: Optional[dict[str, list[float]]] = None
    ):
        """Add baseline priors to the model. The polynomial baseline is evaluated on the normalized data like:
        `baseline_norm = sum_i(coeff[i]/(i+1)**i * spectral_norm**i)`

        :param prior_baseline_coeffs: Width of normal prior distribution on the normalized baseline polynomial
            coefficients. Keys are dataset names and values are lists of length `baseline_degree+1`. If None,
            use `[1.0]*(baseline_degree+1)` for each dataset, defaults to None
        :type prior_baseline_coeffs: Optional[dict[str, list[float]]], optional
        """
        if prior_baseline_coeffs is None:
            prior_baseline_coeffs = {
                key: [1.0] * (self.baseline_degree + 1) for key in self.data.keys()
            }
        for key, coeffs in prior_baseline_coeffs.items():
            if len(coeffs) != self.baseline_degree + 1:
                raise ValueError(
                    f"{key} baseline coefficient prior must have length "
                    + f"`baseline_degree+1` = {self.baseline_degree + 1}"
                )

        with self.model:
            for key in self.data.keys():
                # add the normalized prior
                _ = pm.Normal(
                    f"baseline_{key}_norm",
                    mu=0.0,
                    sigma=prior_baseline_coeffs[key],
                    dims="baseline_coeff",
                )

    def predict_baseline(
        self, baseline_params: Optional[dict[str, list[float]]] = None
    ) -> dict[str, list[float]]:
        """Predict the un-normalized baseline model.

        :param baseline_params: Dictionary of baseline parameters with which to evaluate the baseline model.
            Keys are the same as in the model: "baseline_{key}_norm", where {key} are the supplied datasets. The
            values are lists of length `baseline_degree+1`. If None, evaluate the baseline model using the current
            model state, defaults to None
        :type baseline_params: Optional[dict[str, list[float]]], optional
        :return: Un-normalized baseline models for each dataset. Keys are dataset names and values are
            the un-normalized baseline models.
        :rtype: dict[str, list[float]]
        """
        if baseline_params is None:
            baseline_params = self.model
        baseline_model = {}
        for key, dataset in self.data.items():
            # evaluate the baseline
            baseline_norm = pt.sum(
                [
                    baseline_params[f"baseline_{key}_norm"][i]
                    / (i + 1.0) ** i
                    * dataset.spectral_norm**i
                    for i in range(self.baseline_degree + 1)
                ],
                axis=0,
            )
            baseline_model[key] = dataset.unnormalize_brightness(baseline_norm)
        return baseline_model

    def sample_prior_predictive(self, samples: int = 50) -> az.InferenceData:
        """Generate prior predictive samples

        :param samples: Number of prior predictive samples to draw, defaults to 50
        :type samples: int, optional
        :return: Prior predictive samples
        :rtype: az.InferenceData
        """
        # validate
        assert self._validate()

        with self.model:
            trace = pm.sample_prior_predictive(samples=samples, random_seed=self.seed)

        return trace

    def sample_posterior_predictive(
        self,
        solution: Optional[int] = None,
        thin: int = 100,
    ) -> az.InferenceData:
        """Generate posterior predictive samples

        :param solution: Draw posterior predictive samples from this solution index. If None, draw samples from
            the un-clustered posterior samples, defaults to None
        :type solution: Optional[int], optional
        :param thin: Thin posterior samples by keeping one in :param:thin, defaults to 100
        :type thin: int, optional
        :raises ValueError: No posterior samples
        :return: Posterior predictive samples
        :rtype: az.InferenceData
        """
        # validate
        assert self._validate()

        if self.trace is None:
            raise ValueError("Model has no posterior samples. Try fit() or sample().")

        with self.model:
            if solution is None:
                posterior = self.trace.posterior.sel(draw=slice(None, None, thin))
            else:
                posterior = self.trace[f"solution_{solution}"].sel(
                    draw=slice(None, None, thin)
                )
            trace = pm.sample_posterior_predictive(
                posterior,
                extend_inferencedata=True,
                random_seed=self.seed,
            )

        return trace

    def fit(
        self,
        n: int = 100_000,
        draws: int = 1_000,
        rel_tolerance: float = 0.001,
        abs_tolerance: float = 0.001,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        """Approximate posterior distribution using Variational Inference (VI).

        :param n: Number of VI iterations, defaults to 100_000
        :type n: int, optional
        :param draws: Number of posterior samples to draw, defaults to 1_000
        :type draws: int, optional
        :param rel_tolerance: Relative parameter tolerance for VI convergence, defaults to 0.001
        :type rel_tolerance: float, optional
        :param abs_tolerance: Absolute parameter tolerance for VI convergence, defaults to 0.001
        :type abs_tolerance: float, optional
        :param learning_rate: VI learning rate, defaults to 1e-3
        :type learning_rate: float, optional
        :param `**kwargs`: Additional arguments passed to :func:`pymc.fit`
        """
        # validate
        assert self._validate()

        # reset convergence checks
        self.reset_results()

        with self.model:
            callbacks = [
                CheckParametersConvergence(tolerance=rel_tolerance, diff="relative"),
                CheckParametersConvergence(tolerance=abs_tolerance, diff="absolute"),
            ]
            self.approx = pm.fit(
                n=n,
                random_seed=self.seed,
                progressbar=self.verbose,
                callbacks=callbacks,
                obj_optimizer=pm.adagrad_window(learning_rate=learning_rate),
                **kwargs,
            )
            self.trace = self.approx.sample(draws)

        with self.model:
            pm.compute_log_likelihood(self.trace)

    def sample(
        self,
        init: str = "advi+adapt_diag",
        n_init: int = 100_000,
        chains: int = 4,
        init_kwargs: Optional[dict] = None,
        nuts_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Sample posterior distribution using MCMC.

        :param init: Initialization strategy, defaults to "advi+adapt_diag"
        :type init: str, optional
        :param n_init: Number of initialization iterations, defaults to 100_000
        :type n_init: int, optional
        :param chains: Number of independent Markov chains, defaults to 4
        :type chains: int, optional
        :param init_kwargs: Keyword arguments passed to :func:`init_nuts`, defaults to None
        :type init_kwargs: Optional[dict], optional
        :param nuts_kwargs: Keyword arguments passed to :func:`pymc.NUTS`, defaults to None
        :type nuts_kwargs: Optional[dict], optional
        :param `**kwargs`: Additional arguments passed to :func:`pymc.sample`
        """
        # validate
        assert self._validate()

        # reset convergence checks
        self.reset_results()

        # catch non-standard initialization for non-pymc samplers
        if (
            "nuts_sampler" in kwargs.keys()
            and kwargs["nuts_sampler"] != "pymc"
            and init != "auto"
        ):
            init = "auto"
            warnings.warn("setting init='auto' for non-pymc sampler")

        if init == "auto":
            init = "jitter+adapt_diag"

        if init_kwargs is None:
            init_kwargs = {}
        if nuts_kwargs is None:
            nuts_kwargs = {}

        # freeze model for JAX samplers
        frozen_model = self.model
        if "nuts_sampler" in kwargs.keys() and kwargs["nuts_sampler"] in [
            "numpyro",
            "blackjax",
        ]:
            frozen_model = freeze_dims_and_data(self.model)

        # attempt custom initialization
        initial_points, step = init_nuts(
            self.model,
            init=init,
            n_init=n_init,
            chains=chains,
            nuts_kwargs=nuts_kwargs,
            seed=self.seed,
            verbose=self.verbose,
            **init_kwargs,
        )

        # if we're using custom initialization, then drop nuts
        # arguments from pm.sample
        if initial_points is not None:
            nuts_kwargs = {}

        with frozen_model:
            self.trace = pm.sample(
                init=init,
                initvals=initial_points,
                step=step,
                chains=chains,
                progressbar=self.verbose,
                discard_tuned_samples=False,
                compute_convergence_checks=False,
                random_seed=self.seed,
                **nuts_kwargs,
                **kwargs,
            )

        # add log-likelihood to trace
        if self.verbose:
            print("Adding log-likelihood to trace")
        with frozen_model:
            pm.compute_log_likelihood(self.trace)

        # diagnostics
        if self.verbose:
            # divergences
            num_divergences = self.trace.sample_stats.diverging.data.sum()
            if num_divergences > 0:  # pragma: no cover
                print(f"There were {num_divergences} divergences in converged chains.")

    def sample_smc(
        self,
        **kwargs,
    ):
        """Sample posterior distribution using Sequential Monte Carlo.

        :param `**kwargs`: Additional arguments passed to :func:`pymc.sample_smc`
        """

        # validate
        assert self._validate()

        # reset convergence checks
        self.reset_results()

        with self.model:
            self.trace = pm.sample_smc(
                progressbar=self.verbose,
                compute_convergence_checks=False,
                **kwargs,
            )

        # add log-likelihood to trace
        if self.verbose:
            print("Adding log-likelihood to trace")
        with self.model:
            pm.compute_log_likelihood(self.trace, progressbar=self.verbose)

    def solve(self, num_gmm_samples: float = 10_000, kl_div_threshold: float = 0.1):
        """Identify unique solutions and break the labeling degeneracy.
        Adds new groups to the `trace` called `solution_{idx}` with the label-corrected posterior
        samples of each unique solution.

        :param num_gmm_samples: Number of samples to generate from Gaussian Mixture Model (GMM), defaults to 10_000
        :type num_gmm_samples: int, optional
        :param kl_div_threshold: Kullback-Liebler (KL) divergence threshold, defaults to 0.1
        :type kl_div_threshold: float, optional
        """
        # Drop solutions if they already exist in trace
        for group in list(self.trace.groups()):
            if "solution" in group:  # pragma: no cover
                del self.trace[group]

        self.solutions = []
        solutions = cluster_posterior(
            self.trace.posterior,
            self.n_clouds,
            self._cluster_features,
            num_gmm_samples=num_gmm_samples,
            kl_div_threshold=kl_div_threshold,
            seed=self.seed,
        )
        if len(solutions) < 1 and self.verbose:  # pragma: no cover
            print("No solution found!")

        # convergence checks
        unique_solution = len(solutions) == 1
        if self.verbose:
            if unique_solution:
                print("GMM converged to unique solution")
            elif len(solutions) > 1:  # pragma: no cover
                print(f"GMM found {len(solutions)} unique solutions")
                for solution_idx, solution in enumerate(solutions):
                    print(
                        f"Solution {solution_idx}: chains {list(solution['label_orders'].keys())}"
                    )

        assigned_chains = []
        for solution in solutions:
            assigned_chains += list(solution["label_orders"].keys())
        if self.verbose and len(assigned_chains) < len(
            self.trace.posterior.chain
        ):  # pragma: no cover
            print(
                f"{len(assigned_chains)} of {len(self.trace.posterior.chain)} chains appear converged."
            )

        for solution_idx, solution in enumerate(solutions):
            # report labeling degeneracy
            label_orders = np.array(
                [label_order for label_order in solution["label_orders"].values()]
            )
            if self.verbose and not np.all(
                label_orders == label_orders[0]
            ):  # pragma: no cover
                print(f"Label order mismatch in solution {solution_idx}")
                for chain, label_order in solution["label_orders"].items():
                    print(f"Chain {chain} order: {label_order}")
                print(f"Adopting (first) most common order: {solution['label_order']}")

            # Add solution to the trace
            with warnings.catch_warnings(action="ignore"):
                self.trace.add_groups(
                    **{
                        f"solution_{solution_idx}": solution["posterior_clustered"],
                        "coords": solution["coords"],
                        "dims": solution["dims"],
                    }
                )
                self.solutions.append(solution_idx)
