"""
base_model.py - BaseModel definition

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
Trey Wenger - July 2024 - Add sample_smc
"""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import pymc as pm
from pymc.variational.callbacks import CheckParametersConvergence
from pymc.model.transform.optimization import freeze_dims_and_data

import pytensor.tensor as pt

import arviz as az
import arviz.labels as azl

import numpy as np
from numpy.polynomial import Polynomial

from scipy.stats import norm

import matplotlib.pyplot as plt
import graphviz

from bayes_spec.spec_data import SpecData
from bayes_spec.cluster_posterior import cluster_posterior
from bayes_spec import plots
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
        """
        Initialize a new model

        Inputs:
            data :: dictionary
                Spectral data sets, where the "key" defines the name of the
                dataset, and the value is a SpecData instance.
            n_clouds :: integer
                Number of cloud components
            baseline_degree :: integer
                Degree of the polynomial baseline
            seed :: integer
                Random seed
            verbose :: boolean
                Print extra info
        """
        self.n_clouds = n_clouds
        self.baseline_degree = baseline_degree
        self.seed = seed
        self.verbose = verbose
        self.data = data

        # Initialize the model
        coords = {
            "coeff": range(self.baseline_degree + 1),
            "cloud": range(self.n_clouds),
        }
        self._n_data = 0
        for _, dataset in self.data.items():
            self._n_data += len(dataset.spectral)
        self.model = pm.Model(coords=coords)

        # Model parameters
        self.baseline_params = [f"{key}_baseline_norm" for key in self.data.keys()]
        self.hyper_params = []
        self.cloud_params = []
        self.deterministics = []

        # Parameters used for posterior clustering
        self._cluster_features = []

        # Arviz labeller map
        self.var_name_map = {
            f"{key}_baseline_norm": r"$\beta_{\rm " + key + r"}$"
            for key in self.data.keys()
        }

        # set results and convergence checks
        self.reset_results()

    @abstractmethod
    def add_priors(self, *args, **kwargs):
        pass

    @abstractmethod
    def add_likelihood(self, *args, **kwargs):
        pass

    @property
    def _n_params(self):
        """
        Determine the number of model parameters.
        """
        return (
            len(self.cloud_params) * self.n_clouds
            + len(self.baseline_params) * (self.baseline_degree + 1)
            + len(self.hyper_params)
        )

    @property
    def _get_unique_solution(self):
        """
        Return the unique solution index (0) if there is a unique
        solution, otherwise raise an exception.
        """
        if not self.unique_solution:
            raise ValueError("There is not a unique solution. Must supply solution.")
        return 0

    @property
    def unique_solution(self):
        """
        Check if posterior samples suggest a unique solution
        """
        if self.solutions is None or len(self.solutions) == 0:
            raise ValueError("No solutions. Try solve()")
        return len(self.solutions) == 1

    @property
    def labeller(self):
        """
        Get the arviz labeller.
        """
        return azl.MapLabeller(var_name_map=self.var_name_map)

    def _validate(self):
        """
        Validate the model by checking the log probability at the initial point.
        """
        # check that likelihood has been added
        if len(self.model.observed_RVs) == 0:
            raise ValueError("No observed variables found! Did you add_likelihood()?")

        # check that model can be evaluated
        if not np.isfinite(self.model.logp().eval(self.model.initial_point())):
            raise ValueError(
                "Model initial point is not finite! Mis-specified model or bad priors?"
            )

    def reset_results(self):
        """
        Reset results and convergence checks.

        Inputs: None
        Returns: Nothing
        """
        self.approx: pm.Approximation = None
        self.trace: az.InferenceData = None
        self.solutions = None
        self._good_chains = None
        self._chains_converged: bool = None

    def null_bic(self):
        """
        Evaluate the BIC for the null hypothesis (baseline only, no clouds)

        Inputs: None
        Returns: Nothing
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

    def lnlike_mean_point_estimate(
        self, chain: Optional[int] = None, solution: Optional[int] = None
    ):
        """
        Evaluate model log-likelihood at the mean point estimate of posterior samples.

        Inputs:
            chain :: None or integer
                If None (default), evaluate BIC across all chains using
                clustered posterior samples. Otherwise, evaluate BIC for
                this chain only using un-clustered posterior samples.
            solution :: None or integer
                Solution index
                If chain is None and solution is None:
                    If there is a unique solution, use that
                    Otherwise, raise an exception
                If chain is None and solution is not None:
                    Use this solution index
                If chain is not None:
                    This parameter has no effect

        Returns: lnlike
            lnlike :: scalar
                Log likelihood at point
        """
        if chain is None and solution is None:
            solution = self._get_unique_solution

        # mean point estimate
        if chain is None:
            point = self.trace[f"solution_{solution}"].mean(dim=["chain", "draw"])
        else:
            point = self.trace.posterior.sel(chain=chain).mean(dim=["draw"])

        # RV names and transformations
        params = {}
        for rv in self.model.free_RVs:
            name = rv.name
            param = self.model.rvs_to_values[rv]
            transform = self.model.rvs_to_transforms[rv]
            if transform is None:
                params[param] = point[name].data
            else:
                params[param] = transform.forward(
                    point[name].data, *rv.owner.inputs
                ).eval()

        return float(self.model.logp().eval(params))

    def bic(self, chain: Optional[int] = None, solution: Optional[int] = None):
        """
        Calculate the Bayesian information criterion at the mean point estimate.

        Inputs:
            chain :: None or integer
                If None (default), evaluate BIC across all chains using
                clustered posterior samples. Otherwise, evaluate BIC for
                this chain only using un-clustered posterior samples.
            solution :: None or integer
                Solution index
                If chain is None and solution is None:
                    If there is a unique solution, use that
                    Otherwise, raise an exception
                If chain is None and solution is not None:
                    Use this solution index
                If chain is not None:
                    This parameter has no effect

        Returns: bic
            bic :: scalar
                Bayesian information criterion
        """
        try:
            lnlike = self.lnlike_mean_point_estimate(chain=chain, solution=solution)
            return self._n_params * np.log(self._n_data) - 2.0 * lnlike
        except ValueError as e:
            print(e)
            return np.inf

    def good_chains(self, mad_threshold: float = 5.0):
        """
        Identify bad chains as those with deviant BICs.

        Inputs:
            mad_threshold :: scalar
                Chains are good if they have BICs within {mad_threshold} * MAD of the median BIC.

        Returns: good_chains
            good_chains :: 1-D array of integers
                Chains that appear converged
        """
        if self.trace is None:
            raise ValueError("Model has no posterior samples. Try fit() or sample().")

        # check if already determined
        if self._good_chains is not None:
            return self._good_chains

        # if the trace has fewer than 2 chains, we assume they're both ok so we can run
        # convergence diagnostics
        if len(self.trace.posterior.chain) < 3:
            self._good_chains = self.trace.posterior.chain.data
            return self._good_chains

        # per-chain BIC
        bics = np.array(
            [self.bic(chain=chain) for chain in self.trace.posterior.chain.data]
        )
        mad = np.median(np.abs(bics - np.median(bics)))
        good = np.abs(bics - np.median(bics)) < mad_threshold * mad

        self._good_chains = self.trace.posterior.chain.data[good]
        return self._good_chains

    def add_baseline_priors(self, prior_baseline_coeff=1.0):
        """
        Add baseline priors to model. The baseline priors are spectrally
        normalized, such that
        baseline_norm = coeff[0] + coeff[1]*spectral_norm + coeff[2]*spectral_norm**2 + ...
        where spectral_norm is the normalized spectral axis and baseline_norm is the
        normalized brightness data (both normalized to zero mean and unit variance).
        Thus, prior_baseline_coeff can be assumed near unity.

        Inputs:
            prior_baseline_coeff :: scalar
                Width of the Normal prior distribution on the normalized
                baseline polynomial coefficients

        Returns: Nothing
        """
        with self.model:
            for key in self.data.keys():
                # add the normalized prior
                _ = pm.Normal(
                    f"{key}_baseline_norm",
                    mu=0.0,
                    sigma=prior_baseline_coeff,
                    dims="coeff",
                )

    def predict_baseline(self):
        """
        Predict the un-normalized baseline model.

        Inputs: None

        Returns:
            baseline_model :: dictionary
                Dictionary with keys like those in self.data, where each
                value is the un-normalized baseline model
        """
        baseline_model = {}
        for key, dataset in self.data.items():
            # evaluate the baseline
            baseline_norm = pt.sum(
                [
                    self.model[f"{key}_baseline_norm"][i] * dataset.spectral_norm**i
                    for i in range(self.baseline_degree + 1)
                ],
                axis=0,
            )
            baseline_model[key] = dataset.unnormalize_brightness(baseline_norm)
        return baseline_model

    def prior_predictive_check(self, samples: int = 50, plot_fname: str = None):
        """
        Generate prior predictive samples, and optionally plot the outcomes.

        Inputs:
            samples :: integer
                Number of prior predictive samples to generate
            plot_fname :: string
                If not None, generate a plot of the outcomes over
                the data, and save to this filename.

        Returns: predictive
            predictive :: InferenceData
                Object containing prior and prior predictive samples
        """
        # validate
        self._validate()

        with self.model:
            trace = pm.sample_prior_predictive(samples=samples, random_seed=self.seed)

        if plot_fname is not None:
            plots.plot_predictive(self.data, trace.prior_predictive, plot_fname)

        return trace

    def posterior_predictive_check(
        self,
        solution: Optional[int] = None,
        thin: int = 100,
        plot_fname: Optional[str] = None,
    ):
        """
        Generate posterior predictive samples, and optionally plot the outcomes.

        Inputs:
            solution :: integer
                If None, generate posterior predictive samples from the un-clustered posterior
                samples. Otherwise, generate predictive samples from this solution index.
            thin :: integer
                Thin posterior samples by keeping one in {thin}
            plot_fname :: string
                If not None, generate a plot of the outcomes over
                the data, and save to this filename.

        Returns: predictive
            predictive :: InferenceData
                Object containing posterior and posterior predictive samples
        """
        # validate
        self._validate()

        if self.trace is None:
            raise ValueError("Model has no posterior samples. Try fit() or sample().")

        with self.model:
            if solution is None:
                posterior = self.trace.posterior.sel(
                    chain=self.good_chains(), draw=slice(None, None, thin)
                )
            else:
                posterior = self.trace[f"solution_{solution}"].sel(
                    draw=slice(None, None, thin)
                )
            trace = pm.sample_posterior_predictive(
                posterior,
                extend_inferencedata=True,
                random_seed=self.seed,
            )

        if plot_fname is not None:
            plots.plot_predictive(
                self.data,
                trace.posterior_predictive,
                plot_fname,
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
        """
        Fit posterior using variational inference (VI). If you get NaNs
        during optimization, try increasing the learning rate.

        Inputs:
            n :: integer
                Number of VI iterations
            draws :: integer
                Number of samples to draw from fitted posterior
            rel_tolerance :: scalar
                Relative parameter tolerance for VI convergence
            abs_tolerance :: scalar
                Absolute parameter tolerance for VI convergence
            learning_rate :: scalar
                adagrad_window learning rate. Try increasing if you get NaNs
            **kwargs :: additional keyword arguments
                Additional arguments passed to pymc.fit
                (method)

        Returns: Nothing
        """
        # validate
        self._validate()

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

    def sample(
        self,
        init: str = "advi+adapt_diag",
        n_init: int = 100_000,
        chains: int = 4,
        init_kwargs: Optional[dict] = None,
        nuts_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Sample posterior distribution using MCMC.

        Inputs:
            init :: string
                Initialization strategy
            n_init :: integer
                Number of initialization iterations
            chains :: integer
                Number of chains
            init_kwargs :: dictionary
                Keyword arguments passed to init_nuts
                (tolerance, learning_rate)
            nuts_kwargs :: dictionary
                Keyword arguments passed to pm.NUTS
                (target_accept)
            **kwargs :: additional keyword arguments
                Keyword arguments passed to pm.sample
                (cores, tune, draws)

        Returns: Nothing
        """
        # validate
        self._validate()

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

        # diagnostics
        if self.verbose:
            # converged chains
            good_chains = self.good_chains()
            if len(good_chains) < len(self.trace.posterior.chain):
                print(f"Only {len(good_chains)} chains appear converged.")

            # divergences
            num_divergences = self.trace.sample_stats.diverging.sel(
                chain=self.good_chains()
            ).data.sum()
            if num_divergences > 0:
                print(f"There were {num_divergences} divergences in converged chains.")

    def sample_smc(
        self,
        **kwargs,
    ):
        """
        Sample posterior distribution using Sequential Monte Carlo (SMC).

        Inputs:
            **kwargs :: additional keyword arguments
                Keyword arguments passed to pm.sample_smc
                (draws, chains, cores)

        Returns: Nothing
        """
        # validate
        self._validate()

        # reset convergence checks
        self.reset_results()

        with self.model:
            self.trace = pm.sample_smc(
                progressbar=self.verbose,
                compute_convergence_checks=False,
                **kwargs,
            )

        # diagnostics
        if self.verbose:
            # converged chains
            good_chains = self.good_chains()
            if len(good_chains) < len(self.trace.posterior.chain):
                print(f"Only {len(good_chains)} chains appear converged.")

    def solve(self, p_threshold=0.9):
        """
        Cluster posterior samples and determine unique solutions. Adds
        new groups to self.trace called "solution_{idx}" for the posterior
        samples of each unique solution.

        Inputs:
            p_threshold :: scalar
                p-value threshold for considering a unique solution

        Returns: Nothing
        """
        # Drop solutions if they already exist in trace
        for group in list(self.trace.groups()):
            if "solution" in group:
                del self.trace[group]

        self.solutions = []
        solutions = cluster_posterior(
            self.trace.posterior.sel(chain=self.good_chains()),
            self.n_clouds,
            self._cluster_features,
            p_threshold=p_threshold,
            seed=self.seed,
        )
        if len(solutions) < 1 and self.verbose:
            print("No solution found!")

        # convergence check
        unique_solution = len(solutions) == 1
        if self.verbose:
            if unique_solution:
                print("GMM converged to unique solution")
            else:
                print(f"GMM found {len(solutions)} unique solutions")
                for solution_idx, solution in enumerate(solutions):
                    print(
                        f"Solution {solution_idx}: chains {list(solution['label_orders'].keys())}"
                    )

        # labeling degeneracy check
        for solution_idx, solution in enumerate(solutions):
            label_orders = np.array(
                [label_order for label_order in solution["label_orders"].values()]
            )
            if self.verbose and not np.all(label_orders == label_orders[0]):
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

    def plot_graph(self, dotfile: str, ext: str):
        """
        Generate dot plot of model graph.

        Inputs:
            dotfile :: string
                Where graphviz source is saved
            ext :: string
                Rendered image is {dotfile}.{ext}

        Returns: Nothing
        """
        gviz = pm.model_to_graphviz(self.model)
        gviz.graph_attr["rankdir"] = "TB"
        gviz.graph_attr["splines"] = "ortho"
        gviz.graph_attr["newrank"] = "false"
        source = gviz.unflatten(stagger=3)

        # save and render
        with open(dotfile, "w", encoding="ascii") as f:
            f.write("\n".join(source))
        graphviz.render("dot", ext, dotfile)

    def plot_traces(self, plot_fname: str, warmup: bool = False):
        """
        Plot traces for all chains.

        Inputs:
            plot_fname :: string
                Plot filename
            warmup :: boolean
                If True, plot warmup samples instead

        Returns: Nothing
        """
        posterior = self.trace.warmup_posterior if warmup else self.trace.posterior
        with az.rc_context(rc={"plot.max_subplots": None}):
            var_names = [rv.name for rv in self.model.free_RVs]
            axes = az.plot_trace(
                posterior.sel(chain=self.good_chains()),
                var_names=var_names,
            )
            fig = axes.ravel()[0].figure
            fig.tight_layout()
            fig.savefig(plot_fname, bbox_inches="tight")
            plt.close(fig)

    def plot_pair(self, plot_fname: str, solution: Optional[int] = None):
        """
        Generate pair plots from clustered posterior samples.

        Inputs:
            plot_fname :: string
                Figure filename with the format: {basename}.{ext}
                Several plots are generated:
                {basename}.{ext}
                    Pair plot of non-clustered cloud parameters
                {basename}_determ.{ext}
                    Pair plot of non-clustered cloud deterministic parameters
                {basename}_{cloud}.{ext}
                    Pair plot of clustered cloud with index {cloud} parameters
                {basename}_{num}_determ.{ext}
                    Pair plot of clustered cloud with index {cloud} deterministic parameters
                {basename}_other.{ext}
                    Pair plot of baseline and hyper parameters
            solution :: None or integer
                Plot the posterior samples associated with this solution index. If
                solution is None and there is a unique solution, use that.
                Otherwise, raise an exception.

        Returns: Nothing
        """
        if solution is None:
            solution = self._get_unique_solution
        trace = self.trace[f"solution_{solution}"]

        basename, ext = os.path.splitext(plot_fname)

        # All cloud free parameters
        plots.plot_pair(
            trace,
            self.cloud_params,
            "All Clouds\nFree Parameters",
            plot_fname,
        )
        # All cloud deterministic parameters
        plots.plot_pair(
            trace,
            self.deterministics,
            "All Clouds\nDerived Quantities",
            basename + "_determ" + ext,
        )
        # Baseline & hyper parameters
        plots.plot_pair(
            trace,
            self.baseline_params + self.hyper_params,
            "All Clouds\nDerived Quantities",
            basename + "_other" + ext,
        )
        # Cloud quantities
        for cloud in range(self.n_clouds):
            plots.plot_pair(
                trace.sel(cloud=cloud),
                self.cloud_params,
                f"Cloud {cloud}\nFree Parameters",
                basename + f"_{cloud}" + ext,
            )
            plots.plot_pair(
                trace.sel(cloud=cloud),
                self.deterministics,
                f"Cloud {cloud}\nDerived Quantities",
                basename + f"_{cloud}_determ" + ext,
            )
