"""
cluster_posterior.py - Utilities for clustering posterior samples with
Gaussian Mixture Models.

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import arviz as az
import numpy as np
import xarray
from scipy.stats import mode
from sklearn.mixture import GaussianMixture


def cluster_posterior(
    trace: az.InferenceData,
    n_clusters: int,
    cluster_features: Iterable[str],
    num_gmm_samples: float = 10_000,
    kl_div_threshold: float = 0.1,
    seed: int = 1234,
) -> list:
    """Identify unique solutions and break the labeling degeneracy. To do so, we
    (1) fit a Gaussian Mixture Model (GMM) to the posterior samples of each chain individually.
    (2) calculate the Kullback–Leibler (KL) divergence (mean log-likelihood ratio) between
    chains. If the KD divergence is smaller than the given threshold, then both chains are
    part of the same solution. Otherwise, then each chain belongs to a different solution.
    The KL divergence is calculated from samples drawn from the fitted GMMs following the
    Monte Carlo procedure of Hershey & Olson (2007)
    (3) solve the labeling degeneracy by identifying the most common order of components
    among chains in each solution.

    :param trace: Posterior samples
    :type trace: az.InferenceData
    :param n_clusters: Number of GMM clusters
    :type n_clusters: int
    :param cluster_features: Parameter names to use for clustering
    :type cluster_features: Iterable[str]
    :param num_gmm_samples: Number of samples to generate from Gaussian Mixture Model (GMM), defaults to 10_000
    :type num_gmm_samples: int, optional
    :param kl_div_threshold: Kullback-Liebler (KL) divergence threshold, defaults to 0.1
    :type kl_div_threshold: float, optional
    :param seed: Random seed, defaults to 1234
    :type seed: int, optional
    :return: Solutions, where each element is a dictionary containing posterior samples and other statistics
    :rtype: list
    """
    # Fit GMMs to posterior samples of each chain, generate samples from GMMs
    gmm_results = {}
    for chain in trace.chain.data:
        features = np.array(
            [trace[param].sel(chain=chain).data.flatten() for param in cluster_features]
        ).T
        gmm = GaussianMixture(
            n_components=n_clusters,
            max_iter=100,
            init_params="random_from_data",
            n_init=10,
            verbose=False,
            random_state=seed,
        )
        gmm.fit(features)
        gmm_results[chain] = {"gmm": gmm, "samples": gmm.sample(num_gmm_samples)[0]}

    # Evaluate pair-wise KL divergence
    for chain1 in trace.chain.data:
        gmm_results[chain1]["kl_div"] = {}
        samples = gmm_results[chain1]["samples"]
        lnlike1 = gmm_results[chain1]["gmm"].score(samples)
        for chain2 in trace.chain.data:
            lnlike2 = gmm_results[chain2]["gmm"].score(samples)
            gmm_results[chain1]["kl_div"][chain2] = np.mean(lnlike1 - lnlike2)

    # Group unique solutions based on KL divergence
    solutions = []
    assigned_chains = []
    for chain1 in trace.chain.data:
        # skip if this chain is already assigned
        if chain1 in assigned_chains:
            continue

        # identify chains within KL divergence threshold
        good_kl_div_chains = [
            ch
            for ch, kl_div in gmm_results[chain1]["kl_div"].items()
            if kl_div < kl_div_threshold and ch not in assigned_chains
        ]

        # skip if fewer than two chains are assigned to this solution
        if len(good_kl_div_chains) < 2:  # pragma: no cover
            continue

        # get label order of each chain based on this chain's GMM
        solution = {"label_orders": {}}
        for chain2 in good_kl_div_chains:
            features = np.array(
                [
                    trace[param].sel(chain=chain2).data.flatten()
                    for param in cluster_features
                ]
            ).T
            labels = (
                gmm_results[chain1]["gmm"].predict(features).reshape(-1, n_clusters)
            )
            label_order = mode(labels, axis=0).mode

            # ensure all labels present
            if len(np.unique(label_order)) == len(label_order):
                solution["label_orders"][chain2] = label_order

        # save solution only if at least two chains have all labels present
        if len(solution["label_orders"]) >= 2:
            solutions.append(solution)

            # these chains are now assigned
            assigned_chains += list(solution["label_orders"].keys())

    # Each solution now has the labeling degeneracy broken, in that
    # each cloud has been assigned to a unique GMM cluster. We must
    # now determine which order of GMM clusters is preferred
    good_solutions = []
    for solution in solutions:
        label_orders = np.array(
            [label_order for label_order in solution["label_orders"].values()]
        )
        unique_label_orders, counts = np.unique(
            label_orders,
            axis=0,
            return_counts=True,
        )
        solution["label_order"] = unique_label_orders[np.argmax(counts)]

        # Determine the order of clouds needed to match the adopted label order.
        solution["cloud_orders"] = {}
        for chain, label_order in solution["label_orders"].items():
            xorder = np.argsort(label_order)
            solution["cloud_orders"][chain] = xorder[
                np.searchsorted(label_order[xorder], solution["label_order"])
            ]

        # Add solutions to the trace
        coords = trace.coords.copy()
        coords["chain"] = list(solution["cloud_orders"].keys())
        dims = {}
        posterior_clustered = {}
        for param, samples in trace.data_vars.items():
            if "cloud" in samples.coords:
                # break labeling degeneracy
                posterior_clustered[param] = xarray.concat(
                    [
                        samples.sel(chain=chain, cloud=cloud_order).assign_coords(
                            cloud=range(n_clusters)
                        )
                        for chain, cloud_order in solution["cloud_orders"].items()
                    ],
                    dim="chain",
                )
            else:
                posterior_clustered[param] = xarray.concat(
                    [
                        samples.sel(chain=chain)
                        for chain in solution["cloud_orders"].keys()
                    ],
                    dim="chain",
                )
            dims[param] = list(samples.dims)
        solution["posterior_clustered"] = posterior_clustered
        solution["coords"] = coords
        solution["dims"] = dims
        good_solutions.append(solution)
    return good_solutions
