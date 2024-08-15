"""
cluster_posterior.py - Utilities for clustering posterior samples with
Gaussian Mixture Models.

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
"""

from typing import Iterable

import arviz as az
import numpy as np
import xarray
from scipy.stats import chi2, mode
from scipy.spatial.distance import mahalanobis
from sklearn.mixture import GaussianMixture


def cluster_posterior(
    trace: az.InferenceData,
    n_clusters: int,
    cluster_features: Iterable[str],
    p_threshold: float = 0.9,
    seed: int = 1234,
) -> list:
    """Fit Gaussian Mixture Models (GMM) to the posterior samples in order to (1)
    identify unique solutions and (2) solve the labeling degeneracy.

    :param trace: Posterior samples
    :type trace: az.InferenceData
    :param n_clusters: Number of GMM clusters
    :type n_clusters: int
    :param cluster_features: Parameter names to use for clustering
    :type cluster_features: Iterable[str]
    :param p_threshold: p-value threshold for unique solution identification, defaults to 0.9
    :type p_threshold: float, optional
    :param seed: Random seed, defaults to 1234
    :type seed: int, optional
    :return: Solutions, where each element is a dictionary containing posterior samples and other statistics
    :rtype: list
    """
    # Determine if a chain prefers a unique solution suggested by
    # a significant difference between a GMM fit to only this chain compared
    # to the GMM of previous solutions
    solutions = []
    for chain in trace.chain.data:
        features = np.array([trace[param].sel(chain=chain).data.flatten() for param in cluster_features]).T
        gmm = GaussianMixture(
            n_components=n_clusters,
            max_iter=100,
            init_params="random_from_data",
            n_init=10,
            verbose=False,
            random_state=seed,
        )
        gmm.fit(features)

        # Calculate multivariate z-score between this GMM and other
        # solution means to determine if this is a unique solution.
        # Each GMM could have a different label order, so we compare all
        # combinations of solution GMM cluster and this GMM cluster
        for solution in solutions:
            # cluster_zscore shape (solution clusters, GMM clusters)
            cluster_zscore = np.ones((n_clusters, n_clusters)) * np.nan
            for sol_cluster in range(n_clusters):
                for gmm_cluster in range(n_clusters):
                    # The z-score for MVnormal is mahalanobis distance
                    cov = solution["gmm"].covariances_[sol_cluster] + gmm.covariances_[gmm_cluster]
                    inv_cov = np.linalg.inv(cov)
                    zscore = mahalanobis(
                        solution["gmm"].means_[sol_cluster],
                        gmm.means_[gmm_cluster],
                        inv_cov,
                    )
                    cluster_zscore[sol_cluster, gmm_cluster] = zscore

            # calculate significance from z-score
            matched = cluster_zscore**2.0 < chi2.ppf(p_threshold, df=len(cluster_features))

            # if all GMM clusters are matched to a solution
            # cluster, then this is NOT a unique solution
            if np.all(np.any(matched, axis=0)):
                # adopt GMM labeling from matched solution
                labels = solution["gmm"].predict(features).reshape(-1, n_clusters)
                label_order = mode(labels, axis=0).mode

                # ensure all labels present
                if len(np.unique(label_order)) == len(label_order):
                    solution["label_orders"][chain] = label_order
                    break

        # Otherwise, this is a unique solution
        else:
            labels = gmm.predict(features).reshape(-1, n_clusters)
            label_order = mode(labels, axis=0).mode

            # ensure all labels present+
            if len(np.unique(label_order)) == len(label_order):
                solution = {
                    "gmm": gmm,
                    "label_orders": {chain: label_order},
                }
                solutions.append(solution)

    # Each solution now has the labeling degeneracy broken, in that
    # each cloud has been assigned to a unique GMM cluster. We must
    # now determine which order of GMM clusters is preferred
    good_solutions = []
    for solution in solutions:
        label_orders = np.array([label_order for label_order in solution["label_orders"].values()])
        # no chains have unique feature labels, abort!
        # if len(label_orders) == 0:
        #     continue
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
            solution["cloud_orders"][chain] = xorder[np.searchsorted(label_order[xorder], solution["label_order"])]

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
                        samples.sel(chain=chain, cloud=cloud_order).assign_coords(cloud=range(n_clusters))
                        for chain, cloud_order in solution["cloud_orders"].items()
                    ],
                    dim="chain",
                )
            else:
                posterior_clustered[param] = xarray.concat(
                    [samples.sel(chain=chain) for chain in solution["cloud_orders"].keys()],
                    dim="chain",
                )
            dims[param] = list(samples.dims)
        solution["posterior_clustered"] = posterior_clustered
        solution["coords"] = coords
        solution["dims"] = dims
        good_solutions.append(solution)
    return good_solutions
