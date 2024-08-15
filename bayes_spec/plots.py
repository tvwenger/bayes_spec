"""
plots.py - Plotting helper utilities.

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

import warnings
from typing import Optional, Iterable

import arviz as az
import arviz.labels as azl

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from bayes_spec import SpecData


def plot_traces(posterior: az.InferenceData, var_names: list[str]) -> Iterable[Axes]:
    """Helper function to generate trace plots of posterior samples

    :param posterior: Posterior samples
    :type posterior: az.InferenceData
    :param var_names: Parameters to plot
    :type var_names: list[str]
    :return: `matplotlib` `Axes`
    :rtype: Axes
    """
    with az.rc_context(rc={"plot.max_subplots": None}):
        axes = az.plot_trace(
            posterior,
            var_names=var_names,
        )
    return axes


def plot_predictive(
    data: dict[str, SpecData],
    predictive: az.InferenceData,
) -> Iterable[Axes]:
    """Helper function to generate posterior predictive check plots.

    :param data: Data sets, where the key defines the name of the dataset.
    :type data: dict[str, SpecData]
    :param predictive: Predictive samples
    :type predictive: az.InferenceData
    :return: `matplotlib` `Axes`
    :rtype: Axes
    """
    fig, axes = plt.subplots(len(data), squeeze=False, layout="constrained")
    num_chains = len(predictive.chain)

    # Loop over datasets
    for idx, (key, dataset) in enumerate(data.items()):
        color = iter(plt.cm.rainbow(np.linspace(0, 1, num_chains)))
        # Loop over chains
        for chain in predictive.chain:
            c = next(color)
            # plot predictives
            outcomes = predictive[key].sel(chain=chain).data
            axes[idx][0].plot(
                dataset.spectral,
                outcomes.T,
                linestyle="-",
                color=c,
                alpha=0.5,
                linewidth=2,
            )

        # plot data
        axes[idx][0].plot(
            dataset.spectral,
            dataset.brightness,
            "k-",
        )
        axes[idx][0].set_xlabel(dataset.xlabel)
        axes[idx][0].set_ylabel(dataset.ylabel)
    return axes


def plot_pair(
    trace: az.InferenceData, var_names: list[str], labeller: Optional[azl.MapLabeller] = None
) -> Iterable[Axes]:
    """Helper function to generate sample pair plots.

    :param trace: Samples
    :type trace: az.InferenceData
    :param var_names: Parameter names to plot
    :type var_names: list[str]
    :param labeller: `arviz` labeler, defaults to None
    :type labeller: Optional[azl.MapLabeller], optional
    :return: `matplotlib` `Axes`
    :rtype: Axes
    """
    size = int(2.0 * (len(var_names) + 1))
    textsize = int(np.sqrt(size)) + 8
    with az.rc_context(rc={"plot.max_subplots": None}):
        with warnings.catch_warnings(action="ignore"):
            axes = az.plot_pair(
                trace,
                var_names=var_names,
                combine_dims={"cloud"},
                kind="kde",
                figsize=(size, size),
                labeller=labeller,
                marginals=True,
                marginal_kwargs={"color": "k"},
                textsize=textsize,
                kde_kwargs={
                    "hdi_probs": [
                        0.3,
                        0.6,
                        0.9,
                    ],  # Plot 30%, 60% and 90% HDI contours
                    "contourf_kwargs": {"cmap": "Grays"},
                    "contour_kwargs": {"colors": "k"},
                },
                backend_kwargs={"layout": "constrained"},
            )

    # drop y-label of top left marginal
    axes[0][0].set_ylabel("")
    for ax in axes.flatten():
        ax.grid(False)
    return axes
