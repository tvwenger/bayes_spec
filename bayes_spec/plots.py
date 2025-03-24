"""
plots.py - Plotting helper utilities.

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
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
    trace: az.InferenceData,
    var_names: list[str],
    labeller: Optional[azl.MapLabeller] = None,
    kind: str = "scatter",
    reference_values: Optional[dict] = None,
    kde_kwargs: Optional[dict] = None,
    scatter_kwargs: Optional[dict] = None,
    hexbin_kwargs: Optional[dict] = None,
    reference_values_kwargs: Optional[dict] = None,
) -> Iterable[Axes]:
    """Helper function to generate sample pair plots.

    :param trace: Samples
    :type trace: az.InferenceData
    :param var_names: Parameter names to plot
    :type var_names: list[str]
    :param labeller: `arviz` labeler, defaults to None
    :type labeller: Optional[azl.MapLabeller], optional
    :param kind: plot kind, one of "scatter", "hexbin", or "kde", defaults to "scatter"
    :type kind: str
    :param reference_values: highlight reference values, defaults to None
    :param reference_values: Optional[dict], optional
    :param kde_kwargs: keyword arguments for arviz.plot_kde(), defaults to None
    :param kde_kwargs: Optional[dict], optional
    :param scatter_kwargs: keyword arguments for plt.scatter(), defaults to None
    :param scatter_kwargs: Optional[dict], optional
    :param hexbin_kwargs: keyword arguments for plt.hexbin(), defaults to None
    :param hexbin_kwargs: Optional[dict], optional
    :param reference_values_kwargs: keyword arguments for plt.scatter(), defaults to None
    :param reference_values_kwargs: Optional[dict], optional
    :return: `matplotlib` `Axes`
    :rtype: Axes
    """
    if kde_kwargs is None:
        kde_kwargs = {
            "hdi_probs": [
                0.3,
                0.6,
                0.9,
            ],  # Plot 30%, 60% and 90% HDI contours
            "contourf_kwargs": {"cmap": "Grays"},
            "contour_kwargs": {"colors": "k"},
        }
    if scatter_kwargs is None:
        scatter_kwargs = {
            "marker": ".",
            "color": "k",
            "alpha": 0.1,
        }
    if reference_values_kwargs is None:
        reference_values_kwargs = {
            "marker": "o",
            "color": "r",
            "markersize": 10,
            "linestyle": "none",
        }

    size = int(2.0 * (len(var_names) + 1))
    textsize = int(np.sqrt(size)) + 8
    with az.rc_context(rc={"plot.max_subplots": None}):
        with warnings.catch_warnings(action="ignore"):
            axes = az.plot_pair(
                trace,
                var_names=var_names,
                combine_dims={"cloud"},
                kind=kind,
                figsize=(size, size),
                labeller=labeller,
                marginals=True,
                marginal_kwargs={"color": "k"},
                textsize=textsize,
                kde_kwargs=kde_kwargs,
                scatter_kwargs=scatter_kwargs,
                hexbin_kwargs=hexbin_kwargs,
                backend_kwargs={"layout": "constrained"},
                reference_values=reference_values,
                reference_values_kwargs=reference_values_kwargs,
            )

    # drop y-label of top left marginal
    axes[0][0].set_ylabel("")

    # drop grid lines
    for ax in axes.flatten():
        ax.grid(False)
    return axes
