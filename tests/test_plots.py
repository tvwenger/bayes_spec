"""
test_gauss_models.py - Unit tests for GaussLine models and
bayes_spec sampling functions

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

import numpy as np

from matplotlib.axes import Axes

from bayes_spec import SpecData
from bayes_spec.plots import plot_predictive, plot_pair, plot_traces
from bayes_spec.models import GaussModel

_RNG = np.random.RandomState(seed=1234)


def test_plots():
    # Simulate single-component model
    noise = 1.0
    spectral = np.linspace(-100.0, 100.0, 1000)
    brightness = noise * _RNG.randn(1000)
    data = {"observation": SpecData(spectral, brightness, noise)}
    params = {
        "line_area": [1000.0],
        "fwhm": [25.0],
        "velocity": [10.0],
        "observation_baseline_norm": [0.0],
    }
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors()
    model.add_likelihood()
    brightness = model.model["observation"].eval(params)

    # Test plotting functions
    data = {"observation": SpecData(spectral, brightness, noise)}
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors()
    model.add_likelihood()
    model.sample(
        chains=2,
        cores=2,
        init_kwargs={"rel_tolerance": 0.01, "abs_tolerance": 0.1, "learning_rate": 1e-2},
    )
    model.solve()
    posterior = model.sample_posterior_predictive()
    assert isinstance(plot_predictive(model.data, posterior.posterior_predictive).ravel()[0], Axes)
    assert isinstance(plot_pair(model.trace.solution_0, model.deterministics, labeller=model.labeller).ravel()[0], Axes)
    assert isinstance(plot_traces(model.trace.solution_0, model.deterministics).ravel()[0], Axes)
