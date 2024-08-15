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

Changelog:
Trey Wenger - August 2024
"""

import pytest

import numpy as np

from bayes_spec import SpecData
from bayes_spec.models import GaussModel, GaussNoiseModel

_RNG = np.random.RandomState(seed=1234)


def test_gauss_model():
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
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=False)
    model.add_priors()
    model.add_likelihood()
    brightness = model.model["observation"].eval(params)

    # Model with NaN data
    data = {"observation": SpecData(spectral, brightness * np.nan, noise)}
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=False)
    model.add_priors()
    model.add_likelihood()
    with pytest.raises(ValueError):
        model._validate()

    # Fit single-component model with VI
    data = {"observation": SpecData(spectral, brightness, noise)}
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=False)
    model.add_priors(prior_baseline_coeffs=[1.0])
    model.add_likelihood()
    model.fit()

    # Sample single-component model
    data = {"observation": SpecData(spectral, brightness, noise)}
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=False)
    model.add_priors(prior_baseline_coeffs=[1.0])
    model.add_likelihood()
    model.sample(chains=2, cores=2)
    model.solve()

    # Sample single-component ordered model
    data = {"observation": SpecData(spectral, brightness, noise)}
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=False)
    model.add_priors(ordered=True)
    model.add_likelihood()
    model.sample(chains=2, cores=2)
    model.solve()

    # Test "auto" initialization strategy
    model.sample(init="auto")


def test_gauss_noise_model():
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
        "rms_observation": noise,
    }
    model = GaussNoiseModel(data, 1, baseline_degree=0, seed=1234, verbose=False)
    model.add_priors()
    model.add_likelihood()
    brightness = model.model["observation"].eval(params)

    # Fit single-component model with VI
    data = {"observation": SpecData(spectral, brightness, noise)}
    model = GaussNoiseModel(data, 1, baseline_degree=0, seed=1234, verbose=False)
    model.add_priors(prior_baseline_coeffs=[1.0])
    model.add_likelihood()
    model.fit()
