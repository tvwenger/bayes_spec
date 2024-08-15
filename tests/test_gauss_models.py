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

import pytest

import numpy as np

import arviz as az
import graphviz

from bayes_spec import SpecData
from bayes_spec.models import GaussModel, GaussNoiseModel

import jax
import numpyro

jax.config.update("jax_platform_name", "cpu")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

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
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors()
    model.add_likelihood()
    brightness = model.model["observation"].eval(params)
    assert isinstance(model.graph(), graphviz.sources.Source)
    assert isinstance(model.sample_prior_predictive(), az.InferenceData)
    with pytest.raises(ValueError):
        model.good_chains()
    with pytest.raises(ValueError):
        model.sample_posterior_predictive()

    # Model with NaN data
    data = {"observation": SpecData(spectral, brightness * np.nan, noise)}
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors()
    model.add_likelihood()
    with pytest.raises(ValueError):
        model._validate()

    # Fit single-component model with VI
    data = {"observation": SpecData(spectral, brightness, noise)}
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors(prior_baseline_coeffs=[1.0])
    model.add_likelihood()
    model.fit(rel_tolerance=0.01, abs_tolerance=0.1, learning_rate=1e-2)

    # Sample single-component model
    data = {"observation": SpecData(spectral, brightness, noise)}
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors(prior_baseline_coeffs=[1.0])
    model.add_likelihood()
    model.sample(
        n_init=1000,
        tune=100,
        draws=100,
        chains=4,
        cores=4,
        init_kwargs={"rel_tolerance": 0.01, "abs_tolerance": 0.1, "learning_rate": 1e-2},
    )
    model.solve()
    assert model.unique_solution
    assert model.bic() < model.null_bic()
    assert model.bic(chain=0) < model.null_bic()
    assert isinstance(model.sample_posterior_predictive(), az.InferenceData)
    assert isinstance(model.sample_posterior_predictive(solution=0), az.InferenceData)

    # Sample single-component ordered model
    data = {"observation": SpecData(spectral, brightness, noise)}
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors(ordered=True)
    model.add_likelihood()
    model.sample(
        n_init=1000,
        tune=100,
        draws=100,
        chains=2,
        cores=2,
        init_kwargs={"rel_tolerance": 0.01, "abs_tolerance": 0.1, "learning_rate": 1e-2},
    )
    model.solve()

    # Test "auto" initialization strategy
    model.sample(init="auto", n_init=1000, tune=100, draws=100, chains=2, cores=2)

    # Sample with SMC
    model.sample_smc(chains=2, draws=100)

    # Sample with nutpie
    model.sample(nuts_sampler="nutpie", tune=100, draws=100)

    # Sample with numpyro
    model.sample(nuts_sampler="numpyro", tune=100, draws=100)

    # Sample with blackjax
    model.sample(nuts_sampler="blackjax", tune=100, draws=100)


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
    model = GaussNoiseModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors()
    model.add_likelihood()
    brightness = model.model["observation"].eval(params)

    # Fit single-component model with VI
    data = {"observation": SpecData(spectral, brightness, noise)}
    model = GaussNoiseModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors(prior_baseline_coeffs=[1.0])
    model.add_likelihood()
    model.fit(rel_tolerance=0.01, abs_tolerance=0.1, learning_rate=1e-2)
