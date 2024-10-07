"""
test_gauss_models.py - Unit tests for GaussLine models and
bayes_spec sampling functions

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
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
        "baseline_observation_norm": [0.0],
    }
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors()
    model.add_likelihood()
    brightness = model.model["observation"].eval(params)
    assert isinstance(model.graph(), graphviz.sources.Source)
    assert isinstance(model.sample_prior_predictive(), az.InferenceData)
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
        tune=1000,
        draws=1000,
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

    # Regression tests for https://github.com/tvwenger/bayes_spec/issues/43
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    with pytest.raises(ValueError):
        model.add_priors(prior_line_area=[1.0, 1.0])
    with pytest.raises(ValueError):
        model.add_priors(prior_fwhm=[1.0, 1.0])
    with pytest.raises(ValueError):
        model.add_priors(prior_velocity=10.0)
    with pytest.raises(ValueError):
        model.add_priors(prior_velocity=[10.0])
    with pytest.raises(ValueError):
        model.add_priors(prior_baseline_coeffs=0.0)
    with pytest.raises(ValueError):
        model.add_priors(prior_baseline_coeffs={"observation": [1.0, 2.0, 3.0]})


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
        "baseline_observation_norm": [0.0],
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

    # Regression tests for https://github.com/tvwenger/bayes_spec/issues/43
    model = GaussNoiseModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    with pytest.raises(ValueError):
        model.add_priors(prior_rms=[1.0, 1.0])
