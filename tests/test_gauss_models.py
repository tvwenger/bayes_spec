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

noise = 1.0
spectral = np.linspace(-100.0, 100.0, 1000)
dummy_brightness = noise * _RNG.randn(1000)
params = {
    "line_area": [1000.0],
    "fwhm": [25.0],
    "velocity": [10.0],
    "baseline_observation_norm": [0.0],
}

# Simulate single-component model
dummy_data = {"observation": SpecData(spectral, dummy_brightness, noise)}
_MODEL = GaussModel(dummy_data, 1, baseline_degree=0, seed=1234, verbose=True)
_MODEL.add_priors()
_MODEL.add_likelihood()
sim_brightness = _MODEL.model["observation"].eval(params)
_DATA = {"observation": SpecData(spectral, sim_brightness, noise)}
_MODEL = GaussModel(_DATA, 1, baseline_degree=0, seed=1234, verbose=True)
_MODEL.add_priors()
_MODEL.add_likelihood()


def test_gauss_model():
    assert isinstance(_MODEL.graph(), graphviz.sources.Source)
    assert isinstance(_MODEL.sample_prior_predictive(), az.InferenceData)
    with pytest.raises(ValueError):
        _MODEL.sample_posterior_predictive()


def test_gauss_model_nan():
    # Model with NaN data
    data = {"observation": SpecData(spectral, sim_brightness * np.nan, noise)}
    model = GaussModel(data, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors()
    model.add_likelihood()
    with pytest.raises(ValueError):
        model._validate()


def test_gauss_model_vi():
    # Fit single-component model with VI
    model = GaussModel(_DATA, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors(prior_baseline_coeffs=[1.0])
    model.add_likelihood()
    model.fit(rel_tolerance=0.01, abs_tolerance=0.1, learning_rate=1e-2)


def test_gauss_model_sample():
    # Sample single-component model
    model = GaussModel(_DATA, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors(prior_baseline_coeffs=[1.0])
    model.add_likelihood()
    model.sample(
        tune=1000,
        draws=1000,
        chains=4,
        cores=4,
        init_kwargs={
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.1,
            "learning_rate": 1e-2,
            "start": {"velocity_norm": [-3.0]},
        },
    )
    model.solve()
    assert model.unique_solution
    assert model.bic() < model.null_bic()
    assert model.bic(chain=0) < model.null_bic()
    assert isinstance(model.sample_posterior_predictive(), az.InferenceData)
    assert isinstance(model.sample_posterior_predictive(solution=0), az.InferenceData)


def test_gauss_model_ordered_sample():
    # Sample single-component ordered model
    model = GaussModel(_DATA, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors(ordered=True)
    model.add_likelihood()
    model.sample(
        n_init=1000,
        tune=100,
        draws=100,
        chains=2,
        cores=2,
        init_kwargs={
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.1,
            "learning_rate": 1e-2,
        },
    )
    model.solve()


def test_gauss_model_sample_auto():
    # Test "auto" initialization strategy
    _MODEL.sample(init="auto", n_init=1000, tune=100, draws=100, chains=2, cores=2)


def test_gauss_model_sample_smc():
    # Sample with SMC
    _MODEL.sample_smc(chains=2, draws=100)


def test_gauss_model_nutpie():
    # Sample with nutpie
    _MODEL.sample(nuts_sampler="nutpie", tune=100, draws=100)


def test_gauss_model_numpyro():
    # Sample with numpyro
    _MODEL.sample(nuts_sampler="numpyro", tune=100, draws=100)


def test_gauss_model_blackjax():
    _MODEL.sample(nuts_sampler="blackjax", tune=100, draws=100)


def test_gauss_model_prior_shape():
    # Regression tests for https://github.com/tvwenger/bayes_spec/issues/43
    model = GaussModel(_DATA, 1, baseline_degree=0, seed=1234, verbose=True)
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
    # Fit single-component model with VI
    model = GaussNoiseModel(_DATA, 1, baseline_degree=0, seed=1234, verbose=True)
    model.add_priors(prior_baseline_coeffs=[1.0])
    model.add_likelihood()
    model.fit(rel_tolerance=0.01, abs_tolerance=0.1, learning_rate=1e-2)


def test_gauss_noise_model_prior_shape():
    # Regression tests for https://github.com/tvwenger/bayes_spec/issues/43
    model = GaussNoiseModel(_DATA, 1, baseline_degree=0, seed=1234, verbose=True)
    with pytest.raises(ValueError):
        model.add_priors(prior_rms=[1.0, 1.0])
