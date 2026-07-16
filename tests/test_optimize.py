"""
test_optimize.py - Unit tests for Optimize functionality

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np

from bayes_spec import SpecData, Optimize
from bayes_spec.models import GaussModel

_RNG = np.random.RandomState(seed=1234)

noise = 1.0
spectral = np.linspace(-100.0, 100.0, 1000)
dummy_brightness = noise * _RNG.randn(1000)
dummy_data = {"observation": SpecData(spectral, dummy_brightness, noise)}
params = {
    "line_area": [150.0],
    "fwhm": [25.0],
    "velocity": [10.0],
    "baseline_observation_norm": [0.0],
}
_MODEL = GaussModel(dummy_data, 1, baseline_degree=0, seed=1234, verbose=True)
_MODEL.add_priors()
_MODEL.add_likelihood()
brightness = _MODEL.model["observation"].eval(params)
_DATA = {"observation": SpecData(spectral, brightness, noise)}


def test_fit_all():
    # Test Optimize.fit_all
    opt = Optimize(GaussModel, _DATA, max_n_clouds=1, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    start_spread = {"velocity_norm": [-3.0, 3.0]}
    opt.fit_all(
        start_spread=start_spread,
        **{
            "n": 1000,
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.1,
            "learning_rate": 1e-2,
        },
    )


def test_sample_all():
    # Test Optimize.sample_all
    opt = Optimize(GaussModel, _DATA, max_n_clouds=1, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    start_spread = {"velocity_norm": [-3.0, 3.0]}
    fit_kwargs = {
        "rel_tolerance": 0.01,
        "abs_tolerance": 0.1,
        "learning_rate": 1e-2,
    }
    opt.sample_all(
        start_spread=start_spread,
        sample_kwargs={
            "tune": 100,
            "draws": 100,
            "chains": 2,
            "cores": 2,
            "n_init": 1000,
            "init_kwargs": fit_kwargs,
        },
    )


def test_sample_smc_all():
    # Test Optimize.sample_smc_all
    opt = Optimize(GaussModel, _DATA, max_n_clouds=1, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    opt.sample_smc_all(sample_kwargs={"chains": 2, "draws": 100})


def test_optimize_vi_mcmc():
    # Test Optimize with VI + MCMC
    opt = Optimize(GaussModel, _DATA, max_n_clouds=5, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    start_spread = {"velocity_norm": [-3.0, 3.0]}
    fit_kwargs = {
        "rel_tolerance": 0.01,
        "abs_tolerance": 0.1,
        "learning_rate": 1e-2,
    }
    opt.optimize(
        start_spread=start_spread,
        approx=True,
        smc=False,
        fit_kwargs={
            "n": 10_000,
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.1,
            "learning_rate": 1e-2,
        },
        sample_kwargs={
            "n_init": 10_000,
            "init_kwargs": fit_kwargs,
            "chains": 2,
            "tune": 100,
            "draws": 100,
        },
    )
    assert opt.best_model.n_clouds == 1
    assert opt.null_bic == opt.models[1].null_bic()
    assert len(opt.bics) == 6


def test_optimize_mcmc():
    # MCMC only
    opt = Optimize(GaussModel, _DATA, max_n_clouds=1, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    start_spread = {"velocity_norm": [-3.0, 3.0]}
    fit_kwargs = {
        "rel_tolerance": 0.01,
        "abs_tolerance": 0.1,
        "learning_rate": 1e-2,
    }
    opt.optimize(
        start_spread=start_spread,
        approx=False,
        smc=False,
        sample_kwargs={
            "n_init": 10_000,
            "init_kwargs": fit_kwargs,
            "chains": 2,
            "tune": 500,
            "draws": 500,
        },
    )


def test_optimize_vi_smc():
    # VI + SMC
    opt = Optimize(GaussModel, _DATA, max_n_clouds=1, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    sample_kwargs = {
        "chains": 2,
        "draws": 100,
    }
    opt.optimize(
        sample_kwargs=sample_kwargs,
        approx=True,
        smc=True,
        fit_kwargs={
            "n": 10_000,
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.1,
            "learning_rate": 1e-2,
        },
    )


def test_optimize_smc():
    # SMC only
    opt = Optimize(GaussModel, _DATA, max_n_clouds=1, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    sample_kwargs = {
        "chains": 2,
        "draws": 100,
    }
    opt.optimize(
        sample_kwargs=sample_kwargs,
        approx=False,
        smc=True,
    )
