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
    "line_area": [1000.0],
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
    opt.fit_all(start_spread=start_spread)


def test_sample_all():
    # Test Optimize.sample_all
    opt = Optimize(GaussModel, _DATA, max_n_clouds=1, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    start_spread = {"velocity_norm": [-3.0, 3.0]}
    opt.sample_all(start_spread=start_spread)


def test_sample_smc_all():
    # Test Optimize.sample_smc_all
    opt = Optimize(GaussModel, _DATA, max_n_clouds=1, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    opt.sample_smc_all()


def test_optimize_vi_mcmc():
    # Test Optimize with VI + MCMC
    opt = Optimize(GaussModel, _DATA, max_n_clouds=5, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    start_spread = {"velocity_norm": [-3.0, 3.0]}
    opt.optimize(
        start_spread=start_spread,
        approx=True,
        smc=False,
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
    opt.optimize(
        start_spread=start_spread,
        approx=False,
        smc=False,
    )


def test_optimize_vi_smc():
    # VI + SMC
    opt = Optimize(GaussModel, _DATA, max_n_clouds=1, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    sample_kwargs = {
        "chains": 4,
        "draws": 500,
    }
    opt.optimize(
        sample_kwargs=sample_kwargs,
        approx=True,
        smc=True,
    )


def test_optimize_smc():
    # SMC only
    opt = Optimize(GaussModel, _DATA, max_n_clouds=1, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    sample_kwargs = {
        "chains": 4,
        "draws": 500,
    }
    opt.optimize(
        sample_kwargs=sample_kwargs,
        approx=False,
        smc=True,
    )
