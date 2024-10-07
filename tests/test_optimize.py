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


def test_optimize():
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
    data = {"observation": SpecData(spectral, brightness, noise)}

    # Test Optimize
    opt = Optimize(GaussModel, data, max_n_clouds=2, verbose=True)
    opt.add_priors()
    opt.add_likelihood()
    fit_kwargs = {
        "rel_tolerance": 0.01,
        "abs_tolerance": 0.1,
        "learning_rate": 1e-2,
    }
    sample_kwargs = {
        "chains": 4,
        "cores": 4,
        "tune": 500,
        "draws": 500,
        "init_kwargs": fit_kwargs,
        "nuts_kwargs": {"target_accept": 0.8},
    }

    # VI + MCMC
    opt.optimize(
        fit_kwargs=fit_kwargs,
        sample_kwargs=sample_kwargs,
        approx=True,
        smc=False,
    )
    assert opt.best_model.n_clouds == 1
    assert opt.null_bic == opt.models[1].null_bic()
    assert len(opt.bics) == 3

    # MCMC only
    opt.optimize(
        sample_kwargs=sample_kwargs,
        approx=False,
        smc=False,
    )

    # VI + SMC
    sample_kwargs = {
        "chains": 4,
        "draws": 500,
    }
    opt.optimize(
        fit_kwargs=fit_kwargs,
        sample_kwargs=sample_kwargs,
        approx=True,
        smc=True,
    )

    # SMC only
    opt.optimize(
        sample_kwargs=sample_kwargs,
        approx=False,
        smc=True,
    )
