"""
test_optimize.py - Unit tests for Optimize functionality

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
        "observation_baseline_norm": [0.0],
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
        "chains": 2,
        "cores": 2,
        "tune": 100,
        "draws": 100,
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

    # MCMC only
    opt.optimize(
        sample_kwargs=sample_kwargs,
        approx=False,
        smc=False,
    )

    # VI + SMC
    sample_kwargs = {
        "chains": 2,
        "draws": 100,
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
