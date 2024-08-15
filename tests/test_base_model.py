"""
test_base_model.py - Unit tests for BaseModel functionality

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

from contextlib import ExitStack as does_not_raise
import pytest

import numpy as np
import pymc as pm
import arviz.labels as azl

from bayes_spec import BaseModel, SpecData

_RNG = np.random.RandomState(seed=1234)


class ModelA(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cloud_params += ["x"]
        self.hyper_params += ["y"]
        self.deterministics += ["z"]
        self._cluster_features += ["x", "z"]
        self.var_name_map.update({"x": "X", "y": "Y", "z": "Z"})


class ModelB(ModelA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_priors(self):
        super().add_baseline_priors()
        with self.model:
            x = pm.Normal("x", mu=0.0, sigma=1.0, dims="cloud")
            y = pm.Normal("y", mu=0.0, sigma=1.0)
            _ = pm.Deterministic("z", x + y)


class ModelC(ModelB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_likelihood(self):
        with self.model:
            _ = pm.Normal(
                "observation",
                mu=self.model["z"],
                sigma=self.data["observation"].noise,
                observed=self.data["observation"].brightness,
            )


def test_valid():
    spectral = np.linspace(-5.0, 10.0, 1000)
    brightness = 10.0 * _RNG.randn(1000) + 10.0
    data = {"observation": SpecData(spectral, brightness, 1.0)}

    with pytest.raises(TypeError):
        _ = ModelA(data, 1)
    with pytest.raises(TypeError):
        _ = ModelB(data, 1)
    with does_not_raise():
        _ = ModelC(data, 1)


def test_attributes():
    spectral = np.linspace(-5.0, 10.0, 1000)
    brightness = 10.0 * _RNG.randn(1000) + 10.0
    data = {"observation": SpecData(spectral, brightness, 1.0)}

    model = ModelC(data, 2, baseline_degree=3, seed=1234, verbose=True)
    assert model.n_clouds == 2
    assert model.baseline_degree == 3
    assert model.seed == 1234
    assert model.verbose is True
    assert model.data == data
    assert model._n_data == 1000
    assert model._n_params == 7
    with pytest.raises(ValueError):
        _ = model._get_unique_solution
    with pytest.raises(ValueError):
        _ = model.unique_solution
    assert isinstance(model.labeller, azl.MapLabeller)
    with pytest.raises(ValueError):
        model._validate()
