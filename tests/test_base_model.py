"""
test_base_model.py - Unit tests for BaseModel functionality

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
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
        self._cluster_features += ["x", "z"]
        self.var_name_map.update({"x": "X", "y": "Y", "z": "Z"})


class ModelB(ModelA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_priors(self, prior_baseline_coeffs=None):
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)
        with self.model:
            x = pm.Normal("x", mu=0.0, sigma=1.0, dims="cloud")
            y = pm.Normal("y", mu=0.0, sigma=1.0)
            _ = pm.Deterministic("z", x + y, dims="cloud")


class ModelC(ModelB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_likelihood(self):
        with self.model:
            _ = pm.Normal(
                "observation",
                mu=np.ones_like(self.data["observation"].spectral)
                * self.model["z"].sum(),
                sigma=self.data["observation"].noise,
                observed=self.data["observation"].brightness,
            )


spectral = np.linspace(-5.0, 10.0, 1000)
brightness = 10.0 * _RNG.randn(1000) + 10.0
data = {"observation": SpecData(spectral, brightness, 1.0)}


def test_valid():
    with pytest.raises(TypeError):
        _ = ModelA(data, 1)
    with pytest.raises(TypeError):
        _ = ModelB(data, 1)
    with does_not_raise():
        _ = ModelC(data, 1)


def test_attributes():
    model = ModelC(data, 2, baseline_degree=3, seed=1234, verbose=True)
    model.add_priors()
    with pytest.raises(ValueError):
        model._validate()
    model.add_likelihood()
    assert model.n_clouds == 2
    assert model.baseline_degree == 3
    assert model.seed == 1234
    assert model.verbose is True
    assert model.data == data
    assert model.baseline_freeRVs == ["baseline_observation_norm"]
    assert model.baseline_deterministics == []
    assert model.cloud_freeRVs == ["x"]
    assert model.cloud_deterministics == ["z"]
    assert model.hyper_freeRVs == ["y"]
    assert model.hyper_deterministics == []
    assert model._n_data == 1000
    assert model._n_params == 7
    with pytest.raises(ValueError):
        _ = model._get_unique_solution
    assert not model.unique_solution
    assert isinstance(model.labeller, azl.MapLabeller)
    assert model._validate()


def test_baseline():
    model = ModelC(data, 2, baseline_degree=3, seed=1234, verbose=True)
    model.add_priors()
    model.add_likelihood()

    baseline_params = {
        "baseline_observation_norm": [0.0, 0.0, 0.0, 0.0],
    }
    baseline_model = model.predict_baseline(baseline_params=baseline_params)
    assert len(baseline_model["observation"].eval()) == len(spectral)

    # regression test for #24
    prior_baseline_coeffs = {"observation": [0.0]}
    with pytest.raises(ValueError):
        model.add_priors(prior_baseline_coeffs=prior_baseline_coeffs)
