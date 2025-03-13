"""
test_spec_data.py - Unit tests for spec_data.py

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from contextlib import ExitStack as does_not_raise
import pytest

import numpy as np

from bayes_spec import SpecData

_RNG = np.random.RandomState(seed=1234)


@pytest.mark.parametrize(
    "spectral, brightness, noise, expectation",
    [
        (_RNG.randn(10), _RNG.randn(10), _RNG.randn(10), does_not_raise()),
        (_RNG.randn(10), _RNG.randn(10), 1.0, does_not_raise()),
        (_RNG.randn(10), _RNG.randn(5), 1.0, pytest.raises(ValueError)),
        (_RNG.randn(10), _RNG.randn(10), _RNG.randn(15), pytest.raises(ValueError)),
    ],
)
def test_shape(spectral, brightness, noise, expectation):
    with expectation:
        _ = SpecData(spectral, brightness, noise)


def test_normalize():
    spectral = np.linspace(-5.0, 10.0, 1000)
    brightness = 10.0 * _RNG.randn(1000) + 10.0
    data = SpecData(spectral, brightness, 1.0)
    assert data.spectral_norm.max() == pytest.approx(1.0)
    assert data.spectral_norm.min() == pytest.approx(-1.0)
    assert data.unnormalize_brightness(
        data.normalize_brightness(brightness)
    ) == pytest.approx(brightness)
    assert data.unnormalize_spectral(
        data.normalize_spectral(spectral)
    ) == pytest.approx(spectral)
