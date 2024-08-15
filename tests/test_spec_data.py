"""
test_spec_data.py - Unit tests for spec_data.py

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
    assert data.brightness_norm.mean() == pytest.approx(0.0)
    assert data.brightness_norm.std() == pytest.approx(1.0)
    assert data.unnormalize_brightness(data.normalize_brightness(brightness)) == pytest.approx(brightness)
    assert data.unnormalize_spectral(data.normalize_spectral(spectral)) == pytest.approx(spectral)
