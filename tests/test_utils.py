"""
test_utils.py - Unit tests for utils.py

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pytest

from bayes_spec.utils import gaussian


@pytest.mark.parametrize(
    "x, amp, center, fwhm, expectation",
    [
        (0.0, 1.0, 0.0, 1.0, 1.0),
        (0.5, 1.0, 0.0, 1.0, 0.5),
        (0.0, 2.0, 1.0, 2.0, 1.0),
    ],
)
def test_gaussian(x, amp, center, fwhm, expectation):
    assert gaussian(x, amp, center, fwhm).eval() == pytest.approx(expectation)
