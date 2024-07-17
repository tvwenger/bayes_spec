"""
utils.py - Utility functions

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

Changelog:
Trey Wenger - June 2024
"""

import pytensor.tensor as pt


def gaussian(x: float, amp: float, center: float, fwhm: float) -> float:
    """
    Evaluate a Gaussian.

    Inputs:
        x :: scalar
            Position at which to evaluate
        amp, center, fwhm :: scalars
            Gaussian parameters

    Returns: y
        y :: scalar
            Evaluated Gaussian at x
    """
    return amp * pt.exp(-4.0 * pt.log(2.0) * (x - center) ** 2.0 / fwhm**2.0)
