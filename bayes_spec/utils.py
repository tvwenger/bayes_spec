"""
utils.py - Utility functions

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pytensor.tensor as pt


def gaussian(x: float, amp: float, center: float, fwhm: float) -> float:
    """Evaluate a Gaussian function

    :param x: Position at which to evaluate
    :type x: float
    :param amp: Gaussian amplitude
    :type amp: float
    :param center: Gaussian centroid
    :type center: float
    :param fwhm: Gaussian full-width at half-maximum
    :type fwhm: float
    :return: Gaussian evaluated at :param:x
    :rtype: float
    """
    return amp * pt.exp(-4.0 * pt.log(2.0) * (x - center) ** 2.0 / fwhm**2.0)
