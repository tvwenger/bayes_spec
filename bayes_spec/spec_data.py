"""
spec_data.py - SpecData structure definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np


class SpecData:
    """
    SpecData defines the data structure and utility functions.
    """

    def __init__(
        self,
        spectral: list[float],
        brightness: list[float],
        noise: float | list[float],
        xlabel: str = "Spectral",
        ylabel: str = "Brightness",
    ):
        """Initialize a new `SpecData` instance.

        :param spectral: Spectaral axis
        :type spectral: list[float]
        :param brightness: Brightness data
        :type brightness: list[float]
        :param noise: Noise data. If a scalar, then the noise is assumed constant across the spectrum.
        :type noise: float | list[float]
        :param xlabel: Label for spectral axis, defaults to "Spectral"
        :type xlabel: str, optional
        :param ylabel: Label for brightness axis, defaults to "Brightness"
        :type ylabel: str, optional
        :raises ValueError: Size mis-match between :param:`spectral` and :param:`brightness`
        :raises ValueError: :param:`noise` is not a scalar and there is a size mis-match between :param:`brightness` and :param:`noise`
        """
        if len(spectral) != len(brightness):
            raise ValueError("size mismatch between brightness and spectral")
        self.spectral = spectral
        self.brightness = brightness
        self.xlabel = xlabel
        self.ylabel = ylabel

        if len(np.atleast_1d(noise)) == 1:
            self.noise = np.ones_like(brightness) * noise
        elif len(noise) != len(self.brightness):
            raise ValueError("size mismatch between brightness and noise")
        else:
            self.noise = noise

        # normalize spectral axis to unit domain [-1, 1]
        self._spectral_offset = np.mean(self.spectral)
        self._spectral_scale = np.ptp(self.spectral) / 2.0
        self.spectral_norm = self.normalize_spectral(self.spectral)

        # normalize brightness data using standard normalization
        self._brightness_offset = np.median(self.brightness)
        self._brightness_scale = np.median(self.noise)
        self.brightness_norm = self.normalize_brightness(brightness)

    def _normalize(self, x: float, offset: float, scale: float) -> float:
        """Normalize some data

        :param x: data to normalize
        :type x: float
        :param offset: Normalization offset
        :type offset: float
        :param scale: Normalization scale
        :type scale: float
        :return: Normalized data
        :rtype: float
        """
        return (x - offset) / scale

    def _unnormalize(self, norm_x: float, offset: float, scale: float) -> float:
        """Un-normalize some data

        :param norm_x: Normalized data
        :type norm_x: float
        :param offset: Normalization offset
        :type offset: float
        :param scale: Normalization scale
        :type scale: float
        :return: Un-normalized data
        :rtype: float
        """
        return norm_x * scale + offset

    def normalize_spectral(self, x: float) -> float:
        """Normalize spectral data

        :param x: Spectral data to normalize
        :type x: float
        :return: Normalized spectral data
        :rtype: float
        """
        return self._normalize(x, self._spectral_offset, self._spectral_scale)

    def unnormalize_spectral(self, norm_x: float) -> float:
        """Un-normalize spectral data

        :param norm_x: Normalized spectral data
        :type norm_x: float
        :return: Un-normalized spectral data
        :rtype: float
        """
        return self._unnormalize(norm_x, self._spectral_offset, self._spectral_scale)

    def normalize_brightness(self, x: float) -> float:
        """Normalize brightness data

        :param x: Brightness data to normalize
        :type x: float
        :return: Normalized brightness data
        :rtype: float
        """
        return self._normalize(x, self._brightness_offset, self._brightness_scale)

    def unnormalize_brightness(self, norm_x: float) -> float:
        """Un-normalize brighrtness data

        :param norm_x: Normalized brightness data
        :type norm_x: float
        :return: Un-normalized brightness data
        :rtype: float
        """
        return self._unnormalize(norm_x, self._brightness_offset, self._brightness_scale)
