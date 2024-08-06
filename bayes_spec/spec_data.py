"""
spec_data.py - SpecData structure definition

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
Trey Wenger - July 2024
"""

import numpy as np


class SpecData:
    """
    SpecData defines the data structure and utility functions.
    """

    def __init__(
        self, spectral, brightness, noise, xlabel="Spectral", ylabel="Brightness"
    ):
        """
        Initialize a new data structure

        Inputs:
            spectral :: 1-D array of scalars
                Spectral axis definition (e.g., velocity, frequency)
            brightness :: 1-D array of scalars
                Spectral brightness data (e.g., brightness temperature, flux density)
            noise :: scalar or 1-D array of scalars
                Spectral brightness noise (same units as brightness)
                If a scalar, the noise is assumed constant across the spectrum
            xlabel :: string
                Label for spectral axis
            ylabel :: string
                Label for brightness axis

        Returns:
            data :: SpecData
                New SpecData instance
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
        self._brightness_offset = np.mean(self.brightness)
        self._brightness_scale = np.std(self.brightness)
        self.brightness_norm = self.normalize_brightness(brightness)

    def _normalize(self, x, offset, scale):
        """
        Normalize some data.

        Inputs:
            x :: 1-D array of scalars
                Data to normalize
            offset, scale :: scalars
                Normalization parameters

        Returns:
            norm_x :: 1-D array of scalars
                Normalized data
        """
        return (x - offset) / scale

    def _unnormalize(self, norm_x, offset, scale):
        """
        Un-normalize some data.

        Inputs:
            norm_x :: 1-D array of scalars
                Data to un-normalize
            offset, scale :: scalars
                Normalization parameters

        Returns:
            x :: 1-D array of scalars
                Un-normalized data
        """
        return norm_x * scale + offset

    def normalize_spectral(self, x):
        """
        Normalize spectral axis data.

        Inputs:
            x :: 1-D array of scalars
                Spectral axis data

        Returns:
            norm_x :: 1-D array of scalars
                Normalized spectral axis data
        """
        return self._normalize(x, self._spectral_offset, self._spectral_scale)

    def unnormalize_spectral(self, norm_x):
        """
        Un-normalize spectral axis data.

        Inputs:
            norm_x :: 1-D array of scalars
                Normalized spectral axis data

        Returns:
            x :: 1-D array of scalars
                Un-normalized spectral axis data
        """
        return self._unnormalize(norm_x, self._spectral_offset, self._spectral_scale)

    def normalize_brightness(self, x):
        """
        Normalize brightness axis data.

        Inputs:
            x :: 1-D array of scalars
                brightness axis data

        Returns:
            norm_x :: 1-D array of scalars
                Normalized brightness axis data
        """
        return self._normalize(x, self._brightness_offset, self._brightness_scale)

    def unnormalize_brightness(self, norm_x):
        """
        Un-normalize brightness axis data.

        Inputs:
            norm_x :: 1-D array of scalars
                Normalized brightness axis data

        Returns:
            x :: 1-D array of scalars
                Un-normalized brightness axis data
        """
        return self._unnormalize(
            norm_x, self._brightness_offset, self._brightness_scale
        )
