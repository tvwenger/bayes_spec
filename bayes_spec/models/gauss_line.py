"""
gauss_line.py
Defines GaussLine, a Gaussian line profile model.

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

from typing import Iterable

import numpy as np
import pymc as pm

from bayes_spec import BaseModel
from bayes_spec.utils import gaussian


class GaussLine(BaseModel):
    """
    Definition of a Gaussian line profile model.
    """

    def __init__(self, *args, **kwargs):
        """
        Define model parameters, deterministic quantities, posterior
        clustering features, and TeX parameter representations.

        Inputs: see bayes_spec.BaseModel

        Returns: new GaussModel instance
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Define (normalized) cloud free parameter names
        self.cloud_params += [
            "amplitude_norm",
            "fwhm_norm",
            "velocity_norm",
        ]

        # Define deterministic quantities (including un-normalized cloud free parameters)
        self.deterministics += [
            "amplitude",
            "fwhm",
            "velocity",
            "line_area",
        ]

        # Select features used for posterior clustering
        self._cluster_features += [
            "velocity",
            "line_area",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "amplitude": r"$T_B$ (K)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "line_area": r"$\int\!T_B\,dV$ (K km s$^{-1}$)",
            }
        )

    def define(
        self,
        prior_amplitude: float = 10.0,
        prior_fwhm: float = 25.0,
        prior_velocity: Iterable[float] = [0.0, 25.0],
    ):
        """
        Model definition. The SpecData key must be "observation".

        Inputs:
            prior_amplitude :: scalar
                Width of the half-normal distribution Gaussian amplitude prior
            prior_fwhm :: scalar
                Mode of the k=2 gamma distribution Gaussian FWHM line width prior
            prior_vlsr :: two-element array of scalars
                Center and width of the normal distribution Gaussian centroid prior

        Returns: Nothing
        """
        # add polynomial baseline priors and evaluate the normalized baseline model
        baseline_model_norm = super().add_baseline_priors()

        with self.model:
            # Gaussian amplitude per cloud
            amplitude_norm = pm.HalfNormal(
                "amplitude_norm",
                sigma=1.0,
                dims="cloud",
            )
            amplitude = pm.Deterministic(
                "amplitude", prior_amplitude * amplitude_norm, dims="cloud"
            )

            # Gaussian FWHM line width per cloud
            fwhm_norm = pm.Gamma(
                "fwhm_norm",
                alpha=2.0,
                beta=1.0,
                dims="cloud",
            )
            fwhm = pm.Deterministic("fwhm", prior_fwhm * fwhm_norm, dims="cloud")

            # Gaussian centroid velocity per cloud
            velocity_norm = pm.Normal(
                "velocity_norm",
                mu=0.0,
                sigma=1.0,
                dims="cloud",
            )
            velocity = pm.Deterministic(
                "velocity",
                prior_velocity[0] + prior_velocity[1] * velocity_norm,
                dims="cloud",
            )

            # Deterministic line area per cloud
            _ = pm.Deterministic(
                "line_area",
                amplitude * fwhm * np.sqrt(np.pi / (4.0 * np.log(2.0))),
                dims="cloud",
            )

            # Evaluate line profile model per cloud, sum over clouds
            predicted_line = gaussian(
                self.data["observation"].spectral[:, None], amplitude, velocity, fwhm
            ).sum(axis=1)

            # Normalize line profile model
            predicted_line_norm = self.data["observation"].normalize_brightness(
                predicted_line
            )

            # Add normalized baseline model
            predicted = predicted_line_norm + baseline_model_norm["observation"]

            # Evaluate normalized likelihood
            _ = pm.Normal(
                "observation_norm",
                mu=predicted,
                sigma=1.0,
                observed=self.data["observation"].brightness_norm,
                dims="observation",
            )
