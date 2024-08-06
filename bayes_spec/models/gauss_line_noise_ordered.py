"""
gauss_line_noise_ordered.py
Defines GaussLineNoiseOrdered, a Gaussian line profile model.

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


class GaussLineNoiseOrdered(BaseModel):
    """
    Similar to GaussLineNoise, except the clouds are constrained to
    be ordered according to their velocity.
    """

    def __init__(self, *args, **kwargs):
        """
        Define model parameters, deterministic quantities, posterior
        clustering features, and TeX parameter representations.

        Inputs: see bayes_spec.BaseModel

        Returns: new GaussLine instance
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Define (normalized) cloud free parameter names
        self.cloud_params += [
            "line_area_norm",
            "fwhm_norm",
        ]

        # Define (normalized) hyper-parameter names
        self.hyper_params += [
            "rms_observation_norm",
            "velocity_offset_norm",
        ]

        # Define deterministic quantities (including un-normalized parameters)
        self.deterministics += [
            "line_area",
            "fwhm",
            "velocity",
            "amplitude",
            "rms_observation",
            "velocity",
        ]

        # Select features used for posterior clustering
        self._cluster_features += [
            "velocity",
            "line_area",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "line_area": r"$\int\!T_B\,dV$ (K km s$^{-1}$)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "amplitude": r"$T_B$ (K)",
                "rms_observation": r"rms (K)",
            }
        )

    def add_priors(
        self,
        prior_line_area: float = 100.0,
        prior_fwhm: float = 25.0,
        prior_velocity: Iterable[float] = [0.0, 25.0],
        prior_rms: float = 1.0,
        prior_baseline_coeff: float = 1.0,
    ):
        """
        Add priors to the model.

        Inputs:
            prior_line_area :: scalar
                Prior distribution on line area (K km s-1), where:
                line_area ~ Gamma(alpha=2.0, beta=1.0/prior_line_area)
            prior_fwhm :: scalar
                Prior distribution on line area (K km s-1), where:
                line_area ~ Gamma(alpha=2.0, beta=1.0/prior_line_area)
                Mode of the k=2 gamma distribution Gaussian FWHM line width prior
            prior_velocity :: two-element array of scalars
                Prior distribution on line centroid velocity (km s-1), where:
                velocity(cloud=N) ~ prior_velocity[0] + sum(velocity(cloud<N)) +
                                    Gamma(alpha=2.0, beta=1.0/prior_velocity[1])
                Thus, the velocities of clouds are ordered in increasing order.
            prior_rms :: scalar
                Prior distribution on spectral rms (K), where:
                rms ~ HalfNormal(sigma=prior_rms)
            prior_baseline_coeff :: scalar
                Prior distribution on normalized polynomial baseline coefficients, where:
                coeff ~ Normal(mu=0, sigma=prior_baseline_coeffs)

        Returns: Nothing
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeff=prior_baseline_coeff)

        with self.model:
            # Line area per cloud
            line_area_norm = pm.Gamma(
                "line_area_norm", alpha=2.0, beta=1.0, dims="cloud"
            )
            line_area = pm.Deterministic(
                "line_area", prior_line_area * line_area_norm, dims="cloud"
            )

            # FWHM line width per cloud
            fwhm_norm = pm.Gamma(
                "fwhm_norm",
                alpha=2.0,
                beta=1.0,
                dims="cloud",
            )
            fwhm = pm.Deterministic("fwhm", prior_fwhm * fwhm_norm, dims="cloud")

            # Centroid velocity per cloud
            velocity_offset_norm = pm.Gamma(
                "velocity_offset_norm", alpha=2.0, beta=1.0, dims="cloud"
            )
            velocity_offset = pm.Deterministic(
                "velocity_offset",
                velocity_offset_norm * prior_velocity[1],
                dims="cloud",
            )
            _ = pm.Deterministic(
                "velocity",
                prior_velocity[0] + pm.math.cumsum(velocity_offset),
                dims="cloud",
            )

            # Deterministic amplitude per cloud
            _ = pm.Deterministic(
                "amplitude",
                line_area / fwhm / np.sqrt(np.pi / (4.0 * np.log(2.0))),
                dims="cloud",
            )

            # Spectral rms (K)
            rms_observation_norm = pm.HalfNormal("rms_observation_norm", sigma=1.0)
            _ = pm.Deterministic("rms_observation", rms_observation_norm * prior_rms)

    def predict(self):
        """
        Predict emission spectrum from model parameters.

        Inputs: None

        Returns: predicted
            predicted :: 1-D array of scalars
                Predicted emission spectrum (K)
        """
        # Evaluate line profile model per cloud, sum over clouds
        predicted_line = gaussian(
            self.data["observation"].spectral[:, None],
            self.model["amplitude"],
            self.model["velocity"],
            self.model["fwhm"],
        ).sum(axis=1)

        # Add baseline model
        baseline_models = self.predict_baseline()
        predicted = predicted_line + baseline_models["observation"]
        return predicted

    def add_likelihood(self):
        """
        Add the likelihood to the model. The SpecData key must be "observation".

        Inputs: None
        Returns: Nothing
        """
        # Predict emission
        predicted = self.predict()

        with self.model:
            # Evaluate likelihood
            _ = pm.Normal(
                "observation",
                mu=predicted,
                sigma=self.model["rms_observation"],
                observed=self.data["observation"].brightness,
            )
