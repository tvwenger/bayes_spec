"""
gauss_line_noise.py
Defines GaussLineNoise, a Gaussian line profile model.

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

import pymc as pm

from bayes_spec.models import GaussLine


class GaussLineNoise(GaussLine):
    """
    Definition of a Gaussian line profile model, with noise as an additional
    free parameter.
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

        # Define (normalized) hyper-parameter names
        self.hyper_params += [
            "rms_observation_norm",
        ]

        # Define deterministic quantities (including un-normalized parameters)
        self.deterministics += [
            "rms_observation",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "rms_observation": r"rms (K)",
            }
        )

    def add_priors(
        self,
        prior_rms: float = 1.0,
        **kwargs,
    ):
        """
        Add priors tot he model.

        Inputs:
            prior_rms :: scalar
                Prior distribution on spectral rms (K), where:
                rms ~ HalfNormal(sigma=prior_rms)
            **kwargs :: see GaussLine.add_priors()

        Returns: Nothing
        """
        # Add GaussLine priors
        super().add_priors(**kwargs)

        # Add additional priors
        with self.model:
            # Spectral rms (K)
            rms_observation_norm = pm.HalfNormal("rms_observation_norm", sigma=1.0)
            _ = pm.Deterministic("rms_observation", rms_observation_norm * prior_rms)

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
