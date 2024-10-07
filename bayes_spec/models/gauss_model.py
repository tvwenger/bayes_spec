"""
gauss_model.py
Defines GaussModel, a Gaussian line profile model.

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

import numpy as np
import pymc as pm

from bayes_spec import BaseModel
from bayes_spec.utils import gaussian


class GaussModel(BaseModel):
    """
    Definition of a Gaussian line profile model.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a new GaussModel instance

        :param `*args`: Arguments passed to :class:`BaseModel`
        :param `**kwargs`: Keyword arguments passed to :class:`BaseModel`
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

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
            }
        )

    def add_priors(
        self,
        prior_line_area: float = 100.0,
        prior_fwhm: float = 25.0,
        prior_velocity: Iterable[float] = [0.0, 25.0],
        prior_baseline_coeffs: Optional[Iterable[float]] = None,
        ordered: bool = False,
    ):
        """Add priors to the model.

        :param prior_line_area: Prior distribution on line area (K km s-1), where
            line_area ~ Gamma(alpha=2.0, beta=1.0/prior_line_area)
            defaults to 100.0
        :type prior_line_area: float, optional
        :param prior_fwhm: Prior distribution on line FWHM (km s-1), where
            fwhm ~ Gamma(alpha=2.0, beta=1.0/prior_fwhm)
            defaults to 25.0
        :type prior_fwhm: float, optional
        :param prior_velocity: Prior distribution on line centroid velocity (km s-1), where
            velocity ~ Normal(mu=prior_velocity[0], sigma=prior_velocity[1]) if :param:ordered is `False`
            velocity(cloud=N) ~ prior_velocity[0] + sum(velocity(cloud<N)) + Gamma(alpha=2.0, beta=1.0/prior_velocity[1]) if :param:ordered is `True`
            defaults to [0.0, 25.0]
        :type prior_velocity: Iterable[float], optional
        :param prior_baseline_coeffs: Width of normal prior distribution on the normalized baseline polynomial
            coefficients. If None, use `[1.0]*(baseline_degree+1)`, defaults to None
        :type prior_baseline_coeff: float, optional
        :param ordered: If True, assume ordered velocities, defaults to False
        :type ordered: bool
        """
        # check inputs
        if not isinstance(prior_line_area, float):
            raise ValueError("prior_line_area must be a number")
        if not isinstance(prior_fwhm, float):
            raise ValueError("prior_fwhm must be a number")
        if not isinstance(prior_velocity, list) or len(prior_velocity) != 2:
            raise ValueError("prior_velocity must be a list of two numbers")
        if prior_baseline_coeffs is not None:
            if not isinstance(prior_baseline_coeffs, list) or len(prior_baseline_coeffs) != self.baseline_degree + 1:
                raise ValueError("prior_baseline_coeffs must be a list of length baseline_degree + 1")

        # add polynomial baseline priors
        if prior_baseline_coeffs is not None:
            prior_baseline_coeffs = {"observation": prior_baseline_coeffs}
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # Line area per cloud
            line_area_norm = pm.Gamma("line_area_norm", alpha=2.0, beta=1.0, dims="cloud")
            line_area = pm.Deterministic("line_area", prior_line_area * line_area_norm, dims="cloud")

            # FWHM line width per cloud
            fwhm_norm = pm.Gamma("fwhm_norm", alpha=2.0, beta=1.0, dims="cloud")
            fwhm = pm.Deterministic("fwhm", prior_fwhm * fwhm_norm, dims="cloud")

            # Centroid velocity per cloud
            if ordered:
                velocity_norm = pm.Gamma("velocity_norm", alpha=2.0, beta=1.0, dims="cloud")
                velocity_offset = velocity_norm * prior_velocity[1]
                _ = pm.Deterministic("velocity", prior_velocity[0] + pm.math.cumsum(velocity_offset), dims="cloud")
            else:
                velocity_norm = pm.Normal("velocity_norm", mu=0.0, sigma=1.0, dims="cloud")
                _ = pm.Deterministic("velocity", prior_velocity[0] + prior_velocity[1] * velocity_norm, dims="cloud")

            # Deterministic amplitude per cloud
            _ = pm.Deterministic("amplitude", line_area / fwhm / np.sqrt(np.pi / (4.0 * np.log(2.0))), dims="cloud")

    def predict(self) -> Iterable[float]:
        """Predict observed spectrum from model parameters.

        :return: Predicted spectrum
        :rtype: Iterable[float]
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
        """Add likelihood to the model. Data key must be "observation"."""
        # Predict emission
        predicted = self.predict()

        with self.model:
            # Evaluate likelihood
            _ = pm.Normal(
                "observation",
                mu=predicted,
                sigma=self.data["observation"].noise,
                observed=self.data["observation"].brightness,
            )
