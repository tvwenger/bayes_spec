"""
gauss_noise_model.py
Defines GaussNoiseModel, a Gaussian line profile model.

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pymc as pm

from bayes_spec.models import GaussModel


class GaussNoiseModel(GaussModel):
    """
    Definition of a Gaussian line profile model, with noise as an additional
    free parameter.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a new GaussModel instance

        :param `*args`: Arguments passed to :class:`BaseModel`
        :param `**kwargs`: Keyword arguments passed to :class:`BaseModel`
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

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
        """Add priors to the model.

        :param prior_rms: Prior distribution on spectral rms (K), where
            rms ~ HalfNormal(sigma=prior_rms)
            defaults to 1.0
        :type prior_rms: float, optional
        :param `**kwargs`: Additional keyword arguments passed to :class:`GaussModel.add_priors`
        """
        # Add GaussLine priors
        super().add_priors(**kwargs)

        # check inputs
        if not isinstance(prior_rms, float):
            raise ValueError("prior_rms must be a number")

        # Add additional priors
        with self.model:
            # Spectral rms (K)
            rms_observation_norm = pm.HalfNormal("rms_observation_norm", sigma=1.0)
            _ = pm.Deterministic("rms_observation", rms_observation_norm * prior_rms)

    def add_likelihood(self):
        """Add likelihood to the model. Data key must be "observation"."""
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
