Creating ``bayes_spec`` Models
==============================

This step-by-step guide describes how to create a ``bayes_spec`` model.

Preliminary Steps
-----------------

Before we actually write any code, we must decide how we are going to *parameterize* the model. Things to decide first include:

* What are the spectral line data sets we aim to model? (Data specification)

* What physical conditions do we hope to infer? (Parameter specification)

* What other parameters might be important? (Hyperparameter specification)

* What is the relationship between those parameters and the data? (Likelihood specification)

In general, if you can write down the equations needed to *simulate* the given spectral line data, then you have everything you need to create a ``bayes_spec`` model.

For this guide, we will reproduce the ``GaussModel`` provided by ``bayes_spec``. Our spectral line data is some emission line brightness temperature versus velocity spectrum (data specification), our model parameters are the Gaussian parameters that define the emission line profile (parameter specification) as well as a polynomial baseline (hyperparameter specification), and the relationship between the parameters and the data is simply a Gaussian line profile shape.

We will call the Gaussian line profile parameters ``line_area``, the Gaussian line area, ``fwhm``, the Gaussian full-width at half-maximum line width, and ``velocity``, the line-center velocity. From these free parameters we can derive the Gaussian amplitude, ``amplitude``.

We must also decide how our data will be named and accessed within the model. We choose the name "observation" for our ``SpecData`` key, so our dataset supplied to the model must have key "observation":

.. code-block:: python

    from bayes_spec import SpecData

    data = {"observation": SpecData(
        velocity_axis,
        brightness_data,
        noise,
        xlabel=r"Velocity (km s$^{-1}$)",
        ylabel="Brightness Temperature (K)",
    )}

Model Structure
---------------

All ``bayes_spec`` models are implemented as ``python`` classes that extend the ``BaseModel`` provided by ``bayes_spec``. Models can also extend existing models (e.g., ``GaussLineNoise`` extends ``GaussLine`` to provide additional functionality). In either case, the basic format of a model is the following:

.. code-block:: python

    import pymc as pm
    from bayes_spec import BaseModel

    class GaussModel(BaseModel):
        """Definition of the GaussModel"""

        def __init__(self, *args, **kwargs):
            """Initialize a new GaussModel instance"""
            pass

        def add_priors(self, *args, **kwargs):
            """Add priors to the model"""
            pass

        def add_likelihood(self, *args, **kwargs):
            """Add likelihood to the model."""
            pass

Any ``bayes_spec`` model must include these three functions: ``__init__()``, which initializes the model, ``add_priors()``, which adds priors to the model, and ``add_likelihood()``, which adds the likelihood to the model.

``__init__()``
--------------

The model initialization function must include two parts. The first simply initializes the parent model.

.. code-block:: python

    # Initialize BaseModel
    super().__init__(*args, **kwargs)

Next, we must specify which model parameters will be used for clustering the posterior samples. For efficiency, we should specify the minimum number of parameters that will uniquely identify solutions in the posterior distribution. That is, we should only select parameters that are well-constrained. Note that we could choose to cluster on free parameters (i.e., ``line_area``) or on derived quantities (i.e., ``amplitude``). Here we expect the ``line_area`` and ``velocity`` to be well-constrained, so we cluster on those features.

.. code-block:: python

    # Select features used for posterior clustering
    self._cluster_features += ["velocity", "line_area"]

Finally, we may optionally supply string representations for the model parameters. This is useful to generate LaTeX symbols in the various plots produced by ``bayes_spec``.

.. code-block:: python

    # Define TeX representation of each parameter
    self.var_name_map.update({
        "line_area": r"$\int\!T_B\,dV$ (K km s$^{-1}$)",
        "fwhm": r"$\Delta V$ (km s$^{-1}$)",
        "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
        "amplitude": r"$T_B$ (K)",
    })

Thus our complete ``__init__()`` function looks like this:

.. code-block:: python

    def __init__(self, *args, **kwargs):
        """Initialize a new GaussModel instance"""
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Select features used for posterior clustering
        self._cluster_features += ["velocity", "line_area"]

        # Define TeX representation of each parameter
        self.var_name_map.update({
            "line_area": r"$\int\!T_B\,dV$ (K km s$^{-1}$)",
            "fwhm": r"$\Delta V$ (km s$^{-1}$)",
            "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
            "amplitude": r"$T_B$ (K)",
        })

``add_priors()``
----------------

Next, we must specify the prior distributions on the various model parameters. We do so on the ``add_priors()`` function of our model class. The specifics of how these priors are specified are up you. Some users may simply hard-code the prior distributions in ``add_priors()``, whereas other users may wish to pass parameters to this function in order to adapt their prior distributions to a given dataset. In this case, we will allow the user to specify the shape parameters of the hard-coded prior distributions.

Choosing good prior distributions is an imperative part of Bayesian modeling. Here are three guiding principles:

* Choose physically-allowed prior distributions. If a parameter should not be negative, then ensure that the prior disallows negative values.

* Choose physically-motivated prior distributions. Help MCMC find a good solution by limiting the parameter space as much as possible.

* Normalize your prior distributions. In general for Bayesian modeling and Monte Carlo Markov Chain (MCMC) analyses, it is good practice to normalize the free parameters of a model. MCMC samplers are more efficient when the scale of the various free parameters are similar.

Here we choose the following prior distributions for the free parameters:

* ``line_area``: Gamma distribution with ``alpha=2.0`` is a good choice because it has zero probability density for negative values

* ``fwhm``: Gamma distribution with ``alpha=2.0`` is a good choice because it has zero probability density for negative values

* ``velocity``: Normal distribution

Each of these are "centered" distributions, meaning that changing the *scale* of the distribution is as easy as multiplying samples from those distributions by some scale factor. We can thus define normalized versions of these distributions with a unit scale factor and then alter these normalized distributions into our actual parameter prior distributions.

Creating prior distributions follows the usual ``pymc`` syntax. Notably, any new distributions must be added within a ``with self.model`` block. See `the pymc documentation <https://www.pymc.io/projects/docs/en/stable/api/distributions.html>`_ for more information about the available distributions.

Any derived quantities that you wish to track or that must be used outside of the ``add_priors()`` function (i.e., needed in ``add_likelihood()``) must be wrapped in ``pm.Deterministic()``. Furthermore, cloud-based parameters should have ``dims="cloud"`` to indicate that there is one parameter per cloud.

Internally, ``bayes_spec`` models the baseline structure using polynomial functions. The ``add_priors()`` function must add the priors on the polynomial baseline coefficients via ``super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)`` where ``prior_baseline_coeffs`` can either be ``None``, in which case the normalized coefficient prior distributions all have unit variance, or a dictionary with keys matching the ``SpecData`` and values with the prior width of each baseline coefficient (length ``baseline_degree + 1``). Thus, for ``baseline_degree = 1`` and ``SpecData`` key ``"observation"``, ``prior_baseline_coeffs`` must either be ``None`` or a dictionary like ``{"observation": [1.0, 1.0]}``.

.. code-block:: python

    def add_priors(self,
        prior_line_area = 100.0,
        prior_fwhm = 25.0,
        prior_velocity = [0.0, 25.0],
        prior_baseline_coeffs = None,
    ):
        """Add priors to the model"""
        # add polynomial baseline priors
        if prior_baseline_coeffs is not None:
            prior_baseline_coeffs = {"observation": prior_baseline_coeffs}
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # Line area
            line_area_norm = pm.Gamma("line_area_norm", alpha=2.0, beta=1.0, dims="cloud")
            line_area = pm.Deterministic("line_area", prior_line_area * line_area_norm, dims="cloud")

            # FWHM line width
            fwhm_norm = pm.Gamma("fwhm_norm", alpha=2.0, beta=1.0, dims="cloud")
            fwhm = pm.Deterministic("fwhm", prior_fwhm * fwhm_norm, dims="cloud")

            # Center velocity
            velocity_norm = pm.Normal("velocity_norm", mu=0.0, sigma=1.0, dims="cloud")
            _ = pm.Deterministic("velocity", prior_velocity[0] + prior_velocity[1] * velocity_norm, dims="cloud")

            # Amplitude
            _ = pm.Deterministic("amplitude", line_area / fwhm / np.sqrt(np.pi / (4.0 * np.log(2.0))), dims="cloud")

The first argument of the ``pymc`` distributions or ``Deterministic`` is simply the internal parameter name. These parameters can be accessed in other functions via, for example, ``self.model["amplitude"]``.

``add_likelihood()``
--------------------

Finally, the model must relate the model parameters to the data and evaluate the likelihood. Generally, this is as easy as writing down the *forward model* equations that produce the observed spectral line data. In our case, these equations are simply that of a Gaussian line profile and the polynomial baseline.

Note that mathematical operations on ``pymc`` variables must be implemented via ``pytensor.tensor`` operations. Most ``numpy`` functions are handled implicitly by ``pymc``, but in general it is best to use ``pytensor.tensor`` operations whenever possible. For example, here is the equation of a Gaussian line profile:

.. code-block:: python

    import pytensor.tensor as pt

    def gaussian(x, amp, center, fwhm):
        """Evaluate a Gaussian function"""
        return amp * pt.exp(-4.0 * pt.log(2.0) * (x - center) ** 2.0 / fwhm**2.0)

The ``add_likelihood()`` function then calculates the model-predicted spectra. We also evaluate the polynomial baseline via ``self.predict_baseline()``, which returns a dictionary of baseline models indexed by the ``SpecData`` key(s). Finally, we evaluate the likelihood under the assumption of Normally-distributed noise. The likelihood distribution is identified by the ``observed`` keyword argument.

.. code-block:: python

        def add_likelihood(self):
        """Add likelihood to the model. Data key must be "observation"."""
            # Predict emission, summed over cloud components
            predicted_line = gaussian(
                self.data["observation"].spectral[:, None],
                self.model["amplitude"],
                self.model["velocity"],
                self.model["fwhm"],
            ).sum(axis=1)

            # Baseline model
            baseline_models = self.predict_baseline()
            predicted = predicted_line + baseline_models["observation"]

            with self.model:
                # Evaluate likelihood
                _ = pm.Normal(
                    "observation",
                    mu=predicted,
                    sigma=self.data["observation"].noise,
                    observed=self.data["observation"].brightness,
                )

Complete Model
--------------

Here is our complete example model.

.. code-block:: python

    import pymc as pm
    import pytensor.tensor as pt
    from bayes_spec import BaseModel

    def gaussian(x, amp, center, fwhm):
        """Evaluate a Gaussian function"""
        return amp * pt.exp(-4.0 * pt.log(2.0) * (x - center) ** 2.0 / fwhm**2.0)

    class GaussModel(BaseModel):
        """Definition of the GaussModel"""

        def __init__(self, *args, **kwargs):
            """Initialize a new GaussModel instance"""
            # Initialize BaseModel
            super().__init__(*args, **kwargs)

            # Select features used for posterior clustering
            self._cluster_features += ["velocity", "line_area"]

            # Define TeX representation of each parameter
            self.var_name_map.update({
                "line_area": r"$\int\!T_B\,dV$ (K km s$^{-1}$)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "amplitude": r"$T_B$ (K)",
            })

        def add_priors(self,
            prior_line_area = 100.0,
            prior_fwhm = 25.0,
            prior_velocity = [0.0, 25.0],
            prior_baseline_coeffs = None,
        ):
            """Add priors to the model"""
            # add polynomial baseline priors
            if prior_baseline_coeffs is not None:
                prior_baseline_coeffs = {"observation": prior_baseline_coeffs}
            super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

            with self.model:
                # Line area
                line_area_norm = pm.Gamma("line_area_norm", alpha=2.0, beta=1.0, dims="cloud")
                line_area = pm.Deterministic("line_area", prior_line_area * line_area_norm, dims="cloud")

                # FWHM line width
                fwhm_norm = pm.Gamma("fwhm_norm", alpha=2.0, beta=1.0, dims="cloud")
                fwhm = pm.Deterministic("fwhm", prior_fwhm * fwhm_norm, dims="cloud")

                # Center velocity
                velocity_norm = pm.Normal("velocity_norm", mu=0.0, sigma=1.0, dims="cloud")
                _ = pm.Deterministic("velocity", prior_velocity[0] + prior_velocity[1] * velocity_norm, dims="cloud")

                # Amplitude
                _ = pm.Deterministic("amplitude", line_area / fwhm / np.sqrt(np.pi / (4.0 * np.log(2.0))), dims="cloud")

        def add_likelihood(self):
        """Add likelihood to the model. Data key must be "observation"."""
            # Predict emission, summed over cloud components
            predicted_line = gaussian(
                self.data["observation"].spectral[:, None],
                self.model["amplitude"],
                self.model["velocity"],
                self.model["fwhm"],
            ).sum(axis=1)

            # Baseline model
            baseline_models = self.predict_baseline()
            predicted = predicted_line + baseline_models["observation"]

            with self.model:
                # Evaluate likelihood
                _ = pm.Normal(
                    "observation",
                    mu=predicted,
                    sigma=self.data["observation"].noise,
                    observed=self.data["observation"].brightness,
                )

Checking Model Specification
----------------------------

Writing the model is only the first step! Once your model is written, you should check that all of the parameters and distributions have been specified correctly. Some additional tips and guidance are provided in :doc:`Tips & Tricks <tips>` , but in general we recommend:

* Simulating synthetic observations from your model, following the guide in the :doc:`Basic Tutorial <notebooks/basic_tutorial>`

* Generating prior predictive checks

* Testing MCMC results against synthetic observations with known model parameters