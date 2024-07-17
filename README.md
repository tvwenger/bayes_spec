# bayes_spec <!-- omit in toc -->
A Bayesian Spectral Line Modeling Framework for Astrophysics

`bayes_spec` is a framework for user-defined, cloud-based models of astrophysical systems (e.g., the interstellar medium) that enables spectral line simulation and statistical inference. Built in the [`pymc` probabilistic programming library](https://www.pymc.io/welcome.html), `bayes_spec` uses Monte Carlo Markov Chain techniques to fit user-defined models to data. The user-defined models can be a simple line profile (e.g., a Gaussian profile) or a complicated physical model. The models are "cloud-based", meaning there can be multiple "clouds" or "components" each with a unique set of the model parameters. `bayes_spec` includes algorithms to estimate the optimal number of components in a given dataset.

- [Installation](#installation)
- [Usage](#usage)
  - [Model Definition](#model-definition)
- [Algorithms](#algorithms)
  - [Posterior Sampling: Variational Inference](#posterior-sampling-variational-inference)
  - [Posterior Sampling: MCMC](#posterior-sampling-mcmc)
  - [Posterior Clustering: Gaussian Mixture Models](#posterior-clustering-gaussian-mixture-models)
  - [Optimization](#optimization)
- [Syntax \& Examples](#syntax--examples)
- [Issues and Contributing](#issues-and-contributing)
- [License and Copyright](#license-and-copyright)


# Installation
Preferred: install in a `conda` virtual environment:
```
conda env create -f environment.yml
conda activate bayes_spec
pip install .
```

Alternatively:
```
python setup.py install
```

If you wish to contribute to `bayes_spec`, then you may wish to install additional dependencies and install `bayes_spec` as an "editable" package:
```
conda env create -f environment-dev.yml
conda activate bayes_spec-dev
pip install -e .
```

# Usage

`bayes_spec` assumes that the source of spectral line emission can be decomposed into a series of "clouds" or "components", each of which are defined by a *unique* set of model parameters. For a simple line profile model (like a Gaussian profile), these parameters might be the Gaussian amplitude, center velocity, and line width. For a more complicated model, they might be the optical depth, excitation temperature, and velocity of the cloud. `bayes_spec` also supports hyper-parameters: parameters that influence both non-cloud based features in the data (e.g., spectral baseline structure) as well as overall cloud properties (e.g., a property assumed shared among all clouds) or physical relationships (e.g., an empirical scaling law).

*Users are responsible for defining and testing their models.* Here we briefly describe how to do this, but see [Syntax \& Examples](#syntax--examples) for some practical demonstrations.

Users must specify the following:

1. The data
2. Model parameters and hyper-parameters
3. The parameter and hyper-parameter prior distributions
4. Derived quantities (called "deterministics") that should be saved (e.g., for inference)
5. The relationship between the model parameters and spectral observations (i.e., the likelihood)

## Data Format

Data must be packaged within a `bayes_spec.SpecData` object. `SpecData` takes three arguments: the spectral axis (i.e., frequency, velocity), the brightness data (e.g., brightness temperature, flux density), and the noise (in the same units as the brightness data). The noise can either be a scalar value, in which case it is assumed constant across the spectrum, or an array of the same length as the brightness_data.

```python
from bayes_spec import SpecData

spec = SpecData(spectral_axis, brightness_data, noise, xlabel="Velocity", ylabel="Brightness Temperature")
```

The spectral and brightness data are accessed via `spec.spectral` and `spec.brightness`, respectively.

The data are passed to a `bayes_spec` model in a dictionary, which allows multiple spectra to be included in a single model. For example, if a model is constrained by both an emission-line spectrum and an absorption-line spectrum, then this dictionary might look something like this:

```python
emission = SpecData(emission_spec_axis, emission_data, emission_noise, xlabel="Velocity", ylabel="Brightness Temperature")
absorption = SpecData(absorption_spec_axis, absorption_data, absorption_noise, xlabel="Velocity", ylabel="Optical Depth")

data = {"emission": emission, "absorption": absorption}
```

The keys of this data dictionary (`"emission"` and `"absorption"` in this case) are important, you must remember them and use the same keys in your model definition. 

Internally, `SpecData` normalizes the data by the noise. We can apply the normalization or un-normalization of data via `data['emission'].normalize_brightness` and `data['emission'].unnormalize_brightness` functions, respectively. This will be important for model specification.

## Model Specification

Model specification is made though a class that extends the `bayes_spec.BaseModel` base model class definition. This class must include two methods: `__init__`, which initializes the model, and `define`, which defines the model. Here is a skeleton of what this class definition must look like:

```python
import pymc as pm
import pytensor.tensor as pt
from bayes_spec import BaseModel

# The model name "NewModel" can be anything, but it must extend BaseModel
class NewModel(BaseModel):

    # The class must include __init__ with this signature:
    def __init__(self, *args, **kwargs):
        
        # This function must initialize the BaseModel:
        super().__init__(*args, **kwargs)

        # User-defined cloud-based parameters:
        self.cloud_params += [
            "parameter1",
            "parameter2",
        ]

        # User-defined deterministic (derived) quantities
        self.deterministics += [
            "derived1",
            "derived2",
        ]

        # User-selected features (parameters or deterministics) used for posterior clustering
        self._cluster_features += [
            "parameter2",
            "derived1",
        ]

        # User defined string representation of parameters
        self.var_name_map.update(
            {
                "parameter1": r"$\alpha$",
                "parameter2": r"$\beta$",
                "derived1": r"$A$",
                "derived2": r"$B$",
            }
        )

    # The class must include a method named "define", which can have
    # arbitrary, user-defined arguments:
    def define(self, prior_parameter1,):

        # The define function must add baseline priors to the model
        # The output of the `add_baseline_priors` function is a dictionary
        # where the keys are the same keys as in the data dictionary and the
        # values are the *normalized* baseline models, where the normalization
        # is the same as that used on the data.
        baseline_model_norm = super().add_baseline_priors()

        # Here the user specifies the model
        # This includes priors and the likelihood
        with self.model:

            # Define priors using standard pymc distributions and syntax.
            # dims="cloud" for cloud-based parameters
            parameter1 = pm.Normal("parameter1", mu=0, sigma=prior_parameter1, dims="cloud")
            parameter2 = pm.Normal("parameter2", mu=0, sigma=1, dims="cloud")

            # Derived parameters can be calculated using external functions,
            # ideally written with pytensor
            derived1 = pt.sum([parameter1, parameter2])
            derived2 = my_external_function(parameter1, parameter2)

            # The user must specify the relationship between the model parameters
            # and the observed spectrum/spectra. For example, if `my_emission_function`
            # predicts the emission spectrum and `my_absorption_function` predicts the
            # absorption spectrum per cloud, then the observed spectra might be the
            # sum of the outputs of these functions over all clouds
            emission_spectrum = my_emission_function(
                self.data["emission"].spectral[:, None],
                parameter1,
                derived1,
            ).sum(axis=1)
            absorption_spectrum = my_absorption_function(
                self.data["absorption"].spectral[:, None],
                parameter2,
                derived2,
            ).sum(axis=1)

            # It is good practice to ensure that both the priors and the likelihood
            # are normalized (i.e., mean ~ 0 and variance ~ 1). Thus, we normalize
            # our predicted spectra by the noise and compare it to the similarly-normalized
            # data.
            emission_norm = self.data["emission"].normalize_brightness(emission)
            absorption_norm = self.data["absorption"].normalize_brightness(absorption)

            # We add on the normalized baseline model
            pred_emission = emission_norm + baseline_model_norm["emission"]
            pred_absorption = absorption_norm + baseline_model_norm["absorption"]

            # With normalized data, we can use a unit-variance normal distribution likelihood.
            # The normalized likelihood *must* be included for each spectrum in the data, and
            # the names of these likelihood parameters *must* be like {name}_norm where {name}
            # is the key of the spectrum in the data dictionary. Furthermore, "dims" must be
            # equal to {name}
            _ = pm.Normal(
                "emission_norm",
                mu=pred_emission,
                sigma=1.0,
                observed=self.data["emission"].brightness_norm,
                dims="emission",
            )
            _ = pm.Normal(
                "absorption_norm",
                mu=pred_absorption,
                sigma=1.0,
                observed=self.data["absorption"].brightness_norm,
                dims="absorption",
            )
```

# Algorithms

## Posterior Sampling: Variational Inference

`bayes_spec` can sample from an approximation of model posterior distribution using [variational inference (VI)](https://www.pymc.io/projects/examples/en/latest/variational_inference/variational_api_quickstart.html). The benefit of VI is that it is fast, but the downside is that it often fails to capture complex posterior topologies. We recommend only using VI for quick model tests or MCMC initialization. Draw posterior samples using VI via `model.fit()`.

## Posterior Sampling: MCMC

`bayes_spec` can also use MCMC to sample the posterior distribution. MCMC sampling tends to be much slower but also more accurate. Draw posterior samples using MCMC via `model.sample()`.

## Posterior Clustering: Gaussian Mixture Models

Assuming that we have drawn posterior samples via MCMC using multiple independent Markov chains, then it is possible that each chain disagrees on the order of clouds. This is known as the labeling degeneracy. For some models (e.g., optically thin radiative transfer), the order of clouds along the line-of-sight is arbitrary so each chain may converge to a different label order.

It is also possible that the model solution is degenerate, the posterior distribution is strongly multi-modal, and each chain converges to different, unique solutions.

`bayes_spec` uses Gaussian Mixture Models (GMMs) to break the labeling degeneracy and identify unique solutions. After sampling, execute `model.solve()` to fit a GMM to the posterior samples of each chain individually. Unique solutions are identified by discrepant GMM fits, and we break the labeling degeneracy by adopting the most common cloud order amongst chains.

## Optimization

`bayes_spec` can optimize the number of clouds in addition to the other model parameters. The `Optimize` class will use either VI or MCMC to estimate the preferred number of clouds.

# Syntax & Examples

See the various notebooks under [examples](https://github.com/tvwenger/bayes_spec/tree/main/examples).

# Issues and Contributing

Anyone is welcome to submit issues or contribute to the development
of this software via [Github](https://github.com/tvwenger/bayes_spec).

# License and Copyright

Copyright (c) 2024 Trey Wenger

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

