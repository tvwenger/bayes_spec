# bayes_spec <!-- omit in toc -->
A Bayesian Spectral Line Modeling Framework for Astrophysics

`bayes_spec` is a framework for user-defined, cloud-based models of astrophysical systems (e.g., the interstellar medium) that enables spectral line simulation and statistical inference. Built in the [`pymc` probabilistic programming library](https://www.pymc.io/welcome.html), `bayes_spec` uses Monte Carlo Markov Chain techniques to fit user-defined models to data. The user-defined models can be a simple line profile (e.g., a Gaussian profile) or a complicated physical model. The models are "cloud-based", meaning there can be multiple "clouds" or "components" each with a unique set of the model parameters. `bayes_spec` includes algorithms to estimate the optimal number of components in a given dataset.

- [Installation](#installation)
- [Usage](#usage)
  - [Data Format](#data-format)
  - [Model Specification](#model-specification)
- [Algorithms](#algorithms)
  - [Posterior Sampling: Variational Inference](#posterior-sampling-variational-inference)
  - [Posterior Sampling: MCMC](#posterior-sampling-mcmc)
  - [Posterior Sampling: SMC](#posterior-sampling-smc)
  - [Posterior Clustering: Gaussian Mixture Models](#posterior-clustering-gaussian-mixture-models)
  - [Optimization](#optimization)
- [Syntax \& Examples](#syntax--examples)
- [Issues and Contributing](#issues-and-contributing)
- [License and Copyright](#license-and-copyright)


# Installation
Download and unpack the [latest release](https://github.com/tvwenger/bayes_spec/releases/latest), or [fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and contribute to the development of `bayes_spec`!

Install in a `conda` virtual environment:
```
conda env create -f environment.yml
# or, if you would like to use CUDA (nvidia GPU) samplers:
# conda env create -f environment-cuda.yml
conda activate bayes_spec
pip install .
```

If you wish to contribute to `bayes_spec`, then you may wish to install additional dependencies and install `bayes_spec` as an "editable" package:
```
conda env create -f environment-dev.yml
# or, if you would like to use CUDA (nvidia GPU) samplers:
# conda env create -f environment-cuda-dev.yml
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

The spectral and brightness data are accessed via `spec.spectral` and `spec.brightness`, respectively. The noise is accessed via `spec.noise`.

The data are passed to a `bayes_spec` model in a dictionary, which allows multiple spectra to be included in a single model. For example, if a model is constrained by both an emission-line spectrum and an absorption-line spectrum, then this dictionary might look something like this:

```python
emission = SpecData(emission_spec_axis, emission_data, emission_noise, xlabel="Velocity", ylabel="Brightness Temperature")
absorption = SpecData(absorption_spec_axis, absorption_data, absorption_noise, xlabel="Velocity", ylabel="Optical Depth")

data = {"emission": emission, "absorption": absorption}
```

The keys of this data dictionary (`"emission"` and `"absorption"` in this case) are important, you must remember them and use the same keys in your model definition. 

Internally, `SpecData` normalizes both the spectral axis and the data. Generally, this is only relevant for the polynomial baseline model, which is fit to the normalized data.

## Model Specification

Model specification is made though a class that extends the `bayes_spec.BaseModel` base model class definition. This class must include three methods: `__init__`, which initializes the model, and `add_priors`, which adds the priors to the model, and `add_likelihood`, which adds the likelihood to the model. These priors and likelihood are specified following the usual `pymc` syntax. [See the definition of `GaussLine` for an example.](https://github.com/tvwenger/bayes_spec/blob/main/bayes_spec/models/gauss_line.py) Alternatively, the class can extend an existing `bayes_spec` model, which is convenient for similar models with, for example, added complexity. [See the definition of `GaussLineNoise`, which extends `GaussLine`.](https://github.com/tvwenger/bayes_spec/blob/main/bayes_spec/models/gauss_line_noise.py) 

# Algorithms

## Posterior Sampling: Variational Inference

`bayes_spec` can sample from an approximation of model posterior distribution using [variational inference (VI)](https://www.pymc.io/projects/examples/en/latest/variational_inference/variational_api_quickstart.html). The benefit of VI is that it is fast, but the downside is that it often fails to capture complex posterior topologies. We recommend only using VI for quick model tests or MCMC initialization. Draw posterior samples using VI via `model.fit()`.

## Posterior Sampling: MCMC

`bayes_spec` can also use MCMC to sample the posterior distribution. MCMC sampling tends to be much slower but also more accurate. Draw posterior samples using MCMC via `model.sample()`. Since `bayes_spec` uses `pymc` for sampling, several `pymc` samplers are available, including GPU samplers (see ["other samplers" example notebook](https://github.com/tvwenger/bayes_spec/tree/main/examples)).

## Posterior Sampling: SMC

Finally, `bayes_spec` implements Sequential Monte Carlo (SMC) sampling via `model.sample_smc()`. SMC can significantly improve performance for degenerate models with multi-modal posterior distributions, although it struggles with high dimensional models and models that suffer from a strong labeling degeneracy (see ["other samplers" example notebook]((https://github.com/tvwenger/bayes_spec/tree/main/examples))).

## Posterior Clustering: Gaussian Mixture Models

Assuming that we have drawn posterior samples via MCMC or SMC using multiple independent Markov chains, then it is possible that each chain disagrees on the order of clouds. This is known as the labeling degeneracy. For some models (e.g., optically thin radiative transfer), the order of clouds along the line-of-sight is arbitrary so each chain may converge to a different label order.

It is also possible that the model solution is degenerate, the posterior distribution is strongly multi-modal, and each chain converges to different, unique solutions.

`bayes_spec` uses Gaussian Mixture Models (GMMs) to break the labeling degeneracy and identify unique solutions. After sampling, execute `model.solve()` to fit a GMM to the posterior samples of each chain individually. Unique solutions are identified by discrepant GMM fits, and we break the labeling degeneracy by adopting the most common cloud order amongst chains. The user defines which parameters are used for the GMM clustering.

## Optimization

`bayes_spec` can optimize the number of clouds in addition to the other model parameters. The `Optimize` class will use VI, MCMC, and/or SMC to estimate the preferred number of clouds.

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

