---
title: 'bayes_spec: A Bayesian Spectral Line Modeling Framework for Astrophysics'
tags:
    - Python
    - astronomy
    - astrophysics
    - spectroscopy
    - Bayesian models
authors:
    - name: Trey V. Wenger
      orcid: 0000-0003-0640-7787
      equal-contrib: true
      affiliation: 1
affiliations:
    - name: NSF Astronomy & Astrophysics Postdoctoral Fellow, University of Wisconsin-Madison, USA
      index: 1
date: 15 August 2024
bibliography: paper.bib
---

# Summary

The study of the interstellar medium (ISM) -- the stuff between the stars -- relies heavily on the tools of spectroscopy. Spectral line observations of atoms, ions, and molecules in the ISM reveal the physical conditions and kinematics of the emitting gas. Robust and efficient numerical techniques are thus necessary for inferring the physical conditions of the ISM from observed spectral line data.

# Statement of need

`bayes_spec` is a Bayesian spectral line modeling framework for astrophysics. Given a user-defined model and some data, `bayes_spec` enables inference on the model parameters through different numerical techniques, such as Monte Carlo Markov Chain (MCMC) methods, implemented in the `pymc` probabilistic programming library [@pymc2023]. The API for `bayes_spec` is designed to support astrophysical researchers who wish to "fit" arbitrary, user-defined models, such as simple spectral line profile models or complicated physical models that include a full physical treatment of radiative transfer. These models are "cloud-based", meaning that the spectral line data is decomposed into a series of discrete clouds with parameters defined by the user's model. Importantly, `bayes_spec` provides algorithms to determine the optimal number of clouds for a given model and dataset.

Bayesian models of spectral line observations are rare in astrophysics. Physical inference is typically achieved through inverse modeling: the spectral line data are decomposed into Gaussian components, and then the physical parameters are inferred from the fitted Gaussian parameters under numerous assumptions. For example, such is the strategy of `gausspy` [@lindner2015], `ROHSA` [@marchal2019], `pyspeckit` [@ginsburg2022], and `MWYDYN` [@rigby2024]. This strategy suffers from two problems: (1) the degeneracies of Gaussian decomposition and (2) the assumptions of the inverse model. Bayesian forward models, like those enabled by `bayes_spec`, can overcome both of these limitations: (1) prior knowledge about the physical conditions can reduce the space of possible solutions, and (2) all assumptions are clearly built into the model rather than being applied *a priori*.

`bayes_spec` is inspired by [AMOEBA](https://github.com/AnitaPetzler/AMOEBA) [@petzler2021], an MCMC-based Bayesian model for interstellar hydroxide observations. `McFine` [@williams2024] is a new MCMC-based Bayesian model for hyperfine spectroscopy similar in spirit to `bayes_spec`. With `bayes_spec`, we aim to provide a user-friendly, general-purpose Bayesian modeling framework for *any* astrophysical spectral line observation.

# Usage

Here we demonstrate how to use `bayes_spec` to fit a simple Gaussian line profile model to a synthetic spectrum. For more details, see the [documentation and tutorials](https://readthedocs.org/projects/bayes-spec/badge/?version=latest).

```python
# Generate "dummy" data structure
import numpy as np
from bayes_spec import SpecData

velocity_axis = np.linspace(-250.0, 250.0, 501)
noise = 1.0
brightness_data = noise * np.random.randn(len(velocity_axis))
observation = SpecData(velocity_axis, brightness_data, noise)
dummy_data = {"observation": observation}

# Prepare a three cloud GaussLine model with polynomial baseline degree = 2
from bayes_spec.models import GaussModel

model = GaussModel(dummy_data, n_clouds=3, baseline_degree=2)
model.add_priors()
model.add_likelihood()

# Evaluate the model for a given set of parameters to generate a synthetic "observation"
sim_brightness = model.model.observation.eval({
    "fwhm": [25.0, 40.0, 35.0], # FWHM line width (km/s)
    "line_area": [250.0, 125.0, 175.0], # line area (K km/s)
    "velocity": [-35.0, 10.0, 55.0], # velocity (km/s)
    "baseline_observation_norm": [-0.5, -2.0, 3.0], # normalized baseline coefficients
})
observation = SpecData(velocity_axis, sim_brightness, noise)
data = {"observation": observation}

# Initialize the model with the synthetic observation
model = GaussModel(data, n_clouds=3, baseline_degree=2)
model.add_priors()
model.add_likelihood()

# Draw posterior samples via MCMC
model.sample()

# Solve labeling degeneracy
model.solve()

# visualize posterior distribution
from bayes_spec.plots import plot_pair
plot_pair(model.trace.solution_0, model.cloud_deterministics, labeller=model.labeller)

# get posterior summary statistics
import arviz as az
az.summary(model.trace.solution_0)
```

# References