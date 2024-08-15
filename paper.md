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

`bayes_spec` is a spectral line modeling framework for astrophysics. Given a user-defined model and some data, `bayes_spec` enables inference on the model parameters through different numerical techniques, such as Monte Carlo Markov Chain (MCMC) methods, implemented in the `pymc` probabilistic programming library [@pymc2023]. The API for `bayes_spec` is designed to support astrophysical researchers who wish to "fit" arbitrary, user-defined models, such as simple spectral line profile models or complicated physical models that include a full physical treatment of radiative transfer. These models are "cloud-based", meaning that the spectral line data is decomposed into a series of discrete clouds with parameters defined by the user's model. Importantly, `bayes_spec` provides algorithms to determine the optimal number of clouds for a given model and dataset.

Bayesian models of spectral line observations are rare in astrophysics. `bayes_spec` is inspired by [AMOEBA](https://github.com/AnitaPetzler/AMOEBA) [@petzler2021], an MCMC-based Bayesian model for interstellar hydroxide observations. With `bayes_spec`, we aim to provide a general Bayesian modeling framework for any astrophysical spectral line observation.

# References