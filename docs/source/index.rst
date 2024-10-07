bayes_spec
==========

``bayes_spec`` is a framework for user-defined, cloud-based models of astrophysical systems (e.g., the interstellar medium) that enables spectral line simulation and statistical inference. Built in the ``pymc`` `probabilistic programming library <https://www.pymc.io/welcome.html>`_, ``bayes_spec`` uses Monte Carlo Markov Chain techniques to fit user-defined models to data. The user-defined models can be a simple line profile (e.g., a Gaussian profile) or a complicated physical model. The models are "cloud-based", meaning there can be multiple "clouds" or "components" each with a unique set of the model parameters. ``bayes_spec`` includes algorithms to estimate the optimal number of components in a given dataset.

Useful information can be found in the `Github repository <https://github.com/tvwenger/bayes_spec>`_ and in the tutorials below.

============
Installation
============
.. code-block::
    
    conda create --name bayes_spec -c conda-forge pymc pip
    conda activate bayes_spec
    pip install bayes_spec

.. toctree::
    :maxdepth: 2
    :caption: Guides:

    models
    tips

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   notebooks/basic_tutorial
   notebooks/basic_tutorial_noise
   notebooks/optimization
   notebooks/other_samplers

.. toctree::
   :maxdepth: 2
   :caption: API:
   
   modules