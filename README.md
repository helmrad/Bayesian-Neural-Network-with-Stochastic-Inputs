# Bayesian-Neural-Network-with-Stochastic-Inputs

Given a dataset of samples and targets. This type of Bayesian neural network receieves, additionally to the sample, a stochastic input. The stochastic input, described by a parameterized probability distribution, is meant to account for unobserved features that are not present in the dataset. It allows for modelling complex stochastic effects like multimodality and heteroscedasticity.

The weights of the neural network, also described by a parameterized multidimensional probability distribution, and the aforementioned stochastic inputs z are trained by Bayesian inference. In particular, black-box $\alpha$-divergence minimization is used [Hernández-Lobato et al. 2015] to fit the parameters to a dataset.

This Bayesian neural network was primarily used for multidimensional regression tasks.
