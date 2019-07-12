This code generates a Bayesian feedforward fully-connected neural network. Given a dataset of samples and targets, this type of Bayesian neural network receives, additionally to the sample, a stochastic input. The stochastic input, described by a parameterized probability distribution, is meant to account for unobserved features that are not present in the dataset. It allows for modelling complex stochastic effects like multimodality and heteroscedasticity.

The weights of the neural network, described by a parameterized multidimensional probability distribution, and the aforementioned stochastic inputs are updated by Bayesian inference. In particular, black-box alpha-divergence minimization is used [Hernández-Lobato et al. 2015]. The code is tailored to Python 3.6 and requires the library autograd.

When executed, the code automatically starts to fit a Bayesian neural network to a bimodal sinusoid distribution.
