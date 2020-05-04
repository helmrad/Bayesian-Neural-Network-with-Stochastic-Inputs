This code generates a Bayesian feedforward fully-connected neural network. Given a dataset of samples and targets, this type of Bayesian neural network receives, additionally to the sample, a stochastic input. The stochastic input, described by a parameterized probability distribution, is meant to account for unobserved features that are not present in the dataset. This allows for modelling complex stochastic effects like multimodality and heteroscedasticity, as shown here:
<img src="./images/multimodality_heteroscedasticity.svg">

The weights of the neural network, described by a parameterized multidimensional probability distribution, and the aforementioned stochastic inputs are updated by Bayesian inference. In particular, black-box alpha-divergence minimization [Hernández-Lobato et al. 2015] is used, as detailed in Section 2.1 of the Thesis. The code is tailored to Python 3.6 and requires the library autograd.

When executed, the code automatically starts to fit a Bayesian neural network to a bimodal sinusoid distribution.
