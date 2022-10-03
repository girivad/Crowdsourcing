# Crowdsourcing: Ground-truth Inference Algorithms

This repository contains vectorized and efficient implementations of multiple algorithms for crowdsourced data, particularly for inferring the ground truth from crowdsourced labels.

## **Dawid/Skene**

Implements the Expectation-Maximization Algorithm from the 1979 paper "Maximum Likelihood Estimation of Observer Error Rates Using the EM Algorithm" (A.P. Dawid and A.M. Skene).

Input: Accepts an IxJxK tensor N, where $N_{ijk}$ is the number of times worker k labels task i with label j
Outputs: Produces the estimated label distribution for each task as well as a discretized vector with the best label for each task.

*(Work done at Professor David T. Lee's Tech4Good Lab at UC Santa Cruz; Thank you for letting me build on this!)*
