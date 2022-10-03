# Crowdsourcing: Ground-Truth Inference Algorithms

This repository contains vectorized and efficient implementations of multiple algorithms for crowdsourced data, particularly for inferring the ground truth from crowdsourced labels.

## **Dawid/Skene**

Implements the Expectation-Maximization Algorithm from the 1979 paper "Maximum Likelihood Estimation of Observer Error Rates Using the EM Algorithm" (A.P. Dawid and A.M. Skene).

Input: Accepts an $I \times J \times K$ tensor N, where $N_{ijk}$ is the number of times worker $k$ labels task $i$ with label $j$ ( $I$ is the total number of tasks, $J$ the number of labels, $K$ the number of workers).

Outputs: Produces the estimated label distribution for each task as an $I \times J$ matrix as well as a discretized $I$-vector with the best label for each task.

### Configuration Parameters:

- *max_iter*: The maximum number of iterations of the EM algorithm allowed, may be terminated prior due to convergence. **(Default: 1000)**.
- *collapse*: Any function to post-process the label-distribution at each estimation. The default parameter is the provided function *C_Step* that assigns all the probability to the highest-probability label.
-  *check_convergence*: True/False; whether to terminate upon convergence of estimates or not. **(Default: False)**.
-  *tol*: Maximum difference between estimates not considered convergence. **(Default: 0.001)**
-  *smoothing*: The method of smoothing used for the confusion matrix. **(Default: Laplace)**. *Currently, only Laplace Smoothing is implemented, other forms may be implemented soon if useful.* 
-  *C*: True/False; Indicates whether a 'collapse' step is to be applied **(Default: False)**.

*(Work done at Professor David T. Lee's Tech4Good Lab at UC Santa Cruz; Thank you for letting me build on this!)*
