import numpy as np

def C_step(T):
  b = np.zeros_like(T)
  b[np.arange(len(T)), T.argmax(1)] = 1
  T = b

  return T

def smooth(data, method, coeff):
  
  if method == "Laplace":
    return (data + coeff)/(1 + coeff)
  elif method == "None":
    return data
  else:
    print("Unknown Method " + method + ", returning unchanged data by default.")
    return data

def dawid_skene(N, max_iter = 1000, collapse = C_step, check_convergence = False, tol = 0.001, prior = 1/2, smoothing = "Laplace", C = False):

  N = N.astype(np.float64)
  I, J, K = np.shape(N)
  
  ## Majority Vote initialization
  K_sum = np.sum(N, axis = 2)
  T = K_sum/(np.sum(K_sum, axis = 1).reshape(-1, 1))
  
  ## Collapses the probabilities to a single estimated label per task
  if C:
    T = collapse(T)

  ## EM Algorithm

  for i in range(max_iter):

    ## Maximization Step:

    ## Computes Prior Probability of each class
    p = np.mean(T, axis = 0).reshape(-1, 1)

    ## Unnormalized Confusion Tensor (J X J X K) 
    confusion = np.tensordot(T, N, axes = ([0,0]))
    
    ## Smooths the unnormalized confusion to avoid divide-by-zero when normalizing.
    confusion = smooth(confusion, smoothing, 0.01)

    ## Normalizes the Confusion Tensor
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      confusion /= np.repeat(np.sum(confusion, axis = 1), J, axis = 0).reshape(J, J, K)
    
    ## Expectation Step:
    
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")

      ## Compute Log-Likelihood
      log_conf = np.log(confusion)
      num = np.tensordot(N, log_conf, axes = ((2, 1), (2, 1)))
      denom = np.repeat(np.log(np.matmul(np.exp(num), p.reshape(-1, 1))), J, axis = 1)
      log_like = np.log(p.reshape(1, -1)) + num - denom

    ## Translate Log-Likelihoods to Label Probabilities
    T_new = np.exp(log_like).astype('float64')
 
    if C:
      T_new = collapse(T_new)

    if check_convergence and np.mean(np.abs(T_new - T)) < tol:
      T = T_new
      break
    else:
      T = T_new
    
  ## Return Label Probabilities Tensor and Best-Estimated Labels
  return T, np.argmax(T, axis = 1)
