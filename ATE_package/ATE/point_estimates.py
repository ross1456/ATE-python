import warnings
import numpy as np
from .Algorithms import BFGSAlgorithm

warnings.simplefilter('once', UserWarning)

def GetPointEstSimple(initial_one, initial_two, **kwargs):
  """  
  Function to get point estimates for the simple case of
  binary treatment and NOT the case of treatement effect on
  the treated.
  
  Args:
    initial_one: The initial vector for first BFGS algorithm (treatment arm).
    initial_two: The initial vector for second BFGS algorithm (placebo arm).
    **kwargs: Other arguments to be passed to the function from parent.
  
  Returns:
    A list of estimated lambda values where lambda is the parameter
    vector which we optiize over. It also contains the weights
    obtained from the estimated lambda values used for covariate
    balancing estimates. Finally, the list also contains the
    point estimates and convergence indicator.
  """

  #Obtain extra arguments
  Y = kwargs["Y"]
  X = kwargs["X"]
  treat = kwargs["treat"]
  theta = kwargs["theta"]
  max_iter = kwargs["max_iter"]
  tol = kwargs["tol"]
  backtrack_alpha = kwargs["backtrack_alpha"]
  backtrack_beta = kwargs["backtrack_beta"]
  verbose = kwargs["verbose"]

  N = len(Y)  # Sample size.
  u_mat = np.c_[np.ones(len(X)), X]  # Obtain u matrix of covariates.
  u_bar = np.mean(u_mat, axis=0)  # Get column means.

  # Perform some simple checks.
  if len(initial_one) != len(u_mat[0]):
    raise ValueError("Incorrect length of initial vector")
  if len(initial_two) != len(u_mat[0]):
    raise ValueError("Incorrect length of initial vector")
  if verbose:
    print("Running BFGS algorithm for estimating Weights p: ")

  # Implement the BFGS algorithm to get the optimal lambda
  # and correspoding weights.
  # This corresponds to \lambda_p and \hat{p}_K in vignette.
  treatment_hat = BFGSAlgorithm(initial_one, u_mat, u_bar, treat,
                                theta, backtrack_alpha,
                                backtrack_beta, max_iter,
                                tol)

  if verbose:
    print("\nRunning BFGS algorithm for estimating Weights q: ")

  # Implement the BFGS algorithm to get the optimal lambda_q
  # and correspoding weights \hat{q}_K.
  placebo_hat = BFGSAlgorithm(initial_two, u_mat, u_bar, 1 - treat,
                              theta, backtrack_alpha,
                              backtrack_beta,
                              max_iter, tol)


  # Obtain estimates for tau1 = E[Y(1)], tau2 = E[Y(0)]
  # and tau = E[Y(1)] - E[Y(0)].
  tau_one  = np.sum((treatment_hat.weights * Y)[treat == 1])
  tau_zero = np.sum((placebo_hat.weights * Y)[treat == 0])
  tau = tau_one - tau_zero

  # Obtain estimated weights and lambda's for
  # each treatment arm.
  weights_treat = treatment_hat.weights
  weights_placebo = placebo_hat.weights

  lambda_treat = treatment_hat.est
  lambda_placebo = placebo_hat.est

  # Throw a warning if algorithm did not converge.
  converge = True
  if not treatment_hat.converged or not placebo_hat.converged:
    warnings.warn("BFGS Algorithm did not converge for atleast one objective function.")
    converge = False
  

  # Return list of objects
  return {"lambda_treat": lambda_treat,
          "lambda_placebo": lambda_placebo,
          "weights_treat": weights_treat,
          "weights_placebo": weights_placebo,
          "tau_one": tau_one, "tau_zero": tau_zero,
          "tau": tau, "converge": converge}

def GetPointEstATT(initial, **kwargs):
  '''  
  Function to get point estimates average treatment effect
  on the treated. This case also has binary treatment.

  Args:
    initial: The initial vector for the BFGS algorithm.
    **kwargs: Other arguments to be passed to the function.

  Returns:
    List of objects similar to the previous function. However, this
    function does not return lambda or weights for the treatment arm.
    This is because the we do not need a covariate balancing technique
    for the treatment arm in this case.
  '''

  #Obtain extra arguments
  Y = kwargs["Y"]
  X = kwargs["X"]
  treat = kwargs["treat"]
  theta = kwargs["theta"]
  max_iter = kwargs["max_iter"]
  tol = kwargs["tol"]
  backtrack_alpha = kwargs["backtrack_alpha"]
  backtrack_beta = kwargs["backtrack_beta"]
  verbose = kwargs["verbose"]

  N = len(Y)  # Sample size.
  u_mat = np.c_[np.ones(len(X)), X]  # Design matrix u.

  # Main difference here is the definition of u_bar.
  # u_bar is vector of column means ONLY for those
  # who recieved the treatment.
  u_bar = np.mean(u_mat[treat == 1, :], axis=0)

  if len(initial) != len(u_mat[0]):
    raise ValueError("Incorrect length of initial vector")
  if verbose:
    print("\nRunning BFGS algorithm Raphson for estimating Weights q: ")

  # Run the BFGS algorithm for obtaining the parameter and weights
  # used for covariate balancing.
  placebo_hat = BFGSAlgorithm(initial, u_mat, u_bar,
                              1 - treat, theta,
                              backtrack_alpha, backtrack_beta,
                              max_iter, tol)

  # Note that treatment effect on treated is simple to estimate
  tau_one = np.mean(Y[treat == 1])
  # The calibration estimator for E[Y(0)|T = 1]
  tau_zero = np.sum((placebo_hat.weights * Y)[treat == 0])
  tau = tau_one - tau_zero

  # Weights and lambda vectors
  weights_placebo = placebo_hat.weights
  lambda_placebo = placebo_hat.est

  # Warning message if the algorithm did not converge
  converge = True
  if not placebo_hat.converged:
    warnings.warn("\nBFGS algorithm did not converge for the objective function.")
    converge = False
 
  # Return list
  return {"lambda_placebo": lambda_placebo,
          "weights_placebo": weights_placebo,
          "tau_one": tau_one, "tau_zero": tau_zero,
          "tau": tau, "converge": converge}

def GetPointEstMultiple(initial_mat, **kwargs):
  '''  
  Function to get point estimates for treatment effect
  when we have multiple treatment arms.
  
  Args:
    initial_mat: A matrix of initial values for the different
                 BFGS algorithm. Each row is an inital vector for
                 an algorithm. The total number of rows is J: number of
                 different treatment arms.
    **kwargs: Other arguments to be passed to the function.
  
  Returns:
    A List with the following objects
      lam_mat: A matrix of estimated lambda values. This has the same
               dimensions as initial_mat.
      weights_mat: A matrix estimated weights. The j-th row
                   corresponds to the weights for treatment j.
      tau_treatment_j: A vector of estimates of [EY(0), EY(1),..., EY(J-1)].
      converge: A boolean indicator of convergence status.
  '''

  #Obtain extra arguments
  Y = kwargs["Y"]
  X = kwargs["X"]
  treat = kwargs["treat"]
  theta = kwargs["theta"]
  max_iter = kwargs["max_iter"]
  tol = kwargs["tol"]
  backtrack_alpha = kwargs["backtrack_alpha"]
  backtrack_beta = kwargs["backtrack_beta"]
  verbose = kwargs["verbose"]

  N = len(Y)  # Sample size.
  J = len(np.unique(treat))  # Number of treatment arms.
  u_mat = np.c_[np.ones(len(X)), X]  # Obtain design matrix u.
  u_bar = np.mean(u_mat, axis=0)  # Obtain column means of design matrix.

  # A simple verification of dimensions.
  if np.shape(initial_mat)[1] != len(u_mat[0]):
    raise ValueError("Incorrect length of initial vector")
  if verbose:
    print("\nRunning BFGS for estimating Weights: ")

  # Initialize the matrices and vector which will be returned
  # by this function.
  lam_mat = np.zeros((J, np.shape(initial_mat)[1]))
  weights_mat = np.zeros((J, N))
  tau_treatment_j = np.zeros(J)

  # Loop through the different treatment arms.
  for j in range(J):
    # Inidicator of treatment arm.
    temp_treat = 1 * (treat == j)

    # Implement BFGS algorithm
    treatment_j_hat = BFGSAlgorithm(initial_mat[j],
                                    u_mat, u_bar, temp_treat,
                                    theta,
                                    backtrack_alpha,
                                    backtrack_beta, max_iter, tol)

    # Find estimate for E[Y(j)]
    tau_j_hat = np.sum((treatment_j_hat.weights * Y)[temp_treat == 1])

    tau_treatment_j[j] = tau_j_hat
    lam_mat[j] = treatment_j_hat.est
    weights_mat[j] = treatment_j_hat.weights

    # Warning for non-convergence
    converge = True
    if not treatment_j_hat.converged:
      warnings.warn("BFGS algorithm did not converge for treatment arm {}".format(j))
      converge = False
  # Return result.

  return {"lam_mat": lam_mat,
          "weights_mat": weights_mat,
          "tau_treatment_j": tau_treatment_j,
          "converge": converge}
