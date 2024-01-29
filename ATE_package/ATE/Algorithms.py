import numpy as np
from scipy.optimize import minimize

    
def CRFamily(x, theta):
    """
    A function for evaluating the Cressie Read family.
    See documentation for the exact form for the class of functions.
    Note: In this package we use a modification of this family
      Instead of 1 + theta we use 1 - theta.
    Args:
      x: Vector of points at which function will be evaluated.
      theta: Scalar theta which parametrizes the CR family of
             functions.
   
    Returns:
      The vector with the CR function evaluated for each point
      in x.
    """
    
    # Copy the vector
    xv = np.copy(x)

    if theta == 0:
        # The limiting case of theta = 0
        xv = -np.exp(-x)
    elif theta == -1:
        # The limiting case of theta = -1, use L'Hopital's rule
        xv = np.log(1 + x)
    else:
        # The function for all other values of theta
        xv = -np.power((1 - theta * x), 1 + 1 / theta)
        xv = xv / (1 + theta)

    # Handling non-finite values (e.g., NaN, Inf) by setting them to -Inf
    # For some cases particularly theta == -1, the function is
    # not defined for the entire real line. In order to maintain
    # concavity of the function we assign the value -Inf to regions
    # outside the domain of the function.
    # This is a standard trick in convex optimization.
    xv[~np.isfinite(xv)] = -np.inf

    return xv



def CRFamilyDerivative(x, theta):
    """
    A function for evaluating the first derivative of
    Cressie Read family.
   
    Args:
      x: Vector of points at which derivative will be evaluated.
      theta: Scalar theta which parametrizes the CR family of
             functions.
   
    Returns:
      The vector with the CR function's derivative evaluated
      for each point in x.
    """
    
    # Copy the vector
    xv = np.copy(x)

    if theta == 0:
        # Derivative for the case of theta = 0
        xv = np.exp(-x)
    elif theta == -1:
        # Derivative for the case of theta = -1
        xv = 1 / (1 + x)
    else:
        # Derivative for other values of theta
        xv = np.power((1 - theta * x), 1 / theta)

    # Handling non-finite values (e.g., NaN, Inf) by setting them to -Inf
    xv[~np.isfinite(xv)] = -np.inf

    return xv


def CRFamilySecondDerivative(x, theta):
    """
    A function for evaluating the second derivative of
    the Cressie Read family.
   
    Args:
      x: Vector of points at which the second derivative will
         be evaluated.
      theta: Scalar theta which parametrizes the CR family of
             functions.
   
    Returns:
      The vector with the CR function's 2nd derivative evaluated
      for each point in x.
    """
    
    # Copy the vector
    xv = np.copy(x)

    if theta == 0:
        # Second derivative for the case of theta = 0
        xv = -np.exp(-x)
    elif theta == -1:
        # Second derivative for the case of theta = -1
        xv = -1 / np.power(1 + x, 2)
    else:
        # Second derivative for other values of theta
        xv = -np.power(1 - theta * x, 1 / theta - 1)

    # Handling non-finite values (e.g., NaN, Inf) by setting them to -Inf
    xv[~np.isfinite(xv)] = -np.inf

    return xv



def ObjectiveFunction(lambda_vec, u, ubar, treat, theta):
    """
    The main objective function which we need to optimize over.
    See package Vignette for the objective functions/optimization problem.
   
    Args:
      lam: Lambda vector at which to evaluate objective.
      u: A N * K matrix for the covariates. In our case this is
         this is just the matrix cbind(1, X) for a design matrix X.
         N is the number of subjects with K-1 covariates.
      ubar: The K vector of column means of matrix u.
      treat: The N vector specifying treatment assignment.
      theta: Scalar parametrizing the CR family of functions.
   
    Returns:
      A scalar value of the objective function evaluated at the vector
      lambda.
    """
    
    # Assuming 'u' is already a NumPy matrix and 'lam' is a NumPy vector
  

    # This creates a vector where the i-th term is lambda^T * u_K(X_i)
    lam_t_u = np.dot(u, lambda_vec)
    lam_t_ubar = np.dot(ubar, lambda_vec)

 
    # Assuming CRFamily is already defined and takes a NumPy array and returns a NumPy array
    cr_family_result = CRFamily(lam_t_u, theta)

    # Element-wise multiplication
    result = treat * cr_family_result

    # Mean of result, excluding non-finite values
    finite_result = result[np.isfinite(result)]
    objective_value = -np.mean(finite_result) + lam_t_ubar


    return objective_value

def ObjectiveFirstDerivative(lam, u, ubar, treat, theta):
    """
    Compute the first derivative of the main objective function.

    Args:
    lam: NumPy array (vector) of lambda values.
    u: NumPy array (matrix) representing 'u'.
    ubar: NumPy array (vector) representing 'ubar'.
    treat: NumPy array (vector) representing treatment.
    theta: Scalar theta for the CR family.

    Returns:
    NumPy array (row vector) representing the first derivative of the objective function.
    """

    N = u.shape[0]  # Number of rows in 'u'

    # This creates a vector for the terms lambda^T * u_K(X_i)
    lam_t_u = np.dot(u, lam)
    temp = (treat * CRFamilyDerivative(lam_t_u, theta)).T
    temp[~np.isfinite(temp)] = 0  # Replace non-finite values with 0

    # Compute and return derivative
    return ubar - np.dot(temp, u) / float(N)





def UpdateInverseHessian(old_inv_hessian, diff_in_est, diff_in_derv):
    """
    A simple backtracking line search
    This formulation is taken from Algorithm 9.2 of
    Boyd, Stephen, and Lieven Vandenberghe. Convex optimization. 2004.
    Args:
      kAlpha, kBeta: Constant factors for the backtracking algorithm.
      current_est: The current estimate or parameter
      current_direction: The descent direction vector for our current
                         iteration/step.
      current_derv: The derivative of our objective function at the
                    current estimate (current_est).
      u, ubar, treat, theta: See previous functions above.
   
    Returns:
      A scalar step size for the BFGS algorithm.
    """

    # Calculate scalar values used in the BFGS update formula
    scal1 = np.dot(diff_in_derv.T, np.dot(old_inv_hessian, diff_in_derv))
    scal2 = np.dot(diff_in_est.T, diff_in_derv)

    mat1 = np.dot(np.dot(old_inv_hessian, diff_in_derv).reshape(-1,1), diff_in_est.reshape(1,-1))

    # BFGS update formula
    term1 = ((scal1 + scal2) / scal2**2) * np.outer(diff_in_est, diff_in_est)
    term2 = (mat1 + mat1.T) / scal2

    return old_inv_hessian + term1 - term2

"""
This function is abandoned
"""
def BacktrackLineSearch(kAlpha, kBeta, current_est, current_direction, current_derv, u, ubar, treat, theta):
    """
    Perform a Backtracking Line Search.

    Args:
    kAlpha: Scalar value for alpha in the line search algorithm.
    kBeta: Scalar value for beta in the line search algorithm.
    current_est: NumPy array (vector) representing the current estimate.
    current_direction: NumPy array (vector) representing the current direction of descent.
    current_derv: NumPy array (vector) representing the current derivative.
    u: NumPy array (matrix) representing 'u'.
    ubar: NumPy array (vector) representing 'ubar'.
    treat: NumPy array (vector) representing treatment.
    theta: Scalar theta for the CR family.

    Returns:
    The step size determined by the line search.
    """

    # Assuming 'u' is already a NumPy matrix
    cur_derv = current_derv.T  # Transpose to row vector

    # Begin with an initial step size of 1
    step = 1.0

    # Initialize some quantities
    cur_objective = ObjectiveFunction(current_est, u, ubar, treat, theta)
    dervf_trans_dervx = np.dot(cur_derv, current_direction)

    # Loop until the condition is met
    while ObjectiveFunction(current_est + step * current_direction, u, ubar, treat, theta) > (cur_objective + kAlpha * step * dervf_trans_dervx):
        step *= kBeta

    return step

class BFGSResult:
    def __init__(self, est, weights, converged):
        self.est = est
        self.weights = weights
        self.converged = converged

def BFGSAlgorithm(initial, u, ubar, treat, theta, kAlpha, kBeta, max_iter, tol):
    """
    Perform the BFGS optimization algorithm.

    Args:
    initial: NumPy array (vector) representing the initial estimates.
    u: NumPy array (matrix) representing 'u'.
    ubar: NumPy array (vector) representing 'ubar'.
    treat: NumPy array (vector) representing treatment.
    theta: Scalar theta for the CR family.
    kAlpha: Scalar alpha for the line search algorithm.
    kBeta: Scalar beta for the line search algorithm.
    max_iter: The maximum number of iterations.
    tol: The tolerance level for convergence.

    Returns:
    A dictionary containing the final estimate ('new_est'), weights, and a boolean indicating success.
    """

    N, K = u.shape[0], initial.size
    # u_bar = ubar.T  # Transpose ubar to row vector

    current_est = initial
    # current_inv_hessian = np.eye(K)
    # objective_value = 0
    
    # the original version is by a self-defined BFGS algo, instead, we use scipy version
    result = minimize(ObjectiveFunction, current_est,args=(u, ubar, treat, theta), 
                        method='BFGS', jac=ObjectiveFirstDerivative,
                        options={'maxiter':max_iter,"gtol":tol})
        
    # for i in range(max_iter):
    #     current_derv = ObjectiveFirstDerivative(current_est, u, ubar, treat, theta)
    #     current_direction = -np.dot(current_inv_hessian, current_derv)
    #     objective_value = ObjectiveFunction(current_est, u, ubar, treat, theta)
    #     step_size = BacktrackLineSearch(kAlpha, kBeta, current_est, current_direction, current_derv, u, ubar, treat, theta)
    #     new_est = current_est + step_size * current_direction
    #     new_derv = ObjectiveFirstDerivative(new_est, u, ubar, treat, theta)
    #     diff_in_derv = new_derv - current_derv
    #     diff_in_est = new_est - current_est
    #     new_inv_hessian = UpdateInverseHessian(current_inv_hessian, diff_in_est, diff_in_derv)

        
    #     if objective_value < -1e+30:
    #         # Handle unbounded objective function warning and logic
    #         weights = np.zeros(N)
    #         result = BFGSResult(new_est,weights,False)
    #         return result

    #     if np.linalg.norm(new_derv, 2) < tol:
    #         weights = CRFamilyDerivative(np.dot(u, new_est), theta) / N
    #         result = BFGSResult(new_est,weights,True)
    #         return result

    #     current_est = new_est
    #     current_inv_hessian = new_inv_hessian
    
    new_est = result.x
    weights = CRFamilyDerivative(np.dot(u, new_est), theta) / N
    result = BFGSResult(new_est,weights,result.success)
    
    return result
