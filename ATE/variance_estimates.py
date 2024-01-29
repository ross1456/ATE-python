import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import inv
from .Algorithms import CRFamilyDerivative, CRFamilySecondDerivative

def GetEESimple(u_mat_lambda_treat, u_mat_lambda_placebo, tau_one, tau_zero, Y, treat, u_mat, theta):
    """
    This function calculates the estimating equations for
    the case of simple binary treatments.
    Args:
      u_mat_lambda.treat: The vector u.mat * lambda.treat.
      u_mat_lambda.placebo: The vector u.mat * lambda.placebo.
      ...: Other parameters to be passed to the function.
    Returns:
      A matrix with N rows, each row is the EE for the
      corresponding subject evaluated at the given values of
      lambda.treat and lambda.placebo.
    """
    temp1 = treat * CRFamilyDerivative(u_mat_lambda_treat, theta)
    gk1 = np.multiply(u_mat, temp1[:, np.newaxis]) - u_mat

    temp2 = (1 - treat) * CRFamilyDerivative(u_mat_lambda_placebo, theta)
    gk2 = np.multiply(u_mat, temp2[:, np.newaxis]) - u_mat

    gk3 = (treat * Y) * CRFamilyDerivative(u_mat_lambda_treat, theta) - tau_one
    gk4 = ((1 - treat) * Y) * CRFamilyDerivative(u_mat_lambda_placebo, theta) - tau_zero

    return np.column_stack((gk1, gk2, gk3, gk4))

def GetCovSimple(fitted_point_est, Y, treat, u_mat, theta):
    '''
    This function calculates the estimate of the
    covariance matrix for the parameters taus.
    I.e. cov of E[Y(1)] and E[Y(0)]
    Args:
      fitted_point_est: The output of the function
                        GetPointEstSimple.
      ...: Other parameters to be passed.
    Returns:
      The large covariance matrix for the taus,
      I.e. cov matrix for tau_one and tau_zero.
    '''
    N = len(Y)  # Sample size
    n1 = np.sum(treat)  # Number of subjects in treatment arm
    K = u_mat.shape[1]  # Number of covariates
    # Extract lambda and u * lambda objects
    lambda_treat = fitted_point_est['lambda_treat']
    lambda_placebo = fitted_point_est['lambda_placebo']
    u_mat_lambda_treat = np.dot(u_mat, lambda_treat)
    u_mat_lambda_placebo = np.dot(u_mat, lambda_placebo)

    # Extract point estimates of tau's
    tau_one = fitted_point_est['tau_one']
    tau_zero = fitted_point_est['tau_zero']

    # Calculate the estimating equations
    gk = GetEESimple(u_mat_lambda_treat, u_mat_lambda_placebo, tau_one, tau_zero, Y, treat, u_mat, theta)

    # Calculate the MEAT for the Huber-White sandwich estimate
    meat = np.dot(gk.T, gk) / N

    # Create the 2K*2K matrix A
    temp1 = treat * CRFamilySecondDerivative(u_mat_lambda_treat, theta)
    temp2 = (1 - treat) * CRFamilySecondDerivative(u_mat_lambda_placebo, theta)

    tempA1 = u_mat * temp1[:, np.newaxis]  # Element-wise multiplication
    A1 = np.dot(tempA1.T, u_mat) / N  # Equivalent to crossprod in R
    tempA2 = u_mat * temp2[:, np.newaxis]  # Element-wise multiplication
    A2 = np.dot(tempA2.T, u_mat) / N  # Equivalent to crossprod in R
    A = block_diag(A1, A2)

    # Create the 2*2K matrix C
    C1 = np.dot(u_mat.T, temp1 * Y)
    C2 = np.dot(u_mat.T, temp2 * Y)
    C = block_diag(C1, C2) / N

    # Calculate the bread of the sandwich estimator
    Ainv = block_diag(inv(A1), inv(A2))
    tempMat = np.zeros((2 * K, 2))
    bread = np.hstack((np.vstack((Ainv, np.dot(C, Ainv))), np.vstack((tempMat, - np.eye(2)))))

    # Obtain the large covariance matrix
    large_cov_mat = np.dot(np.dot(bread, np.dot(meat, bread.T)), 1 / N)

    # Return the submatrix of cov for tau1 and tau0
    return large_cov_mat[2 * K:, 2 * K:]

def GetEEATT(u_mat_lambda_placebo, tau_one, tau_zero, Y, treat, u_mat, theta):
    '''    
    This function calculates the estimating equations for
    the case of estimating the treatment effect on the treated.
    Args:
      u_mat_lambda_placebo: The vector u_mat * lambda_placebo.
      ...: Other parameters to be passed to the function.
    Returns:
      A matrix with N rows, each row is the EE for the
      corresponding subject evaluated at the given values of
      lambda_placebo.
    '''

    n1 = np.sum(treat)  # Number in treatment arm
    N = len(Y)  # Sample size
    delta = n1 / N  # Parameter delta

    temp1 = (1 - treat) * CRFamilyDerivative(u_mat_lambda_placebo, theta)
    gk1 = np.multiply(u_mat, temp1[:, np.newaxis]) - np.divide(np.multiply(u_mat, treat[:, np.newaxis]), delta)
    gk2 = treat - delta
    gk3 = treat * (Y - tau_one) / delta
    gk4 = temp1 * Y - tau_zero

    return np.column_stack((gk1, gk2, gk3, gk4))

def GetEEMultiple(u_mat_lambdas, lambdas, taus, Y, treat, u_mat, theta):
    '''
    This function calculates the estimating equations for
    the case of multiple treatment arms.
    Args:
      u_mat_lambdas: The matrix u_mat * lam_mat.
      ...: Other parameters to be passed to the function.
    Returns:
      A matrix with N rows, each row is the EE for the
      corresponding subject evaluated at the given values of
      lambdas.
    '''

    # Get the number of treatment arms
    J = lambdas.shape[0]
    
    # Initialize lists to store EE for lambda_j and tau_j
    gk1_list = [None] * J
    gk2_list = [None] * J
    
    # For each treatment arm, calculate EE for lambda_j and tau_j
    for j in range(J):
        u_mat_lambda_j = u_mat_lambdas[:, j]
        temp1 = 1 * (treat == j) * CRFamilyDerivative(u_mat_lambda_j, theta)
        gk1_list[j] = u_mat * temp1[:, np.newaxis] - u_mat
        gk2_list[j] = 1 * (treat == j) * Y * CRFamilyDerivative(u_mat_lambda_j, theta) - taus[j]
    
    # Combine the EE for each treatment arm
    gk1_matrix = np.column_stack(gk1_list)
    gk2_matrix = np.column_stack(gk2_list)
    
    return np.hstack((gk1_matrix, gk2_matrix))


def GetCovATT(fitted_point_est, Y, treat, u_mat, theta):
    """
    This function calculates the estimate of the
    covariance matrix for the parameter vector.
    Args:
      fitted_point_est: The output list of the function
                        GetPointEstATT.
      ...: Other parameters to be passed.
    Returns:
      The covariance matrix for our parameters
      tau_one and tau_zero.
    """
    N = len(Y)
    n1 = np.sum(treat)
    delta = n1 / N  # Probability of treatment assignment
    K = u_mat.shape[1]

    lambda_placebo = fitted_point_est['lambda_placebo']
    u_mat_lambda_placebo = np.dot(u_mat, lambda_placebo)

    tau_one = fitted_point_est['tau_one']
    tau_zero = fitted_point_est['tau_zero']

    gk = GetEEATT(u_mat_lambda_placebo, tau_one, tau_zero, Y, treat, u_mat, theta)
    meat = np.dot(gk.T, gk) / N

    temp1 = (1 - treat) * CRFamilySecondDerivative(u_mat_lambda_placebo, theta)
    tempA = np.multiply(u_mat, temp1[:, np.newaxis])
    A = np.column_stack((np.dot(tempA.T, u_mat), np.dot(u_mat.T, treat) / (delta ** 2))) / N
    A = np.row_stack((A, np.append(np.zeros(K), -1)))

    C = np.zeros((2, K + 1))
    C[1, :-1] = np.dot(u_mat.T, temp1 * Y) / N

    A_inv = inv(A)
    temp_mat = np.zeros((K + 1, 2))
    bread = np.column_stack((np.row_stack((A_inv, np.dot(C, A_inv))), np.row_stack((temp_mat, -np.eye(2)))))

    large_cov_mat = np.dot(np.dot(bread, meat), bread.T) / N

    return large_cov_mat[-2:, -2:]  # Return the submatrix for covariance of tau0 and tau1

def GetCovMultiple(fitted_point_est, Y, treat, u_mat, theta):
    """
    This function calculates the estimate of the
    covariance matrix for the parameter vector with
    multiple treatments.
    The function returns the J * J covariance matrix
    for the parameters tau_0, tau_1, ..., tau_{J-1}
    Args:
      fitted_point_est: The output list of the function
                        GetPointEstMultiple.
      ...: Other parameters to be passed.
    Returns:
      The covariance matrix for our parameters
      tau0, tau1, ..., tau_{J-1}.
    """
    N = len(Y)
    K = u_mat.shape[1]
    J = np.unique(treat).size
 

    # Obtain estimated lambdas and taus
    lambdas = fitted_point_est['lam_mat']
    taus = fitted_point_est['tau_treatment_j']
    u_mat_lambdas = np.dot(u_mat, lambdas.T)  # Equivalent to tcrossprod in R

    # Calculate the meat matrix
    gk = GetEEMultiple(u_mat_lambdas=u_mat_lambdas, lambdas=lambdas, taus=taus, Y=Y, treat=treat, u_mat=u_mat, theta=theta)
    meat = np.dot(gk.T, gk) / N

    # Initialize A, A inverse, and C lists
    A_list = []
    A_inv_list = []
    C_list = []

    for j in range(J):
        u_mat_lambda_j = u_mat_lambdas[:, j]
        temp1 = (treat == j) * CRFamilySecondDerivative(u_mat_lambda_j, theta)

        tempA = u_mat * temp1[:, np.newaxis]  # Broadcasting for element-wise multiplication
        A_j = np.dot(tempA.T, u_mat) / N
        A_list.append(A_j)
        A_inv_list.append(np.linalg.inv(A_j))

        C_j = np.dot(tempA.T, Y * temp1) / N
        C_list.append(C_j.T)

    # Construct block diagonal matrices for A_inv and C
    A_inv = block_diag(*A_inv_list)
    C = block_diag(*C_list)
    temp_mat = np.zeros((J * K, J))

    # Construct the bread matrix
    bread = np.hstack((np.vstack((A_inv, np.dot(C, A_inv))),
                            np.vstack((temp_mat, -np.eye(J)))))

    # Calculate full covariance matrix
    large_cov_mat = np.dot(np.dot(bread, meat), bread.T) / N

    # Extract and return the submatrix for the covariance of Taus
    return large_cov_mat[J*K:, J*K:]
