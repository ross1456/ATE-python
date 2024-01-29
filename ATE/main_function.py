import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from .point_estimates import GetPointEstSimple, GetPointEstATT, GetPointEstMultiple
from .variance_estimates import GetEESimple, GetCovSimple, GetEEATT, GetCovATT, GetCovMultiple


class ATE:
    # This is the main function available to the user.
    # This creates an ATE self.result. The self.result contains point
    # estimates and variance estimates. It also has
    # generic S3 methods such as plot and summary + print.
    # Args:
    #   Y: Response vector.
    #   X: Design/Covariate matrix.
    #   treat: Vector of treatment indicator. Must be of
    #          the form (0, 1, ...).
    #   ATT: Indicate if treatment effect on the treated is
    #        to be estimated.
    #   verbose: Indicate if extra statments should be printed
    #            to show progress of the function while it runs.
    #   max.iter: Maximum no. of iterations for BFGS algorithms.
    #   tol: Tolerance of the algorithm for stopping conditions.
    #   initial.values: Matrix of initial values for BFGS algorithm.
    #   backtrack.alpha, backtrack.beta: Parameters for backtrack line
    #                                    search algorithm.
    # Returns:
    #   An self.result of class 'ATE'. This contains point estimates and
    #   and variance covariance matrix of our estimates. Along with other
    #   information about our self.result such as data used for estimation.
    def __init__(self, theta=1,
                  ATT: bool = False, verbose: bool = False, 
                  max_iter: int = 200, tol: float = 1e-8, 
                  initial_values = None, backtrack_alpha: float = 0.3,
                  backtrack_beta: float = 0.5):
        
        assert isinstance(ATT, bool), "ATT must be a boolean"
        assert isinstance(verbose, bool), "verbose must be a boolean"
        assert isinstance(max_iter, int), "max_iter must be an integer"
        assert isinstance(theta, (int, float)), "'theta' must be a real number."
        
        self.theta = theta
        self.ATT = ATT
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.names = None
        self.initial_values = initial_values
        self.backtrack_alpha = backtrack_alpha
        self.backtrack_beta = backtrack_beta

        self.is_fit_done = False
        
        #import the necessary functions
        self.get_point_Sim = GetPointEstSimple
        self.get_point_ATT = GetPointEstATT
        self.get_point_Mul = GetPointEstMultiple
        self.get_cov_Sim = GetCovSimple
        self.get_cov_ATT = GetCovATT
        self.get_cov_Mul = GetCovMultiple
        self.get_ee_Sim = GetEESimple
        self.get_ee_ATT = GetEEATT
        
    def fit(self, Y, treat, X):    

        # Check input types and convert pandas DataFrame to numpy matrix
        if isinstance(X, pd.DataFrame):
            X = X.values
            self.names = X.columns
        if isinstance(Y, pd.Series) | isinstance(Y, pd.DataFrame):
            Y = Y.values
        if isinstance(treat, pd.Series) | isinstance(treat, pd.DataFrame):
            treat = treat.values

        self.treat = treat
        self.X = X

        assert isinstance(X, np.ndarray),"Input 'X' should be Numpy array or pandas"
        assert isinstance(Y, np.ndarray),"Input 'Y' should be Numpy array or pandas"
        assert isinstance(treat, np.ndarray), "Input 'treat' should be Numpy array or pandas"


        # Ensure that X is a 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            print("Warning: Data matrix 'X' is a vector, will be treated as n x 1 matrix.")

        # Perform checks
        n, k = X.shape
        if n != len(Y):
            raise ValueError("Dimensions of covariates and response do not match.")
        
        unique_treat = np.unique(treat)
        self.J = len(unique_treat)
        if self.J == 1:
            raise ValueError("There must be at least two treatment arms")
        if not np.array_equal(unique_treat, np.arange(self.J)):
            raise ValueError("The treatment levels must be labelled 0, 1, 2, ...")

        if self.J > 2 and self.ATT:
            raise ValueError("For ATT == True, must have only 2 treatment arms.")

        # Set initial values if not provided
        self.K = k + 1
        if self.initial_values is None:
            initial_values = np.zeros((self.J, self.K)) if not self.ATT else np.zeros(self.K)

        if self.ATT and initial_values.ndim != 1:
            raise ValueError("For ATT == True, only need one vector of initial values.")
        if not self.ATT and (not isinstance(initial_values, np.ndarray) or initial_values.shape != (self.J, self.K)):
            raise ValueError("Initial values must be a matrix with dimensions J x K.")

        # Determine the problem category
        gp = "simple"
        if self.ATT:
            gp = "ATT"
        elif self.J > 2:
            gp = "MT"

        # Perform estimation based on the problem category
        if (gp=="simple"):
            est = self.get_point_Sim(initial_values[0], initial_values[1], X=X, Y=Y, treat=treat,
                                      theta=self.theta, max_iter=self.max_iter, tol=self.tol, backtrack_alpha=self.backtrack_alpha,
                                        backtrack_beta=self.backtrack_beta, verbose=self.verbose)
            if (self.verbose):
                print("\nEstimating Variance")
            
            cov_mat = self.get_cov_Sim(est, Y, treat, np.c_[np.ones(n), X], self.theta)
        elif (gp == "ATT"):
            est = self.get_point_ATT(initial_values, X=X, Y=Y,treat=treat,
                                      theta=self.theta, max_iter=self.max_iter, tol=self.tol, backtrack_alpha=self.backtrack_alpha,
                                      backtrack_beta=self.backtrack_beta, verbose=self.verbose)
            if (self.verbose):
                print("\nEstimating Variance")

            cov_mat = self.get_cov_ATT(est, Y, treat, np.c_[np.ones(n), X], self.theta)
        elif (gp=="MT"):
            est = self.get_point_Mul(initial_values, X=X, Y=Y, treat=treat, 
                                      theta=self.theta, max_iter=self.max_iter, tol=self.tol, backtrack_alpha=self.backtrack_alpha,
                                      backtrack_beta=self.backtrack_beta, verbose=self.verbose)
            if (self.verbose):
                print("\nEstimating Variance")
            cov_mat = self.get_cov_Mul(est, Y, treat, np.c_[np.ones(n), X], self.theta)
        
        self. gp = gp

        
        keys = ["vcov", "X", "Y", "treat","theta","gp","J","K"]
        values = [cov_mat, X, Y, treat, self.theta, gp, self.J, self.K]
        add_dict = dict(zip(keys, values))
        est.update(add_dict)
        self.est = est


    # Construct the ATE self.result    
        result = self.est
        
        if (self.gp=="simple"):
            estimate = {
                "E[Y(1)]": est["tau_one"], 
                "E[Y(0)]": est["tau_zero"],
                "ATE":est["tau"]
            }
            result["estimate"] = estimate
            result["tau_one"] = None
            result["tau_zero"] = None
            result["tau"] = None
        elif (self.gp=="ATT"):
            estimate = {
                "E[Y(1)]": est["tau_one"], 
                "E[Y(0)]": est["tau_zero"],
                "ATE":est["tau"]
            }
            result["estimate"] = estimate
            result["tau_one"] = None
            result["tau_zero"] = None
            result["tau"] = None
        else:
            estimate_names = [f"E[Y({j})]" for j in range(self.J)]
            result['estimate'] = dict(zip(estimate_names, result['tau_treatment_j']))
            result.pop('tau_treatment_j', None)
        
        self.result = result
        self.is_fit_done = True


        return result
          
    def summary(self):
        if not self.is_fit_done:
            print("Fit method must be called before printing results.")
            return

        # summary method for ATE. This function calculates
        # the SE, Z-statistic and confidence intervals
        # and P-values.
        # Args:
        #   object: An object of class 'ATE'.
        #   ...: Other arguments (not used).
        # Returns:
        #   An object of type summary.ATE. This is a list with a matrix for
        #   coefficient estimates. This matrix contains, SE, Z-statistic etc.


        # For binary treatment we calculate the variance of EY(1) - EY(0)
        # using the formula var(a-b) = var(a) + var(b) - 2cov(a,b).

        if self.result['gp'] in ['simple', 'ATT']:
                var_tau = self.result['vcov'][0, 0] + self.result['vcov'][1, 1] - 2 * self.result['vcov'][0, 1]
                se = np.sqrt(np.diag(self.result['vcov']).tolist() + [var_tau])
        else:
            # The case of multiple treatments, now we do not have a
            # specific variable like EY(1) - EY(0). So we just obtain
            # the SE for EY(0), EY(1), EY(2),... .
            se = np.sqrt(np.diag(self.result['vcov']))
        
        est = np.array(list(self.result['estimate'].values()))
        ci_lower = est  - scipy.stats.norm.ppf(0.975) * se
        ci_upper = est  + scipy.stats.norm.ppf(0.975) * se

        z_stat = est  / se
        p_values = 2 * scipy.stats.norm.sf(np.abs(z_stat))

        coef = np.column_stack((est , se, ci_lower, ci_upper, z_stat, p_values))

        # Creating a result dictionary
        result = {
            'Estimate': coef,
            'vcov': self.result['vcov'] if 'vcov' in self.result else self.result['cov'],
            'converge': self.result['converge'],
        }

        if self.result['gp'] =='simple':
            result['weights_treat'] = self.result['weights_treat']
            result['weights_placebo'] = self.result['weights_placebo']
        elif self.result['gp'] =='ATT':
            result['weights_placebo'] = self.result['weights_placebo']
        else:
            result['weights'] = self.result['weights_mat']
        
        printer = pd.DataFrame(coef,index=self.result['estimate'].keys(),columns=["Estimate", "Std. Error", "95%.Lower", "95%.Upper", "z value", "p value"])
        print(printer)
        return result

    def weighted_ecdf(self,  x, t, weights=None):
        """
        Evaluate the empirical CDF or the weighted empirical CDF at a given point.
        
        Args:
        t: A scalar point at which to evaluate the eCDF.
        x: The data points used for evaluating the eCDF.
        weights: An optional array of weights for the weighted CDF.
        
        Returns:
        The value of the empirical CDF or weighted empirical CDF at point t.
        """
        x = np.array(x)
        if weights is None:
            return np.sum(x <= t) / len(x)
        else:
            weights = np.array(weights)
            return np.sum((x <= t) * weights)



    def plot(self, *args):
        """
        # The S3 plot function for ATE.
        # This function plots the empirical CDF for all the
        # covariates for the different
        # tretment groups along with theweighted eCDF.
        # This shows the effect of covariate balancing.
        # Args:
        #   x: An object of class 'ATE'.
        #   ...: Other arguments to be passed.
        # Returns:
        #   Plots for the eCDF and weighted eCDF for each covariate.
        """
        treat = self.treat

        # Binary treatment case
        if self.gp == "simple" or self.gp == "ATT":
            ATT = self.gp == "ATT"
            
            if not ATT:
                weights_treat = self.result["weights_treat"][treat == 1]
            weights_placebo = self.result["weights_placebo"][treat == 0]

            names = self.names if self.names is not None else [f"X{i}" for i in range(1,self.X.shape[1]+1)]

            x_treat = self.X[treat == 1]
            x_placebo = self.X[treat == 0]

            for i in range(x_treat.shape[1]):
                # Special plot for binary covariates
                if len(np.unique(self.X[:, i])) == 2:
                    Treatment = x_treat[:, i]
                    Placebo = x_placebo[:, i]

                    # First plot the unweighted case
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    #Plot Treatment points in blue
                    plt.plot([0.5, 1], [2, np.mean(Treatment)], 'o', c='blue', markersize=8)
                    # Plot Placebo point in red
                    plt.plot([2, 2.5], [np.mean(Placebo), 2], 'o', c='red', markersize=8)

                    plt.ylim(0, 1)
                    plt.ylabel("Mean of group")
                    plt.title(f"Unweighted; variable: {names[i]}")
                    plt.xticks([1, 2], ["Treatment", "Placebo"])
                    if(self.ATT):
                        plt.axhline(y=np.mean(Treatment), color='k', linestyle='--')
                    else:
                        plt.axhline(y=np.mean(self.X[:, i]), color='k', linestyle='--')

                    # Plot for weighted case
                    if not ATT:
                        new_treat = np.sum(weights_treat * Treatment)
                    else:
                        new_treat = np.mean(Treatment)
                    new_placebo = np.sum(weights_placebo * Placebo)

                    plt.subplot(1, 2, 2)
                    plt.plot([0.5, 1], [2, new_treat], 'o', c='blue', markersize=8)
                    plt.plot([2, 2.5], [new_placebo, 2],'o', c='red', markersize=8)
                    plt.ylim(0, 1)
                    plt.title(f"Weighted; variable: {names[i]}")
                    plt.xticks([1, 2], ["Treatment", "Placebo"])

                    if(self.ATT):
                        plt.axhline(y=np.mean(Treatment), color='k', linestyle='--')
                    else:
                        plt.axhline(y=np.mean(self.X[:, i]), color='k', linestyle='--')
                    plt.show()

                else:  # Continuous covariates
                    my_range = [np.min(np.append(x_treat[:, i], x_placebo[:, i])), 
                                np.max(np.append(x_treat[:, i], x_placebo[:, i]))]
                    my_seq = np.linspace(my_range[0], my_range[1], 100)

                    # Calculate eCDF for treatment and placebo

                    ecdf_treat = [self.weighted_ecdf(x_treat[:, i], my_seq_val) for my_seq_val in my_seq]
                    ecdf_placebo = [self.weighted_ecdf(x_placebo[:, i], my_seq_val) for my_seq_val in my_seq]

                    # Plot the unweighted eCDF
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.plot(my_seq, ecdf_treat, 'r-', label="Treatment")
                    plt.plot(my_seq, ecdf_placebo, 'b--', label="Placebo")
                    plt.xlabel(names[i])
                    plt.ylabel("Empirical CDF")
                    plt.title("Unweighted Empirical CDF")
                    plt.legend()

                    # Plot the weighted eCDF
                    if not ATT:
                        weighted_ecdf_treat = [self.weighted_ecdf(x_treat[:, i], my_seq_val, weights_treat) for my_seq_val in my_seq]
                    else:
                        weighted_ecdf_treat = ecdf_treat

                    weighted_ecdf_placebo = [self.weighted_ecdf(x_placebo[:, i], my_seq_val, weights_placebo) for my_seq_val in my_seq]

                    plt.subplot(1, 2, 2)
                    plt.plot(my_seq, weighted_ecdf_treat, 'r-', label="Treatment")
                    plt.plot(my_seq, weighted_ecdf_placebo, 'b--', label="Placebo")
                    plt.xlabel(names[i])
                    plt.ylabel("Empirical CDF")
                    plt.title("Weighted Empirical CDF")
                    plt.legend()
                    plt.show()

        # Multiple Treatments Case
        elif self.gp == "MT":
            weights_mat = self.result["weights_mat"]
            names = self.names if self.names is not None else [f"X{i}" for i in range(self.X.shape[1])]
            J = self.J
            for i in range(self.X.shape[1]):
                # Binary covariates for multiple treatments
                if len(np.unique(self.X[:, i])) == 2:
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    means = [np.mean(self.X[self.treat == j, i]) for j in range(J)]
                    plt.scatter(range(J), means, c=['black','red','green','blue'])
                    plt.axhline(y=self.X[:,i].mean(), color='r', linestyle='--')
                    plt.ylim(0, 1)
                    plt.ylabel("Mean of group")
                    plt.xlabel("Treatment group")
                    plt.title(f"Unweighted; variable: {names[i]}")
                    plt.xticks(list(range(J)))

                    plt.subplot(1, 2, 2)
                    weighted_means = [np.sum(self.X[self.treat == j, i] * weights_mat[j, self.treat == j]) for j in range(J)]
                    plt.scatter(range(J), weighted_means, c=['black','red','green','blue'])
                    plt.axhline(y=self.X[:,i].mean(), color='r', linestyle='--')
                    plt.ylim(0, 1)
                    plt.ylabel("Mean of group")
                    plt.xlabel("Treatment group")
                    plt.title(f"Weighted; variable: {names[i]}")
                    plt.xticks(list(range(J)))
                    plt.show()


                else:  # Continuous covariates for multiple treatments
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    my_range = [np.min(self.X[:, i]), np.max(self.X[:, i])]
                    my_seq = np.linspace(my_range[0], my_range[1], 100)

                    for j in range(J):
                        ecdf_j = [self.weighted_ecdf(self.X[self.treat == j, i], my_seq_val) for my_seq_val in my_seq]
                        plt.plot(my_seq, ecdf_j, label=f"Group {j}")

                    plt.xlabel(names[i])
                    plt.ylabel("Empirical CDF")
                    plt.title("Unweighted Empirical CDF")
                    plt.legend()

                    plt.subplot(1, 2, 2)
                    for j in range(J):
                        weighted_ecdf_j = [self.weighted_ecdf(self.X[self.treat == j, i], my_seq_val, weights_mat[j, self.treat == j]) for my_seq_val in my_seq]
                        plt.plot(my_seq, weighted_ecdf_j, label=f"Group {j}")

                    plt.xlabel(names[i])
                    plt.ylabel("Empirical CDF")
                    plt.title("Weighted Empirical CDF")
                    plt.legend()

                    plt.show()
    # Usage example
    # ate = ATE(...)  # Assuming ATE is properly defined and instantiated
    # ate.plot()


