from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import chi2
'''
Compute statistics to analyze performance of the KF 
'''

def compute_mse(gt, estim):
    return (gt - estim)**2

def compute_confidence_bounds(dof, alpha=0.05):
    """
    Compute the lower and upper confidence bounds for a chi-squared distribution.
    
    :param dof: Degrees of freedom.
    :param alpha: Significance level (default is 0.05 for 95% confidence).
    :return: Tuple containing (lower_bound, upper_bound).
    """
    lower_bound = chi2.ppf(alpha / 2, dof)
    upper_bound = chi2.ppf(1 - alpha / 2, dof)
    return lower_bound, upper_bound

def compute_nees(x_hat, vals, P):
    
    nees = []
    columns_to_select = ['x', 'y', 'lat_vel', 'long_vel']
    # columns_to_select = ['x', 'y', 'vel', 'yaw', 'yaw_rate']
    x_real = vals[columns_to_select].to_numpy()

    xhat = x_hat.to_numpy()

    for i in range(len(xhat)):
        diff = np.subtract(x_real[i], xhat[i])
        nees.append(np.matmul(np.matmul(diff.T, np.linalg.inv(P[i])), diff))

    return nees
    
def compute_nis(y, S):

    nis = []

    for i in range(len(y)):
        nis.append(np.matmul(np.matmul(y[i].T, np.linalg.inv(S[i])), y[i]))

    return np.array([item[0][0] for item in nis])