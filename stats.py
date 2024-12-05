from sklearn.metrics import mean_squared_error
'''
Compute statistics to analyze performance of the KF 
'''

def compute_mse(gt, estim):
    return (gt - estim)**2