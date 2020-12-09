import scipy.stats as st
import scipy.optimize
import warnings
import numpy as np

def log_like_iid_gamma(params, n):
    """Log likelihood for i.i.d. Gamma measurements, parametrized
    by x, a"""

    beta, alpha = params
    if n.any() <= 0:
        return -np.inf
    if beta <= 0: 
        return -np.inf
    if alpha<=0: 
        return -np.inf    
    
    return st.gamma.logpdf(n , alpha, loc=0, scale=1/beta).sum()

#Code based on Bois (2020)

def mle_iid_gamma(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
   Gamma measurements, parametrized by x, a (cov matrix)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_gamma(params, n),
            x0=np.array([2.00, 0.005]),
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
def log_like_iid_bespoke(params, n):
    beta, dbeta = params
    if n.any() <= 0:
        return -np.inf
    if beta <= 0: 
        return -np.inf
    if dbeta <= 0: 
        return -np.inf    
    
    return np.sum(np.log(beta)+np.log(beta+dbeta)-np.log(dbeta)-beta*n+np.log(1-np.exp(-dbeta*n)))

def mle_iid_bespoke(n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_bespoke(params, n),
            x0=np.array([1.00, 0.5]),
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)