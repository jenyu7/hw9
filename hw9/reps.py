import numpy as np
from tqdm import tqdm

#Set up Numpy random generator
rg = np.random.default_rng()

def draw_parametric_bs_reps_mle(
    mle_fun, gen_fun, data, args=(), size=1, progress_bar=False
):
    """Draw parametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(*params, size)`.
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    params = mle_fun(data, *args)

    if progress_bar:
        iterator = tqdm(range(size))
    else:
        iterator = range(size)

    return np.array(
        [mle_fun(gen_fun(*params, size=len(data), *args)) for _ in iterator]
    )

#Generates samples from the model distribution.
def sp_gamma(beta, alpha, size):
    return rg.gamma(alpha, 1/beta, size=size)
    
    