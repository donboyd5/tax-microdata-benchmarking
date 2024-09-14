import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.optimize import minimize, Bounds
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

REGULARIZATION_DELTA = 1.0e-9
OPTIMIZE_FTOL = 1e-8
OPTIMIZE_GTOL = 1e-8
OPTIMIZE_MAXITER = 5000
OPTIMIZE_IPRINT = 1  # 20 is a good diagnostic value; set to 0 for production

def create_A_dense_b(num_targets, num_weights, density, noise, seed):
    np.random.seed(seed)
    
    # targets are rows, tax units are columns
    A_dense = np.random.rand(num_targets, num_weights)
    # Zero out some elements to achieve the desired density
    if density < 1.0:
        mask = np.random.rand(num_targets, num_weights) > density
        A_dense[mask] = 0    
    
    x0 = np.ones(num_weights)        
    Ax0 = A_dense @ x0
    # make b vector close to initial Ax with slight perturbations
    b = Ax0 * np.random.normal(1, noise, num_targets)  # add small noise
    
    return A_dense, b


def objective_function(x, *args):
    """
    Objective function for minimization.
    Search for NOTE in this file for methodological details.
    https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=320
    """
    A, b, delta = args  # A is a jax sparse matrix
    ssq_target_deviations = jnp.sum(jnp.square(A @ x - b))
    ssq_weight_deviations = jnp.sum(jnp.square(x - 1.0))
    return ssq_target_deviations + delta * ssq_weight_deviations


JIT_FVAL_AND_GRAD = jax.jit(jax.value_and_grad(objective_function))


def create_area_weights():

    jax.config.update("jax_platform_name", "cpu")  # ignore GPU/TPU if present
    jax.config.update("jax_enable_x64", True)  # use double precision floats
    
    qtiles = [0.0, 0.01, 0.1, 0.25, .5, 0.75, 0.9, 0.99, 1.0]
    
    # define problem chracteristics
    num_targets, num_weights, density, noise, seed = 20, 200_000, 0.6, 0.3, 123
    
    A_dense_unscaled, b_unscaled = create_A_dense_b(num_targets, num_weights, density, noise, seed)
    
    A_dense_scaled = A_dense_unscaled / b_unscaled[:, np.newaxis] 
    b = np.ones_like(b_unscaled) 

    A = BCOO.from_scipy_sparse(csr_matrix(A_dense_scaled))  # A is JAX sparse matrix      
    
    time0 = time.time()
    res = minimize(
        fun=JIT_FVAL_AND_GRAD,  # objective function and its gradient
        x0=np.ones(num_weights),  # initial guess for weight ratios
        jac=True,  # use gradient from JIT_FVAL_AND_GRAD function
        args=(A, b, REGULARIZATION_DELTA),  # fixed arguments of objective func
        method="L-BFGS-B",  # use L-BFGS-B algorithm
        bounds=Bounds(0.0, np.inf),  # consider only non-negative weight ratios
        options={
            "maxiter": OPTIMIZE_MAXITER,
            "ftol": OPTIMIZE_FTOL,
            "gtol": OPTIMIZE_GTOL,
            "iprint": OPTIMIZE_IPRINT,
        },
    )
    time1 = time.time()
    
    # describe problem characteristics - print AFTER any iteration printing so that
    # we can see them near the optimization results
    density = np.count_nonzero(A_dense_unscaled) / A_dense_unscaled.size
    b0_unscaled=A_dense_unscaled @ np.ones(num_weights)
    init_rmse_targets = np.sqrt(np.mean(np.square(b0_unscaled - b_unscaled)))
    bratio0 = b0_unscaled / b_unscaled
    qbratio0=np.quantile(bratio0, qtiles)    
    
    prob_info = (
        "\nProblem characteristics: \n"
        f"  A shape:                     {A.shape}\n"
        f"  b shape:                     {b.shape}\n"
        f"  A density:                   {density:.3f}\n"
        f"  Ax0 - b noise:               {noise:.3f}\n"
        f"  Random seed:                 {seed}\n"
        f"  Objective function at x0=1:  {objective_function(np.ones(num_weights), A, b, REGULARIZATION_DELTA):.9e}\n"
        f"  Target rmse at x0=1:         {init_rmse_targets:.9e}\n"
        f"  Quantile points:             {', '.join(['{:.2f}'.format(i) for i in (qtiles)])}\n"
        "  Ax / b quantiles at x0=1:\n"
        f"                               {', '.join(['{:.3f}'.format(i) for i in (qbratio0)])}\n"
        )
    print(prob_info)    
    
    prob_params = (
        f"Parameters:\n"        
        f"  REGULARIZATION_DELTA=   {REGULARIZATION_DELTA:e}\n"
        f"  OPTIMIZE_FTOL=          {OPTIMIZE_FTOL:e}\n"
        f"  OPTIMIZE_GTOL=          {OPTIMIZE_GTOL:e}\n"
        )
    print(prob_params)    
    
    bres=A_dense_unscaled @ res.x
    rmse_targets = np.sqrt(np.mean(np.square(bres - b_unscaled)))
    rmse_xratios = np.sqrt(np.mean(np.square(res.x - 1.)))
    bratio = bres / b_unscaled
    qbratio=np.quantile(bratio, qtiles)
    qx=np.quantile(res.x, qtiles)
    
    res_info = (
        f"Key results:\n"
        f"  >>> optimization execution time: {(time1-time0):.1f} secs\n"
        f"  # iterations:          {res.nit}\n"
        f"  objective function:    {res.fun:.9e}\n"
        f"  target rmse:           {rmse_targets:.9e}\n"
        f"  (x - 1) rmse:          {rmse_xratios:.9e}\n"        
        f"  Quantile points:       {', '.join(['{:.2f}'.format(i) for i in (qtiles)])}\n"
        "  Ax / b quantiles at res.x:\n"
        f"                         {', '.join(['{:.3f}'.format(i) for i in (qbratio)])}\n"
        "  res.x quantiles:\n"
        f"                         {', '.join(['{:.3f}'.format(i) for i in (qx)])}\n"
        )
    print(res_info)
    print(">>> full optimization results:\n", res)
    return 

if __name__ == "__main__":
    create_area_weights()
