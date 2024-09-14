import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.optimize import minimize, Bounds
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO


FIRST_YEAR = 2021
LAST_YEAR = 2074
INFILE_PATH = STORAGE_FOLDER / "output" / "tmd.csv.gz"
WTFILE_PATH = STORAGE_FOLDER / "output" / "tmd_weights.csv.gz"
GFFILE_PATH = STORAGE_FOLDER / "output" / "tmd_growfactors.csv"
POPFILE_PATH = STORAGE_FOLDER / "input" / "cbo_population_forecast.yaml"

REGULARIZATION_DELTA = 1.0e-9
OPTIMIZE_FTOL = 1e-8
OPTIMIZE_GTOL = 1e-8
OPTIMIZE_MAXITER = 5000
OPTIMIZE_IPRINT = 1  # 20 is a good diagnostic value; set to 0 for production
OPTIMIZE_RESULTS = False  # set to True to see complete optimization results
DUMP_ALL_TARGET_DEVIATIONS = False  # set to True only for diagnostic work

def create_A_dense_b(num_targets, num_weights, noise=0.1, seed=42):
    np.random.seed(seed)
    
    # targets are rows, tax units are columns
    A_dense = np.random.rand(num_targets, num_weights)    
    x0 = np.ones(num_weights)        
    Ax0 = A_dense @ x0
    # make b vector close to initial Ax with slight perturbations
    b = Ax0 * np.random.normal(1, noise, num_targets)  # add small noise
    
    return A_dense, b


def target_rmse(wght, target_matrix, target_array):
    """
    Return RMSE of the target deviations given specified arguments.
    """
    act = np.dot(wght, target_matrix)
    act_minus_exp = act - target_array
    ratio = act / target_array
    if DUMP_ALL_TARGET_DEVIATIONS:
        for tnum, ratio_ in enumerate(ratio):
            print(
                f"TARGET{(tnum + 1):03d}:ACT-EXP,ACT/EXP= "
                f"{act_minus_exp[tnum]:16.9e}, {ratio_:.3f}"
            )
    # show distribution of target ratios
    bins = [
        0.0,
        0.4,
        0.8,
        0.9,
        0.99,
        0.9995,
        1.0005,
        1.01,
        1.1,
        1.2,
        1.6,
        2.0,
        3.0,
        4.0,
        5.0,
        np.inf,
    ]
    tot = ratio.size
    print(f"DISTRIBUTION OF TARGET ACT/EXP RATIOS (n={tot}):")
    print(f"  with REGULARIZATION_DELTA= {REGULARIZATION_DELTA:e}")
    header = (
        "low bin ratio    high bin ratio"
        "    bin #    cum #     bin %     cum %"
    )
    print(header)
    out = pd.cut(ratio, bins, right=False, precision=6)
    count = pd.Series(out).value_counts().sort_index().to_dict()
    cum = 0
    for interval, num in count.items():
        cum += num
        if cum == 0:
            continue
        line = (
            f">={interval.left:13.6f}, <{interval.right:13.6f}:"
            f"  {num:6d}   {cum:6d}   {num/tot:7.2%}   {cum/tot:7.2%}"
        )
        print(line)
        if cum == tot:
            break
    # return RMSE of ACT-EXP targets
    return np.sqrt(np.mean(np.square(act_minus_exp)))


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


def weight_ratio_distribution(ratio):
    """
    Print distribution of post-optimized to pre-optimized weight ratios.
    """
    bins = [
        0.0,
        1e-6,
        0.1,
        0.2,
        0.5,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
        1.05,
        1.1,
        1.15,
        1.2,
        2.0,
        5.0,
        1e1,
        1e2,
        1e3,
        1e4,
        1e5,
        np.inf,
    ]
    tot = ratio.size
    print(f"DISTRIBUTION OF AREA/US WEIGHT RATIO (n={tot}):")
    print(f"  with REGULARIZATION_DELTA= {REGULARIZATION_DELTA:e}")
    header = (
        "low bin ratio    high bin ratio"
        "    bin #    cum #     bin %     cum %"
    )
    print(header)
    out = pd.cut(ratio, bins, right=False, precision=6)
    count = pd.Series(out).value_counts().sort_index().to_dict()
    cum = 0
    for interval, num in count.items():
        cum += num
        if cum == 0:
            continue
        line = (
            f">={interval.left:13.6f}, <{interval.right:13.6f}:"
            f"  {num:6d}   {cum:6d}   {num/tot:7.2%}   {cum/tot:7.2%}"
        )
        print(line)
        if cum == tot:
            break
    ssqdev = np.sum(np.square(ratio - 1.0))
    print(f"SUM OF SQUARED AREA/US WEIGHT RATIO DEVIATIONS= {ssqdev:e}")


# -- High-level logic of the script:


def create_area_weights_file():

    jax.config.update("jax_platform_name", "cpu")  # ignore GPU/TPU if present
    jax.config.update("jax_enable_x64", True)  # use double precision floats
    
    num_targets, num_weights, noise = 10, 20_000, 0.2
    
    A_dense, b = create_A_dense_b(num_targets, num_weights, noise, seed=123)
    print(A_dense.shape)
    print(b.shape)   
    # density = np.count_nonzero(A_dense) / A_dense.size
    # print(f"target_matrix sparsity ratio = {(1.0 - density):.3f}")

    A = BCOO.from_scipy_sparse(csr_matrix(A_dense))  # A is JAX sparse matrix
    
    print(
       # f"OPTIMIZE_WEIGHT_RATIOS: target_matrix.shape= {target_matrix.shape}\n"
        f"REGULARIZATION_DELTA= {REGULARIZATION_DELTA:e}"
    )

    b0=A_dense @ np.ones(num_weights)
    print("initial proportionate differences")
    print(['{:.3f}'.format(i) for i in (b0 / b)])
    
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
            "disp": OPTIMIZE_IPRINT != 0,
        },
    )
    time1 = time.time()
    res_summary = (
        f">>> optimization execution time: {(time1-time0):.1f} secs"
        f"  iterations={res.nit}  success={res.success}\n"
        f">>> message: {res.message}\n"
        f">>> L-BFGS-B optimized objective function value: {res.fun:.9e}"
    )
    print(res_summary)
    print(">>> full optimization results:\n", res)
    exit()        
        
    wght_area = res.x * wght_us
    rmse = target_rmse(wght_area, target_matrix, target_array)
    print(f"AREA-OPTIMIZED_TARGET_RMSE= {rmse:.9e}")
    weight_ratio_distribution(res.x)

    return rmse


if __name__ == "__main__":
    create_area_weights_file()
