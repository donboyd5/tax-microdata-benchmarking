import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO


# Scaling function for A and b
def scale_A_b(A, b):
    # Compute the scaling factors (L2 norm of each column of A)
    scaling_factors = np.sqrt(A.power(2).sum(axis=0)).A1  # Extract as 1D array
    # Avoid division by zero by setting small values to 1
    scaling_factors[scaling_factors < 1e-10] = 1.0

    # Scale the columns of A
    A_scaled = A.multiply(1 / scaling_factors)

    # Optionally scale b (this step depends on the relative scale of b)
    b_scaled = b / np.linalg.norm(b)

    return A_scaled, b_scaled, scaling_factors


# Example input: large sparse matrix A and vector b
m, n = 500, 200000  # Dimensions of A: m x n
density = 0.001  # Sparsity of A

# Create a random sparse matrix A and a dense vector b
A_scipy = csr_matrix(np.random.rand(m, n) * (np.random.rand(m, n) < density))
b = np.random.rand(m)
lambd = 0.1  # Regularization parameter

# Scale A and b
A_scaled, b_scaled, scaling_factors = scale_A_b(A_scipy, b)

# Convert SciPy sparse matrix to JAX-compatible BCOO format
A_jax = BCOO.from_scipy_sparse(A_scaled)


# Define the residual function using JAX
def residual_function(x, A, b, lambd):
    A_dot_x = A @ (x / scaling_factors)  # Apply scaling factor to x
    residual = A_dot_x - b
    regularization = jnp.sqrt(lambd) * (x - 1)  # Regularization term
    return jnp.concatenate([residual, regularization])


# Objective function for minimization (sum of squared residuals)
def objective_function(x, A, b, lambd):
    res = residual_function(x, A, b, lambd)
    return jnp.sum(jnp.square(res))


# Function to compute the JVP using JAX
def jvp_residual_function(x, A, b, lambd, v):
    # Computes the Jacobian-vector product (JVP) without forming the full Jacobian
    _, jvp = jax.jvp(lambda x: residual_function(x, A, b, lambd), (x,), (v,))
    return jvp


# Define gradient using JAX autodiff
def gradient_function(x, A, b, lambd):
    grad = jax.grad(objective_function)(x, A, b, lambd)
    return np.asarray(grad)


# Initial guess for x
x0 = np.ones(n)

# Set up bounds for non-negative constraints
bounds = [(0, None) for _ in range(n)]

# Call minimize with L-BFGS-B
result = minimize(
    fun=objective_function,  # Objective function
    x0=x0,  # Initial guess
    jac=gradient_function,  # Gradient (JVP-based)
    args=(A_jax, b_scaled, lambd),  # Arguments (use scaled b)
    method="L-BFGS-B",  # Use L-BFGS-B solver for large-scale problems
    bounds=bounds,  # Non-negative bounds
    tol=1e-10,
    options={"disp": True},  # Display convergence info
)

# Extract the optimized solution and apply inverse scaling
x_opt = result.x / scaling_factors  # Apply inverse scaling to x
print(f"Optimized x: {x_opt}")

print(f"Optimized x: {x_opt}")
print(np.quantile(x_opt, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]))

diff = A_scipy @ x_opt - b
pdiff = diff / b
print(np.quantile(pdiff, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]))
