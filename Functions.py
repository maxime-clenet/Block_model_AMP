import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm, truncnorm
import seaborn as sns
# ------------------ Matrix Generation ------------------ #

def elliptic_normal_matrix_opti(n=100, rho=0):
    """
    Generate an elliptic Gaussian random matrix of size (n, n) with correlation term rho.
    
    Parameters
    ----------
    n : int
        Dimension of the square matrix.
    rho : float
        Correlation between upper and lower triangle off-diagonal elements.

    Returns
    -------
    np.ndarray
        Symmetric matrix with entries sampled from an elliptic normal distribution.
    """
    # Define bivariate normal distribution parameters
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    
    # Diagonal entries are standard normal
    A = np.diag(np.random.randn(n))
    
    # Generate upper-triangular off-diagonal entries
    num_off_diag = n * (n - 1) // 2
    upper_samples = np.random.multivariate_normal(mean, cov, num_off_diag)
    
    # Assign values to upper and lower triangle
    upper_indices = np.triu_indices(n, k=1)
    A[upper_indices] = upper_samples[:, 0]
    A.T[upper_indices] = upper_samples[:, 1]
    
    return A

def block_matrix(n_size, beta, s, rho):
    """
    Generate a structured block matrix with elliptic and correlated entries.
    
    Parameters
    ----------
    n_size : int
        Size of the square matrix.
    beta : list of floats
        Relative proportions of each block (must sum to 1).
    s : np.ndarray, shape (B, B)
        Standard deviations for each block.
    rho : np.ndarray, shape (B, B)
        Correlation coefficients for entries in each block.

    Returns
    -------
    np.ndarray
        Block-structured matrix of size (n_size, n_size).
    """
    s = s / np.sqrt(n_size)
    beta = np.array(beta)
    
    # Sanity checks
    if not np.isclose(np.sum(beta), 1):
        raise ValueError("`beta` must sum to 1.")
    
    B = len(beta)
    if s.shape != (B, B) or rho.shape != (B, B):
        raise ValueError("`s` and `rho` must be square matrices of shape (B, B).")
    
    if not np.allclose(rho, rho.T):
        raise ValueError("`rho` must be symmetric.")
    
    if not np.all((-1 <= rho) & (rho <= 1)):
        raise ValueError("All entries in `rho` must be in [-1, 1].")
    
    # Determine block sizes from proportions
    block_sizes = (beta * n_size).astype(int)
    block_sizes[-1] += n_size - np.sum(block_sizes)  # Adjust for rounding error
    
    A = np.zeros((n_size, n_size))  # Initialize final matrix

    row_start = 0
    for i in range(B):
        row_end = row_start + block_sizes[i]
        col_start = 0
        for j in range(B):
            col_end = col_start + block_sizes[j]
            
            if i == j:
                # Diagonal block: elliptic normal distribution
                block = elliptic_normal_matrix_opti(n=block_sizes[i], rho=rho[i, i])
                block *= s[i, i]
                A[row_start:row_end, col_start:col_end] = block
            else:
                # Off-diagonal block: correlated off-diagonal entries
                mean = [0, 0]
                cov = [[1, rho[i, j]], [rho[i, j], 1]]
                
                # Generate matrix of shape (rows_i, cols_j, 2)
                off_diag = np.random.multivariate_normal(mean, cov, 
                                                         size=(block_sizes[i], block_sizes[j]))
                
                block_ij = off_diag[:, :, 0] * s[i, j]
                block_ji = off_diag[:, :, 1] * s[j, i]
                
                # Fill upper and lower blocks
                A[row_start:row_end, col_start:col_end] = block_ij
                A[col_start:col_end, row_start:row_end] = block_ji.T
            
            col_start = col_end
        row_start = row_end

    return A
# ------------------ Fixed-Point Methods ------------------ #

def compute_fixed_point_root(beta, s, rho, r):
    K = len(beta)

    if not np.allclose(rho, rho.T):
        raise ValueError("The `rho` matrix must be symmetric.")

    def fixed_point_equations(x):
        delta = x[:K]
        sigma = x[K:2*K]
        gamma = x[2*K:]

        delta_eqs = np.zeros(K)
        sigma_eqs = np.zeros(K)
        gamma_eqs = np.zeros(K)

        for i in range(K):
            delta_eqs[i] = 1 - delta[i] - sum(
                beta[j] * rho[i, j] * s[i, j] * s[j, i] * (gamma[j] / delta[j])
                for j in range(K)
            )
            sigma_eqs[i] = sigma[i]**2 - sum(
                beta[j] * s[i, j]**2 / delta[j]**2 *
                norm.expect(lambda x: (sigma[j] * x + r[j])**2, loc=0, scale=1)
                for j in range(K)
            )
            gamma_eqs[i] = gamma[i] - (1 - norm.cdf(-r[i] / sigma[i]))

        return np.concatenate([delta_eqs, sigma_eqs, gamma_eqs])

    x0 = np.concatenate([np.ones(K), np.ones(K), np.ones(K)])
    result = optimize.root(fixed_point_equations, x0, method='hybr')

    if not result.success:
        print("Warning: Root solver did not converge.")

    delta = result.x[:K]
    sigma = result.x[K:2*K]
    gamma = result.x[2*K:]
    return delta, sigma, gamma

def compute_fixed_point(beta, s, rho, r, tol=1e-6, max_iter=1000):
    K = len(beta)
    delta = np.ones(K)
    sigma = np.ones(K)
    gamma = np.ones(K)

    if not np.allclose(rho, rho.T):
        raise ValueError("The `rho` matrix must be symmetric.")

    for _ in range(max_iter):
        delta_new = np.zeros(K)
        sigma_new = np.zeros(K)
        gamma_new = np.zeros(K)

        for i in range(K):
            delta_new[i] = 1 - sum(
                beta[j] * rho[i, j] * s[i, j] * s[j, i] * (gamma[j] / delta[j])
                for j in range(K)
            )
            sigma_new[i] = np.sqrt(sum(
                beta[j] * s[i, j]**2 / delta[j]**2 *
                norm.expect(lambda x: max(0, (sigma[j] * x + r[j]))**2, loc=0, scale=1)
                for j in range(K)
            ))
            gamma_new[i] = 1 - norm.cdf(-r[i] / sigma[i])

        if (
            np.max(np.abs(delta_new - delta)) < tol and
            np.max(np.abs(sigma_new - sigma)) < tol and
            np.max(np.abs(gamma_new - gamma)) < tol
        ):
            break

        delta, sigma, gamma = delta_new, sigma_new, gamma_new
    else:
        print("Warning: Did not converge within the maximum number of iterations.")

    return delta, sigma, gamma

def compute_fixed_point_final(beta, s, rho, r):
    delta, sigma, gamma = compute_fixed_point(beta, s, rho, r)
    mu = r / delta
    sigma_new = sigma / delta

    a_scaled = (0 - mu) / sigma_new
    b_scaled = np.full_like(mu, np.inf)

    variance = truncnorm.var(a_scaled, b_scaled, loc=mu, scale=sigma_new)
    return variance, gamma
