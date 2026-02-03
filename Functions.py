"""Functions.py - Block Matrix Model and Fixed-Point Analysis Tools

This module provides functions for generating structured block matrices and computing
fixed-point solutions for the Approximate Message Passing (AMP) algorithm analysis.

Main Components:
    - Matrix Generation: Functions for creating elliptic normal and block-structured matrices
    - Fixed-Point Methods: Algorithms for solving fixed-point equations in AMP theory

Author: Maxime Clenet
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm, truncnorm
import seaborn as sns

# ==================== Matrix Generation ==================== #

def elliptic_normal_matrix_opti(n=100, rho=0):
    """Generate an elliptic Gaussian random matrix with controlled correlation.
    
    Creates a symmetric (n, n) matrix where diagonal entries are standard normal and 
    off-diagonal entries follow a bivariate normal distribution with specified correlation.
    This is useful for modeling covariance structures in block matrix models.
    
    Parameters
    ----------
    n : int, optional
        Dimension of the square matrix (default: 100).
    rho : float, optional
        Correlation coefficient between upper and lower triangle off-diagonal elements.
        Must be in [-1, 1] (default: 0, independent entries).

    Returns
    -------
    np.ndarray
        Symmetric (n, n) matrix with entries sampled from elliptic normal distribution.
        
    Raises
    ------
    ValueError
        If n <= 0 or rho not in [-1, 1].
        
    Examples
    --------
    >>> A = elliptic_normal_matrix_opti(n=50, rho=0.5)
    >>> A.shape
    (50, 50)
    >>> np.allclose(A, A.T)  # Check symmetry
    True
    """
    # Input validation
    if n <= 0:
        raise ValueError(f"Matrix dimension n must be positive, got {n}.")
    if not (-1 <= rho <= 1):
        raise ValueError(f"Correlation rho must be in [-1, 1], got {rho}.")
    
    # Initialize matrix with diagonal entries from standard normal
    A = np.diag(np.random.randn(n))
    
    # Generate correlated off-diagonal entries using bivariate normal
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]  # Bivariate normal with correlation rho
    num_off_diag = n * (n - 1) // 2  # Number of upper-triangular elements
    upper_samples = np.random.multivariate_normal(mean, cov, num_off_diag)
    
    # Fill symmetric matrix: upper triangle with first component, lower with second
    upper_indices = np.triu_indices(n, k=1)
    A[upper_indices] = upper_samples[:, 0]
    A.T[upper_indices] = upper_samples[:, 1]  # Ensures symmetry
    
    return A

def block_matrix(n_size, beta, s, rho):
    """Generate a structured block matrix with elliptic and correlated entries.
    
    Constructs an (n_size, n_size) matrix divided into B×B blocks. Diagonal blocks use
    elliptic normal distributions, while off-diagonal blocks are filled with correlated entries.
    This is the core matrix generation function for block model analysis in AMP theory.
    
    Parameters
    ----------
    n_size : int
        Total dimension of the square matrix.
    beta : array-like, shape (B,)
        Relative proportions of each block. Must sum to 1. Example: [0.5, 0.3, 0.2] for 3 blocks.
    s : np.ndarray, shape (B, B)
        Standard deviation matrix. Element s[i,j] controls the scale of block (i,j).
    rho : np.ndarray, shape (B, B)
        Correlation matrix. Element rho[i,j] controls correlation in block (i,j).
        Must be symmetric with all entries in [-1, 1].

    Returns
    -------
    np.ndarray
        Symmetric block-structured matrix of shape (n_size, n_size).
        
    Raises
    ------
    ValueError
        If beta does not sum to 1, if s and rho are not (B,B) shaped,
        if rho is not symmetric, or if rho entries are not in [-1, 1].
        
    Examples
    --------
    >>> beta = np.array([0.6, 0.4])
    >>> s = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> rho = np.array([[0.3, 0.1], [0.1, 0.3]])
    >>> A = block_matrix(100, beta, s, rho)
    >>> A.shape
    (100, 100)
    """
    # Normalize standard deviations by matrix dimension
    s = np.array(s) / np.sqrt(n_size)
    beta = np.array(beta)
    rho = np.array(rho)
    
    # -------- Input Validation --------
    if not np.isclose(np.sum(beta), 1):
        raise ValueError(f"`beta` must sum to 1, got sum={np.sum(beta)}.")
    
    B = len(beta)
    if s.shape != (B, B) or rho.shape != (B, B):
        raise ValueError(f"`s` and `rho` must be shape ({B}, {B}), got {s.shape} and {rho.shape}.")
    
    if not np.allclose(rho, rho.T):
        raise ValueError("`rho` must be symmetric.")
    
    if not np.all((-1 <= rho) & (rho <= 1)):
        raise ValueError(f"All entries in `rho` must be in [-1, 1], got range [{rho.min()}, {rho.max()}].")
    
    # -------- Block Size Calculation --------
    # Convert proportions to actual block sizes, adjusting last block for rounding errors
    block_sizes = (beta * n_size).astype(int)
    block_sizes[-1] += n_size - np.sum(block_sizes)
    
    A = np.zeros((n_size, n_size))

    # -------- Matrix Construction --------
    row_start = 0
    for i in range(B):
        row_end = row_start + block_sizes[i]
        col_start = 0
        for j in range(B):
            col_end = col_start + block_sizes[j]
            
            if i == j:
                # Diagonal block: use elliptic normal distribution
                block = elliptic_normal_matrix_opti(n=block_sizes[i], rho=rho[i, i])
                block *= s[i, i]
                A[row_start:row_end, col_start:col_end] = block
            else:
                # Off-diagonal block: bivariate normal with specified correlation
                mean = [0, 0]
                cov = [[1, rho[i, j]], [rho[i, j], 1]]
                
                # Generate correlated pairs for all entries in block (i,j) and (j,i)
                off_diag = np.random.multivariate_normal(mean, cov, 
                                                         size=(block_sizes[i], block_sizes[j]))
                
                # Extract and scale components for symmetric blocks
                block_ij = off_diag[:, :, 0] * s[i, j]
                block_ji = off_diag[:, :, 1] * s[j, i]
                
                # Fill both (i,j) and (j,i) blocks to maintain symmetry
                A[row_start:row_end, col_start:col_end] = block_ij
                A[col_start:col_end, row_start:row_end] = block_ji.T
            
            col_start = col_end
        row_start = row_end

    return A

# ==================== Fixed-Point Methods ==================== #

def compute_fixed_point_root(beta, s, rho, r):
    """Solve fixed-point equations using root-finding (Hybr method).
    
    Computes fixed-point solutions for the AMP algorithm's state evolution equations.
    Uses scipy's Hybrid root finder for robust convergence.
    
    Parameters
    ----------
    beta : array-like, shape (K,)
        Block proportion parameters.
    s : np.ndarray, shape (K, K)
        Standard deviation matrix for blocks.
    rho : np.ndarray, shape (K, K)
        Correlation matrix for blocks. Must be symmetric.
    r : array-like, shape (K,)
        Regularization or threshold parameters.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        (delta, sigma, gamma): Fixed-point solution variables
        - delta: denoise parameter (shape K)
        - sigma: noise variance estimate (shape K)
        - gamma: signal correlation (shape K)
        
    Raises
    ------
    ValueError
        If rho matrix is not symmetric.
        
    Notes
    -----
    - Prints warning if root solver does not converge
    - Uses Hybrid method from scipy.optimize
    """
    K = len(beta)
    
    if not np.allclose(rho, rho.T):
        raise ValueError("The `rho` matrix must be symmetric.")

    def fixed_point_equations(x):
        """System of fixed-point equations for AMP state evolution."""
        delta = x[:K]
        sigma = x[K:2*K]
        gamma = x[2*K:]

        delta_eqs = np.zeros(K)
        sigma_eqs = np.zeros(K)
        gamma_eqs = np.zeros(K)

        for i in range(K):
            # Delta equation: denoiser fixed-point
            delta_eqs[i] = 1 - delta[i] - sum(
                beta[j] * rho[i, j] * s[i, j] * s[j, i] * (gamma[j] / delta[j])
                for j in range(K)
            )
            # Sigma equation: variance fixed-point
            sigma_eqs[i] = sigma[i]**2 - sum(
                beta[j] * s[i, j]**2 / delta[j]**2 *
                norm.expect(lambda x: (sigma[j] * x + r[j])**2, loc=0, scale=1)
                for j in range(K)
            )
            # Gamma equation: correlation fixed-point
            gamma_eqs[i] = gamma[i] - (1 - norm.cdf(-r[i] / sigma[i]))

        return np.concatenate([delta_eqs, sigma_eqs, gamma_eqs])

    # Initial guess: all variables equal to 1
    x0 = np.concatenate([np.ones(K), np.ones(K), np.ones(K)])
    result = optimize.root(fixed_point_equations, x0, method='hybr')

    if not result.success:
        print("Warning: Root solver did not converge. Consider checking input parameters.")

    delta = result.x[:K]
    sigma = result.x[K:2*K]
    gamma = result.x[2*K:]
    return delta, sigma, gamma

def compute_fixed_point(beta, s, rho, r, tol=1e-6, max_iter=1000):
    """Solve fixed-point equations using iterative fixed-point iteration.
    
    Computes fixed-point solutions for AMP state evolution using direct iteration.
    This method is often more efficient than root-finding for well-conditioned problems.
    
    Parameters
    ----------
    beta : array-like, shape (K,)
        Block proportion parameters.
    s : np.ndarray, shape (K, K)
        Standard deviation matrix for blocks.
    rho : np.ndarray, shape (K, K)
        Correlation matrix for blocks. Must be symmetric.
    r : array-like, shape (K,)
        Regularization or threshold parameters.
    tol : float, optional
        Convergence tolerance for maximum absolute change (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        (delta, sigma, gamma): Fixed-point solution variables
        - delta: denoise parameter (shape K)
        - sigma: noise variance estimate (shape K)
        - gamma: signal correlation (shape K)
        
    Raises
    ------
    ValueError
        If rho matrix is not symmetric.
        
    Notes
    -----
    - Prints warning if iteration does not converge before max_iter
    - Convergence is checked on maximum absolute change across all variables
    - Initial guess: delta = sigma = gamma = 1
    """
    K = len(beta)
    delta = np.ones(K)
    sigma = np.ones(K)
    gamma = np.ones(K)

    if not np.allclose(rho, rho.T):
        raise ValueError("The `rho` matrix must be symmetric.")

    for iteration in range(max_iter):
        delta_new = np.zeros(K)
        sigma_new = np.zeros(K)
        gamma_new = np.zeros(K)

        for i in range(K):
            # Update delta: denoiser fixed-point
            delta_new[i] = 1 - sum(
                beta[j] * rho[i, j] * s[i, j] * s[j, i] * (gamma[j] / delta[j])
                for j in range(K)
            )
            # Update sigma: variance fixed-point with clipping for numerical stability
            sigma_new[i] = np.sqrt(sum(
                beta[j] * s[i, j]**2 / delta[j]**2 *
                norm.expect(lambda x: max(0, (sigma[j] * x + r[j]))**2, loc=0, scale=1)
                for j in range(K)
            ))
            # Update gamma: correlation fixed-point using truncated normal tail probability
            gamma_new[i] = 1 - norm.cdf(-r[i] / sigma[i])

        # Check for convergence on all variables
        max_change = max(
            np.max(np.abs(delta_new - delta)),
            np.max(np.abs(sigma_new - sigma)),
            np.max(np.abs(gamma_new - gamma))
        )
        
        if max_change < tol:
            break

        delta, sigma, gamma = delta_new, sigma_new, gamma_new
    else:
        print(f"Warning: Did not converge within {max_iter} iterations (final change: {max_change:.2e}).")

    return delta, sigma, gamma

def compute_fixed_point_final(beta, s, rho, r):
    """Compute final AMP estimates including variance of truncated normal distribution.
    
    Solves fixed-point equations and computes the variance of the posterior estimate
    using the truncated normal distribution (tail-bounded estimates).
    
    Parameters
    ----------
    beta : array-like, shape (K,)
        Block proportion parameters.
    s : np.ndarray, shape (K, K)
        Standard deviation matrix for blocks.
    rho : np.ndarray, shape (K, K)
        Correlation matrix for blocks. Must be symmetric.
    r : array-like, shape (K,)
        Regularization or threshold parameters.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (variance, gamma) where:
        - variance: Posterior estimate variance using truncated normal (shape K)
        - gamma: Signal correlation from fixed-point solution (shape K)
        
    Notes
    -----
    - Variance is computed for truncated normal distribution bounded at 0 (from below)
    - Uses iterative fixed-point solver internally
    """
    # Solve fixed-point equations
    delta, sigma, gamma = compute_fixed_point(beta, s, rho, r)
    
    # Compute posterior mean and variance scaling
    mu = r / delta
    sigma_scaled = sigma / delta

    # Setup truncated normal distribution parameters (truncated at 0 from below)
    a_scaled = (0 - mu) / sigma_scaled  # Lower bound scaled
    b_scaled = np.full_like(mu, np.inf)  # No upper bound

    # Compute variance of truncated normal distribution
    variance = truncnorm.var(a_scaled, b_scaled, loc=mu, scale=sigma_scaled)
    
    return variance, gamma
