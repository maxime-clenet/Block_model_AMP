# -*- coding: utf-8 -*-
"""
Author: Maxime Clenet
Description: Solves the LCP for a block-structured interaction matrix and compares
the empirical distribution of persistent species to theoretical truncated normal PDFs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from lemkelcp import lemkelcp

from Benchmark.Theory import compute_fixed_point
from Benchmark.Block_matrix import block_matrix


def zero_LCP_histogram(A, beta):
    """
    Solve the LCP problem for a block-structured matrix and compute histograms
    of persistent species in each block.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Interaction matrix.
    beta : list or array-like
        Block proportions (must sum to 1).

    Returns
    -------
    histograms : list of tuples
        Each tuple is (hist, bin_edges) for one block.
    """
    n = A.shape[0]
    block_sizes = (np.array(beta) * n).astype(int)
    if np.sum(block_sizes) != n:
        block_sizes[-1] += n - np.sum(block_sizes)  # Correct rounding error

    # Solve LCP: find x such that x ≥ 0, Mx + q ≥ 0, and xᵀ(Mx + q) = 0
    q = np.ones(n)
    M = -np.eye(n) + A
    sol = lemkelcp.lemkelcp(-M, -q, maxIter=10000)[0]

    # Histogram computation
    histograms = []
    start = 0
    for block_size in block_sizes:
        end = start + block_size
        positive_values = sol[start:end][sol[start:end] > 0]
        hist, bin_edges = np.histogram(positive_values, bins=25, density=True)
        histograms.append((hist, bin_edges))
        start = end

    return histograms


# --------------------------- Parameters ---------------------------

n_size = 1000
beta = np.array([0.5, 0.5])
s = np.array([[0.5, 0], [0, 0.8]])
rho = np.array([[-0.5, 0], [0, 0]])
r = np.array([1, 1])
K = len(beta)

# --------------------------- Compute Theoretical Distribution ---------------------------

delta, sigma, gamma = compute_fixed_point(beta, s, rho, r)
print("Sigma:", sigma)
print("Delta:", delta)
print("Gamma:", gamma)

mu_k = r / delta
sigma_k = sigma / delta

# --------------------------- Generate Matrix and Compute Histograms ---------------------------

A = block_matrix(n_size, beta, s, rho)
histograms = zero_LCP_histogram(A, beta)

# --------------------------- Plot Results ---------------------------

x = np.linspace(0, 4, 1000)
colors_pdf = ['r', 'g']
colors_hist = ['blue', 'red']

plt.figure(figsize=(10, 5))

# Plot theoretical PDFs
for i in range(K):
    a, b = (0 - mu_k[i]) / sigma_k[i], np.inf
    pdf = truncnorm.pdf(x, a, b, loc=mu_k[i], scale=sigma_k[i])
    plt.plot(x, pdf, color=colors_pdf[i], linestyle='--', linewidth=2, label=f"Theoretical PDF (Block {i+1})")

# Plot empirical histograms
for i, (hist, bin_edges) in enumerate(histograms):
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.3,
            edgecolor='black', color=colors_hist[i], label=f"Empirical Histogram (Block {i+1})")

plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Persistent Species: Theoretical vs Empirical Distributions")
plt.legend()
plt.tight_layout()
plt.show()