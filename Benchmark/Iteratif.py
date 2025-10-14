# -*- coding: utf-8 -*-
"""
Simulation and visualization of LCP solutions in a block-structured interaction model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

from Theory import compute_fixed_point
from Empirical import block_matrix

# ------------------------- Parameters -------------------------

n_size = 10000                      # Size of the full matrix
beta = np.array([0.5, 0.5])        # Block proportions
s = np.array([[0.5, 0], [0, 0.8]]) # Standard deviations of interactions
rho = np.array([[-0.9, 0], [0, 0]])# Correlations between interactions
r = np.array([1, 1])               # Intrinsic growth rates or thresholds

# ------------------------- Fixed Point Computation -------------------------

delta, sigma, gamma = compute_fixed_point(beta, s, rho, r)
print("Sigma:", sigma)
print("Delta:", delta)
print("Gamma:", gamma)

mu_k = r / delta
sigma_k = sigma / delta

# ------------------------- Matrix Generation -------------------------

A = block_matrix(n_size, beta, s, rho)
I = np.eye(n_size)
M = I - A

# Regularization for numerical stability
cond_number = np.linalg.cond(I + M)
if cond_number > 1e12:
    print(f"Matrix is ill-conditioned (cond={cond_number:.2e}). Regularization applied.")
inv_IM = np.linalg.inv(I + M + 1e-3 * np.eye(n_size))

b = -inv_IM @ (-np.ones(n_size))
B = inv_IM @ (I - M)

# ------------------------- Fixed-Point Iteration -------------------------

def fixed_point_iteration(x):
    return b + B @ np.abs(x)

sol = np.zeros(n_size)
alpha = 1.0  # Relaxation factor
n_iterations = 50

for _ in range(n_iterations):
    sol = alpha * fixed_point_iteration(sol) + (1 - alpha) * sol

sol = np.abs(sol) + sol  # Ensure non-negativity

# ------------------------- Block-wise Solution Analysis -------------------------

block_sizes = np.round(beta * n_size).astype(int)
split_indices = np.cumsum(block_sizes)[:-1]
sol_blocks = np.split(sol, split_indices)
sol_blocks = [block[block > 0] for block in sol_blocks]  # Keep only positive values

# ------------------------- Plotting -------------------------

x_vals = np.linspace(0, 4, 1000)
colors = ['b', 'r', 'g', 'm', 'c', 'y']

plt.figure(figsize=(12, 6))

for i, block in enumerate(sol_blocks):
    plt.hist(block, bins=30, density=True, alpha=0.6,
             color=colors[i % len(colors)], edgecolor='black', label=f"Block {i+1}")

for i in range(len(beta)):
    a, b = (0 - mu_k[i]) / sigma_k[i], np.inf  # Lower bound 0, upper ∞
    pdf = truncnorm.pdf(x_vals, a, b, loc=mu_k[i], scale=sigma_k[i])
    plt.plot(x_vals, pdf, color=colors[i % len(colors)], linestyle='--',
             linewidth=2, label=f"Truncated Normal {i+1}")

plt.xlabel("Value of $x$")
plt.ylabel("Density")
plt.title("Histogram of LCP Solutions with Theoretical Distributions")
plt.legend()
plt.tight_layout()
plt.show()