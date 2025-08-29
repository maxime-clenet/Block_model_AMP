# -*- coding: utf-8 -*-
"""
Author: Maxime Clenet
Description: Simulates and visualizes truncated distributions of species abundances
from fixed-point computations in block-structured models.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from Theory import compute_fixed_point

# ------------------------- Parameters -------------------------

n_size = 100                              # Total system size (not used directly here)
beta = np.array([0.5, 0.5])               # Block proportions
s = np.array([[0.5, 0], [0, 0.5]])        # Standard deviations
rho = np.array([[-0.8, 0], [0, 0.8]])     # Correlation structure
r = np.array([1, 1])                      # Growth rates

# ------------------------- Fixed-Point Computation -------------------------

delta, sigma, gamma = compute_fixed_point(beta, s, rho, r)
print("Sigma:", sigma)
print("Delta:", delta)
print("Gamma:", gamma)

mu_k = r / delta
sigma_k = sigma / delta
K = len(beta)

# ------------------------- Simulated Truncated Distributions -------------------------

n_samples = 10000
xi_k = np.random.randn(K, n_samples)  # Standard normal samples
Y_k = (sigma[:, None] * xi_k + r[:, None]) / delta[:, None]
Y_k_truncated = [Y_k[i][Y_k[i] > 0] for i in range(K)]

# ------------------------- Plotting -------------------------

x = np.linspace(0, 3, 1000)
colors = ['blue', 'orange']

plt.figure(figsize=(12, 5))

for i in range(K):
    plt.subplot(1, K, i + 1)

    # Histogram of simulated values
    plt.hist(Y_k_truncated[i], bins=50, density=True, alpha=0.6,
             color=colors[i], label=f"Block {i+1} Samples")

    # Overlay truncated normal PDF
    a, b = (0 - mu_k[i]) / sigma_k[i], np.inf
    pdf = truncnorm.pdf(x, a, b, loc=mu_k[i], scale=sigma_k[i])
    plt.plot(x, pdf, 'k--', linewidth=2, label="Truncated Normal")

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(f"Block {i+1}")
    plt.legend()

plt.tight_layout()
plt.show()