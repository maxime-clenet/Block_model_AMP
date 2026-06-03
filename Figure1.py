# -*- coding: utf-8 -*-
"""
Simulation and visualization of LCP solutions in a block-structured interaction model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

from Functions import compute_fixed_point, block_matrix

# ------------------------- Parameters -------------------------

n_size = 10000                     # Size of the full matrix
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
n_iterations = 100

for _ in range(n_iterations):
    sol = alpha * fixed_point_iteration(sol) + (1 - alpha) * sol

sol = np.abs(sol) + sol  # Ensure non-negativity

# ------------------------- Block-wise Solution Analysis -------------------------

block_sizes = np.round(beta * n_size).astype(int)
split_indices = np.cumsum(block_sizes)[:-1]
sol_blocks = np.split(sol, split_indices)
sol_blocks = [block[block > 0] for block in sol_blocks]  # Keep only positive values

# ------------------------- Plotting -------------------------

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,
    'lines.linewidth': 2.0,
})

x_vals = np.linspace(0, 3.5, 1000)
hist_colors = ['#a6cde8', '#fdbf6f']   # light blue, light orange
line_colors = ['#1f78b4', '#e66101']   # dark blue, dark orange

fig, ax = plt.subplots(figsize=(7, 4.5))

for i, block in enumerate(sol_blocks):
    ax.hist(block, bins=40, density=True, alpha=0.60,
            color=hist_colors[i % len(hist_colors)], edgecolor='none',
            label=f"Community {i + 1}")

for i in range(len(beta)):
    lower_t = (0 - mu_k[i]) / sigma_k[i]
    pdf = truncnorm.pdf(x_vals, lower_t, np.inf, loc=mu_k[i], scale=sigma_k[i])
    ax.plot(x_vals, pdf, color=line_colors[i % len(line_colors)],
            linestyle='-', linewidth=2.0, label=f"Theory {i + 1}")

ax.set_xlabel(r"Abundance $u_*$")
ax.set_ylabel("Probability density")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax.legend(frameon=True, framealpha=0.9, edgecolor='lightgray')
fig.tight_layout()
fig.savefig("Figures/Figure1.png", dpi=300, bbox_inches='tight')
plt.show()