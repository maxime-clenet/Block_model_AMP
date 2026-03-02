"""
Utilities to plot fixed-point quantities computed by the AMP block-model theory.

This module provides three plotting helpers that vary the off-diagonal
interaction variance parameter `s` and visualize the resulting fixed-point
quantities (persistence `gamma_k` and abundance variance `\hat{\sigma}_k^2`) as
returned by `compute_fixed_point_final` from `Benchmark.Theory`.

Functions:
 - plot_gamma_vs_s: plots persistence `gamma_k` vs off-diagonal s
 - plot_sigma_vs_s: plots abundance variance `\hat{\sigma}_k^2` vs s
 - plot_sigma_diff_vs_s: plots difference between community variances

The plotting functions intentionally construct an `s` matrix with identical
off-diagonal entries and fixed diagonal entries (0.5) to explore the effect
of the off-diagonal interaction variance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm, truncnorm
from Functions import compute_fixed_point_final

# ------------------ Plotting Functions ------------------ #

def plot_gamma_vs_s(beta, rho, r, s_min=0.2, s_max=1.4, num_points=100):
    """Plot persistence `gamma_k` as a function of off-diagonal variance s.

    Parameters
    - beta: list-like of community sizes/weights (length K)
    - rho: KxK array of interaction means (signed correlations)
    - r: length-K array of intrinsic growth or scale parameters
    - s_min, s_max: range of scalar off-diagonal variances to sweep
    - num_points: number of points in the sweep

    The function builds an `s` matrix for each scalar value where all
    off-diagonal entries equal the scalar and diagonal entries are fixed to
    0.5. It calls `compute_fixed_point_final(beta, s, rho, r)` which is
    expected to return (variance_array, gamma_array), and plots the gamma
    entries for each community.
    """
    K = len(beta)
    s_values = np.linspace(s_min, s_max, num_points)
    gamma_values = np.zeros((num_points, K))

    for i, s_scalar in enumerate(s_values):
        # Construct interaction-variance matrix: off-diagonal = s_scalar,
        # diagonal = 0.5 (fixed). Shape is K x K.
        s = np.full((K, K), s_scalar)
        np.fill_diagonal(s, 0.5)
        # compute_fixed_point_final returns (variance_array, gamma_array)
        _, gamma = compute_fixed_point_final(beta, s, rho, r)
        gamma_values[i] = gamma

    plt.figure(figsize=(8, 5))
    line_styles = ['-', '--', '-.', ':']
    for k in range(K):
        style = line_styles[k] if k < len(line_styles) else '-'
        plt.plot(
            s_values**2,
            gamma_values[:, k],
            label=f"Community {k + 1}",
            linestyle=style,
            color='black',
        )

    plt.xlabel("Off-diagonal interaction variance ($s^2$)")
    plt.ylabel(r"$\gamma_k$ (persistence)")
    plt.legend(fontsize=13)
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()

def plot_sigma_vs_s(beta, rho, r, s_min=0.2, s_max=1.4, num_points=100):
    """Plot abundance variance `\hat{\sigma}_k^2` as a function of s.

    Parameters are the same as for `plot_gamma_vs_s`.
    """
    K = len(beta)
    s_values = np.linspace(s_min, s_max, num_points)
    sigma_values = np.zeros((num_points, K))

    for i, s_scalar in enumerate(s_values):
        # Build the s matrix (off-diagonal sweep, diagonal fixed to 0.5)
        s = np.full((K, K), s_scalar)
        np.fill_diagonal(s, 0.5)
        # Extract per-community variance from fixed point
        variance, _ = compute_fixed_point_final(beta, s, rho, r)
        sigma_values[i] = variance

    plt.figure(figsize=(8, 5))
    line_styles = ['-', '--', '-.', ':']
    for k in range(K):
        style = line_styles[k] if k < len(line_styles) else '-'
        plt.plot(
            s_values**2,
            sigma_values[:, k],
            label=f"Community {k + 1}",
            linestyle=style,
            color='black',
        )

    plt.xlabel("Off-diagonal interaction variance ($s^2$)")
    plt.ylabel(r"$\hat{\sigma}^2_k$ (abundance variance)")
    plt.legend(fontsize=13)
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()


def plot_sigma_diff_vs_s(beta, rho, r, s_min=0.2, s_max=1.4, num_points=100):
    """Plot the difference in abundance variance between the first two
    communities, i.e. \hat{\sigma}_1^2 - \hat{\sigma}_2^2, as s varies.

    This is useful to detect parameter regimes where one community becomes
    systematically more variable than the other.
    """
    s_values = np.linspace(s_min, s_max, num_points)
    variance_diff = np.zeros(num_points)

    for i, s_scalar in enumerate(s_values):
        s = np.full((len(beta), len(beta)), s_scalar)
        np.fill_diagonal(s, 0.5)
        variance, _ = compute_fixed_point_final(beta, s, rho, r)
        # difference between community 0 and 1 (assumes at least two communities)
        variance_diff[i] = variance[0] - variance[1]

    plt.figure(figsize=(8, 5))
    plt.plot(s_values**2, variance_diff, color='black')
    plt.xlabel("Off-diagonal interaction variance ($s^2$)")
    plt.ylabel(r"Variance difference $(\hat{\sigma}_1^2 - \hat{\sigma}_2^2)$")
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    plt.legend(fontsize=13)
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()

# ------------------ Example Usage ------------------ #

if __name__ == "__main__":
    beta = [0.5, 0.5]
    rho = np.array([[0.8, 0], [0, -0.8]])
    r = np.array([1, 1])

    # Example usage: sweep off-diagonal variance from 0 to 0.8 and plot.
    # These calls are intended as quick demonstrations and can be adapted
    # to different `beta`, `rho`, and `r` configurations as needed.
    plot_gamma_vs_s(beta, rho, r, s_min=0, s_max=0.8, num_points=30)
    plot_sigma_vs_s(beta, rho, r, s_min=0, s_max=0.8, num_points=30)
    plot_sigma_diff_vs_s(beta, rho, r, s_min=0, s_max=0.8, num_points=30)
