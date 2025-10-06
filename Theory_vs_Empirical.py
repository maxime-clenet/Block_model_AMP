# -*- coding: utf-8 -*-
"""
Author: Maxime Clenet
Description: Compare fixed-point predictions vs. empirical estimates of 
gamma and sigma as functions of interaction strength `s` in block-structured systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from Benchmark.Theory import compute_fixed_point
from Benchmark.Empirical import empirical_prop


def plot_gamma_vs_s(beta, rho, r, s_min=0.2, s_max=1.4, num_points=100, n_size=100, mc_prec=300):
    """
    Plot gamma (proportion of persistent species) vs. s for both fixed-point and empirical methods.
    """
    K = len(beta)
    s_values = np.linspace(s_min, s_max, num_points)
    gamma_values = np.zeros((num_points, K))
    empirical_gamma_values = np.zeros((num_points, K))

    for i, s_scalar in enumerate(s_values):
        s = np.full((K, K), s_scalar)

        # Fixed point computation
        _, _, gamma = compute_fixed_point(beta, s, rho, r)
        gamma_values[i] = gamma

        # Empirical estimation
        results = empirical_prop(n_size, beta, s, rho, mc_prec=mc_prec)
        empirical_gamma_values[i] = [res[0] for res in results]

    # Plotting
    plt.figure(figsize=(10, 6))
    for k in range(K):
        plt.plot(s_values, gamma_values[:, k], '--', label=f"Fixed Point Gamma - Block {k + 1}")
        plt.plot(s_values, empirical_gamma_values[:, k], 'o-', label=f"Empirical Gamma - Block {k + 1}")

    plt.xlabel("s (interaction strength)")
    plt.ylabel(r"$\gamma$ (proportion of persistent species)")
    plt.title(r"Comparison of $\gamma$ vs. $s$ (Fixed Point vs Empirical)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_sigma_vs_s(beta, rho, r, s_min=0.2, s_max=1.4, num_points=100, n_size=100, mc_prec=300):
    """
    Plot sigma (variance of persistent species) vs. s for both fixed-point and empirical methods.
    """
    K = len(beta)
    s_values = np.linspace(s_min, s_max, num_points)
    sigma_values = np.zeros((num_points, K))
    empirical_sigma_values = np.zeros((num_points, K))

    for i, s_scalar in enumerate(s_values):
        s = np.full((K, K), s_scalar)

        # Fixed point computation
        delta, sigma, _ = compute_fixed_point(beta, s, rho, r)
        mu = r / delta
        sigma_scaled = sigma / delta

        # Compute variance of truncated normal
        a, b = 0, np.inf
        a_scaled = (a - mu) / sigma_scaled
        b_scaled = (b - mu) / sigma_scaled
        sigma_values[i] = truncnorm.var(a_scaled, b_scaled, loc=mu, scale=sigma_scaled)

        # Empirical estimation
        results = empirical_prop(n_size, beta, s, rho, mc_prec=mc_prec)
        empirical_sigma_values[i] = [res[2] for res in results]

    # Plotting
    plt.figure(figsize=(10, 6))
    for k in range(K):
        plt.plot(s_values, sigma_values[:, k], '--', label=f"Fixed Point Sigma - Block {k + 1}")
        plt.plot(s_values, empirical_sigma_values[:, k], 'o-', label=f"Empirical Sigma - Block {k + 1}")

    plt.xlabel("s (interaction strength)")
    plt.ylabel(r"$\sigma$ (variance of persistent species)")
    plt.title(r"Comparison of $\sigma$ vs. $s$ (Fixed Point vs Empirical)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------- Example Usage ---------------------

# Parameters for gamma comparison
n_size = 100
beta = [0.4, 0.6]
rho = np.array([[0.9, 0], [0, 0]])
r = np.array([1, 1])
plot_gamma_vs_s(beta, rho, r, s_min=0.2, s_max=0.6, num_points=10, n_size=n_size, mc_prec=500)

# Parameters for sigma comparison
rho_sigma = np.array([[0.9, 0], [0, -0.7]])  # different correlation setting
plot_sigma_vs_s(beta, rho_sigma, r, s_min=0.2, s_max=0.6, num_points=10, n_size=n_size, mc_prec=500)