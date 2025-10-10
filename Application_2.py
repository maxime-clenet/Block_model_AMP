import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm, truncnorm
from Benchmark.Theory import compute_fixed_point_final

# ------------------ Plotting Functions ------------------ #

def plot_gamma_vs_s(beta, rho, r, s_min=0.2, s_max=1.4, num_points=100):
    K = len(beta)
    s_values = np.linspace(s_min, s_max, num_points)
    gamma_values = np.zeros((num_points, K))

    for i, s_scalar in enumerate(s_values):
        s = np.full((K, K), s_scalar)
        np.fill_diagonal(s, 0.5)
        _, gamma = compute_fixed_point_final(beta, s, rho, r)
        gamma_values[i] = gamma

    plt.figure(figsize=(8, 5))
    line_styles = ['-', '--', '-.', ':']
    for k in range(K):
        style = line_styles[k] if k < len(line_styles) else '-'
        plt.plot(
            s_values,
            gamma_values[:, k],
            label=f"Community {k + 1}",
            linestyle=style,
            color='black',
        )

    plt.xlabel("Off-diagonal interaction strength (s)")
    plt.ylabel(r"$\gamma_i$ (persistence)")
    plt.legend(fontsize=13)
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()

def plot_sigma_vs_s(beta, rho, r, s_min=0.2, s_max=1.4, num_points=100):
    K = len(beta)
    s_values = np.linspace(s_min, s_max, num_points)
    sigma_values = np.zeros((num_points, K))

    for i, s_scalar in enumerate(s_values):
        s = np.full((K, K), s_scalar)
        np.fill_diagonal(s, 0.5)
        variance, _ = compute_fixed_point_final(beta, s, rho, r)
        sigma_values[i] = variance

    plt.figure(figsize=(8, 5))
    line_styles = ['-', '--', '-.', ':']
    for k in range(K):
        style = line_styles[k] if k < len(line_styles) else '-'
        plt.plot(
            s_values,
            sigma_values[:, k],
            label=f"Community {k + 1}",
            linestyle=style,
            color='black',
        )

    plt.xlabel("Off-diagonal interaction strength (s)")
    plt.ylabel(r"$\sigma^2_i$ (Variance)")
    plt.legend(fontsize=13)
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()


def plot_sigma_diff_vs_s(beta, rho, r, s_min=0.2, s_max=1.4, num_points=100):
    s_values = np.linspace(s_min, s_max, num_points)
    variance_diff = np.zeros(num_points)

    for i, s_scalar in enumerate(s_values):
        s = np.full((len(beta), len(beta)), s_scalar)
        np.fill_diagonal(s, 0.5)
        variance, _ = compute_fixed_point_final(beta, s, rho, r)
        variance_diff[i] = variance[0] - variance[1]

    plt.figure(figsize=(8, 5))
    plt.plot(s_values, variance_diff, color='black')
    plt.xlabel("Off-diagonal interaction strength (s)")
    plt.ylabel(r"Variance difference $(\sigma_1^2 - \sigma_2^2)$")
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

    # Gamma plot
    plot_gamma_vs_s(beta, rho, r, s_min=0, s_max=0.8, num_points=30)

    # Sigma plot
    plot_sigma_vs_s(beta, rho, r, s_min=0, s_max=0.8, num_points=30)

    # Difference of variance plot
    plot_sigma_diff_vs_s(beta, rho, r, s_min=0, s_max=0.8, num_points=30)
