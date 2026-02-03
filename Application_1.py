"""
Helpers to sweep and plot fixed-point quantities versus the self-correlation
parameter rho[0,0].

This module contains small utilities used to investigate how the fixed-point
solutions returned by `compute_fixed_point_final` depend on the within-block
correlation `rho_11`. The typical workflow is:

 - call `sweep_rho11` to compute `gamma_k` (persistence) and
   `variance_k` for a range of `rho_11` values;
 - use `plot_vs_rho11` to plot the resulting curves per community.

The scripts are minimal and intended for exploratory figures — they do not
attempt to be a full plotting library.
"""

import numpy as np
import matplotlib.pyplot as plt

from Benchmark.Theory import compute_fixed_point_final


def sweep_rho11(
    beta: np.ndarray,
    s: np.ndarray,
    rho_template: np.ndarray,
    r: np.ndarray,
    rho11_values: np.ndarray,
):
    """Compute fixed-point variance and persistence while sweeping rho[0,0].

    Parameters
    - beta: 1D array-like of community weights (length K)
    - s: KxK matrix of interaction variances
    - rho_template: KxK template array of interaction means; only entry
      rho_template[0,0] is modified during the sweep
    - r: 1D array-like of intrinsic growth/scale parameters
    - rho11_values: 1D array of scalar values to assign to rho[0,0]

    Returns
    - gamma_values: array shape (len(rho11_values), K) with persistence entries
    - variance_values: array shape (len(rho11_values), K) with variance
      entries (i.e. \hat{\sigma}^2_k)

    Notes
    - Inputs are converted to NumPy arrays of dtype float for safety.
    - The function calls `compute_fixed_point_final(beta, s, rho, r)` which
      is expected to return (variance_array, gamma_array).
    """
    beta = np.asarray(beta, dtype=float)
    s = np.asarray(s, dtype=float)
    rho_template = np.asarray(rho_template, dtype=float)
    r = np.asarray(r, dtype=float)
    rho11_values = np.asarray(rho11_values, dtype=float)

    gamma_values = np.zeros((rho11_values.size, beta.size))
    variance_values = np.zeros_like(gamma_values)

    for idx, rho11 in enumerate(rho11_values):
        # Copy the template and set the (0,0) entry to the current sweep value
        rho = rho_template.copy()
        rho[0, 0] = rho11
        # compute_fixed_point_final returns (variance_array, gamma_array)
        variance, gamma = compute_fixed_point_final(beta, s, rho, r)
        gamma_values[idx] = gamma
        variance_values[idx] = variance

    return gamma_values, variance_values


def plot_vs_rho11(x_values, y_matrix, y_label, ax):
    """Plot multiple community curves against rho[0,0].

    Parameters
    - x_values: 1D array of rho[0,0] sweep values
    - y_matrix: 2D array with shape (len(x_values), K) containing the values
      to plot for each community
    - y_label: label for the y axis (raw string or LaTeX)
    - ax: Matplotlib Axes object on which to plot

    The function cycles through a small set of line styles and draws each
    community curve in black (as in the project's plotting aesthetic).
    """
    line_styles = ['-', '--', '-.', ':']
    for block_idx in range(y_matrix.shape[1]):
        style = line_styles[block_idx] if block_idx < len(line_styles) else '-'
        ax.plot(
            x_values,
            y_matrix[:, block_idx],
            label=f"Community {block_idx + 1}",
            linestyle=style,
            color='black',
        )
    ax.set_xlabel(r"$\rho_{11}$")
    ax.set_ylabel(y_label)
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=13)


if __name__ == "__main__":
    # Minimal example demonstrating how to run the sweep and plot results.
    beta = np.array([0.5, 0.5])
    rho_template = np.array([[0.5, 0.0], [0.0, 0.0]])
    s = np.array([[0.5, 0.5], [0.5, 0.5]])
    r = np.array([1.0, 1.0])

    rho11_values = np.linspace(-0.99, 0.99, 80)
    gamma_values, variance_values = sweep_rho11(beta, s, rho_template, r, rho11_values)

    fig_gamma, ax_gamma = plt.subplots(figsize=(6, 4))
    plot_vs_rho11(
        rho11_values,
        gamma_values,
        y_label=r"$\gamma_k$ (persistence)",
        ax=ax_gamma,
    )
    fig_gamma.tight_layout()

    fig_var, ax_var = plt.subplots(figsize=(6, 4))
    plot_vs_rho11(
        rho11_values,
        variance_values,
        y_label=r"$\hat{\sigma}^2_k$ (abundance variance)",
        ax=ax_var,
    )
    fig_var.tight_layout()
    plt.show()
