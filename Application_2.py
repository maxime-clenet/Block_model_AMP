import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from Benchmark.Theory import compute_fixed_point_final


def sweep_rho_offdiag(
    beta: np.ndarray,
    s: np.ndarray,
    rho_template: np.ndarray,
    r: np.ndarray,
    rho_off_values: np.ndarray,
    pair: Tuple[int, int] = (0, 1),
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute variance and persistence for symmetric off-diagonal correlations."""
    beta = np.asarray(beta, dtype=float)
    s = np.asarray(s, dtype=float)
    rho_template = np.asarray(rho_template, dtype=float)
    r = np.asarray(r, dtype=float)
    rho_off_values = np.asarray(rho_off_values, dtype=float)

    i, j = pair
    if i == j:
        raise ValueError("pair must reference two distinct blocks.")

    gamma_values = np.zeros((rho_off_values.size, beta.size))
    variance_values = np.zeros_like(gamma_values)

    for idx, rho_off in enumerate(rho_off_values):
        rho = rho_template.copy()
        rho[i, j] = rho_off
        rho[j, i] = rho_off
        variance, gamma = compute_fixed_point_final(beta, s, rho, r)
        gamma_values[idx] = gamma
        variance_values[idx] = variance

    return gamma_values, variance_values


def plot_vs_rho_off(x_values, y_matrix, y_label, title, ax):
    """Plot helper mirroring the Application_2 styling."""
    for block_idx in range(y_matrix.shape[1]):
        ax.plot(x_values, y_matrix[:, block_idx], label=f"Block {block_idx + 1}")
    ax.set_xlabel(r"$\rho_{12} = \rho_{21}$")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=12)


if __name__ == "__main__":
    beta = np.array([0.5, 0.5])
    rho_template = np.array([[0.0, 0.0], [0.0, 0.0]])
    s = np.array([[0.5, 0.5], [0.5, 0.5]])
    r = np.array([1.0, 1.0])

    rho_off_values = np.linspace(-0.99, 0.99, 80)
    gamma_values, variance_values = sweep_rho_offdiag(
        beta=beta,
        s=s,
        rho_template=rho_template,
        r=r,
        rho_off_values=rho_off_values,
        pair=(0, 1),
    )

    fig_gamma, ax_gamma = plt.subplots(figsize=(6, 4))
    plot_vs_rho_off(
        rho_off_values,
        gamma_values,
        y_label=r"$\gamma$ (persistence)",
        title=r"Persistence vs $\rho_{12} = \rho_{21}$",
        ax=ax_gamma,
    )
    fig_gamma.tight_layout()

    fig_var, ax_var = plt.subplots(figsize=(6, 4))
    plot_vs_rho_off(
        rho_off_values,
        variance_values,
        y_label=r"$\mathrm{Var}(N)$",
        title=r"Variance vs $\rho_{12} = \rho_{21}$",
        ax=ax_var,
    )
    fig_var.tight_layout()
    plt.show()
