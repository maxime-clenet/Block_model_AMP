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
    """Compute variance and persistence for a range of ``rho_11`` values."""
    beta = np.asarray(beta, dtype=float)
    s = np.asarray(s, dtype=float)
    rho_template = np.asarray(rho_template, dtype=float)
    r = np.asarray(r, dtype=float)
    rho11_values = np.asarray(rho11_values, dtype=float)

    gamma_values = np.zeros((rho11_values.size, beta.size))
    variance_values = np.zeros_like(gamma_values)

    for idx, rho11 in enumerate(rho11_values):
        rho = rho_template.copy()
        rho[0, 0] = rho11
        variance, gamma = compute_fixed_point_final(beta, s, rho, r)
        gamma_values[idx] = gamma
        variance_values[idx] = variance

    return gamma_values, variance_values


def plot_vs_rho11(x_values, y_matrix, y_label, ax):
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
