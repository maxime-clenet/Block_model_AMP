import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Benchmark.Theory import compute_fixed_point_final

def heatmap_gamma1_vs_s12_s21(beta, r, s12_range, s21_range, s_diag=(0.5, 0.5), num_points=50):
    """
    Create a heatmap of gamma_1 as a function of s12 and s21 to analyze feedback effects.

    Parameters
    ----------
    beta : list of floats
        Proportions of block sizes.
    r : ndarray
        Intrinsic growth rates.
    s12_range : tuple
        Min and max values for s12 (interaction from Community 1 to 2).
    s21_range : tuple
        Min and max values for s21 (interaction from Community 2 to 1).
    s_diag : tuple
        Diagonal variances for Community 1 and 2 respectively.
    num_points : int
        Resolution of the heatmap.

    Returns
    -------
    None (displays heatmap)
    """
    K = 2
    assert len(beta) == K and len(r) == K

    s12_values = np.linspace(*s12_range, num_points)
    s21_values = np.linspace(*s21_range, num_points)
    gamma1_grid = np.zeros((num_points, num_points))

    rho = np.zeros((K, K))

    for i, s12 in enumerate(s12_values):
        for j, s21 in enumerate(s21_values):
            s = np.zeros((K, K))
            s[0, 0] = s_diag[0]
            s[1, 1] = s_diag[1]
            s[0, 1] = s12
            s[1, 0] = s21

            _, gamma = compute_fixed_point_final(beta, s, rho, r)
            gamma1_grid[j, i] = gamma[0]  # gamma_1 as Z-axis

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        gamma1_grid,
        xticklabels=np.round(s12_values, 2),
        yticklabels=np.round(s21_values, 2),
        cmap="magma",
        cbar_kws={'label': r'$\gamma_1$'},
        square=True
    )
    plt.xlabel(r"$s_{12}$ (Community 2 → 1)")
    plt.ylabel(r"$s_{21}$ (Community 1 → 2)")
    plt.title(r"Heatmap of $\gamma_1$ vs $s_{12}$ and $s_{21}$")
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    beta = [0.5, 0.5]
    r = np.array([1.0, 1.0])
    heatmap_gamma1_vs_s12_s21(
        beta=beta,
        r=r,
        s12_range=(0.0, 1.0),
        s21_range=(0.0, 1.0),
        s_diag=(0.5, 0.5),
        num_points=15
    )
