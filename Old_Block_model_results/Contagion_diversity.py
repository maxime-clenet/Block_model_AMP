import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Benchmark.Theory import compute_fixed_point_final

def heatmap_gamma2_vs_s21_gamma1(beta, r, s21_range, s11_range, s22=0.5, num_points=50):
    """
    Create a heatmap of gamma_2 as a function of s21 (x-axis) and gamma_1 (y-axis).

    Parameters
    ----------
    beta : list of floats
        Proportions of block sizes.
    r : ndarray
        Intrinsic growth rates.
    s21_range : tuple
        Min and max values for s21 (off-diagonal A21).
    s11_range : tuple
        Min and max values for s11 (diagonal variance of Community 1).
    s22 : float
        Fixed diagonal variance of Community 2.
    num_points : int
        Resolution of the heatmap.

    Returns
    -------
    None (displays heatmap)
    """
    K = 2
    assert len(beta) == K and len(r) == K

    # Set up the grid of interaction values we want to explore.
    s21_values = np.linspace(*s21_range, num_points)
    s11_values = np.linspace(*s11_range, num_points)
    gamma2_grid = np.zeros((num_points, num_points))
    gamma1_labels = []

    # Assume zero correlations between environmental fluctuations.
    rho = np.zeros((K, K))

    for i, s11 in enumerate(s11_values):
        gamma1_row = []
        for j, s21 in enumerate(s21_values):
            # Assemble the variance-covariance matrix for the current grid point.
            s = np.zeros((K, K))
            s[0, 0] = s11
            s[1, 1] = s22
            s[1, 0] = s21
            s[0, 1] = 0.0

            # Solve the deterministic fixed point and keep only the persistence terms.
            _, gamma = compute_fixed_point_final(beta, s, rho, r)
            gamma2_grid[i, j] = gamma[1]  # gamma_2 as Z-axis
            if j == 0:
                gamma1_row.append(gamma[0])
        # Use the last computed gamma_1 as the y-axis label for the entire row.
        gamma1_labels.append(round(gamma[0], 2))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        gamma2_grid,
        xticklabels=np.round(s21_values, 2),
        yticklabels=gamma1_labels,
        cmap="viridis",
        cbar_kws={'label': r'$\gamma_2$'},
        square=True
    )
    plt.xlabel(r"$s_{21}$ (Interaction from Community 2 to 1)")
    plt.ylabel(r"$\gamma_1$ (Persistence of Community 1)")
    plt.title(r"Heatmap of $\gamma_2$ vs $s_{21}$ and $\gamma_1$")
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    beta = [0.5, 0.5]
    r = np.array([1.0, 1.0])
    heatmap_gamma2_vs_s21_gamma1(
        beta=beta,
        r=r,
        s21_range=(0.2, 1.2),
        s11_range=(0.5, 1.2),
        s22=0.5,
        num_points=20
    )
