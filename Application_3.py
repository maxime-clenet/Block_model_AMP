"""
Generate heatmaps exploring how the fixed-point persistence `gamma_k`
depends on asymmetric off-diagonal interaction variances `s_{12}` and
`s_{21}` for a two-block model.

This script provides a single high-level function `heatmap_gamma1_vs_s12_s21`
that constructs an `s` matrix for each pair (s12, s21), calls
`compute_fixed_point_final(beta, s, rho, r)` to obtain the fixed-point
quantities, and displays `gamma_1` as a heatmap. The default plotting
style uses a reversed greyscale colormap and formats ticks for clarity.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from Functions import compute_fixed_point_final

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

    # Discretize the ranges for s12 and s21
    s12_values = np.linspace(*s12_range, num_points)
    s21_values = np.linspace(*s21_range, num_points)
    # Preallocate grid: rows -> s21, cols -> s12
    gamma1_grid = np.zeros((num_points, num_points))

    # Use zero mean interaction template (rho); only s varies here
    rho = np.zeros((K, K))

    # Evaluate the fixed-point for each pair (s12, s21)
    for i, s12 in enumerate(s12_values):
        for j, s21 in enumerate(s21_values):
            s = np.zeros((K, K))
            # set diagonal variances from s_diag tuple
            s[0, 0] = s_diag[0]
            s[1, 1] = s_diag[1]
            # asymmetric off-diagonal entries
            s[0, 1] = s12
            s[1, 0] = s21

            # compute_fixed_point_final returns (variance_array, gamma_array)
            _, gamma = compute_fixed_point_final(beta, s, rho, r)
            # store gamma_1; note the row/col ordering for heatmap display
            gamma1_grid[j, i] = gamma[0]

    # Plot heatmap with consistent color scale for comparability
    plt.figure(figsize=(8, 6))
    vmin, vmax = 0.85, 0.96
    ax = sns.heatmap(
        gamma1_grid,
        xticklabels=np.round(s12_values, 2),
        yticklabels=np.round(s21_values, 2),
        cmap="Greys_r",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': r'$\gamma_k$'},
        square=True,
    )

    # Format colorbar ticks and axis labels for readability
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.linspace(vmin, vmax, num=6))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel(r"$s_{12}$ (Community 2 → 1)")
    plt.ylabel(r"$s_{21}$ (Community 1 → 2)")
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    beta = [0.5, 0.5]
    r = np.array([1.0, 1.0])
    heatmap_gamma1_vs_s12_s21(
        beta=beta,
        r=r,
        s12_range=(0.0, 0.7),
        s21_range=(0.0, 0.7),
        s_diag=(0.7, 0.7),
        num_points=15
    )
