"""
Generate heatmaps exploring how the fixed-point persistence `gamma_k`
depends on asymmetric off-diagonal interaction variances `s^2_{12}` and
`s^2_{21}` for a two-block model.

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
    fig : matplotlib Figure
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
            # set diagonal sd from s_diag tuple
            s[0, 0] = s_diag[0]
            s[1, 1] = s_diag[1]
            # asymmetric off-diagonal entries
            s[0, 1] = s12
            s[1, 0] = s21

            # compute_fixed_point_final returns (variance_array, gamma_array)
            _, gamma = compute_fixed_point_final(beta, s, rho, r)
            # store gamma_1; note the row/col ordering for heatmap display
            gamma1_grid[j, i] = gamma[0]

    # Build sparse tick labels: show ~5 values evenly spaced
    tick_step = max(1, num_points // 5)
    x_sq = np.round(s12_values**2, 2)
    y_sq = np.round(s21_values**2, 2)
    x_labels = [f"{v:.2f}" if i % tick_step == 0 else "" for i, v in enumerate(x_sq)]
    y_labels = [f"{v:.2f}" if i % tick_step == 0 else "" for i, v in enumerate(y_sq)]

    fig, ax_h = plt.subplots(figsize=(6.5, 5.5))
    vmin, vmax = 0.85, 0.96
    sns.heatmap(
        gamma1_grid,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap="Greys_r",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': r'$\gamma_1$', 'shrink': 0.85},
        square=True,
        ax=ax_h,
    )

    # Format colorbar ticks
    cbar = ax_h.collections[0].colorbar
    cbar.set_ticks(np.linspace(vmin, vmax, num=5))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=10)

    ax_h.set_xlabel(r"$s^2_{12}$ (Community 2 $\to$ 1)")
    ax_h.set_ylabel(r"$s^2_{21}$ (Community 1 $\to$ 2)")
    ax_h.tick_params(axis='both', which='both', length=0)

    return fig

# Example usage:
if __name__ == "__main__":
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'font.size': 11,
        'axes.labelsize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
    })

    beta = [0.5, 0.5]
    r = np.array([1.0, 1.0])
    fig = heatmap_gamma1_vs_s12_s21(
        beta=beta,
        r=r,
        s12_range=(0.0, np.sqrt(0.5)),
        s21_range=(0.0, np.sqrt(0.5)),
        s_diag=(np.sqrt(0.5), np.sqrt(0.5)),
        num_points=30,
    )
    fig.tight_layout()
    fig.savefig("Figures/Figure4.png", dpi=300, bbox_inches='tight')
    plt.show()
