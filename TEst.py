import numpy as np
import matplotlib.pyplot as plt
from Theory import compute_fixed_point_final

def compare_structural_vs_statistical(beta, r, s_diag, s_off_values, mode="fixed_diag"):
    """
    Compare persistence with and without inter-block interactions.

    Parameters
    ----------
    beta : list of floats
        Block size proportions.
    r : ndarray, shape (K,)
        Growth rate vector.
    s_diag : ndarray, shape (K, K)
        Base diagonal variance matrix (non-zero only on the diagonal).
    s_off_values : list of tuples
        List of (a, b) values to insert as off-diagonal interaction strengths.
    mode : str
        "fixed_diag" for identical diagonals, "adjust_diag" to rescale diagonals
        to maintain total variance.

    Returns
    -------
    results : list of dict
        Each dict contains A12, A21, gamma_A, gamma_B.
    """
    K = len(beta)
    rho = np.zeros((K, K))
    results = []

    for a, b in s_off_values:
        # Config A: no inter-blocks
        s_A = np.copy(s_diag)

        # Config B: with inter-blocks
        s_B = np.copy(s_diag)
        s_B[0, 1] = a
        s_B[1, 0] = b

        if mode == "adjust_diag":
            var_A = np.sum(s_A**2)
            var_B = np.sum(s_B**2)
            scale = np.sqrt(var_A / var_B) if var_B > 0 else 1.0
            np.fill_diagonal(s_B, np.diagonal(s_B) * scale)

        # Run fixed-point for both
        _, gamma_A = compute_fixed_point_final(beta, s_A, rho, r)
        _, gamma_B = compute_fixed_point_final(beta, s_B, rho, r)

        results.append({
            "A12": a,
            "A21": b,
            "gamma_A": gamma_A,
            "gamma_B": gamma_B,
            "delta_gamma": gamma_B - gamma_A
        })

    return results

def plot_gamma_comparison(results):
    a21_values = [r["A21"] for r in results]
    delta_gammas = [r["delta_gamma"] for r in results]

    delta_g1 = [dg[0] for dg in delta_gammas]
    delta_g2 = [dg[1] for dg in delta_gammas]

    plt.figure(figsize=(8, 5))
    plt.plot(a21_values, delta_g1, label=r"$\Delta \gamma_1$")
    plt.plot(a21_values, delta_g2, label=r"$\Delta \gamma_2$")

    plt.xlabel(r"$s_{21}$ (interaction from block 2 to 1)")
    plt.ylabel(r"$\Delta \gamma$ (with vs. without off-diagonal blocks)")
    plt.title(r"Impact of block structure on persistence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    beta = [0.5, 0.5]
    r = np.array([1.0, 1.0])
    s_diag = np.array([[0.5, 0.0], [0.0, 0.5]])
    s_off_values = [(0.0, b) for b in np.linspace(0.0, 1.2, 30)]

    results_fixed = compare_structural_vs_statistical(beta, r, s_diag, s_off_values, mode="fixed_diag")
    results_adjusted = compare_structural_vs_statistical(beta, r, s_diag, s_off_values, mode="adjust_diag")

    print("Fixed diagonal comparison:")
    plot_gamma_comparison(results_fixed)

    print("Adjusted diagonal comparison:")
    plot_gamma_comparison(results_adjusted)
