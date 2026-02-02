# -*- coding: utf-8 -*-
"""
Author: Maxime Clenet
Description: Simulation and analysis of block-structured interaction matrices 
using elliptic Gaussian distributions and Linear Complementarity Problems (LCP).
"""

import numpy as np
from lemkelcp import lemkelcp
from Benchmark.Block_matrix import block_matrix


def zero_LCP(A, beta):
    """
    Solve the LCP for a block-structured interaction matrix and extract
    persistence properties for each block.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Interaction matrix.
    beta : list of floats
        Proportions of rows/columns in each block. Must sum to 1.

    Returns
    -------
    results : list of tuples
        Each tuple contains (proportion of persistent species, mean, RMS) for a block.
    """
    n = A.shape[0]
    block_sizes = (np.array(beta) * n).astype(int)
    if np.sum(block_sizes) != n:
        block_sizes[-1] += n - np.sum(block_sizes)  # Adjust rounding error

    q = np.ones(n)
    M = -np.eye(n) + A
    sol = lemkelcp.lemkelcp(-M, -q, maxIter=10000)[0]

    results = []
    start = 0
    for block_size in block_sizes:
        end = start + block_size
        res_block = sol[start:end]
        res_pos = res_block[res_block > 0]

        S = len(res_pos)
        m = np.mean(res_pos) if S > 0 else 0
        sigma = np.var(res_pos) if S > 0 else 0
        proportion = S / block_size

        results.append((proportion, m, sigma))
        start = end

    return results


def empirical_prop(n_size, beta, s, rho, mc_prec=500):
    """
    Perform Monte Carlo simulations to empirically estimate persistence
    properties of species within each block.

    Parameters
    ----------
    n_size : int
        Total size of the interaction matrix.
    beta : list of floats
        Block size proportions (must sum to 1).
    s : ndarray, shape (B, B)
        Standard deviations per block.
    rho : ndarray, shape (B, B)
        Correlation coefficients per block.
    mc_prec : int, optional
        Number of Monte Carlo simulations (default: 500).

    Returns
    -------
    results : list of tuples
        Each tuple contains empirical (proportion, mean, RMS) for a block.
    """
    B = len(beta)
    S_p = [np.zeros(mc_prec) for _ in range(B)]
    S_m = [np.zeros(mc_prec) for _ in range(B)]
    S_sigma = [np.zeros(mc_prec) for _ in range(B)]

    for j in range(mc_prec):
        A = block_matrix(n_size, beta, s, rho)
        block_results = zero_LCP(A, beta)

        for i, (p, m, sigma) in enumerate(block_results):
            S_p[i][j] = p
            S_m[i][j] = m
            S_sigma[i][j] = sigma

    results = [
        (np.mean(S_p[i]), np.mean(S_m[i]), np.mean(S_sigma[i]))
        for i in range(B)
    ]
    return results


if __name__ == "__main__":
    # Example usage
    n_size = 1000
    beta = [0.5, 0.5]
    s = np.array([[0.7, 0], [0, 0.7]]) 
    rho = np.array([[0.8, 0], [0, -0.8]]) 
    mc_prec = 100

    results = empirical_prop(n_size, beta, s, rho, mc_prec)

    for i, (prop, mean, rms) in enumerate(results):
        print(f"Block {i + 1}:")
        print(f"  Proportion of persistent species: {prop:.2f}")
        print(f"  Mean of persistent species: {mean:.2f}")
        print(f"  RMS of persistent species: {rms:.2f}")