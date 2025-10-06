import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm, truncnorm

# ------------------ Fixed-Point Methods ------------------ #

def compute_fixed_point_root(beta, s, rho, r):
    K = len(beta)

    if not np.allclose(rho, rho.T):
        raise ValueError("The `rho` matrix must be symmetric.")

    def fixed_point_equations(x):
        delta = x[:K]
        sigma = x[K:2*K]
        gamma = x[2*K:]

        delta_eqs = np.zeros(K)
        sigma_eqs = np.zeros(K)
        gamma_eqs = np.zeros(K)

        for i in range(K):
            delta_eqs[i] = 1 - delta[i] - sum(
                beta[j] * rho[i, j] * s[i, j] * s[j, i] * (gamma[j] / delta[j])
                for j in range(K)
            )
            sigma_eqs[i] = sigma[i]**2 - sum(
                beta[j] * s[i, j]**2 / delta[j]**2 *
                norm.expect(lambda x: (sigma[j] * x + r[j])**2, loc=0, scale=1)
                for j in range(K)
            )
            gamma_eqs[i] = gamma[i] - (1 - norm.cdf(-r[i] / sigma[i]))

        return np.concatenate([delta_eqs, sigma_eqs, gamma_eqs])

    x0 = np.concatenate([np.ones(K), np.ones(K), np.ones(K)])
    result = optimize.root(fixed_point_equations, x0, method='hybr')

    if not result.success:
        print("Warning: Root solver did not converge.")

    delta = result.x[:K]
    sigma = result.x[K:2*K]
    gamma = result.x[2*K:]
    return delta, sigma, gamma

def compute_fixed_point(beta, s, rho, r, tol=1e-6, max_iter=1000):
    K = len(beta)
    delta = np.ones(K)
    sigma = np.ones(K)
    gamma = np.ones(K)

    if not np.allclose(rho, rho.T):
        raise ValueError("The `rho` matrix must be symmetric.")

    for _ in range(max_iter):
        delta_new = np.zeros(K)
        sigma_new = np.zeros(K)
        gamma_new = np.zeros(K)

        for i in range(K):
            delta_new[i] = 1 - sum(
                beta[j] * rho[i, j] * s[i, j] * s[j, i] * (gamma[j] / delta[j])
                for j in range(K)
            )
            sigma_new[i] = np.sqrt(sum(
                beta[j] * s[i, j]**2 / delta[j]**2 *
                norm.expect(lambda x: max(0, (sigma[j] * x + r[j]))**2, loc=0, scale=1)
                for j in range(K)
            ))
            gamma_new[i] = 1 - norm.cdf(-r[i] / sigma[i])

        if (
            np.max(np.abs(delta_new - delta)) < tol and
            np.max(np.abs(sigma_new - sigma)) < tol and
            np.max(np.abs(gamma_new - gamma)) < tol
        ):
            break

        delta, sigma, gamma = delta_new, sigma_new, gamma_new
    else:
        print("Warning: Did not converge within the maximum number of iterations.")

    return delta, sigma, gamma

def compute_fixed_point_final(beta, s, rho, r):
    delta, sigma, gamma = compute_fixed_point(beta, s, rho, r)
    mu = r / delta
    sigma_new = sigma / delta

    a_scaled = (0 - mu) / sigma_new
    b_scaled = np.full_like(mu, np.inf)

    variance = truncnorm.var(a_scaled, b_scaled, loc=mu, scale=sigma_new)
    return variance, gamma
