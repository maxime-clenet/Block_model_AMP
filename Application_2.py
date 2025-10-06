import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm, truncnorm
from Benchmark.Theory import compute_fixed_point_final

beta = [0.5, 0.5]
rho = np.array([[0, 0], [0, 0]])
s = np.array([[0.5, 0.5], [0.5, 0.5]])
r = np.array([1, 1])
compute_fixed_point_final(beta, s, rho, r)