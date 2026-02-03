# Approximate message passing for block-structured ecological systems

## Overview

Code associated to the article: "Approximate message passing for block-structured ecological systems", by Maxime Clenet and Mohammed-Younes Gueddari.

This repository provides a comprehensive toolkit for analyzing species persistence and abundance distributions in block-structured ecosystems. The code implements **Approximate Message Passing (AMP) theory** for high-dimensional random matrix analysis combined with numerical solvers for Linear Complementarity Problems (LCP).

---

## Project Structure

### Core Modules

#### **Functions.py** (Central Library)
Core utility functions for matrix generation and fixed-point computations:
- `elliptic_normal_matrix_opti(n, rho)`: Generate elliptic Gaussian random matrices with controlled correlation
- `block_matrix(n_size, beta, s, rho)`: Construct block-structured interaction matrices
- `compute_fixed_point(beta, s, rho, r)`: Solve fixed-point equations via iterative method
- `compute_fixed_point_root(beta, s, rho, r)`: Solve fixed-point equations via root-finding
- `compute_fixed_point_final(beta, s, rho, r)`: Compute final abundance predictions with variance estimates

### Application Scripts

#### **Figure1.py** 
Simulates and visualizes ecosystem equilibria with comparison between theory and numerics in Figure 1:
- Generates block-structured interaction matrices
- Solves LCP via damped fixed-point iteration
- Computes AMP theory predictions
- Compares numerical solutions with theoretical truncated normal distributions

#### **Application_1.py, Application_2.py, Application_3.py**
Each application file relates to the application mentioned in the article. 

---

## Requirements

- Python 3.8+
- NumPy
- SciPy (scipy.optimize, scipy.stats)
- Matplotlib
- Seaborn (for enhanced visualizations)

---

## Installation

### Clone the repository:
```bash
git clone https://github.com/maxime-clenet/Block_model_AMP.git
cd Block_model_AMP
```

### Install dependencies:
```bash
pip install numpy scipy matplotlib seaborn
```
---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{clenet2026blockamp,
  author = {Clenet, Maxime},
  title = {Block-Structured Ecological Models: AMP Theory Analysis},
  year = {2026},
  url = {https://github.com/maxime-clenet/Block_model_AMP}
}
```

## Contact

For questions or issues, please contact: maxime.clenet@cefe.cnrs.fr

---

**Last Updated**: February 3, 2026
