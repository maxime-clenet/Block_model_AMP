# Block-Structured Ecological Models

This repository provides tools to study species persistence and abundance distributions in ecosystems where species interactions are structured into blocks.  
The code combines matrix generation, Linear Complementarity Problem (LCP) solvers, and statistical analyses to explore the dynamics of block-structured Lotka–Volterra systems.

---

## Files

- **Block_matrix.py**  
  Create block-structured interaction matrices.  
  Blocks are parameterized by size and interaction strength to represent multiple interacting communities.

- **Empirical.py**  
  Compute persistence, variance, and mean abundance of communities by solving the LCP with the **Lemke algorithm**.

- **Iteratif.py**  
  Solve the LCP using an **iterative process** as an alternative to Lemke’s method.

- **Law_comparison.py**  
  Solve the LCP for a block-structured interaction matrix (using Lemke's method) and compare the **empirical distribution of persistent species** to **theoretical truncated normal PDFs**.

- **Law_theory.py**  
  Simulate and visualize **truncated distributions of species abundances** from fixed-point computations in block-structured models (a way to verify that the distribution matches the histogram when we generate the solution from a normal sample).

- **Theory_vs_Empirical.py**  
  Compare **fixed-point predictions** vs. **empirical estimates** of the effective parameters  
  (γ, σ) as functions of interaction strength `s` in block-structured systems.


## Installation

Clone the repository:

```bash
git clone https://github.com/maxime-clenet/Block_model_AMP.git
cd Block_model_AMP
