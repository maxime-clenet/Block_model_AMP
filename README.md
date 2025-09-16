# Block-Structured Ecological Models

This repository contains tools to study ecological communities with block-structured interaction matrices.  
It provides functions to generate interaction matrices by blocks and to analyze persistence and statistical properties of communities by solving the associated Lotka–Volterra equilibria using Linear Complementarity Problems (LCPs).

---

## Files

- **Block_matrix.py**  
  Create interaction matrices with a block structure.  
  Blocks can be parameterized by size and interaction strength to represent multiple interacting communities.

- **Empirical.py**  
  Compute community-level properties (persistence, variance, mean abundance) by solving the LCP associated with the Lotka–Volterra system.  
  Uses the **Lemke algorithm** for solving the LCP.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/maxime-clenet/Block_model_AMP.git
cd Block_model_AMP
