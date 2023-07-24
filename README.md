**Problem Description:**

This problem involves optimizing the sizes of energy resources (wind, solar, battery) to minimize the levelized cost of hydrogen (LCOH). The given data includes historical wind, solar, and battery information for a year. The goal is to find the best site sizes that minimize the LCOH. The optimization formulation is as follows:

Minimize:
```math
\begin{align}
  \quad \min_{ \mathcal S_{wind}, \mathcal S_{solar}, \mathcal S_{bat}, \mathcal P_{wind}, \mathcal P_{solar}, \mathcal P_{bat}} & \frac{AC^{tot}}{F^{tot}} \\
 \textrm{s.t.} 
\quad & AC^{tot} =  C_{t}^{wind}\mathcal S^{t}_{wind} + C_{t}^{solar}\mathcal S^{t}_{solar} + C_{t}^{bat}\mathcal S^{t}_{bat} + \text{Capital Cost}  \\
\quad &  F^{tot} = \sum_{t=t_0}^{t_f}  F_{t} \\
\quad &F^{t} = 20 P^{t}_{elec}  \qquad \forall   t \in  \{t_0, \dots t_f\} \\
&   \mathcal P^{t}_{wind} +  \mathcal P^{t}_{solar} + \mathcal P^{t}_{bat} =  \mathcal P^{t}_{elec} \qquad \forall t \in \{t_0, \dots t_f\}\\
&   \mathcal S_{x} \geq \mathcal P^t_x \qquad \forall   t \in  \{t_0, \dots t_f\} \text{ and } x \in \{wind, solar, bat, elec\}
\end{align}
```

where:
- \$\mathcal{P}\$ indicates used powers.
- \$\mathcal{S}\$ indicates sizes of individual energy sources.

**Code Details:**

The problem can be approached using two different code files, which are provided:

1. `optimize_fractional.py`: This file contains the nonlinear version of the problem, which is solved using the `ipopt` solver.

2. `optimize_linear.py`: This file contains the linearized version of the problem, which is solved using the `cbc` solver. This solver is faster and can handle the optimization for a whole year in approximately 15 seconds.

3. `run_optimization.ipynb`: This is an example notebook that demonstrates how to run the optimization using the provided code files.

**Note:**
Please ensure you have the necessary libraries and dependencies installed to run the optimization code successfully. Also, make sure to set up the correct input data for the historical wind, solar, and battery information for a year before running the optimization.

To summarize, this problem deals with optimizing energy resource sizes to minimize the levelized cost of hydrogen (LCOH), and you can use the provided code files (`optimize_fractional.py` or `optimize_linear.py`) based on your specific requirements and computational capabilities.
