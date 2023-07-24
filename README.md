# Resource Optimization for HOPP

This problem deals with the size optimization of the energy resources such as wind, solar and battery. Given some historical wind, solar, battery data for a year, we find the best site sizes in order to minimize the levelized cost of hydrogen (LCOH).

The optimization formulation is as under:

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

\$\mathcal P\$ indicates used powers and \$\mathcal S\$ indicates sizes of the individual energy sources. 

Code deets:

1. Use `optimize_fractional.py` for the nonlinear version solved using `ipopt`.
2. Use `optimize_linear.py` for the linearized version solved using `cbc` (solves in 15s for a whole year).
3. `run_optimization.ipynb` for the example notebook.
