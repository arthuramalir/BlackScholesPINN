# Project Plan: Curse of Dimensionality in Black-Scholes PDEs

## 1) Core Objective
Evaluate how solution quality and computational cost scale with dimension for Black-Scholes PDEs, comparing Physics-Informed Neural Networks (PINNs) against traditional numerical baselines.

Primary dimensions: d in {1, 2, 5}.

Primary question:
Can PINNs maintain competitive accuracy as d increases while full-grid PDE methods become memory-limited?

## 2) Scope and Positioning
This project is about PDE solvers and scaling, not about proving a universal PINN advantage.

What will be claimed:
- Full-grid finite differences suffer exponential state-space growth (memory wall).
- PINNs show more favorable empirical scaling in tested dimensions.

What will not be claimed:
- PINNs "solve" the curse of dimensionality in general.
- PINNs dominate all classical methods in all regimes.

## 3) Problem Family
Use risk-neutral multi-asset Black-Scholes PDE:

dV/dt + 0.5 * sum_{i=1..d} sum_{j=1..d} rho_ij sigma_i sigma_j S_i S_j d2V/(dS_i dS_j)
+ r * sum_{i=1..d} S_i dV/dS_i - rV = 0

Test payoffs:
- d=1: Vanilla European call (closed-form reference exists).
- d=2: Basket options.
  - Geometric basket call (closed-form reference exists).
  - Arithmetic basket call (no closed form; stochastic reference).
- d=5: Arithmetic basket call (stochastic reference only).

## 4) Experiment Matrix
- Row A (d=1): Vanilla call.
  - References: analytical solution + Crank-Nicolson FDM.
  - Goal: validate baseline correctness.
- Row B (d=2): Basket call.
  - References: geometric closed-form + arithmetic stochastic reference.
  - Goal: validate cross-derivative handling and correlation coupling.
- Row C (d=5): Arithmetic basket.
  - References: stochastic high-accuracy benchmark.
  - Goal: quantify scaling and memory wall impact.

## 5) Baselines and Fairness Protocol
### FDM policy
- Use FDM where feasible:
  - d=1 full baseline.
  - d=2 coarse/moderate baseline.
- For d>=5 full-grid FDM is reported as infeasible due to memory/time, not "skipped without analysis".

Memory accounting for tensor grids:
- Nodes = n^d
- Lower bound memory per field (float64) = 8 * n^d bytes
- Practical lower bound with 3 arrays = 24 * n^d bytes

### PINN policy
- Architecture fixed for comparison runs (e.g., depth 4, width 128, Tanh).
- Sweep only data/collocation budgets and training epochs.
- Track training and inference cost separately.

### Stochastic reference policy
- For no-closed-form cases use a high-quality stochastic benchmark.
- Report estimator uncertainty (confidence intervals or standard error).

## 6) Metrics to Report
Accuracy:
- MAE
- Relative L2 error
- Max error

Compute:
- Training wall-clock time
- Inference wall-clock time
- Peak memory usage

Robustness:
- 3 to 5 seeds per setting
- Report median and IQR

Scaling views:
- Error vs dimension
- Time vs dimension
- Memory vs dimension
- Error vs compute budget

## 7) Execution Order
1. Validate Row A end-to-end (d=1).
2. Implement and test multi-asset PDE residual (d=2).
3. Add basket data generation and stochastic reference.
4. Run Row B and verify consistency against geometric closed form.
5. Run Row C with fixed compute budgets and seed sweeps.
6. Produce final scaling plots and memory-wall table.

## 8) Concrete Outputs for Report
- Table 1: Memory growth for FDM tensor grids (d=1,2,5,10,20).
- Figure 1: Relative L2 error vs dimension.
- Figure 2: Training time vs dimension.
- Figure 3: Peak memory vs dimension.
- Figure 4: Error vs compute budget (PINN vs feasible baselines).
- Discussion: tradeoff between deterministic discretization error and optimization/statistical error.

## 9) Optional Extensions (Not Core)
- BSDE discretization comparison (Euler-Maruyama vs stochastic Heun).
- Path-integral or multi-level Picard comparisons.

These are useful add-ons, but must remain separate from the core PDE scaling claim.

## 10) Repository Mapping (Planned)
- fdm.py: d=1 baseline, d=2 coarse extension if feasible.
- loss.py: pde_residual_multid and total PINN losses.
- data_multi.py: correlated GBM sampling and payoff generation.
- mc_reference.py: stochastic reference with uncertainty estimates.
- compare_methods.py / runner scripts: reproducible experiment execution.
- results/: generated tables and plots.

## 11) Minimal Run Targets
- Pilot budget:
  - N_colloc: 5k
  - N_data: 1k
  - Epochs: 2k
- Medium budget:
  - N_colloc: 20k to 50k
  - N_data: 3k to 10k
  - Epochs: 5k
- Large budget:
  - N_colloc: 100k+
  - N_data: 30k+
  - Epochs: 10k+

Use pilot runs first for correctness, then medium/large for final scaling claims.
