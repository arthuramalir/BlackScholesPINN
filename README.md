# BlackScholesPINN

A Python implementation of Physics-Informed Neural Networks (PINNs) for solving the Black-Scholes partial differential equation used in option pricing.

---

## 📌 What is this?

This repository demonstrates how to use Physics-Informed Neural Networks (PINNs) to learn the solution of the **Black-Scholes equation** — a foundational model in financial mathematics for pricing European call options.

PINNs are neural networks that are trained not just on data, but also on the **underlying physical (or financial) laws** described by differential equations.

---

## 🚀 Features

- ✅ Clean modular design
- ✅ Configurable via `config.json`
- ✅ Supports noisy synthetic data generation
- ✅ Enforces PDE constraint using autograd
- ✅ Lightweight and dependency-free (only PyTorch + NumPy + matplotlib)
- ✅ Fully reproducible

---

## 🧠 What You’ll Learn

- How to generate synthetic financial data using the Black-Scholes formula
- How to train a neural network to obey a PDE using automatic differentiation
- How to combine **data loss** and **PDE loss** in a single objective
- How to modularize ML code for experimentation and reuse

---

## 🗂 Project Structure

```
.
├── black_scholes.py                         # Main wrapper class for training/evaluation
├── compare_methods.py                       # Side-by-side PINN vs FDM comparison script
├── config.json                              # All key hyperparameters
├── data.py                                  # Synthetic data and collocation point generation
├── fdm.py                                   # Crank-Nicolson finite-difference baseline
├── loss.py                                  # PDE residual and total loss function
├── model.py                                 # Neural network architecture (PINN)
├── train.py                                 # Training loop
├── utils.py                                 # Black-Scholes analytical solution
├── example/BlackScholesModel.ipynb          # Notebook for dev or exploration
└── README.md                                # This file
```



## ⚙️ How to Use

1. Install dependencies

```bash
pip install torch numpy matplotlib scipy
```

2. Train the PINN

```bash
python main.py
```

3. Modify configuration

All training and model parameters can be changed in `config.json`, including:

- `K` — Strike price of the option  
- `T` — Time to maturity (in years)  
- `r` — Risk-free interest rate  
- `sigma` — Volatility of the underlying asset  
- `N_data` — Number of synthetic data points to generate  
- `bias` — Constant value added to the synthetic labels  
- `noise_variance` — Standard deviation of Gaussian noise added to synthetic data  
- `min_S`, `max_S` — Range for sampling stock prices (`S`)  
- `epochs` — Number of training iterations  
- `lr` — Learning rate for the optimizer  
- `log_interval` — Number of epochs between log printouts  
- `model_path` — Path where the trained model will be saved  

---

## Output Example

After training, the model compares its predicted call prices with the true Black-Scholes analytical solution at time `t = 0`. A typical output plot shows the learned function overlayed with ground truth.

## Finite-Difference Baseline

To compare the PINN against a traditional solver in a defensible way, this repo now includes a simple Crank-Nicolson finite-difference baseline in `fdm.py`.

Use it as the reference implementation for the same 1D European call problem:

- same PDE
- same parameters `K`, `T`, `r`, `sigma`
- same domain in `S`
- same call-option terminal and boundary conditions

The fair comparison is not "PINN output vs FDM output at different setups". It is:

1. Solve the same PDE with both methods.
2. Evaluate both on the same `S` grid at `t = 0`.
3. Report the same metrics, such as max error, $L^2$ error, runtime, and sensitivity to resolution or sampling.

For this problem, the PINN is mainly interesting as a mesh-free optimization approach, while FDM is the classical accuracy and robustness baseline.

## Study Design Notes

If your goal is a defensible comparison, the study should be organized around the same mathematical problem and not around matching the methods one-to-one.

For the current 1D Black-Scholes case:

- Use the same call option, same parameters, same domain, and the same evaluation grid.
- Compare both methods at `t = 0` against the analytic solution.
- Report accuracy (`MAE`, relative `L2` error, max error), wall-clock time, and sensitivity to discretization or collocation budget.
- Treat PINN training time separately from inference time, because that training cost is part of the method.

For an augmented-dimension study, a clean design is:

1. Define a family of multi-asset Black-Scholes problems with dimension `d = 1, 2, 5`.
2. Keep the payoff family fixed as much as possible, for example a basket call or an arithmetic-average call.
3. Normalize the domain and rescale inputs so the network is not punished by trivial scaling issues.
4. Compare methods under equal information budgets rather than equal code structure.
5. For `d = 1`, use FDM as the reference baseline; for higher `d`, full-grid FDM becomes impractical and you will likely need sparse-grid, Monte Carlo, or semi-analytic references.
6. Report scaling curves: error versus dimension, error versus sample budget, and runtime versus dimension.

The key scientific claim you can support is not that a PINN is universally better than FDM, but that it may degrade more gracefully than grid-based methods as `d` grows.

Run the comparison script with:

```bash
python compare_methods.py
```

It will train or load the PINN, run the Crank-Nicolson baseline, compare both against the analytic solution, and save a plot.

---

## Background

The Black-Scholes model describes the price of a European call option as a solution to the following partial differential equation:

```
∂C/∂t + 0.5 * σ² * S² * ∂²C/∂S² + r * S * ∂C/∂S - r * C = 0
```

Where:
- `C` is the call option price  
- `S` is the stock price  
- `σ` is the volatility  
- `r` is the risk-free interest rate  
- `t` is time to maturity

This project uses a Physics-Informed Neural Network (PINN) to approximate the solution by minimizing both data error and residuals of the PDE.

---

## Author

Piero Paialunga  
PhD in Aerospace Engineering  
Focused on AI for Physics, Finance, and Engineering Problems
