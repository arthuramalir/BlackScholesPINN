This README is designed to keep your project structured according to the "Numerical Analysis" requirements of your course while highlighting the "Novelty" of your BSDE research.

# Project: High-Dimensional BSDEs vs. Traditional Numerical Analysis
**Objective:** Evaluating the "Curse of Dimensionality" in Semi-linear Black-Scholes PDEs: A Comparative Study of FDM, PINNs, and Stochastic Heun-BSDE Solvers.

---

## 📂 Project Structure
* `01_1D_Baseline/`: FDM vs. PINN comparison (The "Validation" phase).
* `02_MultiD_Solver/`: PINN scaled to $d=5, 10, 20, 100$.
* `03_BSDE_Extension/`: Implementation of Euler-Maruyama vs. Heun integration.
* `04_Path_Integral/`: (Optional) Wiener Path Integral Monte Carlo baseline.
* `results/`: Plots for Accuracy, Speed, and Memory usage.

---

## 📈 Phase 1: The 1D Comparison (Baseline)
**Goal:** Prove the PINN works by comparing it to the "Gold Standard" Finite Difference Method.
* **Method:** Solve the 1D Black-Scholes Eq.
* **Metric - Accuracy:** Plot the $L^2$ error of FDM vs. PINN against the analytical solution.
* **Metric - Speed:** Document the time to reach $10^{-3}$ error.
* **Numerical Link:** Discuss the **CFL Condition** and grid spacing ($\Delta x$) for FDM.

---

## 🚀 Phase 2: Scaling to 5D+ (The "Curse" Analysis)
**Goal:** Demonstrate the "Memory Wall" where traditional methods fail.
* **FDM Analysis:** Calculate the theoretical memory requirement for a 5D, 10D, and 20D grid. 
    * *Example:* 20D grid with 10 points/dim = $10^{20}$ nodes. Show that FDM is physically impossible here.
* **PINN Implementation:** Use Latin Hypercube Sampling (LHS) to train in 5D.
* **Speed Gain:** Plot **Dimension ($d$) vs. Training Time**. The PINN should show linear or low-polynomial scaling, proving it "breaks" the curse.



---

## 🧠 Phase 3: The BSDE "Heun" Novelty
**Goal:** Address the 2024-2025 research gap regarding discretization bias.
* **Problem:** Standard Euler-Maruyama (EM) BSDEs have a "bias floor" (they stop getting more accurate after a certain point).
* **Solution:** Implement the **Stochastic Heun** scheme (2nd order).
* **Analysis:** Compare the **Bias vs. Time-step size ($\Delta t$)**. Show that Heun allows for larger time steps (faster training) with higher final accuracy than EM.

---

## 🔗 Extension: Path Integral (PI) Method
**Goal:** Provide a third comparison point using the Feynman-Kac formula.
* **Structure:** Use Monte Carlo sampling of Wiener Paths.
* **Comparison:** * PI is "embarrassingly parallel" but has high variance.
    * PINNs "smooth" this variance through the Neural Network's inductive bias.
* **The Final Argument:** Why the NN-BSDE approach is the most robust for high-dimensional engineering uncertainty.



---

## 📊 Key Results to Include in Report
1.  **Table 1:** Memory usage (MB) for FDM vs. PINN as $d$ increases from 1 to 5.
2.  **Figure 1:** $L^2$ Relative Error vs. Dimension (FDM should stop at $d=4$, PINN should continue).
3.  **Figure 2:** Convergence plot of EM-BSDE vs. Heun-BSDE (The "Novelty" proof).
4.  **Discussion:** The trade-off between the "stochastic noise" of path integrals and the "optimization difficulty" of PINNs.

---

### 🛠 Quick Commands
* `python train.py --dim 1 --mode fdm` (Run 1D baseline)
* `python train.py --dim 20 --mode pinn` (Run high-D study)
* `python train.py --dim 20 --mode bsde --scheme heun` (Run the novelty test)