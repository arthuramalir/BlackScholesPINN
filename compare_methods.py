import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from black_scholes import BlackScholesPINN
from fdm import black_scholes_fdm


def black_scholes_closed_form(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    safe_S = np.maximum(S, 1e-12)
    d1 = (np.log(safe_S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def l2_error(reference, prediction):
    numerator = np.linalg.norm(prediction - reference)
    denominator = np.linalg.norm(reference)
    if denominator == 0:
        return float(numerator)
    return float(numerator / denominator)


def load_or_train_pinn(config, force_train=False):
    pinn = BlackScholesPINN(config)
    model_path = config.get("model_path", "model.pth")

    if os.path.exists(model_path) and not force_train:
        print(f"[1/5] Loading existing PINN weights from {model_path} ...")
        state_dict = torch.load(model_path, map_location="cpu")
        pinn.model.load_state_dict(state_dict)
        pinn.model.eval()
        print("[1/5] Loaded existing model. Skipping training.")
        return pinn, 0.0, False

    if force_train and os.path.exists(model_path):
        print(f"[1/5] --force-train set. Ignoring existing model at {model_path}.")
    else:
        print("[1/5] No saved model found. Training PINN from scratch.")

    print("[2/5] Training PINN ...")
    start = time.perf_counter()
    pinn.train()
    pinn.export()
    pinn.model.eval()
    elapsed = time.perf_counter() - start
    print(f"[2/5] PINN training complete in {elapsed:.2f} s.")
    return pinn, elapsed, True


def main(config_path, force_train=False):
    print("[0/5] Loading configuration ...")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"[0/5] Loaded config from {config_path}")

    torch.manual_seed(config.get("seed", 1234))
    np.random.seed(config.get("seed", 1234))

    pinn, training_time, trained_now = load_or_train_pinn(config, force_train=force_train)

    n_eval = config.get("n_eval", 400)
    S_grid = np.linspace(config["min_S"], config["max_S"], n_eval)
    t_grid = np.zeros_like(S_grid)
    S_eval = torch.tensor(S_grid, dtype=torch.float32).view(-1, 1)
    t_eval = torch.tensor(t_grid, dtype=torch.float32).view(-1, 1)

    print(f"[3/5] Running PINN inference on {n_eval} points ...")
    pinn_start = time.perf_counter()
    with torch.no_grad():
        pinn_pred = pinn.predict(S_eval, t_eval).cpu().numpy().flatten()
    pinn_time = time.perf_counter() - pinn_start
    print(f"[3/5] PINN inference complete in {pinn_time:.4f} s.")

    print(f"[4/5] Solving FDM baseline with n_S={n_eval - 1}, n_T={config.get('fdm_n_T', n_eval - 1)} ...")
    fdm_start = time.perf_counter()
    fdm_S, fdm_pred, _ = black_scholes_fdm(
        config["K"],
        config["T"],
        config["r"],
        config["sigma"],
        S_max=config["max_S"],
        n_S=n_eval - 1,
        n_T=config.get("fdm_n_T", n_eval - 1),
    )
    fdm_time = time.perf_counter() - fdm_start
    print(f"[4/5] FDM solve complete in {fdm_time:.4f} s.")

    print("[5/5] Computing metrics and saving plot ...")
    reference = black_scholes_closed_form(S_grid, config["K"], config["T"], config["r"], config["sigma"])

    pinn_mae = float(np.mean(np.abs(pinn_pred - reference)))
    fdm_mae = float(np.mean(np.abs(fdm_pred - reference)))
    pinn_l2 = l2_error(reference, pinn_pred)
    fdm_l2 = l2_error(reference, fdm_pred)

    print("=== Black-Scholes comparison at t = 0 ===")
    if trained_now:
        print(f"PINN training time: {training_time:.4f} s")
    else:
        print("PINN training time: skipped (existing model loaded)")
    print(f"PINN inference time: {pinn_time:.4f} s")
    print(f"FDM solve time: {fdm_time:.4f} s")
    print(f"PINN MAE vs analytic: {pinn_mae:.6e}")
    print(f"FDM MAE vs analytic:  {fdm_mae:.6e}")
    print(f"PINN relative L2 error: {pinn_l2:.6e}")
    print(f"FDM relative L2 error:  {fdm_l2:.6e}")
    print(f"PINN vs FDM MAE: {np.mean(np.abs(pinn_pred - fdm_pred)):.6e}")

    plt.figure(figsize=(10, 6))
    plt.plot(S_grid, reference, label="Analytic Black-Scholes", color="black", linewidth=2)
    plt.plot(S_grid, pinn_pred, label="PINN", color="tab:blue", linestyle="--")
    plt.plot(fdm_S, fdm_pred, label="FDM (Crank-Nicolson)", color="tab:orange", linestyle=":")
    plt.xlabel("Stock Price S")
    plt.ylabel("Call Option Price")
    plt.title("PINN vs FDM for the 1D Black-Scholes Call Option")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = config.get("comparison_plot", "comparison_pinn_fdm.png")
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")
    print("[5/5] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PINN and FDM on the Black-Scholes problem")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force PINN retraining even if model_path already exists",
    )
    args = parser.parse_args()
    main(args.config, force_train=args.force_train)
