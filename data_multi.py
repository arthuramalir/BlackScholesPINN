import numpy as np
import torch
from scipy.stats import norm, qmc


def _as_vector(value, d, name):
    """Convert scalar/list input to a length-d numpy vector."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(d, float(arr), dtype=float)
    arr = arr.reshape(-1)
    if arr.size != d:
        raise ValueError(f"{name} must have length {d}, got {arr.size}")
    return arr


def build_correlation_matrix(d, rho=None, corr_matrix=None):
    """Build a valid d x d correlation matrix.

    Priority:
    1) explicit corr_matrix
    2) scalar rho for constant off-diagonal correlation
    3) identity
    """
    if corr_matrix is not None:
        corr = np.asarray(corr_matrix, dtype=float)
        if corr.shape != (d, d):
            raise ValueError(f"corr_matrix must be shape ({d}, {d})")
        return corr

    if rho is None:
        return np.eye(d, dtype=float)

    rho = float(rho)
    corr = np.full((d, d), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    return corr


def _sample_standard_normals(n_samples, d, seed=None, method="lhs"):
    """Sample N(0, I_d) with either LHS or pseudo-random sampling."""
    if method not in {"lhs", "random"}:
        raise ValueError("method must be either 'lhs' or 'random'")

    if method == "lhs":
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        u = sampler.random(n=n_samples)
        # Clamp away from 0/1 to avoid inf from inverse CDF.
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        z = norm.ppf(u)
    else:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_samples, d))

    return z


def sample_terminal_gbm(
    n_samples,
    d,
    S0,
    T,
    r,
    sigma,
    corr_matrix=None,
    rho=None,
    seed=None,
    method="lhs",
):
    """Sample correlated terminal prices S(T) under risk-neutral GBM.

    S_i(T) = S0_i * exp((r - 0.5 sigma_i^2) T + sigma_i sqrt(T) Z_i)
    with correlated Z ~ N(0, Corr).

    Returns:
        ndarray of shape (n_samples, d)
    """
    S0_vec = _as_vector(S0, d, "S0")
    sigma_vec = _as_vector(sigma, d, "sigma")
    corr = build_correlation_matrix(d=d, rho=rho, corr_matrix=corr_matrix)

    z_independent = _sample_standard_normals(
        n_samples=n_samples,
        d=d,
        seed=seed,
        method=method,
    )

    # Cholesky factorization for correlation coupling.
    chol = np.linalg.cholesky(corr)
    z_corr = z_independent @ chol.T

    drift = (r - 0.5 * sigma_vec**2) * T
    diffusion = sigma_vec * np.sqrt(T)
    log_ST = np.log(S0_vec) + drift + z_corr * diffusion
    return np.exp(log_ST)


def basket_payoff_arithmetic(ST, K, weights=None, option_type="call"):
    """Arithmetic basket payoff for terminal prices ST.

    ST: shape (n_samples, d)
    """
    ST = np.asarray(ST, dtype=float)
    if ST.ndim != 2:
        raise ValueError("ST must be a 2D array of shape (n_samples, d)")

    d = ST.shape[1]
    if weights is None:
        weights = np.ones(d, dtype=float) / d
    else:
        weights = _as_vector(weights, d, "weights")

    basket = ST @ weights
    if option_type == "call":
        return np.maximum(basket - K, 0.0)
    if option_type == "put":
        return np.maximum(K - basket, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def basket_payoff_geometric(ST, K, weights=None, option_type="call"):
    """Geometric basket payoff for terminal prices ST.

    ST: shape (n_samples, d)
    """
    ST = np.asarray(ST, dtype=float)
    if ST.ndim != 2:
        raise ValueError("ST must be a 2D array of shape (n_samples, d)")

    d = ST.shape[1]
    if weights is None:
        weights = np.ones(d, dtype=float) / d
    else:
        weights = _as_vector(weights, d, "weights")

    # Weighted geometric mean: exp(sum_i w_i log(S_i))
    log_g = np.log(np.maximum(ST, 1e-12)) @ weights
    basket = np.exp(log_g)

    if option_type == "call":
        return np.maximum(basket - K, 0.0)
    if option_type == "put":
        return np.maximum(K - basket, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def generate_multid_terminal_data(
    config,
    n_samples=None,
    payoff_type="arithmetic",
    option_type="call",
    method="lhs",
    seed=None,
):
    """Generate terminal-condition data for multi-asset PINN training.

    Returns:
        S_T_torch: shape (N, d), requires_grad=True
        t_T_torch: shape (N, 1), all equal to T, requires_grad=True
        payoff_torch: shape (N, 1)
    """
    d = int(config["d"])
    N = int(n_samples if n_samples is not None else config.get("N_data", 3000))

    S0 = config.get("S0", config.get("K", 1.0))
    sigma = config["sigma"]
    corr_matrix = config.get("corr_matrix")
    rho = config.get("rho")

    ST = sample_terminal_gbm(
        n_samples=N,
        d=d,
        S0=S0,
        T=float(config["T"]),
        r=float(config["r"]),
        sigma=sigma,
        corr_matrix=corr_matrix,
        rho=rho,
        seed=seed,
        method=method,
    )

    weights = config.get("weights")
    K = float(config["K"])
    if payoff_type == "arithmetic":
        payoff = basket_payoff_arithmetic(ST, K=K, weights=weights, option_type=option_type)
    elif payoff_type == "geometric":
        payoff = basket_payoff_geometric(ST, K=K, weights=weights, option_type=option_type)
    else:
        raise ValueError("payoff_type must be 'arithmetic' or 'geometric'")

    S_torch = torch.tensor(ST, dtype=torch.float32, requires_grad=True)
    t_torch = torch.full((N, 1), float(config["T"]), dtype=torch.float32, requires_grad=True)
    payoff_torch = torch.tensor(payoff.reshape(-1, 1), dtype=torch.float32)

    return S_torch, t_torch, payoff_torch


def generate_lhs_collocation_points(config, n_colloc=10000, seed=None):
    """Generate collocation points (S, t) with Latin Hypercube Sampling.

    S is sampled in [min_S, max_S]^d.
    t is sampled in [0, T].

    Returns:
        S_colloc_torch: shape (N, d), requires_grad=True
        t_colloc_torch: shape (N, 1), requires_grad=True
    """
    d = int(config["d"])
    T = float(config["T"])

    min_S = _as_vector(config.get("min_S", 1.0), d, "min_S")
    max_S = _as_vector(config.get("max_S", 2.0), d, "max_S")
    if np.any(max_S <= min_S):
        raise ValueError("Each max_S component must be > min_S")

    sampler = qmc.LatinHypercube(d=d + 1, seed=seed)
    u = sampler.random(n=int(n_colloc))

    # First d dimensions for S-space, last one for t-space.
    S_u = u[:, :d]
    t_u = u[:, d: d + 1]

    S = min_S + S_u * (max_S - min_S)
    t = t_u * T

    S_torch = torch.tensor(S, dtype=torch.float32, requires_grad=True)
    t_torch = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    return S_torch, t_torch
