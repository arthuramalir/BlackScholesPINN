import numpy as np
from scipy.stats import norm


def _sigma_eff(sigma_vec, corr_matrix=None, weights=None):
    sigma_vec = np.asarray(sigma_vec, dtype=float)
    d = sigma_vec.size
    if weights is None:
        weights = np.ones(d) / d
    w = np.asarray(weights).reshape(-1, 1)
    if corr_matrix is None:
        cov = np.outer(sigma_vec, sigma_vec)
    else:
        corr = np.asarray(corr_matrix, dtype=float)
        cov = np.outer(sigma_vec, sigma_vec) * corr
    sigma2 = float((w.T @ cov @ w).squeeze())
    return np.sqrt(sigma2)


def geometric_basket_price(S, K, T, r, sigma_vec, corr_matrix=None, weights=None):
    """Closed-form price for a geometric-weighted basket call (equal weights default).

    S : array-like of spot prices, shape (d,) or (n,d)
    K : strike (scalar)
    T : time to maturity
    r : risk-free rate
    sigma_vec : volatilities per asset, length d
    corr_matrix : optional correlation matrix (d x d)
    weights : optional weights (length d). If None, equal weights (1/d).

    Returns: price(s) as float or ndarray matching first dim of S.
    """
    S = np.asarray(S, dtype=float)
    single = (S.ndim == 1)
    if single:
        S = S.reshape(1, -1)
    d = S.shape[1]
    if weights is None:
        weights = np.ones(d) / d
    weights = np.asarray(weights, dtype=float)

    # geometric mean initial value for each sample
    logS = np.log(S)
    lnG0 = (logS * weights).sum(axis=1)
    G0 = np.exp(lnG0)

    sigma_eff = _sigma_eff(sigma_vec, corr_matrix=corr_matrix, weights=weights)

    # Black-Scholes formula applied to geometric basket G0
    if sigma_eff <= 0:
        # degenerate: price is discounted intrinsic
        price = np.exp(-r * T) * np.maximum(G0 - K, 0.0)
    else:
        sqrtT = np.sqrt(T)
        d1 = (np.log(G0 / K) + (r + 0.5 * sigma_eff**2) * T) / (sigma_eff * sqrtT)
        d2 = d1 - sigma_eff * sqrtT
        price = G0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return price[0] if single else price


def black_scholes_price(S, K, T, r, sigma_vec, corr_matrix=None, payoff="geometric"):
    """Wrapper: compute price for d-dimension problem.

    - If `payoff` == 'geometric' and d>=1 -> uses geometric_basket_price closed form.
    - If `payoff` == 'vanilla' and d==1 -> uses standard Black-Scholes closed form.
    - For arithmetic basket or unsupported cases, raises NotImplementedError.
    """
    S = np.asarray(S, dtype=float)
    d = S.size if S.ndim == 1 else S.shape[1]
    if payoff == "geometric":
        return geometric_basket_price(S, K, T, r, sigma_vec, corr_matrix=corr_matrix)
    if payoff == "vanilla" and d == 1:
        # apply Black-Scholes on single underlying
        S0 = float(S) if S.ndim == 0 else float(S.flatten()[0])
        sigma = float(np.asarray(sigma_vec).ravel()[0])
        sqrtT = np.sqrt(T)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    raise NotImplementedError("Arithmetic-basket closed form is not available. Use MC reference.")


def pde_equation_string(d):
    """Return a human-readable PDE string for multi-asset Black-Scholes of dimension d.

    The generic form is:
    \partial_t V + 1/2 \sum_{i,j} a_{ij} S_i S_j \partial_{S_i,S_j} V + \sum_i r S_i \partial_{S_i} V - r V = 0

    where a_{ij} = sigma_i sigma_j rho_{ij}.
    """
    lines = []
    lines.append("General multi-asset Black–Scholes PDE (risk-neutral):")
    lines.append("∂V/∂t + 1/2 Σ_{i=1..d}Σ_{j=1..d} a_ij S_i S_j ∂^2V/(∂S_i ∂S_j) + Σ_{i=1..d} r S_i ∂V/∂S_i - r V = 0")
    lines.append("")
    lines.append("where a_ij = sigma_i * sigma_j * rho_ij (covariance of log-returns).")
    lines.append("")
    lines.append("Explicit for d={}:".format(d))
    if d <= 5:
        for i in range(1, d + 1):
            for j in range(1, d + 1):
                coef = f"(1/2) a_{{{i}{j}}} S_{i} S_{j} ∂^2V/∂S_{i}∂S_{j}"
                if i == 1 and j == 1:
                    lines.append("PDE terms:")
                lines.append(f"  + {coef}")
        for i in range(1, d + 1):
            lines.append(f"  + r S_{i} ∂V/∂S_{i}")
        lines.append("  - r V")
    else:
        lines.append("d > 5: use the generic summation form above.")
    return "\n".join(lines)


if __name__ == "__main__":
    # Small self-test and demonstration
    print(pde_equation_string(1))
    print()
    print(pde_equation_string(2))
    print()
    # Example geometric basket price for d=3
    S = [100, 95, 105]
    K = 100
    T = 1.0
    r = 0.01
    sigma_vec = [0.2, 0.25, 0.22]
    print("Geometric basket price (d=3):", geometric_basket_price(S, K, T, r, sigma_vec))
