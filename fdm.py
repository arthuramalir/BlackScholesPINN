import numpy as np


def _thomas_algorithm(lower, diagonal, upper, rhs):
    """Solve a tridiagonal linear system with the Thomas algorithm."""
    n = len(diagonal)
    if n == 1:
        return rhs / diagonal

    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    c_prime[0] = upper[0] / diagonal[0]
    d_prime[0] = rhs[0] / diagonal[0]

    for i in range(1, n - 1):
        denominator = diagonal[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / denominator
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denominator

    denominator = diagonal[-1] - lower[-1] * c_prime[-1]
    d_prime[-1] = (rhs[-1] - lower[-1] * d_prime[-2]) / denominator

    solution = np.zeros(n)
    solution[-1] = d_prime[-1]

    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]

    return solution


def black_scholes_fdm(
    K,
    T,
    r,
    sigma,
    S_max=None,
    n_S=400,
    n_T=400,
):
    """Price a European call with a Crank-Nicolson finite-difference scheme.

    The PDE is solved in forward time-to-maturity tau = T - t, with terminal
    payoff V(S, 0) = max(S - K, 0).

    Returns
    -------
    S_grid : ndarray
        Asset price grid.
    V : ndarray
        Option values at tau = T, corresponding to t = 0.
    surface : ndarray
        Full solution surface with shape (n_T + 1, n_S + 1).
    """
    if S_max is None:
        S_max = 4.0 * K

    S_grid = np.linspace(0.0, S_max, n_S + 1)
    dS = S_grid[1] - S_grid[0]
    dt = T / n_T

    # tau = 0 is the terminal payoff.
    surface = np.zeros((n_T + 1, n_S + 1), dtype=float)
    surface[0, :] = np.maximum(S_grid - K, 0.0)

    interior_count = n_S - 1
    asset_interior = S_grid[1:-1]

    for n in range(n_T):
        tau_now = n * dt
        tau_next = (n + 1) * dt

        # European call boundaries.
        left_now = 0.0
        left_next = 0.0
        right_now = S_max - K * np.exp(-r * tau_now)
        right_next = S_max - K * np.exp(-r * tau_next)

        v_now = surface[n, 1:-1]

        alpha = 0.25 * dt * (
            sigma**2 * (asset_interior**2) / (dS**2) - r * asset_interior / dS
        )
        beta = -0.5 * dt * (sigma**2 * (asset_interior**2) / (dS**2) + r)
        gamma = 0.25 * dt * (
            sigma**2 * (asset_interior**2) / (dS**2) + r * asset_interior / dS
        )

        lower = -alpha[1:]
        diagonal = 1.0 - beta
        upper = -gamma[:-1]

        rhs = alpha * surface[n, :-2] + (1.0 + beta) * v_now + gamma * surface[n, 2:]

        rhs[0] += alpha[0] * left_next
        rhs[-1] += gamma[-1] * right_next

        surface[n + 1, 1:-1] = _thomas_algorithm(lower, diagonal, upper, rhs)
        surface[n + 1, 0] = left_next
        surface[n + 1, -1] = right_next

    return S_grid, surface[-1, :], surface
