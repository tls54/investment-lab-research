"""
option_pricing_carr_madan.py
----------------------------
Heston model calibration via Carr-Madan FFT pricing.

Sections
--------
1. Black-Scholes pricing & implied volatility
2. Heston characteristic function (vectorised, Carr-Madan formulation)
3. Carr-Madan FFT pricing
4. Vectorised implied volatility
5. IV surface & calibration utilities
"""

import numpy as np
import scipy.stats
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, differential_evolution
import itertools


# ---------------------------------------------------------------------------
# 1. Black-Scholes pricing & implied volatility
# ---------------------------------------------------------------------------

def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_call_price(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * scipy.stats.norm.cdf(d1) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2) - S * scipy.stats.norm.cdf(-d1)


def _vega(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    return S * scipy.stats.norm.pdf(d1) * np.sqrt(T)


def implied_volatility(option_price, S, K, T, r, sigma_init=None, option_type='call'):
    """
    Newton-Raphson implied volatility solver.

    Parameters
    ----------
    option_price : float
    S, K, T, r : float
    sigma_init : float, optional  (default 0.25)
    option_type : {'call', 'put'}

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If vega collapses or solver does not converge in 15 iterations.
    """
    price_func = bs_call_price if option_type.lower() == 'call' else bs_put_price
    sigma = sigma_init if sigma_init is not None else 0.25

    for _ in range(15):
        diff = price_func(S, K, T, r, sigma) - option_price
        if abs(diff) < 1e-8:
            return sigma
        v = _vega(S, K, T, r, sigma)
        if v < 1e-10:
            raise ValueError(f"Vega collapsed to {v}")
        sigma = max(0.001, min(sigma - diff / v, 5.0))

    raise ValueError(f"IV did not converge. Last diff: {diff}")


def safe_implied_volatility(price, S, K, T, r, option_type='call'):
    """Return NaN instead of raising if IV calculation fails."""
    try:
        return implied_volatility(price, S, K, T, r, option_type=option_type)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# 2. Heston characteristic function (vectorised, Carr-Madan formulation)
# ---------------------------------------------------------------------------

def heston_d(omega, kappa, xi, rho):
    """
    Discriminant d in the Heston characteristic function.

    Vectorised over omega. Branch-cut safe: Re(d) >= 0.
    """
    d = np.sqrt(
        (rho * xi * 1j * omega - kappa) ** 2
        + xi ** 2 * (1j * omega + omega ** 2)
    )
    return np.where(np.real(d) < 0, -d, d)


def heston_g(omega, kappa, xi, rho, d=None):
    """Auxiliary ratio g = (kappa - rho*xi*i*omega - d) / (kappa - rho*xi*i*omega + d)."""
    if d is None:
        d = heston_d(omega, kappa, xi, rho)
    num = kappa - rho * xi * 1j * omega - d
    den = kappa - rho * xi * 1j * omega + d
    return num / den


def heston_D(omega, tau, kappa, xi, rho, d=None, g=None):
    """D component of the Heston characteristic exponent."""
    if d is None:
        d = heston_d(omega, kappa, xi, rho)
    if g is None:
        g = heston_g(omega, kappa, xi, rho, d=d)
    return ((kappa - rho * xi * 1j * omega - d) / xi ** 2) * \
           ((1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau)))


def heston_C(omega, tau, kappa, theta, xi, rho, r, d=None, g=None):
    """C component of the Heston characteristic exponent."""
    if d is None:
        d = heston_d(omega, kappa, xi, rho)
    if g is None:
        g = heston_g(omega, kappa, xi, rho, d=d)
    return (r * 1j * omega * tau) + \
           (kappa * theta / xi ** 2) * \
           ((kappa - rho * xi * 1j * omega - d) * tau
            - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))


def heston_char_func(omega, tau, S, r, v0, kappa, theta, xi, rho):
    """
    Heston characteristic function phi(omega).

    Vectorised over omega; d and g are computed once and shared.

    Parameters
    ----------
    omega : array-like
        Frequency argument(s).
    tau : float
        Time to maturity in years.
    S, r, v0, kappa, theta, xi, rho : float
        Spot, risk-free rate, and Heston parameters.

    Returns
    -------
    np.ndarray (complex)
    """
    omega = np.asarray(omega)
    d = heston_d(omega, kappa, xi, rho)
    g = heston_g(omega, kappa, xi, rho, d=d)
    C = heston_C(omega, tau, kappa, theta, xi, rho, r, d=d, g=g)
    D = heston_D(omega, tau, kappa, xi, rho, d=d, g=g)
    return np.exp(C + D * v0 + 1j * omega * np.log(S))


# ---------------------------------------------------------------------------
# 3. Carr-Madan FFT pricing
# ---------------------------------------------------------------------------

def heston_psi(omega, tau, S, r, v0, kappa, theta, xi, rho, alpha):
    """
    Carr-Madan dampened call Fourier transform psi(omega).

    psi(omega) = e^{-rT} * phi(omega - (alpha+1)*i)
                 / (alpha^2 + alpha - omega^2 + i*(2*alpha+1)*omega)

    Parameters
    ----------
    omega : array-like
    tau, S, r, v0, kappa, theta, xi, rho : float
    alpha : float
        Dampening exponent (typically 1.0–1.5).

    Returns
    -------
    np.ndarray (complex)
    """
    omega_shifted = omega - (alpha + 1) * 1j
    phi = heston_char_func(omega_shifted, tau, S, r, v0, kappa, theta, xi, rho)
    denom = alpha ** 2 + alpha - omega ** 2 + 1j * (2 * alpha + 1) * omega
    return np.exp(-r * tau) * phi / denom


def heston_fft_prices(tau, S, r, v0, kappa, theta, xi, rho,
                      alpha=1.5, N=4096, eta=0.25):
    """
    Compute Heston call prices across a log-strike grid via the Carr-Madan FFT method.

    The grid is centred around log(K) = 0 (ATM) with:
        lambda = 2*pi / (N*eta)   [log-strike spacing]
        b      = N*lambda/2       [grid half-width]
        k_u    = -b + lambda*u,   u = 0, ..., N-1

    Parameters
    ----------
    tau : float
        Time to maturity.
    S, r, v0, kappa, theta, xi, rho : float
        Market and Heston parameters.
    alpha : float, optional
        Dampening exponent (default 1.5).
    N : int, optional
        FFT grid size (default 4096).
    eta : float, optional
        Frequency-grid spacing (default 0.25); controls upper truncation
        omega_max = N*eta and log-strike spacing lambda = 2*pi/(N*eta).

    Returns
    -------
    log_strikes : np.ndarray  shape (N,)
    call_prices : np.ndarray  shape (N,)
    """
    j = np.arange(N)
    omega = eta * j + 1e-8          # avoid omega=0

    # Simpson weights
    w = np.ones(N)
    w[1:-1:2] = 4
    w[2:-2:2] = 2
    w /= 3

    b = np.pi / eta
    lambda_val = 2 * np.pi / (N * eta)
    log_strikes = -b + lambda_val * j

    x = (np.exp(1j * j * eta * b)
         * heston_psi(omega, tau, S, r, v0, kappa, theta, xi, rho, alpha)
         * w * eta)

    call_prices = (np.exp(-alpha * log_strikes) / np.pi) * np.real(np.fft.fft(x))
    return log_strikes, call_prices


# ---------------------------------------------------------------------------
# 4. Vectorised implied volatility
# ---------------------------------------------------------------------------

def _bs_price_vec(S, K, T, r, sigma, option_type):
    """Black-Scholes price vectorised over K and sigma."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * scipy.stats.norm.cdf(d1) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)
    return K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2) - S * scipy.stats.norm.cdf(-d1)


def _vega_vec(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * scipy.stats.norm.pdf(d1) * np.sqrt(T)


def vec_implied_volatility(prices, S, K, T, r, option_type='call',
                           sigma_init=0.25, max_iter=15, tol=1e-8):
    """
    Vectorised Newton-Raphson implied volatility solver.

    Parameters
    ----------
    prices : array-like
        Option prices.
    S : float
    K : array-like
        Strikes (same shape as prices).
    T : float
        Single maturity shared across all prices.
    r : float
    option_type : {'call', 'put'}
    sigma_init, max_iter, tol : float, int, float

    Returns
    -------
    np.ndarray
        Implied volatilities; NaN where not converged.
    """
    prices = np.asarray(prices, dtype=float)
    K = np.asarray(K, dtype=float)

    sigma = np.full_like(prices, sigma_init)
    converged = np.zeros(prices.shape, dtype=bool)

    for _ in range(max_iter):
        bs = _bs_price_vec(S, K, T, r, sigma, option_type)
        diff = bs - prices
        v = _vega_vec(S, K, T, r, sigma)

        converged |= np.abs(diff) < tol
        if converged.all():
            break

        can_step = ~converged & (v > 1e-10)
        safe_v = np.where(can_step, v, 1.0)   # avoid div/0 before np.where masks
        sigma = np.where(can_step, np.clip(sigma - diff / safe_v, 0.001, 5.0), sigma)

    return np.where(converged, sigma, np.nan)


# ---------------------------------------------------------------------------
# 5. IV surface & calibration utilities
# ---------------------------------------------------------------------------

def check_feller(kappa, theta, xi):
    """
    Check the Feller condition: 2*kappa*theta > xi^2.

    Returns True if satisfied.
    """
    feller = 2 * kappa * theta
    threshold = xi ** 2
    if feller <= threshold:
        print(f"Feller condition VIOLATED: 2kT={feller:.4f} <= xi^2={threshold:.4f}")
    else:
        print(f"Feller condition satisfied: 2kT={feller:.4f} > xi^2={threshold:.4f}")
    return feller > threshold


def iv_loss(market_ivs, heston_ivs):
    """
    Mean-squared error between market and model IV surfaces.

    Masks NaN entries from either surface before computing.

    Returns
    -------
    float
        MSE, or inf if no valid pairs exist.
    """
    mask = np.isfinite(market_ivs) & np.isfinite(heston_ivs)
    if mask.sum() == 0:
        return np.inf
    return ((market_ivs[mask] - heston_ivs[mask]) ** 2).sum() / mask.sum()


def heston_fft_iv_surface(strikes, maturities, S, r, v0, kappa, theta, xi, rho,
                           opt_type='call', alpha=1.5, N=4096, eta=0.25):
    """
    Compute Heston implied volatilities using Carr-Madan FFT pricing.

    For each unique maturity, a single FFT call prices all strikes simultaneously,
    making this significantly faster than per-option numerical integration.

    Parameters
    ----------
    strikes, maturities : array-like
        Parallel 1-D arrays of (K, T) pairs.
    S, r : float
    v0, kappa, theta, xi, rho : float
        Heston parameters.
    opt_type : {'call', 'put'}
    alpha, N, eta : float, int, float
        Carr-Madan FFT parameters.

    Returns
    -------
    np.ndarray
        Implied volatilities (NaN where computation fails).
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    ivs = np.zeros(len(strikes))

    for maturity in np.unique(maturities):
        mask = maturities == maturity
        strikes_at_tau = strikes[mask]

        log_strikes, call_prices = heston_fft_prices(
            maturity, S, r, v0, kappa, theta, xi, rho, alpha, N, eta
        )
        cs = CubicSpline(log_strikes, call_prices)
        prices = cs(np.log(strikes_at_tau))

        if opt_type == 'put':
            prices = prices - strikes_at_tau + strikes_at_tau * np.exp(-r * maturity)

        ivs[mask] = vec_implied_volatility(prices, S, strikes_at_tau, maturity, r, opt_type)

    return ivs


def objective(heston_params, strikes, maturities, market_ivs, S, r, opt_type='call',
              alpha=1.5, N=4096, eta=0.25):
    """
    Calibration objective: IV MSE between market surface and Heston FFT surface.

    Parameters
    ----------
    heston_params : array-like
        [v0, kappa, theta, xi, rho]
    strikes, maturities : array-like
    market_ivs : np.ndarray
    S, r : float
    opt_type : {'call', 'put'}
    alpha, N, eta : Carr-Madan FFT parameters.

    Returns
    -------
    float
    """
    v0, kappa, theta, xi, rho = heston_params
    heston_ivs = heston_fft_iv_surface(
        strikes, maturities, S, r, v0, kappa, theta, xi, rho,
        opt_type=opt_type, alpha=alpha, N=N, eta=eta,
    )
    return iv_loss(market_ivs, heston_ivs)


def _surface_flatness(strikes, maturities, market_ivs, S):
    """
    Measure IV surface flatness along three axes to determine whether
    a global search is needed before local refinement.

    Returns
    -------
    dict with keys:
        atm_term_struct_std  : std of ATM IVs across maturities (identifies kappa)
        mean_smile_curvature : mean wing IV − ATM IV            (identifies xi)
        mean_skew            : mean abs put/call wing asymmetry  (identifies rho)
        use_global_search    : True if any axis is flat enough to
                               risk L-BFGS-B missing the right basin
    """
    strikes    = np.asarray(strikes,    dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)

    atm_ivs    = []
    curvatures = []
    skews      = []
    wing_dist  = 0.05 * S

    for T in np.unique(maturities):
        mask     = maturities == T
        K_slice  = strikes[mask]
        iv_slice = market_ivs[mask]
        valid    = np.isfinite(iv_slice)

        if valid.sum() < 2:
            continue

        atm_idx = np.argmin(np.abs(K_slice - S))
        atm_iv  = iv_slice[atm_idx]
        atm_ivs.append(atm_iv)

        wing_mask = valid & (np.abs(K_slice - S) > wing_dist)
        if wing_mask.sum() >= 2:
            curvatures.append(np.mean(iv_slice[wing_mask]) - atm_iv)

        put_wing  = valid & (K_slice < S - wing_dist)
        call_wing = valid & (K_slice > S + wing_dist)
        if put_wing.sum() >= 1 and call_wing.sum() >= 1:
            skews.append(np.mean(iv_slice[put_wing]) - np.mean(iv_slice[call_wing]))

    atm_std   = float(np.std(atm_ivs))             if len(atm_ivs)    >= 2 else 0.0
    mean_curv = float(np.mean(curvatures))          if len(curvatures) >= 1 else 0.0
    mean_skew = float(np.mean(np.abs(skews)))       if len(skews)      >= 1 else 0.0

    # Thresholds in raw IV units (calibrated from observed surface diagnostics):
    #   calm:  atm_std=0.00016, mean_curv=0.00326, mean_skew=0.03458
    #   event: atm_std=0.01449, mean_curv=0.00023, mean_skew=0.08020
    TERM_THRESH = 0.001   # flags calm (0.00016) not event (0.01449)
    CURV_THRESH = 0.001   # flags event (0.00023) not calm (0.00326)
    SKEW_THRESH = 0.001   # both well above — neither flagged

    use_global = (atm_std   < TERM_THRESH or
                  mean_curv < CURV_THRESH or
                  mean_skew < SKEW_THRESH)

    return {
        'atm_term_struct_std':  atm_std,
        'mean_smile_curvature': mean_curv,
        'mean_skew':            mean_skew,
        'use_global_search':    use_global,
    }


def _global_search_x0(objective, obj_args, all_bounds,
                       popsize=8, maxiter=100, tol=1e-3):
    """
    Run a full 5D differential evolution search to find a good starting
    basin for L-BFGS-B. Used when the IV surface is too flat for the
    local solver to reliably identify the correct parameter region.

    Returns
    -------
    best_x : np.ndarray [v0, kappa, theta, xi, rho]
    """
    de_result = differential_evolution(
        objective,
        bounds=all_bounds,
        args=obj_args,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        seed=0,
        polish=False,
        workers=-1, 

    )

    print(f"  [DE global search — loss={de_result.fun:.6f}, "
          f"nfev={de_result.nfev}]")
    print(f"    v0={de_result.x[0]:.4f}  kappa={de_result.x[1]:.4f}  "
          f"theta={de_result.x[2]:.4f}  xi={de_result.x[3]:.4f}  "
          f"rho={de_result.x[4]:.4f}")

    return de_result.x


def calibrate_heston(strikes, maturities, market_ivs, S, r,
                     x0=None, opt_type='call', alpha=1.5, N=4096, eta=0.25):
    """
    Calibrate Heston parameters to a market IV surface using Carr-Madan FFT pricing.

    Flat surfaces (where L-BFGS-B is likely to miss the correct basin) are
    detected automatically and trigger a full 5D differential evolution global
    search before local refinement. Rich surfaces go straight to L-BFGS-B.

    Parameters
    ----------
    strikes, maturities : array-like
    market_ivs          : np.ndarray
    S, r                : float
    x0 : array-like, optional
        Initial guess [v0, kappa, theta, xi, rho].
        Defaults to [0.04, 2.0, 0.04, 0.3, -0.7] for rich surfaces.
        Ignored when a global search is triggered.
    opt_type            : {'call', 'put'}
    alpha, N, eta       : Carr-Madan FFT parameters.

    Returns
    -------
    scipy.optimize.OptimizeResult
        result.x   = [v0, kappa, theta, xi, rho]
        result.fun = MSE loss
    """
    all_bounds = [(1e-4, 1.0), (0.1, 15.0), (1e-4, 1.0), (1e-4, 2.0), (-0.99, 0.99)]
    obj_args   = (strikes, maturities, market_ivs, S, r, opt_type, alpha, N, eta)

    flatness = _surface_flatness(strikes, maturities, market_ivs, S)

    if flatness['use_global_search']:
        print(f"  [flat surface detected — "
              f"ATM term-struct std={flatness['atm_term_struct_std']*100:.3f} vpts, "
              f"smile curv={flatness['mean_smile_curvature']*100:.3f} vpts, "
              f"skew={flatness['mean_skew']*100:.3f} vpts]")
        x0 = _global_search_x0(objective, obj_args, all_bounds)
    else:
        if x0 is None:
            x0 = np.array([0.04, 2.0, 0.04, 0.3, -0.7])
        x0 = np.asarray(x0, dtype=float)

    return minimize(
        fun=objective,
        x0=x0,
        args=obj_args,
        method='L-BFGS-B',
        bounds=all_bounds,
    )


def residual_surface(strikes, maturities, market_ivs, S, r,
                     v0, kappa, theta, xi, rho, opt_type='call'):
    """
    Compute per-strike/maturity IV residuals (market - model) as a 2-D grid.

    Parameters
    ----------
    strikes, maturities : array-like
    market_ivs : np.ndarray
    S, r, v0, kappa, theta, xi, rho : float
    opt_type : {'call', 'put'}

    Returns
    -------
    unique_strikes : np.ndarray  shape (n_K,)
    unique_maturities : np.ndarray  shape (n_T,)
    grid : np.ndarray  shape (n_T, n_K), NaN where missing
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    heston_ivs = heston_fft_iv_surface(
        strikes, maturities, S, r, v0, kappa, theta, xi, rho, opt_type=opt_type
    )
    residuals = market_ivs - heston_ivs

    unique_strikes = np.sort(np.unique(strikes))
    unique_maturities = np.sort(np.unique(maturities))

    grid = np.full((len(unique_maturities), len(unique_strikes)), np.nan)
    for i, T in enumerate(unique_maturities):
        for j, K in enumerate(unique_strikes):
            mask = (maturities == T) & (strikes == K)
            if mask.any():
                grid[i, j] = residuals[mask][0]

    return unique_strikes, unique_maturities, grid
