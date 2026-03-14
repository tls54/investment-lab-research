"""
option_pricing_quad.py
-----------------
Core pricing, Greeks, implied-volatility, and Heston calibration functions
extracted from Black-Scholes.ipynb for acceleration and productionisation.

Sections
--------
1. Black-Scholes pricing
2. Greeks
3. Implied volatility
4. Heston model (characteristic-function / Fourier inversion)
5. Calibration utilities
6. Synthetic surface & data utilities
"""

import numpy as np
import pandas as pd
import scipy.stats
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# 1. Black-Scholes pricing
# ---------------------------------------------------------------------------

def d1_func(S, K, T, r, sigma):
    return (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))


def d2_func(d1, T, sigma):
    return d1 - (sigma * np.sqrt(T))


def bs_call_price(S, K, T, r, sigma):
    d1 = d1_func(S, K, T, r, sigma)
    d2 = d2_func(d1, T, sigma)
    return S * scipy.stats.norm.cdf(d1) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    d1 = d1_func(S, K, T, r, sigma)
    d2 = d2_func(d1, T, sigma)
    return K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2) - S * scipy.stats.norm.cdf(-d1)


# ---------------------------------------------------------------------------
# 2. Greeks
# ---------------------------------------------------------------------------

def delta_call(d1):
    return scipy.stats.norm.cdf(d1)


def delta_put(d1):
    return scipy.stats.norm.cdf(d1) - 1


def gamma(d1, S, sigma, T):
    return scipy.stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    d1 = d1_func(S, K, T, r, sigma)
    return S * scipy.stats.norm.pdf(d1) * np.sqrt(T)


# ---------------------------------------------------------------------------
# 3. Implied volatility
# ---------------------------------------------------------------------------

def implied_volatility(option_price, S, K, T, r, sigma_init=None, option_type='call'):
    """
    Newton-Raphson implied volatility solver.

    Parameters
    ----------
    option_price : float
        Observed market price of the option.
    S, K, T, r : float
        Spot, strike, time-to-expiry (years), risk-free rate.
    sigma_init : float, optional
        Starting guess; defaults to 0.25.
    option_type : {'call', 'put'}

    Returns
    -------
    float
        Implied volatility (annualised).

    Raises
    ------
    ValueError
        If vega collapses or the solver does not converge within 15 iterations.
    """
    tol = 1e-8
    if option_type.lower() == 'call':
        price_func = bs_call_price
    elif option_type.lower() == 'put':
        price_func = bs_put_price
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")

    if sigma_init is None:
        sigma_init = 0.25

    num_iters = 15
    sigma = sigma_init
    for _ in range(num_iters):
        BS_price = price_func(S, K, T, r, sigma)
        diff = BS_price - option_price

        if abs(diff) < tol:
            return sigma

        vega_val = vega(S, K, T, r, sigma)
        if vega_val < 1e-10:
            raise ValueError(f"Vega collapsed to {vega_val} - likely deep ITM/OTM or near expiry")

        sigma -= diff / vega_val
        sigma = max(0.001, min(sigma, 5.0))

    raise ValueError(f"IV did not converge after {num_iters} iterations. Last diff: {diff}")


def safe_implied_volatility(price, S, K, T, r, option_type):
    """Return NaN instead of raising if IV calculation fails."""
    try:
        return implied_volatility(price, S, K, T, r, option_type=option_type)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# 4. Heston model (characteristic-function / Fourier inversion)
# ---------------------------------------------------------------------------

def heston_b(kappa, rho, xi, j_mode):
    """Auxiliary parameter b_j (numeraire-dependent)."""
    if j_mode == 1:
        return kappa - rho * xi
    elif j_mode == 2:
        return kappa
    else:
        raise ValueError(f'j_mode value must be 1 or 2, not {j_mode}')


def heston_u(j_mode):
    """Return u_j parameter for numeraire j."""
    if j_mode == 1:
        return 0.5
    elif j_mode == 2:
        return -0.5
    else:
        raise ValueError(f'j_mode value must be 1 or 2, not {j_mode}')


def heston_d(omega, kappa, xi, rho, j_mode):
    """
    Compute discriminant d_j with Lord-Kahl branch selection (Re(d) > 0).

    d_j = sqrt[(rho*xi*i*omega - b_j)^2 + xi^2*(omega^2 - 2*u_j*i*omega)]
    """
    b_j = heston_b(kappa, rho, xi, j_mode)
    u_j = heston_u(j_mode)

    d = np.sqrt(
        (rho * xi * 1j * omega - b_j) ** 2
        + xi ** 2 * (omega ** 2 - 2 * u_j * 1j * omega)
    )

    if np.real(d) < 0:
        d = -d

    return d


def heston_g(omega, kappa, xi, rho, j_mode):
    """Auxiliary parameter g_j in the Heston characteristic function."""
    b_j = heston_b(kappa, rho, xi, j_mode)
    d_j = heston_d(omega, kappa, xi, rho, j_mode)
    return (b_j - rho * xi * 1j * omega - d_j) / (b_j - rho * xi * 1j * omega + d_j)


def heston_D(tau, omega, kappa, xi, rho, j_mode):
    """D function in the Heston characteristic exponent."""
    b_j = heston_b(kappa, rho, xi, j_mode)
    d_j = heston_d(omega, kappa, xi, rho, j_mode)
    g_j = heston_g(omega, kappa, xi, rho, j_mode)

    return ((b_j - rho * xi * 1j * omega - d_j) / (xi ** 2)) * \
           (1 - np.exp(-d_j * tau)) / (1 - g_j * np.exp(-d_j * tau))


def heston_C(tau, omega, kappa, theta, xi, rho, r, j_mode):
    """C function in the Heston characteristic exponent."""
    b_j = heston_b(kappa, rho, xi, j_mode)
    d_j = heston_d(omega, kappa, xi, rho, j_mode)
    g_j = heston_g(omega, kappa, xi, rho, j_mode)

    return r * tau * 1j * omega + \
           (kappa * theta / (xi ** 2)) * \
           ((b_j - rho * xi * 1j * omega - d_j) * tau
            - 2 * np.log((1 - g_j * np.exp(-d_j * tau)) / (1 - g_j)))


def heston_integrand(omega, K, S, tau, r, v0, kappa, theta, xi, rho, j_mode):
    """Re[exp(-i*omega*log(K)) * phi_j / (i*omega)] for Fourier inversion."""
    C_j = heston_C(tau, omega, kappa, theta, xi, rho, r, j_mode)
    D_j = heston_D(tau, omega, kappa, xi, rho, j_mode)
    phi_j = np.exp(C_j + D_j * v0 + (1j * omega * np.log(S)))

    return np.real((np.exp(-1j * omega * np.log(K)) * phi_j) / (1j * omega))


def heston_probability(K, S, T, r, v0, kappa, theta, xi, rho, j_mode, omega_max=100):
    """
    Compute risk-neutral probability P_j via numerical Fourier inversion.

    Returns P_j = 0.5 + (1/pi) * integral_{0}^{omega_max} Re[...] d_omega
    """
    result, error = quad(
        heston_integrand,
        1e-10,
        omega_max,
        args=(K, S, T, r, v0, kappa, theta, xi, rho, j_mode)
    )
    if error > 1e-6:
        print(f"(Warning) Integration error: {error}")
    return 0.5 + result / np.pi


def heston_call_price(S, K, T, r, v0, kappa, theta, xi, rho):
    """Heston call price: S*P1 - K*exp(-rT)*P2."""
    P_1 = heston_probability(K, S, T, r, v0, kappa, theta, xi, rho, 1)
    P_2 = heston_probability(K, S, T, r, v0, kappa, theta, xi, rho, 2)
    return S * P_1 - K * np.exp(-r * T) * P_2


def heston_put_price(S, K, T, r, v0, kappa, theta, xi, rho):
    """Heston put price via put-call parity: P = C - S + K*exp(-rT)."""
    call = heston_call_price(S, K, T, r, v0, kappa, theta, xi, rho)
    return call - S + K * np.exp(-r * T)


# ---------------------------------------------------------------------------
# 5. Calibration utilities
# ---------------------------------------------------------------------------

def check_feller(kappa, theta, xi):
    """
    Check the Feller condition: 2*kappa*theta > xi^2.

    If violated the variance process can reach zero, causing numerical issues.

    Returns
    -------
    bool
        True if condition is satisfied.
    """
    feller = 2 * kappa * theta
    threshold = xi ** 2

    if feller <= threshold:
        print(f"⚠️  Feller condition VIOLATED: 2κθ={feller:.4f} ≤ ξ²={threshold:.4f}")
        print("    Variance can reach zero - may cause numerical issues")
    else:
        print(f"✓ Feller condition satisfied: 2κθ={feller:.4f} > ξ²={threshold:.4f}")

    return feller > threshold


def heston_iv_surface(strikes, maturities, S, r, v0, kappa, theta, xi, rho, opt_type='call'):
    """
    Compute Heston implied volatilities for parallel arrays of strikes and maturities.

    Parameters
    ----------
    strikes, maturities : array-like
        Parallel 1-D arrays of (K, T) pairs.
    S, r : float
        Spot and risk-free rate.
    v0, kappa, theta, xi, rho : float
        Heston parameters.
    opt_type : {'call', 'put'}

    Returns
    -------
    np.ndarray
        Implied volatilities (NaN where computation fails).
    """
    pricers = {'call': heston_call_price, 'put': heston_put_price}
    pricer = pricers[opt_type]

    IVs = []
    for strike, maturity in zip(strikes, maturities):
        price = pricer(S=S, K=strike, T=maturity, r=r,
                       v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        iv = safe_implied_volatility(price, S=S, K=strike, T=maturity, r=r, option_type=opt_type)
        IVs.append(iv)

    return np.array(IVs)


def iv_loss(market_ivs, heston_ivs):
    """
    Mean-squared error between market and model IV surfaces.

    Masks out any NaN entries from either surface before computing the loss.

    Returns
    -------
    float
        MSE loss, or inf if no valid pairs exist.
    """
    mask = np.isfinite(market_ivs) & np.isfinite(heston_ivs)

    if mask.sum() == 0:
        return np.inf

    return ((market_ivs[mask] - heston_ivs[mask]) ** 2).sum() / mask.sum()


def objective(heston_params, strikes, maturities, market_ivs, S, r, opt_type):
    """
    Objective function for Heston parameter optimisation.

    Parameters
    ----------
    heston_params : array-like
        [v0, kappa, theta, xi, rho]
    strikes, maturities : array-like
        Parallel 1-D arrays matching market_ivs.
    market_ivs : np.ndarray
        Target implied volatilities from the market.
    S, r : float
        Spot and risk-free rate.
    opt_type : {'call', 'put'}

    Returns
    -------
    float
        IV MSE loss.
    """
    v0, kappa, theta, xi, rho = heston_params
    heston_ivs = heston_iv_surface(strikes, maturities, S, r, v0, kappa, theta, xi, rho, opt_type=opt_type)
    return iv_loss(market_ivs, heston_ivs)


# ---------------------------------------------------------------------------
# 6. Synthetic surface & data utilities
# ---------------------------------------------------------------------------

def sigma_true_full(K, T, S=100, skew_slope=-0.15, curvature=0.05):
    """
    Synthetic implied-volatility surface with skew, smile, and term structure.

    sigma(K, T) = sigma_atm(T) + skew_slope*(K/S - 1) + curvature*(K/S - 1)^2
    sigma_atm(T) = 0.15 + 0.05*sqrt(T)

    Useful for generating ground-truth surfaces in tests and calibration demos.
    """
    sigma_atm = 0.15 + 0.05 * np.sqrt(T)
    moneyness = K / S
    offset = moneyness - 1
    return sigma_atm + skew_slope * offset + curvature * offset ** 2


def generate_option_chain(S, r, strikes, maturities, option_types, sigma_func):
    """
    Generate a synthetic option chain priced from a given IV surface function.

    Parameters
    ----------
    S, r : float
        Spot and risk-free rate.
    strikes, maturities : array-like
    option_types : list of {'call', 'put'}
    sigma_func : callable
        Signature: sigma_func(K, T, S) -> implied_volatility.

    Returns
    -------
    pd.DataFrame
        Columns: maturity, strike, moneyness, option_type, true_iv,
                 bs_price, intrinsic_value.
    """
    rows = []
    for T in maturities:
        for K in strikes:
            for opt_type in option_types:
                true_iv = sigma_func(K, T, S)
                if opt_type == 'call':
                    price = bs_call_price(S, K, T, r, true_iv)
                    intrinsic = max(S - K, 0)
                else:
                    price = bs_put_price(S, K, T, r, true_iv)
                    intrinsic = max(K - S, 0)

                rows.append({
                    'maturity': T,
                    'strike': K,
                    'moneyness': K / S,
                    'option_type': opt_type,
                    'true_iv': true_iv,
                    'bs_price': price,
                    'intrinsic_value': intrinsic,
                })

    return pd.DataFrame(rows)


def compute_heston_iv_surface_df(df, S, r, params):
    """
    Apply Heston pricing to every row of an option-chain DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: strike, maturity, option_type.
    S, r : float
        Spot and risk-free rate.
    params : dict
        Keys: v0, kappa, theta, xi, rho.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns: heston_price, heston_iv.
    """
    results = df.copy()
    heston_prices = []

    for _, row in df.iterrows():
        try:
            if row['option_type'] == 'call':
                price = heston_call_price(
                    S=S, K=row['strike'], T=row['maturity'], r=r,
                    v0=params['v0'], kappa=params['kappa'],
                    theta=params['theta'], xi=params['xi'], rho=params['rho'],
                )
            else:
                price = heston_put_price(
                    S=S, K=row['strike'], T=row['maturity'], r=r,
                    v0=params['v0'], kappa=params['kappa'],
                    theta=params['theta'], xi=params['xi'], rho=params['rho'],
                )
        except Exception as e:
            print(f"Failed at K={row['strike']}, T={row['maturity']}, type={row['option_type']}: {e}")
            price = np.nan

        heston_prices.append(price)

    results['heston_price'] = heston_prices
    results['heston_iv'] = results.apply(
        lambda row: safe_implied_volatility(
            row['heston_price'], S, row['strike'], row['maturity'], r,
            option_type=row['option_type'],
        ),
        axis=1,
    )
    return results


def make_surface_grids(df, value_col):
    """
    Pivot an option-chain DataFrame into a 2-D grid suitable for surface plots.

    Uses calls only to produce a clean (maturity x strike) grid.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: option_type, strike, maturity, <value_col>.
    value_col : str
        Column name to populate the grid with (e.g. 'true_iv', 'heston_iv').

    Returns
    -------
    strikes : np.ndarray  shape (n_strikes,)
    maturities : np.ndarray  shape (n_maturities,)
    grid : np.ndarray  shape (n_maturities, n_strikes), NaN where missing
    """
    calls = df[df['option_type'] == 'call'].copy()
    strikes = np.sort(calls['strike'].unique())
    maturities = np.sort(calls['maturity'].unique())

    grid = np.full((len(maturities), len(strikes)), np.nan)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            mask = (calls['maturity'] == T) & (calls['strike'] == K)
            if mask.any():
                grid[i, j] = calls.loc[mask, value_col].values[0]

    return strikes, maturities, grid
