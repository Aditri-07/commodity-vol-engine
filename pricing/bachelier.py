"""
Bachelier (normal) model for European options on futures/forwards.

F is assumed normally distributed rather than log-normal, so unlike Black-76,
sigma here is a PRICE volatility (e.g. $/yr for WTI, not a percentage), and
F/K can go negative without breaking the model. This is the reason the model
exists in this project: April 20, 2020 WTI settled at -$37.63 -- Black-76 is
mathematically undefined there (log of a negative number), but Bachelier
prices straight through it. Use this model, not Black-76, for any scenario
where the forward can approach or cross zero.

Same vectorization/day-count conventions as pricing/black76.py.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EPS = 1e-12


def _d(F, K, T, sigma):
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    T_safe = np.maximum(T, _EPS)
    sigma_safe = np.maximum(sigma, _EPS)
    return (F - K) / (sigma_safe * np.sqrt(T_safe))


def price(F, K, T, sigma, r, is_call):
    """
    Bachelier price. sigma is a PRICE (normal) vol, not a percentage vol --
    do not feed it a Black-76-calibrated sigma without converting first.

    At T<=0 or sigma<=0, collapses to discounted intrinsic value.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r = np.asarray(r, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    T_safe = np.maximum(T, _EPS)
    sigma_safe = np.maximum(sigma, _EPS)
    d = _d(F, K, T, sigma)
    disc = np.exp(-r * T)
    stdev = sigma_safe * np.sqrt(T_safe)

    call_val = disc * ((F - K) * norm.cdf(d) + stdev * norm.pdf(d))
    put_val = disc * ((K - F) * norm.cdf(-d) + stdev * norm.pdf(d))
    val = np.where(is_call, call_val, put_val)

    intrinsic_call = np.maximum(F - K, 0.0)
    intrinsic_put = np.maximum(K - F, 0.0)
    intrinsic = np.where(is_call, intrinsic_call, intrinsic_put)
    degenerate = (T <= 0) | (sigma <= 0)
    val = np.where(degenerate, disc * intrinsic, val)

    return val


def greeks(F, K, T, sigma, r, is_call):
    """
    Analytical Greeks for Bachelier. Note vega/gamma have a different shape
    to Black-76's since sigma is a price vol, not a percentage vol -- don't
    compare magnitudes directly across the two models without converting.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r = np.asarray(r, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    T_safe = np.maximum(T, _EPS)
    sigma_safe = np.maximum(sigma, _EPS)
    d = _d(F, K, T, sigma)
    disc = np.exp(-r * T)
    sqrtT = np.sqrt(T_safe)
    pdf_d = norm.pdf(d)

    delta_call = disc * norm.cdf(d)
    delta_put = disc * (norm.cdf(d) - 1.0)
    delta = np.where(is_call, delta_call, delta_put)

    gamma = disc * pdf_d / (sigma_safe * sqrtT)

    vega = disc * sqrtT * pdf_d

    theta_call = -disc * sigma_safe * pdf_d / (2.0 * sqrtT) - r * disc * (
        (F - K) * norm.cdf(d) + sigma_safe * sqrtT * pdf_d
    )
    theta_put = -disc * sigma_safe * pdf_d / (2.0 * sqrtT) - r * disc * (
        (K - F) * norm.cdf(-d) + sigma_safe * sqrtT * pdf_d
    )
    theta = np.where(is_call, theta_call, theta_put)

    rho_call = -T * disc * ((F - K) * norm.cdf(d) + sigma_safe * sqrtT * pdf_d)
    rho_put = -T * disc * ((K - F) * norm.cdf(-d) + sigma_safe * sqrtT * pdf_d)
    rho = np.where(is_call, rho_call, rho_put)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def normal_vol_from_lognormal(F, sigma_ln):
    """
    Quick ATM-style conversion sigma_normal ~= F * sigma_lognormal, useful as
    an initial guess when bootstrapping a Bachelier IV solve from a Black-76
    surface. This is a first-order approximation (exact only in the limit of
    small vol / short maturity), not an identity -- always re-solve for the
    true Bachelier IV afterward rather than relying on this conversion alone.
    """
    return np.asarray(F, dtype=float) * np.asarray(sigma_ln, dtype=float)


if __name__ == "__main__":
    # Sanity check including a negative-forward case (the WTI April 2020 regime).
    F, K, T, sigma, r = 74.0, 75.0, 0.5, 22.0, 0.045  # sigma in $/sqrt(yr)
    c = price(F, K, T, sigma, r, True)
    p = price(F, K, T, sigma, r, False)
    print(f"Normal case  -- Call: {float(c):.6f}  Put: {float(p):.6f}")
    parity_lhs = float(c) - float(p)
    parity_rhs = np.exp(-r * T) * (F - K)
    print(f"Put-call parity check: {parity_lhs:.8f} vs {float(parity_rhs):.8f}")

    F_neg = -37.63
    c_neg = price(F_neg, 40.0, 0.02, 15.0, r, True)
    p_neg = price(F_neg, 40.0, 0.02, 15.0, r, False)
    print(f"Negative-forward case (F={F_neg}) -- Call: {float(c_neg):.6f}  Put: {float(p_neg):.6f}")
