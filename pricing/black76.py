"""
Black-76 model for European options on futures/forwards.

This is the standard model for exchange-listed commodity futures options
(WTI/CL, Henry Hub/NG on CME): the option is on the futures price F, not a
spot with a cost-of-carry, so there's no dividend/yield term -- discounting
is the only place r enters.

Everything is vectorized over NumPy arrays so a full vol surface (hundreds
of quotes) prices/greeks in one call rather than a Python loop per point.
All time-to-maturity inputs are expected to already be ACT/365 year
fractions from data.schema.year_fraction, to stay consistent with how the
synthetic adapter generates prices.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EPS = 1e-12


def _d1_d2(F, K, T, sigma):
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    T_safe = np.maximum(T, _EPS)
    sigma_safe = np.maximum(sigma, _EPS)
    sqrtT = np.sqrt(T_safe)

    d1 = (np.log(F / K) + 0.5 * sigma_safe ** 2 * T_safe) / (sigma_safe * sqrtT)
    d2 = d1 - sigma_safe * sqrtT
    return d1, d2


def price(F, K, T, sigma, r, is_call):
    """
    Black-76 price. F, K, T, sigma, r broadcast against each other;
    is_call is a bool or bool array.

    At T<=0 or sigma<=0, collapses to discounted intrinsic value.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r = np.asarray(r, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    d1, d2 = _d1_d2(F, K, T, sigma)
    disc = np.exp(-r * T)

    call_val = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put_val = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    val = np.where(is_call, call_val, put_val)

    # Degenerate cases: no time value or no vol -> discounted intrinsic.
    intrinsic_call = np.maximum(F - K, 0.0)
    intrinsic_put = np.maximum(K - F, 0.0)
    intrinsic = np.where(is_call, intrinsic_call, intrinsic_put)
    degenerate = (T <= 0) | (sigma <= 0)
    val = np.where(degenerate, disc * intrinsic, val)

    return val


def greeks(F, K, T, sigma, r, is_call):
    """
    Analytical Greeks, all discounted (r enters as a pure discount factor,
    since Black-76 has no separate cost-of-carry on F).

    Returns a dict of arrays (same broadcast shape as inputs):
      delta  d(price)/dF          -- per unit of futures price
      gamma  d2(price)/dF2
      vega   d(price)/d(sigma)    -- for a 1.0 (100 vol pt) move; divide by
                                     100 for a "per vol point" quote convention
      theta  d(price)/dt          -- per year; divide by 365 for per-day
      rho    d(price)/dr          -- sensitivity to the discount rate
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r = np.asarray(r, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    d1, d2 = _d1_d2(F, K, T, sigma)
    T_safe = np.maximum(T, _EPS)
    sigma_safe = np.maximum(sigma, _EPS)
    disc = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    delta_call = disc * norm.cdf(d1)
    delta_put = disc * (norm.cdf(d1) - 1.0)
    delta = np.where(is_call, delta_call, delta_put)

    gamma = disc * pdf_d1 / (F * sigma_safe * np.sqrt(T_safe))

    vega = disc * F * pdf_d1 * np.sqrt(T_safe)

    # theta: d(price)/dt = -d(price)/dT, decomposed into vol-decay + discount-drift terms.
    term1 = -disc * F * pdf_d1 * sigma_safe / (2.0 * np.sqrt(T_safe))
    theta_call = term1 - r * disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    theta_put = term1 - r * disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    theta = np.where(is_call, theta_call, theta_put)

    rho_call = -T * disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    rho_put = -T * disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    rho = np.where(is_call, rho_call, rho_put)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


if __name__ == "__main__":
    # Quick sanity check against the synthetic adapter's own scalar formula.
    F, K, T, sigma, r = 74.0, 75.0, 0.5, 0.30, 0.045
    c = price(F, K, T, sigma, r, True)
    p = price(F, K, T, sigma, r, False)
    print(f"Call: {float(c):.6f}  Put: {float(p):.6f}")
    # Put-call parity for Black-76: C - P = disc * (F - K)
    parity_lhs = float(c) - float(p)
    parity_rhs = np.exp(-r * T) * (F - K)
    print(f"Put-call parity check: {parity_lhs:.8f} vs {float(parity_rhs):.8f}")
    g = greeks(F, K, T, sigma, r, True)
    print({k: float(v) for k, v in g.items()})
