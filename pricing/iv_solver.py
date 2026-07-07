"""
Implied volatility solvers.

SciPy's brentq is scalar-only, so this module loops it over arrays of
quotes -- but does the loop once here, centrally, with proper bounds
handling and NaN-on-failure semantics, rather than every caller writing
its own ad hoc version (which is how day-count / model mismatches like the
one roundtrip_test.py guards against tend to creep in).

Design choices:
  * No-arbitrage bounds are checked before solving. A price outside
    [intrinsic, discounted forward] (for a call) can't correspond to any
    sigma >= 0 under the model; brentq would either fail loudly or, worse,
    silently walk to a bracket edge. We return NaN and flag it instead --
    exactly the kind of bad quote the vol surface's arb filters should see.
  * Black-76 and Bachelier are solved with separate bracket logic since
    Bachelier sigma is a price vol (much larger magnitude, no upper
    no-arbitrage bound the way log-normal vol has in practice).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from pricing import black76, bachelier

_CALL, _PUT = True, False


def _bounds_ok_black76(F, K, T, r, price, is_call):
    """No-arb price bounds for Black-76 (undiscounted intrinsic <= price <= disc*F for calls)."""
    disc = np.exp(-r * T)
    intrinsic = disc * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    upper = disc * F if is_call else disc * K
    return (price >= intrinsic - 1e-10) and (price <= upper + 1e-10)


def implied_vol_black76(price, F, K, T, r, is_call, lo=1e-6, hi=8.0):
    """
    Solve for Black-76 sigma given a single scalar quote. Returns NaN (does
    not raise) if the price violates no-arbitrage bounds or brentq fails to
    bracket a root -- callers doing bulk surface work should filter NaNs
    rather than let one bad quote kill a whole calibration pass.
    """
    if T <= 0 or price <= 0:
        return np.nan
    if not _bounds_ok_black76(F, K, T, r, price, is_call):
        return np.nan
    f = lambda s: black76.price(F, K, T, s, r, is_call) - price
    try:
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            return np.nan
        return brentq(f, lo, hi, maxiter=200, xtol=1e-10)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_bachelier(price, F, K, T, r, is_call, lo=1e-8, hi=None):
    """
    Solve for Bachelier (normal, price-space) sigma. hi defaults to a wide
    multiple of |F| + |K| so it comfortably brackets typical solutions even
    when F is near/at/below zero (the 2020 negative-WTI regime), where a
    Black-76-style fixed upper bound like 8.0 would be meaningless.
    """
    if T <= 0 or price <= 0:
        return np.nan
    if hi is None:
        hi = 50.0 * (abs(F) + abs(K) + 1.0)
    disc = np.exp(-r * T)
    intrinsic = disc * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    if price < intrinsic - 1e-10:
        return np.nan
    f = lambda s: bachelier.price(F, K, T, s, r, is_call) - price
    try:
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            return np.nan
        return brentq(f, lo, hi, maxiter=200, xtol=1e-10)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_surface(prices, forwards, strikes, maturities, r, is_call,
                         model="black76"):
    """
    Bulk solve: all inputs are same-length 1D array-likes (one row per
    quote, matching the shape of a QuoteSnapshot frame). r may be scalar or
    array-like. Returns a NumPy array of IVs (NaN where inversion failed).

    model: "black76" or "bachelier".
    """
    prices = np.asarray(prices, dtype=float)
    forwards = np.asarray(forwards, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    r_arr = np.broadcast_to(np.asarray(r, dtype=float), prices.shape)

    n = len(prices)
    out = np.full(n, np.nan)
    solver = implied_vol_black76 if model == "black76" else implied_vol_bachelier

    for i in range(n):
        out[i] = solver(
            prices[i], forwards[i], strikes[i], maturities[i], r_arr[i], bool(is_call[i])
        )
    return out


if __name__ == "__main__":
    # Round-trip sanity check mirroring roundtrip_test.py's spirit, but going
    # through this module's public API rather than the adapter's private
    # scalar formula, to prove the two independently agree.
    from data.adapters.synthetic import SyntheticAdapter
    from data.schema import Commodity, OptionType, year_fraction

    adapter = SyntheticAdapter(inject_noise=False)
    snap = adapter.fetch([Commodity.WTI, Commodity.HENRY_HUB])
    true_iv = snap.meta["true_iv"]
    raw_mid = snap.meta["raw_mid"]
    r = snap.meta["r"]
    f = snap.frame.reset_index(drop=True)

    T = np.array([year_fraction(snap.as_of, e) for e in f["expiry"]])
    is_call = (f["option_type"] == OptionType.CALL).to_numpy()

    iv = implied_vol_surface(raw_mid, f["forward"].to_numpy(), f["strike"].to_numpy(),
                              T, r, is_call, model="black76")

    err = np.abs(iv - true_iv)
    print(f"pricing/iv_solver.py round-trip over {len(f)} quotes:")
    print(f"  max err: {err.max():.2e}  mean err: {err.mean():.2e}  NaNs: {np.isnan(iv).sum()}")
    assert err.max() < 1e-3, "iv_solver disagrees with synthetic generator beyond date granularity!"
    print("PASS: pricing.iv_solver matches data.adapters.synthetic's Black-76 formula.")
