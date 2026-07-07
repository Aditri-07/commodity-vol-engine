# Commodity Options Pricing & Volatility Surface Analytics Engine

A Python engine for pricing commodity **futures options** (WTI crude, Henry Hub
natural gas) with Black-76 and Bachelier models, and building/validating implied
volatility surfaces.

## Data access & honesty note
This project is designed around a **source-agnostic quote schema**. Live vendor
data (Bloomberg BQuant, CME public settlements, yfinance, EIA) plugs into the
same schema, but the **primary runnable path is a synthetic fallback generator**
so the whole pipeline runs end-to-end with zero external access. Synthetic
prices are generated *from* a known volatility surface via Black-76, so they are
arbitrage-consistent by construction — inverting them recovers the input smile
(see `roundtrip_test.py`). This is documented rather than hidden.

## What's built so far

### Step 1: data layer
- `data/schema.py` — canonical quote schema + shared ACT/365 `year_fraction`
  (single day-count convention for the whole project).
- `data/adapters/base.py` — `QuoteAdapter` protocol; downstream code depends
  only on this interface, never on a specific vendor.
- `data/adapters/synthetic.py` — arb-consistent synthetic chain generator in the
  futures-option (Black-76) shape: WTI backwardation + mild put skew, Henry Hub
  contango + call skew, maturity-dependent smile, widening wings, optional noise
  injection to exercise downstream validation/arb filters.
- `roundtrip_test.py` — inverts generated prices back to IV; proves the data is
  arbitrage-consistent.

**Note:** `coerce_to_schema` explicitly casts the `commodity`/`option_type`
columns to `object` dtype. Pandas' modern string-dtype inference otherwise
silently stores these `str`-subclassing Enum columns via `str(x)`
(`"OptionType.CALL"` instead of the enum's `.value` `"C"`), which makes any
vectorized `df[col] == Enum.X` comparison always evaluate `False` while a
scalar `.iloc[i]` lookup looks fine — a nasty one to debug downstream.

### Step 2: pricing core
- `pricing/black76.py` — vectorized Black-76 price + Greeks (delta, gamma,
  vega, theta, rho). Standard model for exchange-listed futures options.
- `pricing/bachelier.py` — vectorized Bachelier (normal) price + Greeks.
  Prices straight through zero/negative forwards (April 2020 WTI at
  -$37.63), where Black-76 is undefined.
- `pricing/iv_solver.py` — brentq-based inversion for both models, with
  no-arbitrage price-bound checks before solving (returns NaN rather than
  a garbage root on a bad quote) and a vectorized bulk-surface entry point.

### Step 3: volatility surface (current focus)
- `vol_surface/smile_models/` — four interchangeable per-maturity smile
  parametrizations behind a common `SmileModel` interface:
  - `quadratic.py` — sigma(k) = a + bk + ck²; cheap sanity-check baseline,
    matches the synthetic generator's own functional form.
  - `sabr.py` — Hagan et al.'s lognormal SABR (industry-standard for
    futures/rates/commodities). `beta` is a per-commodity choice (0.5 WTI,
    0.7 Henry Hub) rather than fit, per standard market practice; beta=1 is
    lognormal-like, beta=0 is normal-like, bridging naturally to Bachelier
    for stressed/near-zero forwards.
  - `svi.py` — raw SVI (Gatheral), fit in total-variance space. Comes with
    a closed-form (Gatheral-Jacquier) necessary condition for single-slice
    butterfly-arbitrage-freedom, checked directly on fitted params.
  - `spline.py` — monotone PCHIP interpolation; deliberately model-free,
    with no arbitrage-free guarantee, used as a contrast case.
- `vol_surface/surface.py` — `VolSurface`: quotes → per-maturity IV
  inversion → per-slice smile fit → cross-maturity interpolation in total
  variance (flat-extrapolated past the first/last quoted maturity).
- `vol_surface/arb_filters.py` — flag-only (no auto-repair) detection:
  numerical Breeden-Litzenberger butterfly check on any model, closed-form
  SVI cross-check, and a calendar (non-decreasing total variance) check
  across maturities. Verified to fire on Henry Hub under injected quote
  noise (least-squares models like SABR/SVI absorb it; the spline, forced
  through every point, does not).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
Always run from the project root so package imports resolve.

```bash
# Data layer
python -m data.adapters.synthetic     # snapshot + summary (12 maturities x 9 strikes = 108 quotes)
python roundtrip_test.py              # arbitrage-consistency of the synthetic generator

# Pricing core
python -m pricing.black76             # put-call parity + Greeks sanity check
python -m pricing.bachelier           # parity check + negative-forward case
python -m pricing.iv_solver           # bulk IV round-trip over the full synthetic snapshot

# Vol surface
python -m vol_surface.smile_models.sabr   # SABR self-consistency fit
python -m vol_surface.smile_models.svi    # SVI self-consistency fit + closed-form arb check
python -m vol_surface.surface             # calibrate all 4 models, compare ATM IV
python -m vol_surface.arb_filters         # arb-filter comparison across models/commodities
```

Expected round-trip output:
```
Round-trip over 108 quotes (WTI + Henry Hub):
  raw price  -> IV : max ~5e-05, mean ~7e-06
  quoted mid -> IV : max ~6e-04, mean ~2e-05
PASS: synthetic prices invert cleanly; data is arbitrage-consistent by construction.
```

## Roadmap
- Forward curve construction (no-arb interpolation across the futures strip)
- Greeks aggregation for multi-leg structures on top of the calibrated surface
- Scenario/stress engine (2020 oil collapse via Bachelier, gas vol spikes,
  non-parallel curve shocks)
- Streamlit dashboard
