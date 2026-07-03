"""
Source-agnostic quote schema for commodity options.

Every data adapter (yfinance, EIA, CME public settlements, synthetic fallback)
must return data conforming to this schema. This keeps the pricing/surface code
decoupled from where the data came from, and lets you swap in Bloomberg BQuant
later without touching downstream modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

import pandas as pd


# ACT/365 fixed. This is THE day-count convention for the whole project; every
# module (synthetic generator, pricing, surface) must derive time-to-maturity
# through this function so generated prices and re-derived IVs stay consistent.
ACT_365 = 365.0


def year_fraction(start: date, end: date) -> float:
    """Time in years between two dates under ACT/365 fixed."""
    return (end - start).days / ACT_365


class OptionType(str, Enum):
    CALL = "C"
    PUT = "P"


class Commodity(str, Enum):
    WTI = "WTI"          # CME symbol CL, underlying futures CL=F
    HENRY_HUB = "HH"     # CME symbol NG, underlying futures NG=F


# Canonical column set every adapter must produce.
# One row = one option quote on one strike/maturity for one commodity.
QUOTE_COLUMNS = [
    "commodity",         # Commodity
    "as_of",             # date the snapshot was taken
    "expiry",            # option expiry date
    "future_expiry",     # expiry of the underlying futures contract
    "forward",           # F: futures/forward price of the underlying
    "strike",            # K
    "option_type",       # OptionType
    "bid",               # option bid (price, same units as forward)
    "ask",               # option ask
    "mid",               # (bid+ask)/2, filled by adapter or normalizer
    "volume",            # contracts traded (0/NaN allowed)
    "open_interest",     # OI (0/NaN allowed)
    "source",            # provenance string, e.g. "synthetic", "yfinance"
]


@dataclass
class QuoteSnapshot:
    """A validated container of option quotes conforming to QUOTE_COLUMNS."""

    frame: pd.DataFrame
    as_of: date
    source: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(QUOTE_COLUMNS) - set(self.frame.columns)
        if missing:
            raise ValueError(f"Quote frame missing required columns: {sorted(missing)}")
        # Enforce column order for reproducibility.
        self.frame = self.frame[QUOTE_COLUMNS].reset_index(drop=True)

    # ---- convenience accessors ---------------------------------------
    def for_commodity(self, commodity: Commodity) -> pd.DataFrame:
        return self.frame[self.frame["commodity"] == commodity].copy()

    def maturities(self, commodity: Commodity) -> list[date]:
        sub = self.for_commodity(commodity)
        return sorted(sub["expiry"].unique().tolist())

    def summary(self) -> pd.DataFrame:
        g = (
            self.frame.groupby(["commodity", "expiry"])
            .agg(
                n_strikes=("strike", "nunique"),
                n_quotes=("strike", "size"),
                forward=("forward", "first"),
                k_min=("strike", "min"),
                k_max=("strike", "max"),
            )
            .reset_index()
        )
        return g


def coerce_to_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Fill mid, tag source, and order columns. Raises on missing essentials."""
    df = df.copy()
    essential = {"commodity", "as_of", "expiry", "forward", "strike", "option_type", "bid", "ask"}
    missing = essential - set(df.columns)
    if missing:
        raise ValueError(f"Adapter output missing essential fields: {sorted(missing)}")

    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
    for opt in ("volume", "open_interest"):
        if opt not in df.columns:
            df[opt] = pd.NA
    if "future_expiry" not in df.columns:
        df["future_expiry"] = df["expiry"]
    df["source"] = source

    return df[QUOTE_COLUMNS]
