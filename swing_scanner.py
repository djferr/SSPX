# swing_scanner.py
# Minervini-style swing trading scanner
#
# Install:
#   pip install yfinance pandas numpy
#
# Run:
#   python swing_scanner.py
#
# Optional:
#   - Edit SYMBOLS below
#   - Or load from a CSV/watchlist later

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# CONFIG
# ----------------------------
SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA", "AVGO", "AMD",
    "CRM", "PANW", "NOW", "ANET", "UBER", "MELI", "SHOP", "PLTR", "SMCI",
    "TTD", "CRWD", "LULU", "NFLX", "CELH", "ARM", "ZS", "DDOG", "SNOW",
    "IOT", "APP", "MDB", "FTNT", "TEAM", "ABNB", "COST", "ELF", "DECK",
]

BENCHMARK = "SPY"
LOOKBACK = "2y"          # enough for 200-day MA and 52-week stats
MIN_DOLLAR_VOLUME = 20_000_000
MIN_PRICE = 10.0
MAX_RESULTS = 30
OUTPUT_CSV = "swing_scan_results.csv"

# Scoring weights
WEIGHTS = {
    "trend_template": 30,
    "relative_strength": 20,
    "volume_quality": 10,
    "tightness": 15,
    "pivot_setup": 15,
    "distance_from_high": 10,
}


@dataclass
class ScanResult:
    symbol: str
    close: float
    volume: float
    dollar_volume_20d: float
    ma50: float
    ma150: float
    ma200: float
    high_52w: float
    low_52w: float
    pct_from_52w_high: float
    pct_above_52w_low: float
    rs_vs_spy_3m: float
    rs_vs_spy_6m: float
    avg_down_vol_ratio: float
    contraction_1: float
    contraction_2: float
    contraction_3: float
    tight_close_range_10d: float
    pivot: float
    breakout_ready: bool
    score: int
    notes: str


# ----------------------------
# DATA HELPERS
# ----------------------------
def download_data(symbols: List[str], period: str = LOOKBACK) -> dict[str, pd.DataFrame]:
    data = {}
    for s in symbols:
        try:
            df = yf.download(
                s,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                print(f"[WARN] No data for {s}")
                continue

            # Flatten if yfinance returns multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(set(df.columns)):
                print(f"[WARN] Missing columns for {s}: {df.columns.tolist()}")
                continue

            df = df.dropna().copy()
            data[s] = df
        except Exception as e:
            print(f"[ERROR] Failed downloading {s}: {e}")
    return data


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA150"] = df["Close"].rolling(150).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["DollarVol20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    df["52W_HIGH"] = df["High"].rolling(252).max()
    df["52W_LOW"] = df["Low"].rolling(252).min()

    return df


# ----------------------------
# FEATURE CALCULATIONS
# ----------------------------
def percent_change(a: float, b: float) -> float:
    if a == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (b / a - 1.0) * 100.0


def compute_rs(stock_df: pd.DataFrame, bench_df: pd.DataFrame, days: int) -> float:
    if len(stock_df) < days + 1 or len(bench_df) < days + 1:
        return np.nan

    stock_ret = stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-days - 1]
    bench_ret = bench_df["Close"].iloc[-1] / bench_df["Close"].iloc[-days - 1]

    if bench_ret == 0:
        return np.nan

    # ratio > 1 means outperforming benchmark
    return (stock_ret / bench_ret - 1.0) * 100.0


def average_down_volume_ratio(df: pd.DataFrame, lookback: int = 30) -> float:
    recent = df.tail(lookback).copy()
    if recent.empty:
        return np.nan

    recent["chg"] = recent["Close"].pct_change()
    up = recent[recent["chg"] > 0]["Volume"]
    down = recent[recent["chg"] < 0]["Volume"]

    if len(up) == 0 or len(down) == 0:
        return np.nan

    up_avg = up.mean()
    down_avg = down.mean()

    if up_avg == 0:
        return np.nan

    return down_avg / up_avg  # lower is better


def contraction_measure(df: pd.DataFrame, span: int) -> float:
    """
    Measures average daily range over a window.
    Lower number = tighter action.
    """
    recent = df.tail(span)
    if recent.empty:
        return np.nan
    return ((recent["High"] - recent["Low"]) / recent["Close"]).mean() * 100.0


def tight_close_range(df: pd.DataFrame, span: int = 10) -> float:
    recent = df.tail(span)
    if recent.empty:
        return np.nan
    hi = recent["Close"].max()
    lo = recent["Close"].min()
    if lo == 0:
        return np.nan
    return (hi / lo - 1.0) * 100.0


def detect_pivot(df: pd.DataFrame, base_lookback: int = 30) -> Tuple[float, bool]:
    """
    Simple pivot logic:
    - pivot = highest high of prior base window excluding current day
    - breakout_ready = price within ~2% of pivot or above it on strong-ish volume
    """
    if len(df) < base_lookback + 1:
        return np.nan, False

    recent = df.tail(base_lookback + 1).copy()
    prior = recent.iloc[:-1]
    today = recent.iloc[-1]

    pivot = prior["High"].max()
    close = today["Close"]
    vol = today["Volume"]
    avg_vol = prior["Volume"].tail(20).mean()

    near_pivot = close >= pivot * 0.98
    breakout = (close > pivot) and (avg_vol > 0) and (vol >= avg_vol * 1.2)

    return float(pivot), bool(near_pivot or breakout)


def ma200_rising(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < 200 + lookback:
        return False
    current = df["MA200"].iloc[-1]
    past = df["MA200"].iloc[-lookback]
    if pd.isna(current) or pd.isna(past):
        return False
    return current > past


# ----------------------------
# SCORING
# ----------------------------
def score_stock(df: pd.DataFrame, bench_df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
    if len(df) < 252:
        return None

    row = df.iloc[-1]

    close = float(row["Close"])
    volume = float(row["Volume"])
    ma50 = float(row["MA50"]) if not pd.isna(row["MA50"]) else np.nan
    ma150 = float(row["MA150"]) if not pd.isna(row["MA150"]) else np.nan
    ma200 = float(row["MA200"]) if not pd.isna(row["MA200"]) else np.nan
    high_52w = float(row["52W_HIGH"]) if not pd.isna(row["52W_HIGH"]) else np.nan
    low_52w = float(row["52W_LOW"]) if not pd.isna(row["52W_LOW"]) else np.nan
    dollar_volume_20d = float(row["DollarVol20"]) if not pd.isna(row["DollarVol20"]) else np.nan

    if any(pd.isna(x) for x in [ma50, ma150, ma200, high_52w, low_52w, dollar_volume_20d]):
        return None

    if close < MIN_PRICE or dollar_volume_20d < MIN_DOLLAR_VOLUME:
        return None

    pct_from_high = percent_change(high_52w, close)  # negative if below high
    pct_above_low = percent_change(low_52w, close)

    rs_3m = compute_rs(df, bench_df, 63)
    rs_6m = compute_rs(df, bench_df, 126)

    down_vol_ratio = average_down_volume_ratio(df, 30)
    c1 = contraction_measure(df, 60)
    c2 = contraction_measure(df, 30)
    c3 = contraction_measure(df, 15)
    tight10 = tight_close_range(df, 10)
    pivot, breakout_ready = detect_pivot(df)

    score = 0
    notes = []

    # 1) Trend template
    trend_pass = (
        close > ma50 > ma150 > ma200 and
        ma200_rising(df) and
        close >= high_52w * 0.75 and
        close >= low_52w * 1.30
    )
    if trend_pass:
        score += WEIGHTS["trend_template"]
        notes.append("trend_template_pass")
    else:
        # partial credit
        partial = 0
        if close > ma50: partial += 6
        if ma50 > ma150: partial += 6
        if ma150 > ma200: partial += 6
        if ma200_rising(df): partial += 6
        if close >= high_52w * 0.75: partial += 6
        score += min(partial, WEIGHTS["trend_template"])

    # 2) RS vs SPY
    rs_points = 0
    if not pd.isna(rs_3m):
        if rs_3m > 15: rs_points += 10
        elif rs_3m > 5: rs_points += 6
    if not pd.isna(rs_6m):
        if rs_6m > 20: rs_points += 10
        elif rs_6m > 8: rs_points += 6
    score += min(rs_points, WEIGHTS["relative_strength"])
    if rs_points >= 12:
        notes.append("strong_rs")

    # 3) Volume quality
    # lower down/up volume ratio is better
    vol_points = 0
    if not pd.isna(down_vol_ratio):
        if down_vol_ratio < 0.85:
            vol_points = WEIGHTS["volume_quality"]
            notes.append("constructive_volume")
        elif down_vol_ratio < 1.0:
            vol_points = 6
    score += vol_points

    # 4) Tightness / contraction
    # We want contraction to shrink from older window to newer window
    tight_points = 0
    if not any(pd.isna(x) for x in [c1, c2, c3, tight10]):
        contracting = (c3 < c2 < c1)
        very_tight = tight10 < 3.5

        if contracting:
            tight_points += 8
            notes.append("volatility_contracting")
        if very_tight:
            tight_points += 7
            notes.append("tight_10d")
    score += min(tight_points, WEIGHTS["tightness"])

    # 5) Pivot / breakout setup
    pivot_points = 0
    if breakout_ready:
        pivot_points += 10
        notes.append("near_or_above_pivot")
    if not pd.isna(pivot) and close >= pivot * 0.97:
        pivot_points += 5
    score += min(pivot_points, WEIGHTS["pivot_setup"])

    # 6) Distance from 52W high
    # closer to highs is preferred
    high_points = 0
    if not pd.isna(pct_from_high):
        # pct_from_high is negative if below high
        distance_below_high = abs(min(pct_from_high, 0))
        if distance_below_high <= 5:
            high_points = 10
        elif distance_below_high <= 10:
            high_points = 7
        elif distance_below_high <= 15:
            high_points = 4
    score += high_points

    return ScanResult(
        symbol=symbol,
        close=round(close, 2),
        volume=round(volume, 0),
        dollar_volume_20d=round(dollar_volume_20d, 0),
        ma50=round(ma50, 2),
        ma150=round(ma150, 2),
        ma200=round(ma200, 2),
        high_52w=round(high_52w, 2),
        low_52w=round(low_52w, 2),
        pct_from_52w_high=round(pct_from_high, 2) if not pd.isna(pct_from_high) else np.nan,
        pct_above_52w_low=round(pct_above_low, 2) if not pd.isna(pct_above_low) else np.nan,
        rs_vs_spy_3m=round(rs_3m, 2) if not pd.isna(rs_3m) else np.nan,
        rs_vs_spy_6m=round(rs_6m, 2) if not pd.isna(rs_6m) else np.nan,
        avg_down_vol_ratio=round(down_vol_ratio, 2) if not pd.isna(down_vol_ratio) else np.nan,
        contraction_1=round(c1, 2) if not pd.isna(c1) else np.nan,
        contraction_2=round(c2, 2) if not pd.isna(c2) else np.nan,
        contraction_3=round(c3, 2) if not pd.isna(c3) else np.nan,
        tight_close_range_10d=round(tight10, 2) if not pd.isna(tight10) else np.nan,
        pivot=round(pivot, 2) if not pd.isna(pivot) else np.nan,
        breakout_ready=breakout_ready,
        score=int(score),
        notes=", ".join(notes),
    )


# ----------------------------
# MAIN
# ----------------------------
def run_scan(symbols: List[str]) -> pd.DataFrame:
    all_symbols = sorted(set(symbols + [BENCHMARK]))
    print(f"[INFO] Downloading data for {len(all_symbols)} symbols...")
    raw = download_data(all_symbols, LOOKBACK)

    if BENCHMARK not in raw:
        raise RuntimeError(f"Could not download benchmark data for {BENCHMARK}")

    processed = {s: add_indicators(df) for s, df in raw.items()}
    bench_df = processed[BENCHMARK]

    results: List[ScanResult] = []

    for symbol in symbols:
        try:
            df = processed.get(symbol)
            if df is None or df.empty:
                continue

            result = score_stock(df, bench_df, symbol)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed processing {symbol}: {e}")
            traceback.print_exc()

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame([asdict(r) for r in results])
    out = out.sort_values(
        by=["score", "rs_vs_spy_3m", "pct_from_52w_high"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return out


if __name__ == "__main__":
    df = run_scan(SYMBOLS)

    if df.empty:
        print("No qualifying stocks found.")
    else:
        print("\nTop scan results:\n")
        print(df.head(MAX_RESULTS).to_string(index=False))
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[INFO] Saved results to {OUTPUT_CSV}")