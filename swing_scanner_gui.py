from __future__ import annotations

import io
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, date
from textwrap import dedent
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import ttkbootstrap as tb
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from ttkbootstrap.constants import *

APP_TITLE = "Swing Scanner Pro X"

SP500_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
BENCHMARKS = ["SPY", "VOO", "^GSPC"]

LOOKBACK = "1y"
MAX_WORKERS = 16
TOP_N_DEFAULT = 30
ENRICH_TOP_N = 20

MIN_PRICE = 10.0
MIN_DOLLAR_VOLUME = 20_000_000
A_PLUS_SCORE = 72

WATCHLIST_FILE = "swing_watchlist.json"
JOURNAL_FILE = "trade_journal.csv"
SNAPSHOT_DIR = "chart_snapshots"

DETAILS_HELP_TEXT = dedent("""
Selected Stock Details Guide

score
The higher the score, the more bullish traits the stock has. The scanner rewards trend quality,
strength versus the market, tight price action, liquidity, and breakout positioning.

setup_type
The scanner's best guess at the pattern:
- Breakout Setup
- Tight Base
- Trend Continuation
- Pullback Setup
- Mixed Setup

breakout_ready
True means price is sitting near a recent pivot area.

market_ok
Whether the market benchmark trend is supportive for swing longs.

rs_vs_benchmark_3m / 6m
Relative strength versus the benchmark over about 3 and 6 months. Positive numbers mean the
stock has outperformed the market.

pct_from_52w_high
How far the stock is from its 52-week high. Strong leaders usually stay relatively close to highs.

tight_close_range_10d
A lower number means the stock has traded more tightly over the last 10 sessions.

volume_ratio_today
Today's volume divided by the 20-day average volume.

atr_pct
Average True Range as a percentage of price. Lower often means cleaner price action.

entry / stop / risk_per_share / position_size_shares
Trade-planning numbers based on your account size and risk percent.

How to use it
The scanner is for finding candidates, not auto-buying. Focus on top scores, strong relative strength,
tight action, liquidity, and clean entries near pivots.
""")

DEFAULT_SYMBOLS_TEXT = "AAPL\nMSFT\nNVDA\nMETA\nAMZN"


@dataclass
class ScanResult:
    symbol: str
    close: float
    score: int
    setup_type: str
    breakout_ready: bool
    market_ok: bool
    rs_vs_benchmark_3m: float
    rs_vs_benchmark_6m: float
    pct_from_52w_high: float
    tight_close_range_10d: float
    pivot: float
    dollar_volume_20d: float
    volume_ratio_today: float
    atr_pct: float
    trend_strength: float
    sector: str
    earnings_warning: str
    entry: float
    stop: float
    risk_per_share: float
    position_size_shares: int
    explanation: str
    notes: str


class UniverseLoader:
    @staticmethod
    def get_sp500_symbols() -> List[str]:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(SP500_URL, headers=headers, timeout=20)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        if "Symbol" not in df.columns:
            raise RuntimeError("Could not find Symbol column in S&P 500 source.")
        symbols = []
        for value in df["Symbol"].tolist():
            sym = str(value).strip().upper().replace(".", "-")
            if sym:
                symbols.append(sym)
        return list(dict.fromkeys(symbols))


class Storage:
    @staticmethod
    def load_watchlist() -> List[str]:
        if not os.path.exists(WATCHLIST_FILE):
            return []
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [str(x).upper() for x in data]
        except Exception:
            return []

    @staticmethod
    def save_watchlist(symbols: List[str]) -> None:
        cleaned = sorted(list(dict.fromkeys([str(s).upper() for s in symbols if str(s).strip()])))
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)

    @staticmethod
    def ensure_journal() -> None:
        if not os.path.exists(JOURNAL_FILE):
            cols = [
                "date", "symbol", "entry", "stop", "target", "shares",
                "setup_type", "score", "status", "notes"
            ]
            pd.DataFrame(columns=cols).to_csv(JOURNAL_FILE, index=False)

    @staticmethod
    def load_journal() -> pd.DataFrame:
        Storage.ensure_journal()
        return pd.read_csv(JOURNAL_FILE)

    @staticmethod
    def add_journal_row(row: dict) -> None:
        Storage.ensure_journal()
        df = pd.read_csv(JOURNAL_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(JOURNAL_FILE, index=False)


class MetaHelper:
    @staticmethod
    def get_sector_and_earnings(symbol: str) -> Tuple[str, str]:
        sector = "Unknown"
        earnings_warning = "Unknown"
        try:
            ticker = yf.Ticker(symbol)

            try:
                info = ticker.info or {}
                sector = info.get("sector", "Unknown") or "Unknown"
            except Exception:
                pass

            try:
                cal = ticker.calendar
                values = []
                if isinstance(cal, pd.DataFrame) and not cal.empty:
                    values = cal.values.flatten().tolist()
                elif isinstance(cal, dict):
                    values = list(cal.values())

                dates: List[date] = []
                for value in values:
                    try:
                        ts = pd.to_datetime(value)
                        if not pd.isna(ts):
                            dates.append(ts.date())
                    except Exception:
                        pass

                future_dates = sorted([d for d in dates if d >= date.today()])
                if future_dates:
                    delta = (future_dates[0] - date.today()).days
                    if delta <= 21:
                        earnings_warning = f"Earnings in {delta} day(s)"
                    else:
                        earnings_warning = "No near earnings"
                else:
                    earnings_warning = "No near earnings"
            except Exception:
                pass
        except Exception:
            pass

        return sector, earnings_warning


class ScannerEngine:
    @staticmethod
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in out.columns for col in required):
            return pd.DataFrame()

        out = out[required].copy()
        for col in required:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna()
        return out

    @staticmethod
    def download_symbol(symbol: str) -> pd.DataFrame:
        df = yf.download(
            symbol,
            period=LOOKBACK,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return ScannerEngine.normalize_df(df)

    @staticmethod
    def download_data(symbols: List[str], log_callback=None) -> Dict[str, pd.DataFrame]:
        unique = list(dict.fromkeys(symbols))
        results: Dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(ScannerEngine.download_symbol, symbol): symbol for symbol in unique}
            total = len(future_map)
            done = 0

            for future in as_completed(future_map):
                symbol = future_map[future]
                done += 1
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                        if log_callback:
                            log_callback(f"Downloaded {symbol} ({done}/{total})")
                    else:
                        if log_callback:
                            log_callback(f"No usable data for {symbol} ({done}/{total})")
                except Exception as e:
                    if log_callback:
                        log_callback(f"Download failed for {symbol}: {e}")

        return results

    @staticmethod
    def compute_rs(stock_df: pd.DataFrame, bench_df: Optional[pd.DataFrame], days: int) -> float:
        if bench_df is None:
            return 0.0
        if len(stock_df) < days + 1 or len(bench_df) < days + 1:
            return np.nan

        stock_start = float(stock_df["Close"].iloc[-days - 1])
        stock_end = float(stock_df["Close"].iloc[-1])
        bench_start = float(bench_df["Close"].iloc[-days - 1])
        bench_end = float(bench_df["Close"].iloc[-1])

        if stock_start <= 0 or bench_start <= 0:
            return np.nan

        stock_ret = stock_end / stock_start
        bench_ret = bench_end / bench_start
        if bench_ret == 0:
            return np.nan

        return round((stock_ret / bench_ret - 1.0) * 100.0, 2)

    @staticmethod
    def atr_pct(df: pd.DataFrame, period: int = 14) -> float:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(period).mean().iloc[-1]
        latest_close = close.iloc[-1]
        if pd.isna(atr) or latest_close <= 0:
            return np.nan

        return float(atr / latest_close * 100.0)

    @staticmethod
    def volume_ratio_today(df: pd.DataFrame) -> float:
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        today_vol = df["Volume"].iloc[-1]
        if pd.isna(avg_vol) or avg_vol <= 0:
            return np.nan
        return float(today_vol / avg_vol)

    @staticmethod
    def trend_strength(df: pd.DataFrame) -> float:
        close = df["Close"]
        latest = close.iloc[-1]
        ma21 = close.rolling(21).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma50

        if any(pd.isna(x) for x in [latest, ma21, ma50, ma200]) or latest <= 0:
            return np.nan

        val = ((latest / ma21 - 1) + (latest / ma50 - 1) + (latest / ma200 - 1)) * 100.0
        return round(float(val), 2)

    @staticmethod
    def market_health(bench_df: Optional[pd.DataFrame]) -> bool:
        if bench_df is None or bench_df.empty or len(bench_df) < 200:
            return False

        close = float(bench_df["Close"].iloc[-1])
        ma21 = float(bench_df["Close"].rolling(21).mean().iloc[-1])
        ma50 = float(bench_df["Close"].rolling(50).mean().iloc[-1])
        ma200 = float(bench_df["Close"].rolling(200).mean().iloc[-1])

        return bool(close > ma21 > ma50 > ma200)

    @staticmethod
    def classify_setup(
        close: float,
        pivot: float,
        ma21: float,
        ma50: float,
        tight_10d: float,
        pct_from_high: float,
    ) -> str:
        if close >= pivot * 0.98 and tight_10d <= 4.0:
            return "Breakout Setup"
        if close > ma21 and close > ma50 and tight_10d <= 3.5:
            return "Tight Base"
        if close > ma50 and close < pivot * 0.98 and pct_from_high >= -12:
            return "Trend Continuation"
        if close > ma21 and close > ma50 and close < pivot * 0.95:
            return "Pullback Setup"
        return "Mixed Setup"

    @staticmethod
    def exclude_junk(symbol: str, df: pd.DataFrame) -> bool:
        if df.empty or len(df) < 200:
            return True
        close = float(df["Close"].iloc[-1])
        dollar_vol = float((df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1])
        if pd.isna(dollar_vol) or close < MIN_PRICE or dollar_vol < MIN_DOLLAR_VOLUME:
            return True
        if "^" in symbol:
            return True
        return False

    @staticmethod
    def build_trade_plan(
        close: float,
        pivot: float,
        ma21: float,
        atr_pct: float,
        account_size: float,
        risk_percent: float,
    ) -> Tuple[float, float, float, int]:
        entry = round(max(close, pivot), 2)

        stop_candidate_1 = pivot * 0.97
        stop_candidate_2 = ma21 * 0.99
        stop = round(min(stop_candidate_1, stop_candidate_2), 2)

        if atr_pct > 0:
            atr_stop = close * (1 - min(0.06, max(0.02, atr_pct / 100.0)))
            stop = round(min(stop, atr_stop), 2)

        risk_per_share = round(max(0.01, entry - stop), 2)
        max_risk_dollars = max(1.0, account_size * (risk_percent / 100.0))
        shares = int(max_risk_dollars // risk_per_share) if risk_per_share > 0 else 0
        return entry, stop, risk_per_share, max(0, shares)

    @staticmethod
    def build_explanation(
        symbol: str,
        score: int,
        setup_type: str,
        market_ok: bool,
        reasons: List[str],
        cautions: List[str],
        entry: float,
        stop: float,
        shares: int,
    ) -> str:
        lines = [f"Why {symbol} is a candidate:"]
        lines.append(f"- Setup type: {setup_type}")
        lines.append(f"- Scanner score: {score}")
        lines.append(f"- Market health filter: {'Healthy' if market_ok else 'Not ideal'}")

        for reason in reasons:
            lines.append(f"- {reason}")

        if cautions:
            lines.append("")
            lines.append("Things to watch:")
            for caution in cautions:
                lines.append(f"- {caution}")

        lines.append("")
        lines.append("Trade planning ideas:")
        lines.append(f"- Suggested entry area: {entry:.2f}")
        lines.append(f"- Suggested stop area: {stop:.2f}")
        lines.append(f"- Approx position size: {shares} shares")
        lines.append("- Wait for clean price behavior near the pivot instead of forcing a trade.")
        lines.append("- Use this as a shortlist tool, not an automatic buy signal.")
        return "\n".join(lines)

    @staticmethod
    def score_stock(
        symbol: str,
        df: pd.DataFrame,
        bench_df: Optional[pd.DataFrame],
        market_ok: bool,
        account_size: float,
        risk_percent: float,
        sector: str,
        earnings_warning: str,
    ) -> Optional[ScanResult]:
        if ScannerEngine.exclude_junk(symbol, df):
            return None

        close = float(df["Close"].iloc[-1])
        ma10 = float(df["Close"].rolling(10).mean().iloc[-1])
        ma21 = float(df["Close"].rolling(21).mean().iloc[-1])
        ma50 = float(df["Close"].rolling(50).mean().iloc[-1])
        ma150 = float(df["Close"].rolling(150).mean().iloc[-1]) if len(df) >= 150 else ma50
        ma200 = float(df["Close"].rolling(200).mean().iloc[-1]) if len(df) >= 200 else ma50

        high_52w = float(df["High"].rolling(min(252, len(df))).max().iloc[-1])
        low_52w = float(df["Low"].rolling(min(252, len(df))).min().iloc[-1])

        dollar_volume_20d = float((df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1])
        pivot = float(df["High"].tail(30).max())

        range10_hi = float(df["Close"].tail(10).max())
        range10_lo = float(df["Close"].tail(10).min())
        tight_10d = ((range10_hi / range10_lo) - 1.0) * 100.0 if range10_lo > 0 else np.nan

        rs_3m = ScannerEngine.compute_rs(df, bench_df, 63)
        rs_6m = ScannerEngine.compute_rs(df, bench_df, 126)
        vol_ratio = ScannerEngine.volume_ratio_today(df)
        atrp = ScannerEngine.atr_pct(df)
        trend = ScannerEngine.trend_strength(df)

        pct_from_high = ((close / high_52w) - 1.0) * 100.0 if high_52w > 0 else np.nan
        pct_above_low = ((close / low_52w) - 1.0) * 100.0 if low_52w > 0 else np.nan

        required = [close, ma10, ma21, ma50, high_52w, low_52w, dollar_volume_20d]
        if any(np.isnan(x) for x in required):
            return None

        setup_type = ScannerEngine.classify_setup(close, pivot, ma21, ma50, tight_10d, pct_from_high)

        score = 0
        reasons: List[str] = []
        cautions: List[str] = []
        notes: List[str] = []
        breakout_ready = False

        if market_ok:
            score += 6
            notes.append("market_ok")
            reasons.append("The benchmark trend is healthy, which supports long swing setups.")
        else:
            cautions.append("The overall market trend is not ideal, so long setups deserve more caution.")

        if close > ma10 > ma21:
            score += 8
            notes.append("short_term_trend")
            reasons.append("Price is above the 10-day and 21-day averages, showing short-term control by buyers.")

        if close > ma50:
            score += 10
            notes.append("above_50ma")
            reasons.append("Price is above the 50-day average, which is a common swing-trading trend filter.")

        if ma50 > ma150:
            score += 8
            notes.append("50_over_150")
        else:
            cautions.append("The 50-day average is not clearly above the 150-day average.")

        if close > ma200:
            score += 10
            notes.append("above_200ma")
            reasons.append("Price is above the 200-day average, which keeps the stock in a longer-term uptrend.")
        else:
            cautions.append("Price is not clearly above the 200-day average.")

        if not np.isnan(pct_from_high):
            dist = abs(min(pct_from_high, 0))
            if dist <= 5:
                score += 15
                notes.append("within_5pct_of_high")
                reasons.append("The stock is very close to its 52-week high, which is often where true leaders live.")
            elif dist <= 10:
                score += 10
                notes.append("within_10pct_of_high")
                reasons.append("The stock is still reasonably close to its 52-week high.")
            elif dist <= 15:
                score += 5
                notes.append("within_15pct_of_high")
            else:
                cautions.append("The stock is not especially close to its highs.")

        if not np.isnan(pct_above_low) and pct_above_low >= 30:
            score += 5
            notes.append("well_above_52w_low")

        if not np.isnan(rs_3m):
            if rs_3m > 15:
                score += 10
                notes.append("strong_rs_3m")
                reasons.append("It has strongly outperformed the market over the last 3 months.")
            elif rs_3m > 5:
                score += 5
                notes.append("good_rs_3m")

        if not np.isnan(rs_6m):
            if rs_6m > 20:
                score += 10
                notes.append("strong_rs_6m")
                reasons.append("It has also outperformed over 6 months, which makes the strength more believable.")
            elif rs_6m > 8:
                score += 5
                notes.append("good_rs_6m")

        if not np.isnan(tight_10d):
            if tight_10d < 3.5:
                score += 12
                notes.append("tight_10d")
                reasons.append("Price has tightened over the last 10 days, which can be constructive before a move.")
            elif tight_10d < 6:
                score += 6
                notes.append("fairly_tight_10d")
            else:
                cautions.append("Recent price action is a bit wide, so the setup is less clean.")

        if close >= pivot * 0.98:
            score += 10
            notes.append("near_pivot")
            breakout_ready = True
            reasons.append("The stock is close to its pivot area, so it is in a position where a breakout could matter.")
        else:
            cautions.append("Price is not especially close to the pivot yet.")

        if not np.isnan(vol_ratio):
            if vol_ratio >= 1.5:
                score += 8
                notes.append("volume_surge")
                reasons.append("Volume is clearly above normal today, which can signal attention and sponsorship.")
            elif vol_ratio >= 1.1:
                score += 4
                notes.append("volume_above_avg")

        if dollar_volume_20d >= 50_000_000:
            score += 6
            notes.append("very_liquid")
            reasons.append("Liquidity is strong, which usually leads to cleaner entries and exits.")

        if not np.isnan(atrp):
            if atrp <= 4.5:
                score += 4
                notes.append("controlled_volatility")
            elif atrp > 8:
                cautions.append("ATR is relatively high, so the stock may swing around more than ideal.")

        if not np.isnan(trend) and trend > 8:
            score += 4
            notes.append("trend_strength")

        if "Earnings in" in earnings_warning:
            cautions.append("Earnings are coming soon, which adds event risk.")

        entry, stop, risk_per_share, shares = ScannerEngine.build_trade_plan(
            close=close,
            pivot=pivot,
            ma21=ma21,
            atr_pct=0.0 if np.isnan(atrp) else atrp,
            account_size=account_size,
            risk_percent=risk_percent,
        )

        explanation = ScannerEngine.build_explanation(
            symbol=symbol,
            score=score,
            setup_type=setup_type,
            market_ok=market_ok,
            reasons=reasons,
            cautions=cautions,
            entry=entry,
            stop=stop,
            shares=shares,
        )

        return ScanResult(
            symbol=symbol,
            close=round(close, 2),
            score=int(score),
            setup_type=setup_type,
            breakout_ready=breakout_ready,
            market_ok=market_ok,
            rs_vs_benchmark_3m=0.0 if np.isnan(rs_3m) else rs_3m,
            rs_vs_benchmark_6m=0.0 if np.isnan(rs_6m) else rs_6m,
            pct_from_52w_high=0.0 if np.isnan(pct_from_high) else round(pct_from_high, 2),
            tight_close_range_10d=0.0 if np.isnan(tight_10d) else round(tight_10d, 2),
            pivot=round(pivot, 2),
            dollar_volume_20d=round(dollar_volume_20d, 0),
            volume_ratio_today=0.0 if np.isnan(vol_ratio) else round(vol_ratio, 2),
            atr_pct=0.0 if np.isnan(atrp) else round(atrp, 2),
            trend_strength=0.0 if np.isnan(trend) else round(trend, 2),
            sector=sector,
            earnings_warning=earnings_warning,
            entry=entry,
            stop=stop,
            risk_per_share=risk_per_share,
            position_size_shares=shares,
            explanation=explanation,
            notes=", ".join(notes),
        )


class ChartWindow(tb.Toplevel):
    def __init__(self, parent, symbol: str, df: pd.DataFrame, explanation: str, result_row: dict):
        super().__init__(parent)
        self.title(f"Chart - {symbol}")
        self.geometry("1260x760")

        outer = tb.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(0, weight=1)

        chart_frame = tb.Frame(outer)
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        side_frame = tb.Frame(outer)
        side_frame.grid(row=0, column=1, sticky="nsew")
        side_frame.columnconfigure(0, weight=1)
        side_frame.rowconfigure(1, weight=1)

        top_btns = tb.Frame(side_frame)
        top_btns.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top_btns.columnconfigure(0, weight=1)

        tb.Button(
            top_btns,
            text="Save Snapshot",
            bootstyle=SUCCESS,
            command=lambda: self.save_snapshot(symbol, df, result_row),
        ).grid(row=0, column=1, sticky="e")

        expl_frame = tb.Labelframe(side_frame, text="Why this is a candidate", padding=10)
        expl_frame.grid(row=1, column=0, sticky="nsew")
        expl_frame.rowconfigure(0, weight=1)
        expl_frame.columnconfigure(0, weight=1)

        fig = plt.Figure(figsize=(8.5, 6), dpi=100)
        ax = fig.add_subplot(211)
        rs_ax = fig.add_subplot(212)

        plot_df = df.tail(180).copy()
        x = range(len(plot_df))

        ax.plot(x, plot_df["Close"], label="Close")
        ax.plot(x, plot_df["Close"].rolling(10).mean(), label="MA10")
        ax.plot(x, plot_df["Close"].rolling(21).mean(), label="MA21")
        ax.plot(x, plot_df["Close"].rolling(50).mean(), label="MA50")
        if len(plot_df) >= 200:
            ax.plot(x, plot_df["Close"].rolling(200).mean(), label="MA200")

        pivot = result_row.get("pivot", None)
        if pivot:
            ax.axhline(pivot, linestyle="--", label="Pivot")

        recent_high = plot_df["High"].max()
        ax.axhline(recent_high, linestyle=":", label="Recent High")

        ax.set_title(symbol)
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.legend()

        rs_line = (plot_df["Close"] / plot_df["Close"].iloc[0]) * 100
        rs_ax.plot(x, rs_line, label="Relative Performance Proxy")
        rs_ax.set_title("Strength Line")
        rs_ax.set_xlabel("Days")
        rs_ax.set_ylabel("Index=100")
        rs_ax.grid(True, alpha=0.3)
        rs_ax.legend()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        expl_box = ScrolledText(expl_frame, wrap="word", font=("Segoe UI", 10))
        expl_box.grid(row=0, column=0, sticky="nsew")
        expl_box.insert("1.0", explanation)
        expl_box.configure(state="disabled")

    def save_snapshot(self, symbol: str, df: pd.DataFrame, row: dict):
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        path = os.path.join(SNAPSHOT_DIR, f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        plot_df = df.tail(180)
        ax.plot(plot_df["Close"], label="Close")
        ax.plot(plot_df["Close"].rolling(21).mean(), label="MA21")
        ax.plot(plot_df["Close"].rolling(50).mean(), label="MA50")
        if len(plot_df) >= 200:
            ax.plot(plot_df["Close"].rolling(200).mean(), label="MA200")
        if row.get("pivot"):
            ax.axhline(row["pivot"], linestyle="--", label="Pivot")
        ax.set_title(f"{symbol} | Score: {row.get('score', '')}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        messagebox.showinfo("Snapshot Saved", f"Saved to:\n{path}")


class App(tb.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title(APP_TITLE)
        self.geometry("1520x920")
        self.minsize(1260, 800)

        self.results_df = pd.DataFrame()
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.is_scanning = False

        self.only_a_plus_var = tb.BooleanVar(value=False)
        self.only_breakout_var = tb.BooleanVar(value=False)
        self.market_filter_var = tb.BooleanVar(value=False)

        self.top_n_var = tb.IntVar(value=TOP_N_DEFAULT)
        self.account_size_var = tb.DoubleVar(value=10000.0)
        self.risk_percent_var = tb.DoubleVar(value=1.0)

        self.watchlist = Storage.load_watchlist()

        self._build_ui()
        self._configure_tree_tags()
        self.refresh_watchlist_tab()
        self.refresh_journal_tab()

    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.scan_tab = tb.Frame(self.notebook)
        self.watchlist_tab = tb.Frame(self.notebook)
        self.journal_tab = tb.Frame(self.notebook)

        self.notebook.add(self.scan_tab, text="Scanner")
        self.notebook.add(self.watchlist_tab, text="Watchlist")
        self.notebook.add(self.journal_tab, text="Trade Journal")

        self._build_scan_tab()
        self._build_watchlist_tab()
        self._build_journal_tab()

    def _build_scan_tab(self):
        self.scan_tab.columnconfigure(1, weight=1)
        self.scan_tab.rowconfigure(0, weight=1)

        left = tb.Frame(self.scan_tab, padding=14)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(11, weight=1)

        right = tb.Frame(self.scan_tab, padding=(0, 14, 14, 14))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)

        tb.Label(left, text="Swing Scanner Pro X", font=("Segoe UI", 20, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))
        tb.Label(left, text="Paste symbols, load the S&P 500, or scan your watchlist.", bootstyle=SECONDARY).grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.symbols_box = ScrolledText(left, height=10, font=("Consolas", 11), wrap="word")
        self.symbols_box.grid(row=2, column=0, sticky="nsew")
        self.symbols_box.insert("1.0", DEFAULT_SYMBOLS_TEXT)

        row1 = tb.Frame(left)
        row1.grid(row=3, column=0, sticky="ew", pady=(10, 6))
        row1.columnconfigure((0, 1), weight=1)
        self.run_btn = tb.Button(row1, text="Run Scan", bootstyle=SUCCESS, command=self.run_scan)
        self.run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tb.Button(row1, text="Clear Symbols", bootstyle=SECONDARY, command=self.clear_symbols).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        row2 = tb.Frame(left)
        row2.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        row2.columnconfigure((0, 1), weight=1)
        tb.Button(row2, text="Load S&P 500", bootstyle=PRIMARY, command=self.load_sp500).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tb.Button(row2, text="Scan S&P 500 Now", bootstyle=WARNING, command=self.scan_sp500).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        row3 = tb.Frame(left)
        row3.grid(row=5, column=0, sticky="ew", pady=(0, 6))
        row3.columnconfigure((0, 1), weight=1)
        tb.Button(row3, text="Load Watchlist", bootstyle=INFO, command=self.load_watchlist_into_symbols).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tb.Button(row3, text="Save Results CSV", bootstyle=SUCCESS, command=self.save_results).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        planner = tb.Labelframe(left, text="Risk Planner", padding=10)
        planner.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        planner.columnconfigure(1, weight=1)

        tb.Label(planner, text="Account Size").grid(row=0, column=0, sticky="w")
        tb.Entry(planner, textvariable=self.account_size_var).grid(row=0, column=1, sticky="ew", padx=(10, 0))

        tb.Label(planner, text="Risk % Per Trade").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tb.Entry(planner, textvariable=self.risk_percent_var).grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))

        filters = tb.Labelframe(left, text="Filters", padding=10)
        filters.grid(row=7, column=0, sticky="ew", pady=(0, 8))

        tb.Checkbutton(filters, text=f"Only show A+ setups ({A_PLUS_SCORE}+)", variable=self.only_a_plus_var, command=self.refresh_table, bootstyle="round-toggle").pack(anchor="w")
        tb.Checkbutton(filters, text="Only show breakout-ready", variable=self.only_breakout_var, command=self.refresh_table, bootstyle="round-toggle").pack(anchor="w", pady=(6, 0))
        tb.Checkbutton(filters, text="Only show market-healthy names", variable=self.market_filter_var, command=self.refresh_table, bootstyle="round-toggle").pack(anchor="w", pady=(6, 0))

        topn_row = tb.Frame(filters)
        topn_row.pack(fill="x", pady=(10, 0))
        tb.Label(topn_row, text="Top rows:").pack(side="left")
        self.top_n_combo = tb.Combobox(topn_row, values=[10, 20, 30, 50, 100, 200], textvariable=self.top_n_var, width=8, state="readonly")
        self.top_n_combo.pack(side="left", padx=(8, 0))
        self.top_n_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_table())

        self.status_var = tb.StringVar(value="Ready")
        tb.Label(left, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).grid(row=8, column=0, sticky="w", pady=(2, 8))

        stat_frame = tb.Labelframe(left, text="Quick Stats", padding=10)
        stat_frame.grid(row=9, column=0, sticky="ew", pady=(0, 8))
        self.stats_var = tb.StringVar(value="No scan yet")
        tb.Label(stat_frame, textvariable=self.stats_var, justify="left").pack(anchor="w")

        alerts_frame = tb.Labelframe(left, text="Alerts", padding=10)
        alerts_frame.grid(row=10, column=0, sticky="ew", pady=(0, 8))
        self.alerts_box = ScrolledText(alerts_frame, height=6, font=("Consolas", 9), wrap="word")
        self.alerts_box.pack(fill="both", expand=True)

        log_header = tb.Frame(left)
        log_header.grid(row=11, column=0, sticky="ew", pady=(4, 6))
        log_header.columnconfigure(0, weight=1)
        tb.Label(log_header, text="Scanner Log", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        tb.Button(log_header, text="Info", bootstyle=INFO, command=self.show_info, width=8).grid(row=0, column=1, sticky="e")

        self.log_box = ScrolledText(left, height=12, font=("Consolas", 10), wrap="word")
        self.log_box.grid(row=12, column=0, sticky="nsew")

        top_right = tb.Frame(right)
        top_right.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top_right.columnconfigure(0, weight=1)

        header_row = tb.Frame(top_right)
        header_row.grid(row=0, column=0, sticky="ew")
        header_row.columnconfigure(0, weight=1)
        tb.Label(header_row, text="Top Candidates", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        tb.Button(header_row, text="Info", bootstyle=INFO, command=self.show_info, width=8).grid(row=0, column=1, sticky="e")

        self.summary_var = tb.StringVar(value="No results yet")
        tb.Label(top_right, textvariable=self.summary_var, bootstyle=SECONDARY).grid(row=1, column=0, sticky="w")

        columns = (
            "symbol", "score", "close", "setup_type", "breakout_ready", "market_ok",
            "rs_vs_benchmark_3m", "rs_vs_benchmark_6m", "pct_from_52w_high",
            "tight_close_range_10d", "pivot", "volume_ratio_today",
            "earnings_warning", "notes"
        )

        table_frame = tb.Frame(right)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll = tb.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=yscroll.set)

        widths = {
            "symbol": 80,
            "score": 70,
            "close": 90,
            "setup_type": 130,
            "breakout_ready": 100,
            "market_ok": 85,
            "rs_vs_benchmark_3m": 115,
            "rs_vs_benchmark_6m": 115,
            "pct_from_52w_high": 115,
            "tight_close_range_10d": 120,
            "pivot": 90,
            "volume_ratio_today": 110,
            "earnings_warning": 150,
            "notes": 280,
        }

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=widths[col], anchor="center")

        details_frame = tb.Labelframe(right, text="Selected Stock Details", padding=10)
        details_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(1, weight=1)

        details_header = tb.Frame(details_frame)
        details_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        details_header.columnconfigure(0, weight=1)

        tb.Button(details_header, text="Open Chart + Explanation", bootstyle=PRIMARY, command=self.open_chart).grid(row=0, column=1, sticky="e")
        tb.Button(details_header, text="Add to Watchlist", bootstyle=INFO, command=self.add_selected_to_watchlist).grid(row=0, column=2, sticky="e", padx=(8, 0))
        tb.Button(details_header, text="Add to Journal", bootstyle=SUCCESS, command=self.add_selected_to_journal).grid(row=0, column=3, sticky="e", padx=(8, 0))

        self.details_box = ScrolledText(details_frame, height=12, font=("Consolas", 10), wrap="word")
        self.details_box.grid(row=1, column=0, sticky="nsew")

        self.tree.bind("<<TreeviewSelect>>", self.show_details)
        self.tree.bind("<Double-1>", lambda _e: self.open_chart())

    def _build_watchlist_tab(self):
        self.watchlist_tab.columnconfigure(0, weight=1)
        self.watchlist_tab.rowconfigure(1, weight=1)

        header = tb.Frame(self.watchlist_tab, padding=14)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        tb.Label(header, text="Watchlist / Favorites", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")

        btns = tb.Frame(header)
        btns.grid(row=0, column=1, sticky="e")
        tb.Button(btns, text="Load into Scanner", bootstyle=PRIMARY, command=self.load_watchlist_into_symbols).pack(side="left", padx=4)
        tb.Button(btns, text="Remove Selected", bootstyle=DANGER, command=self.remove_selected_watchlist).pack(side="left", padx=4)

        self.watchlist_tree = ttk.Treeview(self.watchlist_tab, columns=("symbol",), show="headings")
        self.watchlist_tree.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.watchlist_tree.heading("symbol", text="Symbol")
        self.watchlist_tree.column("symbol", width=220, anchor="center")

    def _build_journal_tab(self):
        self.journal_tab.columnconfigure(0, weight=1)
        self.journal_tab.rowconfigure(1, weight=1)

        header = tb.Frame(self.journal_tab, padding=14)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        tb.Label(header, text="Trade Journal", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        tb.Button(header, text="Refresh Journal", bootstyle=INFO, command=self.refresh_journal_tab).grid(row=0, column=1, sticky="e")

        cols = ("date", "symbol", "entry", "stop", "target", "shares", "setup_type", "score", "status", "notes")
        self.journal_tree = ttk.Treeview(self.journal_tab, columns=cols, show="headings")
        self.journal_tree.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        for col in cols:
            self.journal_tree.heading(col, text=col)
            self.journal_tree.column(col, width=110, anchor="center")

    def _configure_tree_tags(self):
        self.tree.tag_configure("elite", background="#17321d", foreground="#d8ffd8")
        self.tree.tag_configure("good", background="#2f2a12", foreground="#fff4c9")
        self.tree.tag_configure("weak", background="#331818", foreground="#ffd6d6")

    def write_log(self, msg: str):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.update_idletasks()

    def show_info(self):
        win = tb.Toplevel(self)
        win.title("Selected Stock Details Guide")
        win.geometry("760x700")
        txt = ScrolledText(win, wrap="word", font=("Segoe UI", 10))
        txt.pack(fill="both", expand=True, padx=12, pady=12)
        txt.insert("1.0", DETAILS_HELP_TEXT)
        txt.configure(state="disabled")

    def set_symbols(self, symbols: List[str]):
        self.symbols_box.delete("1.0", "end")
        self.symbols_box.insert("1.0", "\n".join(symbols))

    def get_symbols(self) -> List[str]:
        raw = self.symbols_box.get("1.0", "end").strip()
        if not raw:
            return []
        parts = []
        for line in raw.splitlines():
            for piece in line.replace(",", " ").split():
                cleaned = piece.strip().upper()
                if cleaned:
                    parts.append(cleaned)
        return list(dict.fromkeys(parts))

    def clear_symbols(self):
        self.symbols_box.delete("1.0", "end")

    def load_sp500(self):
        try:
            self.status_var.set("Loading S&P 500 symbols...")
            self.update_idletasks()
            symbols = UniverseLoader.get_sp500_symbols()
            self.set_symbols(symbols)
            self.status_var.set(f"Loaded {len(symbols)} S&P 500 symbols")
            self.write_log(f"Loaded {len(symbols)} S&P 500 symbols")
        except Exception as e:
            self.status_var.set("Failed to load S&P 500")
            messagebox.showerror("S&P 500 Load Error", str(e))

    def scan_sp500(self):
        try:
            symbols = UniverseLoader.get_sp500_symbols()
            self.set_symbols(symbols)
            self.write_log(f"Loaded {len(symbols)} S&P 500 symbols")
            self.run_scan()
        except Exception as e:
            messagebox.showerror("S&P 500 Scan Error", str(e))

    def load_watchlist_into_symbols(self):
        self.set_symbols(self.watchlist)
        self.notebook.select(self.scan_tab)

    def save_results(self):
        if self.results_df.empty:
            messagebox.showinfo("No Results", "Run a scan first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="swing_scan_results.csv",
        )
        if not path:
            return
        self._filtered_results().to_csv(path, index=False)
        self.write_log(f"Saved results to {path}")

    def run_scan(self):
        if self.is_scanning:
            return

        symbols = self.get_symbols()
        if not symbols:
            messagebox.showwarning("No Symbols", "Enter at least one ticker.")
            return

        self.is_scanning = True
        self.run_btn.configure(state="disabled")
        self.status_var.set("Scanning...")
        self.summary_var.set(f"Scanning {len(symbols)} symbols")
        self.log_box.delete("1.0", "end")
        self.alerts_box.delete("1.0", "end")
        self.details_box.delete("1.0", "end")
        self.clear_table()

        threading.Thread(target=self._scan_worker, args=(symbols,), daemon=True).start()

    def _scan_worker(self, symbols: List[str]):
        try:
            universe = list(dict.fromkeys([s.strip().upper() for s in symbols if s.strip()]))

            self.price_data = ScannerEngine.download_data(
                universe,
                log_callback=lambda msg: self.after(0, self.write_log, msg),
            )

            bench_df = None
            bench_used = None
            for bench in BENCHMARKS:
                self.after(0, self.write_log, f"Trying benchmark: {bench}")
                bench_data = ScannerEngine.download_data([bench])
                candidate = bench_data.get(bench)
                if candidate is not None and not candidate.empty and len(candidate) >= 130:
                    bench_df = candidate
                    bench_used = bench
                    self.price_data[bench] = candidate
                    break

            if bench_used:
                self.after(0, self.write_log, f"Using benchmark: {bench_used}")
            else:
                self.after(0, self.write_log, "No benchmark available. Falling back to zero RS values.")

            market_ok = ScannerEngine.market_health(bench_df)
            account_size = float(self.account_size_var.get())
            risk_percent = float(self.risk_percent_var.get())

            results: List[ScanResult] = []
            for idx, symbol in enumerate(universe, start=1):
                self.after(0, self.write_log, f"Scoring {symbol} ({idx}/{len(universe)})")
                df = self.price_data.get(symbol)
                if df is None or df.empty:
                    continue
                try:
                    result = ScannerEngine.score_stock(
                        symbol=symbol,
                        df=df,
                        bench_df=bench_df,
                        market_ok=market_ok,
                        account_size=account_size,
                        risk_percent=risk_percent,
                        sector="Unknown",
                        earnings_warning="Unknown",
                    )
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    self.after(0, self.write_log, f"Scoring failed for {symbol}: {e}")

            out = pd.DataFrame([asdict(r) for r in results]) if results else pd.DataFrame()
            if not out.empty:
                out = out.sort_values(
                    by=["score", "rs_vs_benchmark_3m", "pct_from_52w_high"],
                    ascending=[False, False, False],
                ).reset_index(drop=True)

                top_symbols = out["symbol"].head(ENRICH_TOP_N).tolist()
                self.after(0, self.write_log, f"Enriching top {len(top_symbols)} names with sector / earnings...")
                for sym in top_symbols:
                    sector, earnings_warning = MetaHelper.get_sector_and_earnings(sym)
                    out.loc[out["symbol"] == sym, "sector"] = sector
                    out.loc[out["symbol"] == sym, "earnings_warning"] = earnings_warning

                    if "Earnings in" in earnings_warning:
                        current_expl = str(out.loc[out["symbol"] == sym, "explanation"].iloc[0])
                        if "Earnings are coming soon" not in current_expl:
                            current_expl += (
                                "\n\nThings to watch:\n"
                                f"- {earnings_warning}, which adds event risk."
                            )
                            out.loc[out["symbol"] == sym, "explanation"] = current_expl

            self.after(0, self._scan_complete, out, None)

        except Exception:
            err = traceback.format_exc()
            self.after(0, self._scan_complete, pd.DataFrame(), err)

    def _scan_complete(self, df: pd.DataFrame, error: Optional[str]):
        self.is_scanning = False
        self.run_btn.configure(state="normal")

        if error:
            self.status_var.set("Scan failed")
            self.summary_var.set("Error while scanning")
            self.write_log(error)
            messagebox.showerror("Scan Error", error)
            return

        self.results_df = df.copy()
        self.refresh_table()
        self.refresh_alerts()

        if df.empty:
            self.status_var.set("Done - no qualifying stocks")
            self.summary_var.set("0 candidates found")
            self.stats_var.set("No scan results")
            self.write_log("Scan complete. No qualifying stocks found.")
        else:
            self.status_var.set("Done")
            top_score = int(df["score"].max())
            avg_score = round(float(df["score"].mean()), 1)
            breakout_count = int(df["breakout_ready"].sum())
            self.stats_var.set(
                f"Candidates: {len(df)}\n"
                f"Top score: {top_score}\n"
                f"Average score: {avg_score}\n"
                f"Breakout-ready: {breakout_count}"
            )
            self.write_log(f"Scan complete. {len(df)} candidates found.")

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _filtered_results(self) -> pd.DataFrame:
        df = self.results_df.copy()
        if df.empty:
            return df

        if self.only_a_plus_var.get():
            df = df[df["score"] >= A_PLUS_SCORE]
        if self.only_breakout_var.get():
            df = df[df["breakout_ready"] == True]
        if self.market_filter_var.get():
            df = df[df["market_ok"] == True]

        try:
            top_n = int(self.top_n_var.get())
        except Exception:
            top_n = TOP_N_DEFAULT

        return df.head(top_n).reset_index(drop=True)

    def refresh_table(self):
        df = self._filtered_results()
        self.clear_table()

        if df.empty:
            self.summary_var.set("0 visible candidates")
            return

        self.summary_var.set(f"{len(df)} visible candidates")

        for _, row in df.iterrows():
            tag = "weak"
            if row.get("score", 0) >= A_PLUS_SCORE:
                tag = "elite"
            elif row.get("score", 0) >= 45:
                tag = "good"

            self.tree.insert(
                "",
                "end",
                values=(
                    row.get("symbol", ""),
                    row.get("score", ""),
                    row.get("close", ""),
                    row.get("setup_type", ""),
                    row.get("breakout_ready", ""),
                    row.get("market_ok", ""),
                    row.get("rs_vs_benchmark_3m", ""),
                    row.get("rs_vs_benchmark_6m", ""),
                    row.get("pct_from_52w_high", ""),
                    row.get("tight_close_range_10d", ""),
                    row.get("pivot", ""),
                    row.get("volume_ratio_today", ""),
                    row.get("earnings_warning", ""),
                    row.get("notes", ""),
                ),
                tags=(tag,),
            )

    def refresh_alerts(self):
        self.alerts_box.delete("1.0", "end")
        if self.results_df.empty:
            return

        alerts = []
        for _, row in self.results_df.head(15).iterrows():
            if row.get("breakout_ready", False):
                alerts.append(f"{row['symbol']}: near pivot / breakout-ready")
            if row.get("volume_ratio_today", 0) >= 1.5:
                alerts.append(f"{row['symbol']}: volume surge ({row['volume_ratio_today']}x)")
            if "Earnings in" in str(row.get("earnings_warning", "")):
                alerts.append(f"{row['symbol']}: {row['earnings_warning']}")

        if not alerts:
            alerts = ["No notable alerts from current results."]

        self.alerts_box.insert("1.0", "\n".join(alerts))

    def _selected_symbol(self) -> Optional[str]:
        selection = self.tree.selection()
        if not selection:
            return None
        values = self.tree.item(selection[0], "values")
        if not values:
            return None
        return str(values[0])

    def show_details(self, _event=None):
        symbol = self._selected_symbol()
        if not symbol or self.results_df.empty:
            return

        match = self.results_df[self.results_df["symbol"] == symbol]
        if match.empty:
            return

        row = match.iloc[0].to_dict()

        quality = "Weak"
        if row["score"] >= A_PLUS_SCORE:
            quality = "A+ setup"
        elif row["score"] >= 45:
            quality = "Decent setup"

        lines = [f"quality_assessment: {quality}"]
        ordered_keys = [
            "symbol", "score", "close", "setup_type", "market_ok", "breakout_ready",
            "sector", "earnings_warning", "rs_vs_benchmark_3m", "rs_vs_benchmark_6m",
            "pct_from_52w_high", "tight_close_range_10d", "pivot",
            "dollar_volume_20d", "volume_ratio_today", "atr_pct", "trend_strength",
            "entry", "stop", "risk_per_share", "position_size_shares", "notes"
        ]
        for key in ordered_keys:
            lines.append(f"{key}: {row.get(key, '')}")

        lines.append("")
        lines.append(str(row.get("explanation", "")))

        self.details_box.delete("1.0", "end")
        self.details_box.insert("1.0", "\n".join(lines))

    def open_chart(self):
        symbol = self._selected_symbol()
        if not symbol:
            messagebox.showinfo("No Selection", "Select a stock first.")
            return

        df = self.price_data.get(symbol)
        if df is None or df.empty:
            messagebox.showerror("No Chart Data", f"No price data available for {symbol}.")
            return

        match = self.results_df[self.results_df["symbol"] == symbol]
        explanation = "No explanation available."
        row_dict = {}
        if not match.empty:
            row_dict = match.iloc[0].to_dict()
            explanation = str(row_dict.get("explanation", explanation))

        ChartWindow(self, symbol, df, explanation, row_dict)

    def add_selected_to_watchlist(self):
        symbol = self._selected_symbol()
        if not symbol:
            messagebox.showinfo("No Selection", "Select a stock first.")
            return

        self.watchlist.append(symbol)
        self.watchlist = sorted(list(dict.fromkeys([s.upper() for s in self.watchlist])))
        Storage.save_watchlist(self.watchlist)
        self.refresh_watchlist_tab()
        messagebox.showinfo("Watchlist", f"Added {symbol} to watchlist.")

    def refresh_watchlist_tab(self):
        for item in self.watchlist_tree.get_children():
            self.watchlist_tree.delete(item)
        for symbol in self.watchlist:
            self.watchlist_tree.insert("", "end", values=(symbol,))

    def remove_selected_watchlist(self):
        selected = self.watchlist_tree.selection()
        if not selected:
            return

        symbols = []
        for item in selected:
            values = self.watchlist_tree.item(item, "values")
            if values:
                symbols.append(str(values[0]))

        self.watchlist = [s for s in self.watchlist if s not in symbols]
        Storage.save_watchlist(self.watchlist)
        self.refresh_watchlist_tab()

    def add_selected_to_journal(self):
        symbol = self._selected_symbol()
        if not symbol or self.results_df.empty:
            messagebox.showinfo("No Selection", "Select a stock first.")
            return

        row = self.results_df[self.results_df["symbol"] == symbol]
        if row.empty:
            return

        r = row.iloc[0].to_dict()
        journal_row = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "symbol": r.get("symbol", ""),
            "entry": r.get("entry", ""),
            "stop": r.get("stop", ""),
            "target": round(float(r.get("entry", 0)) * 1.12, 2) if r.get("entry", 0) else "",
            "shares": r.get("position_size_shares", ""),
            "setup_type": r.get("setup_type", ""),
            "score": r.get("score", ""),
            "status": "planned",
            "notes": r.get("notes", ""),
        }

        Storage.add_journal_row(journal_row)
        self.refresh_journal_tab()
        self.notebook.select(self.journal_tab)
        messagebox.showinfo("Journal", f"Added {symbol} to trade journal.")

    def refresh_journal_tab(self):
        for item in self.journal_tree.get_children():
            self.journal_tree.delete(item)

        df = Storage.load_journal()
        for _, row in df.iterrows():
            self.journal_tree.insert(
                "",
                "end",
                values=(
                    row.get("date", ""),
                    row.get("symbol", ""),
                    row.get("entry", ""),
                    row.get("stop", ""),
                    row.get("target", ""),
                    row.get("shares", ""),
                    row.get("setup_type", ""),
                    row.get("score", ""),
                    row.get("status", ""),
                    row.get("notes", ""),
                ),
            )


if __name__ == "__main__":
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    app = App()
    app.mainloop()