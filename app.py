from __future__ import annotations

import io
import json
import os
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

APP_TITLE = "Swing Scanner Pro X"
APP_TAGLINE = "Institutional-style swing trade discovery for retail traders."
SP500_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
BENCHMARKS = ["SPY", "VOO", "^GSPC"]

LOOKBACK = "1y"
MAX_WORKERS = 16
ENRICH_TOP_N = 20

MIN_PRICE = 10.0
MIN_DOLLAR_VOLUME = 20_000_000
A_PLUS_SCORE = 72

WATCHLIST_FILE = "watchlist.json"
JOURNAL_FILE = "trade_journal.csv"
SNAPSHOT_DIR = "chart_snapshots"

INFO_TEXT = (
    "### Scanner Controls\n"
    "- Run the full S&P 500 scan with one click.\n"
    "- Filter by score, breakout readiness, or sector.\n"
    "- Click a ticker in the results table to inspect it.\n"
    "- Save names to your Ideas list or Journal.\n\n"
    "### Ranking Logic\n"
    "The score rewards trend quality, relative strength, tight price action, "
    "liquidity, and proximity to a breakout pivot."
)


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
    target: float
    risk_per_share: float
    position_size_shares: int
    explanation: str
    notes: str


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
                "date",
                "symbol",
                "entry",
                "stop",
                "target",
                "shares",
                "setup_type",
                "score",
                "status",
                "notes",
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


class UniverseLoader:
    @staticmethod
    @st.cache_data(show_spinner=False, ttl=3600)
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


class MetaHelper:
    @staticmethod
    @st.cache_data(show_spinner=False, ttl=1800)
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
        return out.dropna()

    @staticmethod
    @st.cache_data(show_spinner=False, ttl=900)
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
    def download_data(symbols: List[str], log_list: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        unique = list(dict.fromkeys(symbols))

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
                        if log_list is not None:
                            log_list.append(f"Downloaded {symbol} ({done}/{total})")
                    else:
                        if log_list is not None:
                            log_list.append(f"No usable data for {symbol} ({done}/{total})")
                except Exception as e:
                    if log_list is not None:
                        log_list.append(f"Download failed for {symbol}: {e}")

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
    def classify_setup(close: float, pivot: float, ma21: float, ma50: float, tight_10d: float, pct_from_high: float) -> str:
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
    ) -> Tuple[float, float, float, float, int]:
        entry = round(max(close, pivot), 2)
        stop_candidate_1 = pivot * 0.97
        stop_candidate_2 = ma21 * 0.99
        stop = round(min(stop_candidate_1, stop_candidate_2), 2)

        if atr_pct > 0:
            atr_stop = close * (1 - min(0.06, max(0.02, atr_pct / 100.0)))
            stop = round(min(stop, atr_stop), 2)

        risk_per_share = round(max(0.01, entry - stop), 2)
        target = round(entry + (risk_per_share * 2.0), 2)
        max_risk_dollars = max(1.0, account_size * (risk_percent / 100.0))
        shares = int(max_risk_dollars // risk_per_share) if risk_per_share > 0 else 0
        return entry, stop, target, risk_per_share, max(0, shares)

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
        target: float,
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
        lines.append(f"- Suggested first target / sell zone: {target:.2f}")
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

        entry, stop, target, risk_per_share, shares = ScannerEngine.build_trade_plan(
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
            target=target,
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
            target=target,
            risk_per_share=risk_per_share,
            position_size_shares=shares,
            explanation=explanation,
            notes=", ".join(notes),
        )


def inject_dark_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(30,64,175,0.18), transparent 22%),
                radial-gradient(circle at top left, rgba(14,165,233,0.10), transparent 18%),
                linear-gradient(180deg, #08101c 0%, #0b1220 100%);
            color: #e5e7eb;
        }

        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0c1323 0%, #0a1120 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1540px;
        }

        h1, h2, h3, h4, h5, h6, p, label, div, span {
            color: #e5e7eb;
        }

        .scanner-card {
            background: linear-gradient(180deg, rgba(17,24,39,0.96) 0%, rgba(15,23,42,0.96) 100%);
            padding: 1rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.16);
        }

        .scanner-subtle {
            color: #9ca3af;
            font-size: 0.95rem;
        }

        .sspx-hero {
            background:
                linear-gradient(135deg, rgba(15,23,42,0.96) 0%, rgba(17,24,39,0.96) 58%, rgba(23,37,84,0.96) 100%);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 24px;
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 18px 50px rgba(0,0,0,0.22);
        }

        .sspx-chip {
            display:inline-block;
            padding:8px 12px;
            border-radius:999px;
            background:#111827;
            border:1px solid rgba(255,255,255,0.08);
            color:#d1d5db;
            font-size:0.84rem;
            font-weight:700;
            margin-right:8px;
            margin-bottom:8px;
        }

        .sspx-locked-card,
        .sspx-panel,
        .sspx-kpi {
            background: linear-gradient(180deg, rgba(17,24,39,0.96) 0%, rgba(15,23,42,0.96) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.16);
        }

        .sspx-locked-card {
            padding: 18px;
            min-height: 170px;
        }

        .sspx-panel {
            padding: 16px;
            margin-bottom: 16px;
        }

        .sspx-kpi {
            padding: 14px 16px;
            min-height: 96px;
        }

        .sspx-section-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 10px;
        }

        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            background-color: #0f172a !important;
            color: #e5e7eb !important;
            border-radius: 12px !important;
        }

        .stSlider [data-baseweb="slider"] {
            padding-top: 8px;
            padding-bottom: 8px;
        }

        .stButton button,
        .stDownloadButton button {
            background: linear-gradient(180deg, #1f2937 0%, #182130 100%) !important;
            color: #f9fafb !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 12px !important;
            min-height: 42px !important;
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            border-color: rgba(147,197,253,0.45) !important;
            background: #273449 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: #111827;
            border-radius: 12px 12px 0 0;
            padding: 12px 18px;
        }

        .streamlit-expanderHeader {
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%) !important;
            color: #f9fafb !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 16px !important;
        }

        [data-testid="stExpander"] {
            background: linear-gradient(180deg, rgba(17,24,39,0.96) 0%, rgba(15,23,42,0.96) 100%) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 18px !important;
            overflow: hidden !important;
            margin-bottom: 16px !important;
        }

        [data-testid="stExpander"] details,
        [data-testid="stExpander"] details summary {
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%) !important;
            color: #f9fafb !important;
        }

        [data-testid="stExpander"] details summary {
            border-bottom: 1px solid rgba(255,255,255,0.08) !important;
            padding: 0.75rem 1rem !important;
        }

        [data-testid="stExpander"] details summary:hover {
            background: linear-gradient(180deg, #182130 0%, #111827 100%) !important;
        }

        [data-testid="stExpander"] svg,
        [data-testid="stExpander"] summary svg {
            fill: #e5e7eb !important;
            color: #e5e7eb !important;
        }

        [data-testid="stExpanderDetails"] {
            background: linear-gradient(180deg, rgba(17,24,39,0.96) 0%, rgba(15,23,42,0.96) 100%) !important;
            color: #e5e7eb !important;
            padding-top: 0.75rem !important;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.16);
        }

        [data-testid="stMetricLabel"] {
            color: #9ca3af !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"] {
            color: #f9fafb !important;
            font-size: 1.4rem !important;
            font-weight: 800 !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            overflow: hidden;
        }

        .sspx-divider {
            height: 1px;
            background: rgba(255,255,255,0.08);
            margin: 8px 0 16px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(label: str, kind: str = "neutral") -> str:
    colors = {
        "green": ("#052e16", "#86efac"),
        "red": ("#3f0d0d", "#fca5a5"),
        "yellow": ("#3b2f0e", "#fde68a"),
        "blue": ("#0c2440", "#93c5fd"),
        "neutral": ("#1f2937", "#d1d5db"),
        "purple": ("#2e1065", "#c4b5fd"),
    }
    bg, fg = colors.get(kind, colors["neutral"])
    return (
        f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;"
        f"background:{bg};color:{fg};font-size:0.82rem;font-weight:700;"
        f"margin-right:8px;margin-bottom:6px;'>{label}</span>"
    )


def metric_card(label: str, value, positive: Optional[bool] = None) -> str:
    if positive is True:
        border = "#166534"
        value_color = "#86efac"
    elif positive is False:
        border = "#7f1d1d"
        value_color = "#fca5a5"
    else:
        border = "rgba(255,255,255,0.08)"
        value_color = "#f9fafb"

    return f"""
    <div style="
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border: 1px solid {border};
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
    ">
        <div style="font-size:0.82rem;color:#9ca3af;margin-bottom:6px;">{label}</div>
        <div style="font-size:1.25rem;font-weight:800;color:{value_color};">{value}</div>
    </div>
    """


def info_tile(title: str, value: str, subtitle: str = "") -> str:
    return f"""
    <div class="sspx-kpi">
        <div style="font-size:0.82rem;color:#9ca3af;margin-bottom:7px;">{title}</div>
        <div style="font-size:1.35rem;font-weight:850;color:#f8fafc;line-height:1.1;">{value}</div>
        <div style="font-size:0.82rem;color:#93c5fd;margin-top:8px;">{subtitle}</div>
    </div>
    """


def panel_open(title: str = "", subtitle: str = "") -> str:
    heading = ""
    if title:
        heading += f'<div class="sspx-section-title">{title}</div>'
    if subtitle:
        heading += f'<div style="font-size:0.9rem;color:#9ca3af;margin-bottom:10px;">{subtitle}</div>'
    return f'<div class="sspx-panel">{heading}'


def panel_close() -> str:
    return ""


def render_scoring_explainer():
    with st.expander("ℹ️ How picks are chosen"):
        st.markdown(
            f"""
### How the scanner picks stocks

This scanner finds **liquid swing trade setups** with:
- strong trends
- relative strength vs the market
- tight price action
- proximity to breakout levels

### Score breakdown

**Trend**
- price above 10 / 21 / 50 / 200-day moving averages
- stronger moving average alignment scores better

**Strength**
- 3M and 6M outperformance vs the benchmark
- trading close to 52-week highs

**Setup quality**
- tighter 10-day price action
- close to pivot / breakout zone
- constructive setup types rank higher

**Volume & liquidity**
- higher 20-day dollar volume
- above-average current volume
- controlled ATR / volatility

### What gets filtered out

Names are usually excluded if they:
- trade below ${MIN_PRICE}
- have average dollar volume below ${MIN_DOLLAR_VOLUME:,.0f}
- do not have enough clean price history

### What breakout-ready means

A name becomes breakout-ready when it is trading close enough to its pivot that
a valid breakout could matter.

### What risk does

**Risk does not affect the score.**

Risk is only used for **trade planning**:
- Account Size = total account
- Risk % Per Trade = max amount you are willing to lose on one trade
- that max dollar risk is divided by **risk per share**

### Position size logic

Example:
- Account = $10,000
- Risk = 1% = $100
- Risk per share = $2

Estimated position size = **50 shares**

### Bottom line

This is a **ranking + planning tool**, not an automatic buy signal.
Higher score means a cleaner setup for this style, not a guaranteed winner.
"""
        )


def ensure_state() -> None:
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = Storage.load_watchlist()
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if "price_data" not in st.session_state:
        st.session_state.price_data = {}
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = None
    if "selected_benchmark" not in st.session_state:
        st.session_state.selected_benchmark = "Auto"
    if "min_score_filter" not in st.session_state:
        st.session_state.min_score_filter = 0




def ensure_results_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    out = df.copy()
    defaults = {
        "technical_score": out["score"] if "score" in out.columns else 0,
        "news_score": 0,
        "breakout_ready": False,
        "rs_vs_benchmark_3m": 0.0,
        "volume_ratio_today": 0.0,
        "confidence_score": out["score"] if "score" in out.columns else 0,
        "confidence_label": "Watch",
        "sector": "Unknown",
        "setup_type": "Mixed Setup",
        "close": 0.0,
        "entry": 0.0,
        "stop": 0.0,
        "target": 0.0,
        "position_size_shares": 0,
        "news_headline": "No recent headline found",
        "catalyst_tags": "",
        "why_pick": "",
        "earnings_warning": "Unknown",
        "market_ok": False,
    }

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    if "technical_score" in out.columns and "score" in out.columns:
        out["technical_score"] = out["technical_score"].fillna(out["score"])
    if "confidence_score" in out.columns and "score" in out.columns:
        out["confidence_score"] = out["confidence_score"].fillna(out["score"])
    if "confidence_label" in out.columns:
        out["confidence_label"] = out["confidence_label"].replace("", np.nan).fillna("Watch")
    return out

def make_sparkline_data_uri(price_df: Optional[pd.DataFrame]) -> str:
    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return ""

    closes = price_df["Close"].tail(30).dropna().tolist()
    if len(closes) < 2:
        return ""

    width = 220
    height = 60
    pad = 4

    min_v = min(closes)
    max_v = max(closes)
    span = max(max_v - min_v, 1e-9)

    points = []
    for i, v in enumerate(closes):
        x = pad + (i / (len(closes) - 1)) * (width - 2 * pad)
        y = height - pad - ((v - min_v) / span) * (height - 2 * pad)
        points.append(f"{x:.1f},{y:.1f}")

    up = closes[-1] >= closes[0]
    line_color = "#86efac" if up else "#fca5a5"

    svg = f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <polyline
            fill="none"
            stroke="{line_color}"
            stroke-width="3"
            points="{' '.join(points)}"
            stroke-linecap="round"
            stroke-linejoin="round"
        />
    </svg>
    """
    return "data:image/svg+xml;utf8," + quote(svg)


def run_scan_logic(
    symbols: List[str],
    account_size: float,
    risk_percent: float,
    selected_benchmark: str = "Auto",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str], List[str]]:
    logs: List[str] = []
    alerts: List[str] = []

    universe = list(dict.fromkeys([s.strip().upper() for s in symbols if s.strip()]))
    price_data = ScannerEngine.download_data(universe, log_list=logs)

    bench_df = None
    bench_used = None

    if selected_benchmark != "Auto":
        logs.append(f"Using chosen benchmark: {selected_benchmark}")
        bench_data = ScannerEngine.download_data([selected_benchmark])
        candidate = bench_data.get(selected_benchmark)
        if candidate is not None and not candidate.empty and len(candidate) >= 130:
            bench_df = candidate
            bench_used = selected_benchmark
            price_data[selected_benchmark] = candidate

    if bench_df is None:
        for bench in BENCHMARKS:
            logs.append(f"Trying benchmark: {bench}")
            bench_data = ScannerEngine.download_data([bench])
            candidate = bench_data.get(bench)
            if candidate is not None and not candidate.empty and len(candidate) >= 130:
                bench_df = candidate
                bench_used = bench
                price_data[bench] = candidate
                break

    if bench_used:
        logs.append(f"Using benchmark: {bench_used}")
    else:
        logs.append("No benchmark available. Falling back to zero RS values.")

    market_ok = ScannerEngine.market_health(bench_df)
    results: List[ScanResult] = []

    for idx, symbol in enumerate(universe, start=1):
        logs.append(f"Scoring {symbol} ({idx}/{len(universe)})")
        df = price_data.get(symbol)
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
            logs.append(f"Scoring failed for {symbol}: {e}")

    out = pd.DataFrame([asdict(r) for r in results]) if results else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(
            by=["score", "rs_vs_benchmark_3m", "pct_from_52w_high"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        top_symbols = out["symbol"].head(ENRICH_TOP_N).tolist()
        logs.append(f"Enriching top {len(top_symbols)} names with sector / earnings...")
        for sym in top_symbols:
            sector, earnings_warning = MetaHelper.get_sector_and_earnings(sym)
            out.loc[out["symbol"] == sym, "sector"] = sector
            out.loc[out["symbol"] == sym, "earnings_warning"] = earnings_warning
            if "Earnings in" in earnings_warning:
                current_expl = str(out.loc[out["symbol"] == sym, "explanation"].iloc[0])
                if "Earnings are coming soon" not in current_expl:
                    current_expl += f"\n\nThings to watch:\n- {earnings_warning}, which adds event risk."
                    out.loc[out["symbol"] == sym, "explanation"] = current_expl

        alert_set = set()
        for _, row in out.head(20).iterrows():
            if row.get("breakout_ready", False):
                alert_set.add(f"{row['symbol']}: near pivot / breakout-ready")
            if row.get("volume_ratio_today", 0) >= 1.5:
                alert_set.add(f"{row['symbol']}: volume surge ({row['volume_ratio_today']}x)")
            if "Earnings in" in str(row.get("earnings_warning", "")):
                alert_set.add(f"{row['symbol']}: {row['earnings_warning']}")

        alerts = sorted(alert_set)

    if not alerts:
        alerts = ["No notable alerts from current results."]

    return out, price_data, logs, alerts


def build_chart(symbol: str, df: pd.DataFrame, result_row: dict):
    fig, (ax, rs_ax) = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    fig.patch.set_facecolor("#0f172a")

    plot_df = df.tail(180).copy()
    x = range(len(plot_df))

    ax.set_facecolor("#0f172a")
    rs_ax.set_facecolor("#0f172a")

    ax.plot(x, plot_df["Close"], label="Close")
    ax.plot(x, plot_df["Close"].rolling(10).mean(), label="MA10")
    ax.plot(x, plot_df["Close"].rolling(21).mean(), label="MA21")
    ax.plot(x, plot_df["Close"].rolling(50).mean(), label="MA50")
    if len(plot_df) >= 200:
        ax.plot(x, plot_df["Close"].rolling(200).mean(), label="MA200")

    pivot = result_row.get("pivot")
    if pivot:
        ax.axhline(pivot, linestyle="--", label="Pivot")
    recent_high = plot_df["High"].max()
    ax.axhline(recent_high, linestyle=":", label="Recent High")

    ax.set_title(symbol, color="white")
    ax.set_xlabel("Days", color="white")
    ax.set_ylabel("Price", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.25)
    ax.legend()

    rs_line = (plot_df["Close"] / plot_df["Close"].iloc[0]) * 100
    rs_ax.plot(x, rs_line, label="Relative Performance Proxy")
    rs_ax.set_title("Strength Line", color="white")
    rs_ax.set_xlabel("Days", color="white")
    rs_ax.set_ylabel("Index=100", color="white")
    rs_ax.tick_params(colors="white")
    rs_ax.grid(True, alpha=0.25)
    rs_ax.legend()

    return fig


def save_snapshot(symbol: str, df: pd.DataFrame, row: dict) -> str:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor("#0f172a")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0f172a")

    plot_df = df.tail(180)
    ax.plot(plot_df["Close"], label="Close")
    ax.plot(plot_df["Close"].rolling(21).mean(), label="MA21")
    ax.plot(plot_df["Close"].rolling(50).mean(), label="MA50")
    if len(plot_df) >= 200:
        ax.plot(plot_df["Close"].rolling(200).mean(), label="MA200")
    if row.get("pivot"):
        ax.axhline(row["pivot"], linestyle="--", label="Pivot")

    ax.set_title(f"{symbol} | Score: {row.get('score', '')}", color="white")
    ax.tick_params(colors="white")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def render_app_header():
    st.markdown(
        f"""
        <div class="sspx-hero">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:2.15rem;font-weight:900;color:#f8fafc;line-height:1.05;letter-spacing:-0.02em;">
                        {APP_TITLE}
                    </div>
                    <div style="font-size:1rem;color:#93c5fd;margin-top:8px;">
                        {APP_TAGLINE}
                    </div>
                    <div style="margin-top:14px;">
                        <span class="sspx-chip">Pro Scanner</span>
                        <span class="sspx-chip">Institutional Ranking Engine</span>
                        <span class="sspx-chip">AI-Assisted Trade Planning</span>
                    </div>
                </div>
                <div style="min-width:260px;text-align:right;">
                    <div class="sspx-chip">Universe: S&amp;P 500</div>
                    <div class="sspx-chip">Style: Momentum / Breakout</div>
                    <div class="sspx-chip">Product Tier: Professional</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_status_bar(results_df: pd.DataFrame):
    market_label = "Waiting for scan"
    market_kind = "neutral"

    if not results_df.empty and "market_ok" in results_df.columns:
        market_ok = bool(results_df["market_ok"].mode().iloc[0])
        if market_ok:
            market_label = "Market Status: Healthy"
            market_kind = "green"
        else:
            market_label = "Market Status: Caution"
            market_kind = "yellow"

    c1, c2, c3 = st.columns([1, 1, 1.3])
    with c1:
        st.markdown(badge_html(market_label, market_kind), unsafe_allow_html=True)
    with c2:
        st.markdown(badge_html("Universe: S&P 500", "blue"), unsafe_allow_html=True)
    with c3:
        st.markdown(badge_html("Mode: Pro Momentum Scanner", "purple"), unsafe_allow_html=True)


def render_scan_kpis(results_df: pd.DataFrame):
    if results_df.empty:
        return

    total_names = len(results_df)
    avg_score = round(float(results_df["score"].mean()), 1) if "score" in results_df.columns else 0
    breakout_count = int(results_df["breakout_ready"].sum()) if "breakout_ready" in results_df.columns else 0
    a_plus_count = int((results_df["score"] >= A_PLUS_SCORE).sum()) if "score" in results_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(info_tile("Names in view", str(total_names), "Filtered active results"), unsafe_allow_html=True)
    with c2:
        st.markdown(info_tile("Average score", f"{avg_score}", "Across displayed names"), unsafe_allow_html=True)
    with c3:
        st.markdown(info_tile("Breakout-ready", str(breakout_count), "Near pivot / trigger zone"), unsafe_allow_html=True)
    with c4:
        st.markdown(info_tile("A+ setups", str(a_plus_count), f"Score {A_PLUS_SCORE}+"), unsafe_allow_html=True)



def get_pick_badges(row: dict) -> List[str]:
    badges = []
    if int(row.get("score", 0)) >= A_PLUS_SCORE:
        badges.append(badge_html("🟢 A+ setup", "green"))
    if bool(row.get("breakout_ready", False)):
        badges.append(badge_html("🔥 Breakout ready", "red"))
    if int(row.get("news_score", 0)) >= 8:
        badges.append(badge_html("📰 News confirmed", "green"))
    elif int(row.get("news_score", 0)) <= -4:
        badges.append(badge_html("📰 Headline risk", "yellow"))
    if str(row.get("confidence_label", "")) == "High":
        badges.append(badge_html("🎯 High confidence", "green"))
    elif str(row.get("confidence_label", "")) == "Medium":
        badges.append(badge_html("🎯 Medium confidence", "blue"))
    if "Earnings in" in str(row.get("earnings_warning", "")):
        badges.append(badge_html(f"⚠️ {row['earnings_warning']}", "yellow"))
    return badges


def build_clean_why_pick_items(why_pick: str, explanation: str = "") -> List[str]:
    raw = str(why_pick or "").strip()
    items: List[str] = []

    if raw:
        parts = [part.strip(" •-") for part in raw.split("|") if part.strip()]
        for part in parts:
            clean = str(part).strip()
            if clean and clean.lower() != "no explanation available yet":
                items.append(clean)

    if not items:
        exp = str(explanation or "")
        extracted: List[str] = []
        mapping = [
            ("benchmark trend is healthy", "Supportive market backdrop"),
            ("above the 10-day and 21-day", "Above 10/21-day moving averages"),
            ("above the 50-day average", "Above 50-day moving average"),
            ("above the 200-day average", "Above 200-day moving average"),
            ("very close to its 52-week high", "Trading near 52-week highs"),
            ("reasonably close to its 52-week high", "Trading near 52-week highs"),
            ("strongly outperformed the market over the last 3 months", "Strong 3-month relative strength"),
            ("outperformed the market over the last 3 months", "Strong 3-month relative strength"),
            ("outperformed over 6 months", "Strong 6-month relative strength"),
            ("price has tightened over the last 10 days", "Tight recent price action"),
            ("close to its pivot area", "Near breakout / pivot zone"),
            ("volume is clearly above normal today", "Volume confirming the move"),
            ("volume is above normal today", "Volume above normal"),
            ("liquidity is strong", "Strong liquidity"),
        ]

        for line in exp.splitlines():
            s = line.strip()
            if not s.startswith("-"):
                continue
            content = s.lstrip("-").strip()
            lower = content.lower()
            if lower.startswith((
                "setup type:",
                "scanner score:",
                "market health filter:",
                "suggested entry area:",
                "suggested stop area:",
                "suggested first target / sell zone:",
                "approx position size:",
                "wait for clean price behavior",
                "use this as a shortlist tool",
            )):
                continue
            if lower in {"things to watch:", "trade planning ideas:"}:
                continue
            if any(lower.startswith(prefix) for prefix in ["the overall market trend is not ideal", "recent price action is a bit wide", "earnings are coming soon"]):
                continue

            mapped = None
            for needle, replacement in mapping:
                if needle in lower:
                    mapped = replacement
                    break
            if mapped:
                extracted.append(mapped)

        seen = set()
        items = []
        for item in extracted:
            if item not in seen:
                seen.add(item)
                items.append(item)

    if not items:
        fallback_checks = [
            (float(explanation.count("")) if False else None),
        ]

    if not items:
        items = ["Strong technical structure", "Defined risk / reward plan"]

    return items[:5]


def render_why_pick_list(why_pick: str, explanation: str = "") -> str:
    items = build_clean_why_pick_items(why_pick, explanation)
    return "".join(
        f"<li style='margin-bottom:6px;color:#cbd5e1;'>{item}</li>"
        for item in items[:5]
    )


def render_why_pick_inline(why_pick: str, explanation: str = "") -> str:
    items = build_clean_why_pick_items(why_pick, explanation)
    return " • ".join(items[:4])


def render_pick_detail_panel(row: dict, price_data: Dict[str, pd.DataFrame], rank_label: str):
    symbol = str(row.get("symbol", ""))
    price_df = price_data.get(symbol)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 12px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.16);
        ">
            <div style="font-size:0.82rem;color:#93c5fd;margin-bottom:8px;">{rank_label} details</div>
            <div style="display:grid;grid-template-columns:repeat(2,minmax(120px,1fr));gap:10px 18px;color:#d1d5db;font-size:0.95rem;line-height:1.55;">
                <div><b>Entry:</b> {row.get('entry')}</div>
                <div><b>Stop:</b> {row.get('stop')}</div>
                <div><b>Target:</b> {row.get('target')}</div>
                <div><b>Position Size:</b> {row.get('position_size_shares')} shares</div>
                <div><b>Sector:</b> {row.get('sector')}</div>
                <div><b>Confidence:</b> {row.get('confidence_score')} ({row.get('confidence_label')})</div>
                <div><b>News:</b> {row.get('news_sentiment')} ({row.get('news_score')})</div>
                <div><b>3M RS:</b> {row.get('rs_vs_benchmark_3m')}%</div>
                <div><b>Volume Ratio:</b> {row.get('volume_ratio_today')}x</div>
                <div style="grid-column:1 / -1;"><b>Catalysts:</b> {row.get('catalyst_tags') or 'None'}</div>
                <div style="grid-column:1 / -1;"><b>Headline:</b> {row.get('news_headline') or 'No recent headline found'}</div>
            </div>
            <div style="margin-top:16px;">
                <div style="font-size:0.85rem;color:#9ca3af;margin-bottom:8px;">Why this pick</div>
                <ul style="margin:0;padding-left:18px;">
                    {render_why_pick_list(row.get('why_pick', ''), row.get('explanation', ''))}
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button(f"⭐ Save {symbol} to Watchlist", key=f"ideas_{rank_label}_{symbol}", use_container_width=True):
            st.session_state.watchlist.append(symbol)
            st.session_state.watchlist = sorted(list(dict.fromkeys([s.upper() for s in st.session_state.watchlist])))
            Storage.save_watchlist(st.session_state.watchlist)
            st.success(f"Added {symbol}")

    with action_cols[1]:
        if st.button(f"Add {symbol} to Journal", key=f"journal_{rank_label}_{symbol}", use_container_width=True):
            journal_row = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "symbol": row.get("symbol", ""),
                "entry": row.get("entry", ""),
                "stop": row.get("stop", ""),
                "target": row.get("target", ""),
                "shares": row.get("position_size_shares", ""),
                "setup_type": row.get("setup_type", ""),
                "score": row.get("score", ""),
                "status": "planned",
                "notes": f"{row.get('notes', '')} | news={row.get('news_sentiment', 'Neutral')} ({row.get('news_score', 0)})",
            }
            Storage.add_journal_row(journal_row)
            st.success(f"Added {symbol} to journal")

    with action_cols[2]:
        if price_df is not None and not price_df.empty:
            if st.button(f"Save {symbol} Snapshot", key=f"snap_{rank_label}_{symbol}", use_container_width=True):
                path = save_snapshot(symbol, price_df, row)
                st.success(f"Saved to {path}")
        else:
            st.button("No Chart Data", key=f"nodata_{rank_label}_{symbol}", use_container_width=True, disabled=True)

    if price_df is not None and not price_df.empty:
        fig = build_chart(symbol, price_df, row)
        st.pyplot(fig, clear_figure=True)

    st.markdown("#### Trade Notes")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 16px;
            color: #e5e7eb;
            white-space: pre-wrap;
            line-height: 1.55;
            box-shadow: 0 10px 28px rgba(0,0,0,0.16);
            margin-bottom: 10px;
        ">{row.get('explanation', '')}</div>
        """,
        unsafe_allow_html=True,
    )



def render_feed_pick_card(row: dict, price_data: Dict[str, pd.DataFrame], rank_number: int, hero: bool = False):
    symbol = str(row.get("symbol", ""))
    spark_uri = make_sparkline_data_uri(price_data.get(symbol))
    badges = get_pick_badges(row)
    badges.append(badge_html("Top Pick", "purple") if hero else badge_html(f"Pick #{rank_number}", "purple"))

    title = "Top Pick of the Day" if hero else f"Pick #{rank_number}"
    confidence_label = str(row.get("confidence_label", ""))
    if confidence_label == "High":
        confidence_text = "High confidence"
    elif confidence_label == "Medium":
        confidence_text = "Medium confidence"
    else:
        confidence_text = "Tracking"

    setup_type = str(row.get("setup_type", "Mixed Setup"))
    score_value = row.get("score", 0)
    target_value = row.get("target", 0)
    close_value = row.get("close", 0)
    rs_value = row.get("rs_vs_benchmark_3m", 0)

    st.markdown(
        f"""
        <div style="
            background: {'linear-gradient(135deg, #081425 0%, #0f172a 48%, #172554 100%)' if hero else 'linear-gradient(180deg, #0f172a 0%, #0c1527 100%)'};
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: {'26px' if hero else '20px'};
            padding: {'20px' if hero else '16px'};
            margin-bottom: 12px;
            box-shadow: {'0 22px 60px rgba(0,0,0,0.26)' if hero else '0 14px 34px rgba(0,0,0,0.20)'};
        ">
            <div style="font-size:0.86rem;color:#93c5fd;font-weight:800;margin-bottom:8px;letter-spacing:0.03em;">{title}</div>
            <div style="font-size:{'2.5rem' if hero else '1.6rem'};font-weight:950;color:#f8fafc;margin-bottom:12px;letter-spacing:-0.03em;line-height:1;">
                {symbol}
            </div>
            <div style="margin-bottom:12px;">{''.join(badges)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_cols = st.columns([1.15, 1.15, 1.15, 1.1] if hero else [1, 1, 1, 1])
    with info_cols[0]:
        st.markdown(info_tile("Score", f"{score_value}"), unsafe_allow_html=True)
    with info_cols[1]:
        st.markdown(info_tile("Setup", f"{setup_type}"), unsafe_allow_html=True)
    with info_cols[2]:
        st.markdown(info_tile("Target", f"{target_value}"), unsafe_allow_html=True)
    with info_cols[3]:
        st.markdown(info_tile("Confidence", confidence_text), unsafe_allow_html=True)

    meta_left, meta_right = st.columns([1.35, 0.65])
    with meta_left:
        st.markdown(
            f"""
            <div class="sspx-panel" style="margin-top:4px;">
                <div style="display:flex;gap:18px;flex-wrap:wrap;color:#cbd5e1;font-size:0.95rem;">
                    <div><span style="color:#94a3b8;">Close</span> <span style="font-weight:800;color:#f8fafc;">{close_value}</span></div>
                    <div><span style="color:#94a3b8;">3M RS</span> <span style="font-weight:800;color:#f8fafc;">{rs_value}%</span></div>
                    <div><span style="color:#94a3b8;">Confidence</span> <span style="font-weight:800;color:#f8fafc;">{confidence_text}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with meta_right:
        st.markdown('<div style="font-size:0.8rem;color:#9ca3af;margin:6px 0 8px 0;">' + ('30-day momentum' if hero else '30-day sparkline') + '</div>', unsafe_allow_html=True)
        if spark_uri:
            st.markdown(
                f"""
                <div style="background:#09111f;border:1px solid rgba(255,255,255,0.08);border-radius:18px;padding:12px;">
                    <img src="{spark_uri}" style="width:100%;height:{96 if hero else 76}px;display:block;" />
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="sspx-panel">No sparkline</div>', unsafe_allow_html=True)

    quick_label = "Tap for quick view" if hero else f"Tap for Pick #{rank_number} quick view"
    detail_key = f"show_breakdown_{symbol}_{rank_number}_{'hero' if hero else 'feed'}"
    with st.expander(quick_label, expanded=hero):
        q1, q2 = st.columns(2)
        with q1:
            st.markdown(metric_card("Score", row.get("score")), unsafe_allow_html=True)
            st.markdown(metric_card("Confidence", row.get("confidence_label") or "Tracking"), unsafe_allow_html=True)
            st.markdown(metric_card("Close", row.get("close")), unsafe_allow_html=True)
        with q2:
            st.markdown(metric_card("Setup", row.get("setup_type")), unsafe_allow_html=True)
            st.markdown(metric_card("Target", row.get("target")), unsafe_allow_html=True)
            st.markdown(metric_card("3M RS", f"{row.get('rs_vs_benchmark_3m')}%"), unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="sspx-panel" style="margin-top:10px;">
                <div><b>Tags:</b> {row.get('catalyst_tags') or 'None'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_label = "Show full breakdown" if hero else f"Show Pick #{rank_number} full breakdown"
        detail_state_key = f"{detail_key}_open"
        button_key = f"{detail_key}_button"
        if st.button(btn_label, key=button_key, use_container_width=True):
            st.session_state[detail_state_key] = not st.session_state.get(detail_state_key, False)
        if st.session_state.get(detail_state_key, False):
            render_pick_detail_panel(row, price_data, "Top Pick" if hero else f"Pick #{rank_number}")


def render_top_pick_hero(results_df: pd.DataFrame, price_data: Dict[str, pd.DataFrame]):
    results_df = ensure_results_schema(results_df)
    if results_df.empty:
        return

    hero_df = results_df.sort_values(
        by=["score", "technical_score", "breakout_ready", "rs_vs_benchmark_3m", "volume_ratio_today"],
        ascending=[False, False, False, False, False],
    ).head(1)

    if hero_df.empty:
        return

    row = hero_df.iloc[0].to_dict()
    render_feed_pick_card(row, price_data, rank_number=1, hero=True)


def render_ranked_feed(results_df: pd.DataFrame, price_data: Dict[str, pd.DataFrame], top_n: int = 10):
    results_df = ensure_results_schema(results_df)
    if results_df.empty:
        return

    st.markdown("#### Ranked Picks")

    ranked = results_df.sort_values(
        by=["score", "technical_score", "rs_vs_benchmark_3m", "volume_ratio_today"],
        ascending=[False, False, False, False],
    ).head(top_n).reset_index(drop=True)

    for idx, row in ranked.iterrows():
        render_feed_pick_card(row.to_dict(), price_data, rank_number=idx + 1, hero=False)

def render_stock_summary_card(row: dict):
    badges = []
    if row["score"] >= A_PLUS_SCORE:
        badges.append(badge_html("🟢 A+ setup", "green"))
    if bool(row.get("breakout_ready", False)):
        badges.append(badge_html("🔥 Breakout ready", "red"))
    if "Earnings in" in str(row.get("earnings_warning", "")):
        badges.append(badge_html(f"⚠️ {row['earnings_warning']}", "yellow"))
    if bool(row.get("market_ok", False)):
        badges.append(badge_html("Market healthy", "blue"))

    st.markdown(panel_open(f"{row['symbol']} Summary", "Selected idea at a glance"), unsafe_allow_html=True)
    st.markdown("".join(badges), unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sspx-divider"></div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px 16px;font-size:0.95rem;color:#d1d5db;">
            <div><b>Score</b><br>{row['score']}</div>
            <div><b>Setup</b><br>{row['setup_type']}</div>
            <div><b>Sector</b><br>{row['sector']}</div>
            <div><b>Close</b><br>{row['close']}</div>
            <div><b>Entry</b><br>{row['entry']}</div>
            <div><b>Stop</b><br>{row['stop']}</div>
            <div><b>Target</b><br>{row.get('target', '')}</div>
            <div><b>Risk / Share</b><br>{row['risk_per_share']}</div>
            <div><b>Position Size</b><br>{row['position_size_shares']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(panel_close(), unsafe_allow_html=True)


def render_stock_details_cards(row: dict):
    st.markdown(f"#### {row['symbol']} Overview")

    row1 = st.columns(4)
    with row1[0]:
        st.markdown(metric_card("Score", row["score"], row["score"] >= A_PLUS_SCORE), unsafe_allow_html=True)
    with row1[1]:
        st.markdown(metric_card("Close", row["close"]), unsafe_allow_html=True)
    with row1[2]:
        st.markdown(metric_card("Setup", row["setup_type"]), unsafe_allow_html=True)
    with row1[3]:
        st.markdown(
            metric_card("Breakout Ready", "Yes" if row["breakout_ready"] else "No", row["breakout_ready"]),
            unsafe_allow_html=True,
        )

    row2 = st.columns(4)
    with row2[0]:
        rs3 = float(row.get("rs_vs_benchmark_3m", 0))
        st.markdown(metric_card("RS vs Benchmark (3M)", f"{rs3}%", rs3 >= 0), unsafe_allow_html=True)
    with row2[1]:
        rs6 = float(row.get("rs_vs_benchmark_6m", 0))
        st.markdown(metric_card("RS vs Benchmark (6M)", f"{rs6}%", rs6 >= 0), unsafe_allow_html=True)
    with row2[2]:
        pct_high = float(row.get("pct_from_52w_high", 0))
        st.markdown(metric_card("% From 52W High", f"{pct_high}%", pct_high >= -10), unsafe_allow_html=True)
    with row2[3]:
        vol_ratio = float(row.get("volume_ratio_today", 0))
        st.markdown(metric_card("Volume Ratio", f"{vol_ratio}x", vol_ratio >= 1.1), unsafe_allow_html=True)

    row3 = st.columns(4)
    with row3[0]:
        st.markdown(metric_card("Entry", row.get("entry", "")), unsafe_allow_html=True)
    with row3[1]:
        st.markdown(metric_card("Stop", row.get("stop", "")), unsafe_allow_html=True)
    with row3[2]:
        st.markdown(metric_card("Risk / Share", row.get("risk_per_share", "")), unsafe_allow_html=True)
    with row3[3]:
        st.markdown(metric_card("Position Size", row.get("position_size_shares", "")), unsafe_allow_html=True)

    row4 = st.columns(4)
    with row4[0]:
        atr_pct = float(row.get("atr_pct", 0))
        st.markdown(metric_card("ATR %", f"{atr_pct}%", atr_pct <= 5), unsafe_allow_html=True)
    with row4[1]:
        tight = float(row.get("tight_close_range_10d", 0))
        st.markdown(metric_card("10D Tightness", f"{tight}%", tight <= 4), unsafe_allow_html=True)
    with row4[2]:
        trend = float(row.get("trend_strength", 0))
        st.markdown(metric_card("Trend Strength", trend, trend >= 0), unsafe_allow_html=True)
    with row4[3]:
        st.markdown(metric_card("Sector", row.get("sector", "")), unsafe_allow_html=True)

    st.markdown("#### Trade Notes")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px;
            color: #e5e7eb;
            white-space: pre-wrap;
            line-height: 1.55;
            box-shadow: 0 10px 28px rgba(0,0,0,0.16);
        ">{row.get('explanation', '')}</div>
        """,
        unsafe_allow_html=True,
    )


def render_premium_modules():
    st.markdown("#### Premium Modules")

    cols = st.columns(4)
    cards = [
        ("AI Market Regime", "Pro", "Dynamic market condition model with trend, breadth, and volatility overlay.", "Available soon"),
        ("Earnings Risk Map", "Pro", "Visualize upcoming event risk across your active setups and watchlist.", "Locked"),
        ("Relative Strength Leaders", "AI", "AI-ranked sector and stock leadership map for faster idea generation.", "Locked"),
        ("Smart Signal Engine", "Pro", "Prioritized alert stack for breakout readiness, volume expansion, and follow-through.", "Locked"),
    ]

    for col, (title, badge, text, status) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="sspx-locked-card">
                    <div style="margin-bottom:10px;">
                        {badge_html(badge, "purple")}
                    </div>
                    <div style="font-size:1.05rem;font-weight:800;color:#f8fafc;margin-bottom:8px;">
                        {title}
                    </div>
                    <div style="font-size:0.92rem;color:#cbd5e1;line-height:1.55;margin-bottom:14px;">
                        {text}
                    </div>
                    <div style="font-size:0.82rem;color:#93c5fd;font-weight:700;">
                        {status}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_results_table(results_df: pd.DataFrame):
    view_cols = [
        "symbol",
        "score",
        "setup_type",
        "sector",
    ]

    table_df = results_df[view_cols].copy()

    ticker_cell = JsCode(
        """
        class TickerCellRenderer {
            init(params) {
                this.params = params;
                this.eGui = document.createElement('span');
                this.eGui.innerText = params.value;
                this.eGui.style.color = '#93c5fd';
                this.eGui.style.fontWeight = '700';
                this.eGui.style.cursor = 'pointer';
                this.eGui.style.textDecoration = 'underline';
                this.eGui.addEventListener('click', () => {
                    params.node.setSelected(true, true);
                });
            }
            getGui() {
                return this.eGui;
            }
        }
        """
    )

    score_style = JsCode(
        """
        function(params) {
            if (params.value >= 72) {
                return {color: '#86efac', fontWeight: '800'};
            } else if (params.value >= 45) {
                return {color: '#fde68a', fontWeight: '700'};
            } else {
                return {color: '#fca5a5', fontWeight: '700'};
            }
        }
        """
    )

    gb = GridOptionsBuilder.from_dataframe(table_df)
    gb.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
        floatingFilter=True,
    )
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_column("symbol", headerName="Ticker", cellRenderer=ticker_cell)
    gb.configure_column("score", cellStyle=score_style, sort="desc")

    grid_options = gb.build()
    grid_options["rowHeight"] = 42
    grid_options["animateRows"] = True

    response = AgGrid(
        table_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        theme="balham-dark",
        height=440,
        reload_data=False,
    )
    return response


def sidebar_controls():
    with st.sidebar:
        st.title(APP_TITLE)
        st.caption("Professional Edition")

        account_size = st.number_input("Account Size", min_value=1000.0, value=10000.0, step=500.0)
        risk_percent = st.number_input("Risk % Per Trade", min_value=0.1, value=1.0, step=0.1)
        top_n = st.selectbox("Top rows", [10, 20, 30, 50, 100, 200], index=2)
        selected_benchmark = st.selectbox(
            "Benchmark",
            ["Auto"] + BENCHMARKS,
            index=(["Auto"] + BENCHMARKS).index(st.session_state.selected_benchmark)
            if st.session_state.selected_benchmark in (["Auto"] + BENCHMARKS)
            else 0,
        )

        only_a_plus = st.toggle(f"Only show A+ setups ({A_PLUS_SCORE}+)", value=False)
        only_breakout = st.toggle("Only show breakout-ready", value=False)
        only_market_ok = st.toggle("Only show market-healthy names", value=False)
        min_score_filter = st.slider("Minimum score", min_value=0, max_value=100, value=0, step=1)

        st.session_state.selected_benchmark = selected_benchmark
        st.session_state.min_score_filter = min_score_filter

        st.divider()
        st.markdown(badge_html("Pro Workspace", "purple"), unsafe_allow_html=True)
        st.markdown(INFO_TEXT)

    return account_size, risk_percent, top_n, only_a_plus, only_breakout, only_market_ok, selected_benchmark, min_score_filter





def render_clean_header():
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;margin-top:8px;margin-bottom:18px;">
            <div>
                <div style="font-size:2.45rem;font-weight:900;line-height:1.05;color:#f8fafc;">{APP_TITLE}</div>
                <div style="font-size:1.05rem;color:#9fb4d9;margin-top:10px;">{APP_TAGLINE}</div>
            </div>
            <div style="display:flex;gap:10px;align-items:center;">
                <span class="sspx-chip">👤 Profile</span>
                <span class="sspx-chip">⚙️ Settings</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def run_scan_with_premium_feedback(account_size: float, risk_percent: float, selected_benchmark: str):
    symbols = UniverseLoader.get_sp500_symbols()
    status_box = st.empty()
    progress_box = st.empty()
    stage_steps = [
        ("Loading S&P 500 universe", 8),
        ("Pulling price history", 28),
        ("Scoring technical setups", 58),
        ("Ranking strongest candidates", 82),
        ("Packaging top ideas", 100),
    ]
    progress = progress_box.progress(0, text="Preparing scan...")
    for message, pct in stage_steps[:2]:
        status_box.markdown(
            f"""
            <div class="sspx-panel" style="margin-top:12px;padding:14px 16px;">
                <div style="font-size:0.88rem;color:#93c5fd;font-weight:700;letter-spacing:0.03em;text-transform:uppercase;">Scanning engine</div>
                <div style="font-size:1.12rem;color:#f8fafc;font-weight:800;margin-top:4px;">{message}</div>
                <div style="font-size:0.92rem;color:#9ca3af;margin-top:6px;">Reviewing {len(symbols)} names with your current rules.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        progress.progress(pct, text=message)

    with st.spinner(f"Scanning {len(symbols)} S&P 500 stocks..."):
        results_df, price_data, logs, alerts = run_scan_logic(
            symbols,
            account_size,
            risk_percent,
            selected_benchmark=selected_benchmark,
        )

    for message, pct in stage_steps[2:]:
        status_box.markdown(
            f"""
            <div class="sspx-panel" style="margin-top:12px;padding:14px 16px;">
                <div style="font-size:0.88rem;color:#93c5fd;font-weight:700;letter-spacing:0.03em;text-transform:uppercase;">Scanning engine</div>
                <div style="font-size:1.12rem;color:#f8fafc;font-weight:800;margin-top:4px;">{message}</div>
                <div style="font-size:0.92rem;color:#9ca3af;margin-top:6px;">Finalizing today's ranked opportunity feed.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        progress.progress(pct, text=message)

    status_box.markdown(
        f"""
        <div class="sspx-panel" style="margin-top:12px;padding:14px 16px;border:1px solid rgba(134,239,172,0.25);">
            <div style="font-size:0.88rem;color:#86efac;font-weight:700;letter-spacing:0.03em;text-transform:uppercase;">Scan complete</div>
            <div style="font-size:1.12rem;color:#f8fafc;font-weight:800;margin-top:4px;">{len(results_df)} ranked setups ready</div>
            <div style="font-size:0.92rem;color:#9ca3af;margin-top:6px;">Opening your top pick and ranked feed now.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    progress.progress(100, text="Scan complete")
    return results_df, price_data, logs, alerts


def scanner_tab(account_size, risk_percent, top_n, only_a_plus, only_breakout, only_market_ok, selected_benchmark, min_score_filter):
    results_df = ensure_results_schema(st.session_state.results_df.copy())
    has_results = not results_df.empty

    if not has_results:
        _, mid, _ = st.columns([1.2, 2.4, 1.2])
        with mid:
            if st.button("Scan Now", type="primary", use_container_width=True):
                results_df, price_data, logs, alerts = run_scan_with_premium_feedback(
                    account_size=account_size,
                    risk_percent=risk_percent,
                    selected_benchmark=selected_benchmark,
                )
                st.session_state.results_df = results_df
                st.session_state.price_data = price_data
                st.session_state.logs = logs
                st.session_state.alerts = alerts
                if not results_df.empty:
                    st.session_state.selected_symbol = results_df.iloc[0]["symbol"]
                st.rerun()

        c1, c2 = st.columns([1.05, 1.35])
        with c1:
            render_scoring_explainer()
        with c2:
            st.markdown(
                f"""
                <div class="sspx-panel" style="min-height:124px;">
                    <div style="font-size:1rem;color:#f8fafc;font-weight:800;margin-bottom:8px;">How this version works</div>
                    <div style="font-size:0.95rem;color:#d1d5db;line-height:1.75;">
                        Tap <span style="color:#93c5fd;font-weight:700;">Scan Now</span> to launch the ranking engine. We’ll surface your top setup first, then open the full ranked feed.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.stop()

    if only_a_plus:
        results_df = results_df[results_df["score"] >= A_PLUS_SCORE]
    if only_breakout:
        results_df = results_df[results_df["breakout_ready"] == True]
    if only_market_ok:
        results_df = results_df[results_df["market_ok"] == True]
    if min_score_filter > 0:
        results_df = results_df[results_df["score"] >= min_score_filter]

    sectors = ["All"] + sorted([s for s in results_df["sector"].dropna().unique().tolist() if str(s).strip()])
    sector_filter = st.selectbox("Sector filter", sectors)
    if sector_filter != "All":
        results_df = results_df[results_df["sector"] == sector_filter]

    results_df = results_df.head(max(10, top_n)).reset_index(drop=True)

    render_market_status_bar(results_df)

    if results_df.empty:
        st.info("No scan results match your current filters.")
    else:
        render_scan_kpis(results_df)
        render_top_pick_hero(results_df, st.session_state.price_data)
        render_ranked_feed(results_df, st.session_state.price_data, top_n=min(10, len(results_df)))

        st.markdown("#### Feed Stats")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Top score", int(results_df["score"].max()))
        with c2:
            st.metric("A+ names", int((results_df["score"] >= A_PLUS_SCORE).sum()))
        with c3:
            st.metric("Median score", round(float(results_df["score"].median()), 1))
        with c4:
            st.metric("Breakout-ready", int(results_df["breakout_ready"].sum()))

        with st.expander("Advanced Table View"):
            st.markdown("For desktop review only.")
            render_results_table(results_df)

        render_premium_modules()

    with st.expander("Scanner Log"):
        if st.session_state.logs:
            st.code("\n".join(st.session_state.logs))
        else:
            st.write("No logs yet.")


def ideas_tab():
    st.subheader("Ideas")
    watchlist = st.session_state.watchlist
    if not watchlist:
        st.info("No saved ideas yet.")
        return

    top_left, top_right = st.columns([1.25, 1])
    with top_left:
        st.markdown(panel_open("Saved Ideas", "Your shortlist / watchlist"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"ticker": watchlist}), use_container_width=True, height=320, hide_index=True)
        st.markdown(panel_close(), unsafe_allow_html=True)

    with top_right:
        st.markdown(panel_open("Idea Actions", "Manage and rescan"), unsafe_allow_html=True)
        remove_symbol = st.selectbox("Remove idea", watchlist)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Remove Idea", use_container_width=True):
                st.session_state.watchlist = [s for s in watchlist if s != remove_symbol]
                Storage.save_watchlist(st.session_state.watchlist)
                st.rerun()
        with c2:
            if st.button("Scan Ideas Only", use_container_width=True):
                with st.spinner(f"Scanning {len(watchlist)} saved ideas..."):
                    results_df, price_data, logs, alerts = run_scan_logic(
                        watchlist,
                        account_size=10000.0,
                        risk_percent=1.0,
                        selected_benchmark=st.session_state.selected_benchmark,
                    )
                    st.session_state.results_df = results_df
                    st.session_state.price_data = price_data
                    st.session_state.logs = logs
                    st.session_state.alerts = alerts
                    if not results_df.empty:
                        st.session_state.selected_symbol = results_df.iloc[0]["symbol"]
                st.rerun()
        st.markdown(panel_close(), unsafe_allow_html=True)


def journal_tab():
    st.subheader("Journal")
    journal_df = Storage.load_journal()
    if journal_df.empty:
        st.info("Trade journal is empty.")
    else:
        st.markdown(panel_open("Trade Journal", "Planned and logged ideas"), unsafe_allow_html=True)
        st.dataframe(journal_df, use_container_width=True, height=300, hide_index=True)
        st.markdown(panel_close(), unsafe_allow_html=True)

        st.markdown("#### Performance Snapshot")
        working_df = journal_df.copy()

        if "score" in working_df.columns:
            working_df["score"] = pd.to_numeric(working_df["score"], errors="coerce")
        if "shares" in working_df.columns:
            working_df["shares"] = pd.to_numeric(working_df["shares"], errors="coerce")
        if "entry" in working_df.columns:
            working_df["entry"] = pd.to_numeric(working_df["entry"], errors="coerce")
        if "target" in working_df.columns:
            working_df["target"] = pd.to_numeric(working_df["target"], errors="coerce")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Journal Rows", len(working_df))
        with c2:
            avg_score = working_df["score"].dropna().mean()
            st.metric("Average Score", round(float(avg_score), 1) if not np.isnan(avg_score) else 0)
        with c3:
            planned_count = int((working_df["status"].astype(str).str.lower() == "planned").sum()) if "status" in working_df.columns else 0
            st.metric("Planned Trades", planned_count)

        if "setup_type" in working_df.columns and not working_df["setup_type"].dropna().empty:
            st.markdown(panel_open("Setup Breakdown", "Distribution of logged setups"), unsafe_allow_html=True)
            st.bar_chart(working_df["setup_type"].value_counts())
            st.markdown(panel_close(), unsafe_allow_html=True)

        csv = journal_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Journal CSV", csv, file_name="trade_journal.csv", mime="text/csv")


def signals_tab():
    st.subheader("Signals")
    alerts = st.session_state.alerts
    if not alerts:
        st.info("No signals yet.")
        return

    st.markdown(badge_html("Live Signal Feed", "purple"), unsafe_allow_html=True)
    st.markdown(panel_open("Signal Stream", "Scanner-generated alerts"), unsafe_allow_html=True)
    for alert in alerts:
        st.write(f"- {alert}")
    st.markdown(panel_close(), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ensure_state()
    inject_dark_theme()
    render_clean_header()

    (
        account_size,
        risk_percent,
        top_n,
        only_a_plus,
        only_breakout,
        only_market_ok,
        selected_benchmark,
        min_score_filter,
    ) = sidebar_controls()

    has_results = not st.session_state.results_df.empty

    if not has_results:
        scanner_tab(
            account_size,
            risk_percent,
            top_n,
            only_a_plus,
            only_breakout,
            only_market_ok,
            selected_benchmark,
            min_score_filter,
        )
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Scanner", "Ideas", "Journal", "Signals"])
        with tab1:
            scanner_tab(
                account_size,
                risk_percent,
                top_n,
                only_a_plus,
                only_breakout,
                only_market_ok,
                selected_benchmark,
                min_score_filter,
            )
        with tab2:
            ideas_tab()
        with tab3:
            journal_tab()
        with tab4:
            signals_tab()


if __name__ == "__main__":
    main()
