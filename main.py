# pip install pandas numpy requests

import time
import math
import requests
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta, timezone

# ----------------------------
# Settings
# ----------------------------
BASE_URL = "https://data-api.binance.vision"  # market-data-only base (public)
KLINES_EP = "/api/v3/klines"                  # spot klines endpoint
INTERVAL = "15m"
LOOKBACK_DAYS = 90

# CMC Top10 (today, based on CoinMarketCap "Top 100 Crypto Coins" page)
CMC_TOP10 = ["BTC", "ETH", "USDT", "XRP", "BNB", "USDC", "SOL", "TRX", "DOGE", "BCH"]

# We will trade spot pairs vs USDT (on Binance spot)
SYMBOLS = [f"{c}USDT" for c in CMC_TOP10 if c not in ["USDT"]]  # USDT itself excluded
# USDCUSDT exists but is near-stable; it'll be filtered by volatility anyway.
# Also note: some pairs might not exist on spot; code will skip if API errors.

# Capital: 10,000,000 KRW -> ~ 6,944 USDT (using 1 USD ~ 1440 KRW)
START_CAPITAL_USDT = 10000000 / 1440.49  # from Wise table example; adjust if you want
# Fees (VIP 0 default): 0.10% per trade side
FEE_RATE = 0.001  # 0.1%

# Strategy parameters (CS-VR Breakout)
RS_LOOKBACK_BARS = 48          # 12h on 15m bars
HH_LOOKBACK = 12               # breakout window
ATR_N = 14
REGIME_MEAN_BARS = 288         # ~3 days on 15m
MAX_HOLD_BARS = 16             # 4 hours
STOP_ATR = 0.8
TRAIL_ATR = 1.2
VOL_REGIME_MIN_RATIO = 0.85    # relax volatility regime filter from 1.0

# Portfolio constraints
MAX_OPEN_POSITIONS = 2         # open up to two positions

# Candidate scoring / TOP filtering
TOP_MAX_CANDIDATES = 2         # choose up to this many top-ranked symbols each bar
TOP_SCORE_MIN = 0.08           # minimum signal score required to enter
SCORE_RS_WEIGHT = 1.0          # relative-strength weight
SCORE_BREAKOUT_WEIGHT = 80.0   # breakout strength weight (distance over HH)
SCORE_VOL_PENALTY = 0.003      # penalty when short-term vol is too high
SCORE_MOM_WEIGHT = 0.5        # short-term momentum confirmation weight

# Trading quality controls
VOL_REGIME_HARD_MAX = 2.5      # hard cap for unstable volatility regime
MIN_HOLD_BARS = 2              # hold at least this many bars unless TP reached
TAKE_PROFIT_ATR = 1.6          # ATR-based take-profit target
SYMBOL_COOLDOWN_BARS = 6       # no re-entry for same symbol after exit
REMOVE_STABLECOIN = True       # skip USDCUSDT entirely

# ----------------------------
# Binance fetch helpers
# ----------------------------
def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def fetch_klines(symbol: str, start_ms: int, end_ms: int, interval: str = "15m") -> pd.DataFrame:
    """
    Fetches klines in chunks (limit max 1000).
    """
    out = []
    limit = 1000
    cur = start_ms

    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": limit}
        r = requests.get(BASE_URL + KLINES_EP, params=params, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"{symbol} HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        if not data:
            break
        out.extend(data)

        last_open = data[-1][0]
        # Move forward by 1ms to avoid duplicates
        cur = last_open + 1

        # If fewer than limit returned, we are done
        if len(data) < limit:
            break

        # gentle rate limiting
        time.sleep(0.2)

    df = pd.DataFrame(out, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","tbbav","tbqav","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open","high","low","close","volume"]]

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ----------------------------
# Strategy + Backtest
# ----------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = atr(df, ATR_N)
    df["vol"] = df["atr"] / df["close"]
    df["ret24h"] = df["close"].pct_change(RS_LOOKBACK_BARS)
    df["hh"] = df["high"].rolling(HH_LOOKBACK).max().shift(1)
    vol_mean = df["vol"].rolling(REGIME_MEAN_BARS).mean()
    df["vol_ratio"] = df["vol"] / vol_mean.replace(0, np.nan)
    df["regime"] = df["vol_ratio"] >= VOL_REGIME_MIN_RATIO
    df["mom4"] = df["close"].pct_change(4)
    df["mom16"] = df["close"].pct_change(16)
    return df

def backtest(dfs: Dict[str, pd.DataFrame]) -> dict:
    # align common index
    common = None
    for sym, df in dfs.items():
        common = df.index if common is None else common.intersection(df.index)
    for sym in list(dfs.keys()):
        dfs[sym] = dfs[sym].loc[common].copy()

    # features
    for sym in dfs:
        dfs[sym] = compute_features(dfs[sym])

    if "BTCUSDT" not in dfs:
        raise RuntimeError("BTCUSDT required as benchmark for RS.")

    btc_ret = dfs["BTCUSDT"]["ret24h"]

    equity = START_CAPITAL_USDT
    equity_curve = []
    trades = []

    positions = []
    symbol_cooldown_until = {}
    skip_symbols = {"USDCUSDT"} if REMOVE_STABLECOIN else set()

    for bar_i, t in enumerate(common):
        # 1) Manage open positions
        updated_positions = []
        for pos in positions:
            sym = pos["sym"]
            df = dfs[sym]
            if t not in df.index:
                updated_positions.append(pos)
                continue

            price = float(df.loc[t, "close"])
            hold_bars = bar_i - pos["entry_idx"]
            atrv_raw = df.loc[t, "atr"]
            if pd.isna(atrv_raw):
                if hold_bars >= MAX_HOLD_BARS:
                    exit_px = price
                    gross = pos["qty"] * (exit_px - pos["entry_px"])
                    fee = FEE_RATE * (pos["qty"] * pos["entry_px"]) + FEE_RATE * (pos["qty"] * exit_px)
                    net = gross - fee
                    equity += net
                    trades.append({
                        "sym": sym,
                        "entry_t": pos["entry_t"],
                        "exit_t": t,
                        "entry_px": pos["entry_px"],
                        "exit_px": exit_px,
                        "qty": pos["qty"],
                        "gross_pnl": gross,
                        "fee": fee,
                        "net_pnl": net,
                        "hold_bars": hold_bars,
                        "entry_score": pos["entry_score"],
                    })
                    symbol_cooldown_until[sym] = bar_i + SYMBOL_COOLDOWN_BARS
                    continue
                updated_positions.append(pos)
                continue

            atrv = float(atrv_raw)
            pos["peak_px"] = max(pos["peak_px"], price)

            stop = pos["entry_px"] - STOP_ATR * atrv
            trail = pos["peak_px"] - TRAIL_ATR * atrv
            take_profit = pos["entry_px"] + TAKE_PROFIT_ATR * atrv
            protect = hold_bars < MIN_HOLD_BARS

            exit_cond = (
                (price >= take_profit)
                or (hold_bars >= MAX_HOLD_BARS)
                or ((not protect) and ((price < stop) or (price < trail)))
            )

            if exit_cond:
                exit_px = price
                gross = pos["qty"] * (exit_px - pos["entry_px"])
                fee = FEE_RATE * (pos["qty"] * pos["entry_px"]) + FEE_RATE * (pos["qty"] * exit_px)
                net = gross - fee
                equity += net
                trades.append({
                    "sym": sym,
                    "entry_t": pos["entry_t"],
                    "exit_t": t,
                    "entry_px": pos["entry_px"],
                    "exit_px": exit_px,
                    "qty": pos["qty"],
                    "gross_pnl": gross,
                    "fee": fee,
                    "net_pnl": net,
                    "hold_bars": hold_bars,
                    "entry_score": pos["entry_score"],
                })
                symbol_cooldown_until[sym] = bar_i + SYMBOL_COOLDOWN_BARS
            else:
                updated_positions.append(pos)
        positions = updated_positions

        # 2) Enter new positions
        if len(positions) < MAX_OPEN_POSITIONS:
            used_symbols = {pos["sym"] for pos in positions}
            candidates = []
            for sym, df in dfs.items():
                if sym in used_symbols or sym in skip_symbols:
                    continue
                if symbol_cooldown_until.get(sym, -1) >= bar_i:
                    continue

                row = df.loc[t]
                if (
                    pd.isna(row["ret24h"]) or pd.isna(row["atr"]) or pd.isna(row["hh"]) or
                    pd.isna(row["vol_ratio"]) or pd.isna(row["mom4"]) or pd.isna(row["mom16"]) or
                    pd.isna(btc_ret.loc[t])
                ):
                    continue

                if not bool(row["regime"]) or row["vol_ratio"] > VOL_REGIME_HARD_MAX:
                    continue
                if row["close"] <= row["hh"]:
                    continue
                if row["mom4"] <= 0:
                    continue

                rs = float(row["ret24h"] - btc_ret.loc[t])
                if rs <= 0:
                    continue

                breakout = float(row["close"] / row["hh"] - 1.0)
                if breakout < 0.0005:
                    continue

                vol_ratio = float(row["vol_ratio"])
                score = (
                    SCORE_RS_WEIGHT * rs
                    + SCORE_BREAKOUT_WEIGHT * breakout
                    - SCORE_VOL_PENALTY * max(0.0, vol_ratio - 1.25)
                    + SCORE_MOM_WEIGHT * float(row["mom16"])
                )

                if score > TOP_SCORE_MIN:
                    candidates.append((score, sym, rs, float(df.loc[t, "open"])))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                open_slots = MAX_OPEN_POSITIONS - len(positions)
                for score, pick_sym, rs, entry_px in candidates[:min(TOP_MAX_CANDIDATES, open_slots)]:
                    if len(positions) >= MAX_OPEN_POSITIONS:
                        break

                    capital_base = equity / MAX_OPEN_POSITIONS
                    if capital_base <= 0:
                        break

                    score_norm = min(1.0, max(0.0, (score - TOP_SCORE_MIN) / TOP_SCORE_MIN))
                    alloc = capital_base * (0.45 + 0.55 * score_norm)
                    qty = (alloc * (1 - FEE_RATE)) / entry_px
                    if qty <= 0 or not np.isfinite(qty):
                        continue

                    positions.append({
                        "sym": pick_sym,
                        "entry_px": entry_px,
                        "entry_t": t,
                        "entry_idx": bar_i,
                        "peak_px": entry_px,
                        "qty": qty,
                        "entry_score": score,
                        "entry_rs": rs,
                    })

        equity_curve.append((t, equity))

    ec = pd.DataFrame(equity_curve, columns=["t","equity"]).set_index("t")
    mdd = (ec["equity"] / ec["equity"].cummax() - 1.0).min()

    tdf = pd.DataFrame(trades)
    if len(tdf) > 0:
        win_rate = (tdf["net_pnl"] > 0).mean()
        avg_trade = tdf["net_pnl"].mean()
        total_net = tdf["net_pnl"].sum()
    else:
        win_rate = np.nan
        avg_trade = 0.0
        total_net = 0.0

    return {
        "final_equity_usdt": float(ec["equity"].iloc[-1]) if len(ec) else equity,
        "start_equity_usdt": float(START_CAPITAL_USDT),
        "net_pnl_usdt": float(total_net),
        "mdd": float(mdd) if not np.isnan(mdd) else np.nan,
        "trades": int(len(tdf)),
        "win_rate": float(win_rate) if not np.isnan(win_rate) else np.nan,
        "avg_trade_usdt": float(avg_trade),
        "equity_curve": ec,
        "trades_df": tdf,
    }

def main():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    start_ms, end_ms = ms(start), ms(end)

    dfs = {}
    for sym in SYMBOLS:
        try:
            df = fetch_klines(sym, start_ms, end_ms, INTERVAL)
            dfs[sym] = df
            print(f"Fetched {sym}: {len(df)} bars")
        except Exception as e:
            print(f"Skip {sym}: {e}")

    # Need BTCUSDT benchmark; if missing, abort
    if "BTCUSDT" not in dfs:
        raise RuntimeError("BTCUSDT not fetched; cannot run RS benchmark strategy.")

    res = backtest(dfs)

    print("\n--- Backtest Result (CS-VR Breakout) ---")
    print(f"Start equity (USDT): {res['start_equity_usdt']:.2f}")
    print(f"Final equity (USDT): {res['final_equity_usdt']:.2f}")
    print(f"Net PnL (USDT):      {res['net_pnl_usdt']:.2f}")
    print(f"Trades:              {res['trades']}")
    print(f"Win rate:            {res['win_rate']:.2%}" if not np.isnan(res["win_rate"]) else "Win rate:            N/A")
    print(f"Avg trade (USDT):    {res['avg_trade_usdt']:.2f}")
    print(f"Max drawdown:        {res['mdd']:.2%}" if not np.isnan(res["mdd"]) else "Max drawdown:        N/A")

    # Optional: save outputs
    res["equity_curve"].to_csv("equity_curve.csv")
    res["trades_df"].to_csv("trades.csv", index=False)
    print("\nSaved: equity_curve.csv, trades.csv")

if __name__ == "__main__":
    main()
