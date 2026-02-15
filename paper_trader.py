#!/usr/bin/env python3

"""Paper trading runner for the strategy in main.py.

This script runs the same core signal logic as the current strategy, but with
trade simulation details you can tune (slippage, fee, output logs) for paper
validation before moving to live/testnet.
"""

import argparse
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _load_strategy_module():
    spec = importlib.util.spec_from_file_location("bt", "/workspaces/tradingbot/main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data(mod, lookback_days: int):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    start_ms, end_ms = mod.ms(start), mod.ms(end)

    dfs = {}
    for sym in mod.SYMBOLS:
        try:
            df = mod.fetch_klines(sym, start_ms, end_ms, mod.INTERVAL)
            dfs[sym] = df
            print(f"Fetched {sym}: {len(df)}")
        except Exception as e:
            print(f"Skip {sym}: {e}")
    if "BTCUSDT" not in dfs:
        raise RuntimeError("BTCUSDT required as benchmark for RS.")
    return dfs


def _apply_slippage(price: float, side: str, bps: float) -> float:
    slip = price * (bps / 10000.0)
    if side == "buy":
        return price + slip
    if side == "sell":
        return max(0.0, price - slip)
    raise ValueError("side must be buy or sell")


def run_paper_backtest(mod, dfs: Dict[str, pd.DataFrame], slippage_bps: float = 1.0, entry_fee: float = None, exit_fee: float = None):
    if entry_fee is None:
        entry_fee = getattr(mod, "FEE_RATE", 0.001)
    if exit_fee is None:
        exit_fee = getattr(mod, "FEE_RATE", 0.001)

    # align common index
    common = None
    for sym, df in dfs.items():
        common = df.index if common is None else common.intersection(df.index)
    for sym in list(dfs.keys()):
        dfs[sym] = dfs[sym].loc[common].copy()

    # features
    for sym in dfs:
        dfs[sym] = mod.compute_features(dfs[sym])

    btc_ret = dfs["BTCUSDT"]["ret24h"]

    equity = mod.START_CAPITAL_USDT
    equity_curve = []
    trades = []

    positions = []
    symbol_cooldown_until = {}
    skip_symbols = {"USDCUSDT"} if getattr(mod, "REMOVE_STABLECOIN", True) else set()

    for bar_i, t in enumerate(common):
        updated_positions = []
        for pos in positions:
            sym = pos["sym"]
            df = dfs[sym]
            if t not in df.index:
                updated_positions.append(pos)
                continue

            row = df.loc[t]
            price = float(row["close"])
            hold_bars = bar_i - pos["entry_idx"]
            atrv = row["atr"]

            if pd.isna(atrv):
                if hold_bars >= mod.MAX_HOLD_BARS:
                    exit_px = _apply_slippage(price, "sell", slippage_bps)
                    gross = pos["qty"] * (exit_px - pos["entry_px"])
                    fee = exit_fee * (pos["qty"] * pos["entry_px"]) + exit_fee * (pos["qty"] * exit_px)
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
                        "mode": "paper",
                        "reason": "atr_na_timeout",
                    })
                    symbol_cooldown_until[sym] = bar_i + mod.SYMBOL_COOLDOWN_BARS
                    continue
                updated_positions.append(pos)
                continue

            atrv = float(atrv)
            pos["peak_px"] = max(pos["peak_px"], price)

            stop = pos["entry_px"] - mod.STOP_ATR * atrv
            trail = pos["peak_px"] - mod.TRAIL_ATR * atrv
            take_profit = pos["entry_px"] + mod.TAKE_PROFIT_ATR * atrv
            protect = hold_bars < mod.MIN_HOLD_BARS

            exit_cond = (
                (price >= take_profit)
                or (hold_bars >= mod.MAX_HOLD_BARS)
                or ((not protect) and ((price < stop) or (price < trail)))
            )

            if exit_cond:
                exit_px = _apply_slippage(price, "sell", slippage_bps)
                gross = pos["qty"] * (exit_px - pos["entry_px"])
                fee = exit_fee * (pos["qty"] * pos["entry_px"]) + exit_fee * (pos["qty"] * exit_px)
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
                    "mode": "paper",
                    "reason": "take_profit_or_stop_or_timeout",
                })
                symbol_cooldown_until[sym] = bar_i + mod.SYMBOL_COOLDOWN_BARS
            else:
                updated_positions.append(pos)
        positions = updated_positions

        if len(positions) < mod.MAX_OPEN_POSITIONS:
            used_symbols = {p["sym"] for p in positions}
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

                if not bool(row["regime"]) or row["vol_ratio"] > mod.VOL_REGIME_HARD_MAX:
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
                    mod.SCORE_RS_WEIGHT * rs
                    + mod.SCORE_BREAKOUT_WEIGHT * breakout
                    - mod.SCORE_VOL_PENALTY * max(0.0, vol_ratio - 1.25)
                    + mod.SCORE_MOM_WEIGHT * float(row["mom16"])
                )

                if score > mod.TOP_SCORE_MIN:
                    candidates.append((score, sym, rs, float(row["open"])))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                slots = mod.MAX_OPEN_POSITIONS - len(positions)
                for score, pick_sym, rs, open_px in candidates[:min(mod.TOP_MAX_CANDIDATES, slots)]:
                    if len(positions) >= mod.MAX_OPEN_POSITIONS:
                        break

                    capital_base = equity / mod.MAX_OPEN_POSITIONS
                    if capital_base <= 0:
                        break

                    score_norm = min(1.0, max(0.0, (score - mod.TOP_SCORE_MIN) / max(mod.TOP_SCORE_MIN, 1e-12)))
                    alloc = capital_base * (0.45 + 0.55 * score_norm)
                    entry_px = _apply_slippage(open_px, "buy", slippage_bps)
                    qty = (alloc * (1 - entry_fee)) / entry_px
                    if qty <= 0 or not np.isfinite(qty):
                        continue
                    entry_fee_cost = entry_fee * (alloc)

                    fee = entry_fee_cost
                    equity -= fee
                    positions.append({
                        "sym": pick_sym,
                        "entry_px": entry_px,
                        "entry_t": t,
                        "entry_idx": bar_i,
                        "peak_px": entry_px,
                        "qty": qty,
                        "entry_score": score,
                        "entry_rs": rs,
                        "entry_fee": fee,
                    })

        equity_curve.append((t, equity))

    ec = pd.DataFrame(equity_curve, columns=["t", "equity"]).set_index("t")
    if len(ec) == 0:
        raise RuntimeError("No bars were processed in paper run.")

    mdd = (ec["equity"] / ec["equity"].cummax() - 1.0).min()

    tdf = pd.DataFrame(trades)
    if len(tdf) > 0:
        win_rate = (tdf["net_pnl"] > 0).mean()
        avg_trade = tdf["net_pnl"].mean()
        total_net = tdf["net_pnl"].sum()
        realized_notional = (tdf["entry_px"] * tdf["qty"]).sum()
    else:
        win_rate = np.nan
        avg_trade = 0.0
        total_net = 0.0
        realized_notional = 0.0

    return {
        "start_equity_usdt": float(mod.START_CAPITAL_USDT),
        "final_equity_usdt": float(ec["equity"].iloc[-1]),
        "net_pnl_usdt": float(total_net),
        "mdd": float(mdd),
        "trades": int(len(tdf)),
        "win_rate": float(win_rate) if not np.isnan(win_rate) else np.nan,
        "avg_trade_usdt": float(avg_trade),
        "equity_curve": ec,
        "trades_df": tdf,
        "open_positions": positions,
        "realized_notional_usdt": float(realized_notional),
    }


def main():
    parser = argparse.ArgumentParser(description="Run paper trading simulation for strategy")
    parser.add_argument("--lookback-days", type=int, default=None, help="override LOOKBACK_DAYS")
    parser.add_argument("--slippage-bps", type=float, default=1.0, help="Execution slippage in bps")
    parser.add_argument("--entry-fee", type=float, default=None, help="Entry fee rate (default: FEE_RATE)")
    parser.add_argument("--exit-fee", type=float, default=None, help="Exit fee rate (default: FEE_RATE)")
    parser.add_argument("--out-dir", default="/workspaces/tradingbot", help="Output directory")
    args = parser.parse_args()

    mod = _load_strategy_module()
    lookback_days = args.lookback_days or mod.LOOKBACK_DAYS

    dfs = load_data(mod, lookback_days)
    res = run_paper_backtest(mod, dfs, slippage_bps=args.slippage_bps, entry_fee=args.entry_fee, exit_fee=args.exit_fee)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    equity_path = out_dir / "paper_equity_curve.csv"
    trades_path = out_dir / "paper_trades.csv"

    res["equity_curve"].to_csv(equity_path)
    res["trades_df"].to_csv(trades_path, index=False)

    print("\n--- Paper Run Result (Spot Strategy Logic) ---")
    print(f"Start equity (USDT): {res['start_equity_usdt']:.2f}")
    print(f"Final equity (USDT): {res['final_equity_usdt']:.2f}")
    print(f"Net PnL (USDT):      {res['net_pnl_usdt']:.2f}")
    print(f"Trades:              {res['trades']}")
    print(f"Win rate:            {res['win_rate']:.2%}" if not np.isnan(res["win_rate"]) else "Win rate:            N/A")
    print(f"Avg trade (USDT):    {res['avg_trade_usdt']:.2f}")
    print(f"Max drawdown:        {res['mdd']:.2%}")
    print(f"Realized notional:   {res['realized_notional_usdt']:.2f}")
    print(f"Saved: {equity_path}, {trades_path}")


if __name__ == "__main__":
    main()
