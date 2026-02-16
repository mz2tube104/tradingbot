#!/usr/bin/env python3
"""Run the strategy from `main.py` on Binance Spot Testnet in live mode."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests


def _load_strategy_module():
    mod_path = Path(__file__).resolve().parent / "main.py"
    spec = importlib.util.spec_from_file_location("bt", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BinanceTestnetClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://testnet.binance.vision", timeout: float = 20.0):
        self.api_key = api_key
        self.api_secret = (api_secret or "").strip().encode("utf-8")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})
        self.recv_window = 5000

    def _request(self, method: str, endpoint: str, params: Dict[str, object] = None, signed: bool = False):
        params = {} if params is None else dict(params)
        url = self.base_url + endpoint

        if signed:
            params["timestamp"] = int(datetime.now(timezone.utc).timestamp() * 1000)
            params["recvWindow"] = self.recv_window
            params["signature"] = self._sign(params)

        resp = self.session.request(method, url, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"{method} {endpoint} HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if isinstance(data, dict) and data.get("code"):
            raise RuntimeError(f"{method} {endpoint} API error: {data}")
        return data

    def _sign(self, params: Dict[str, object]) -> str:
        payload = urlencode([(str(k), str(v)) for k, v in params.items()], doseq=True)
        return hmac.new(self.api_secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def get_account(self) -> Dict[str, object]:
        return self._request("GET", "/api/v3/account", signed=True)

    def market_buy(self, symbol: str, quote_order_qty: float) -> Dict[str, object]:
        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": self._fmt_qty(quote_order_qty, 2),
        }
        return self._request("POST", "/api/v3/order", params=params, signed=True)

    def market_sell(self, symbol: str, quantity: float) -> Dict[str, object]:
        params = {
            "symbol": symbol,
            "side": "SELL",
            "type": "MARKET",
            "quantity": self._fmt_qty(quantity, 8),
        }
        return self._request("POST", "/api/v3/order", params=params, signed=True)

    @staticmethod
    def _fmt_qty(v: float, digits: int) -> str:
        return f"{Decimal(str(v)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_DOWN):.{digits}f}"


def _extract_order_stats(order: Dict[str, object], side: str):
    fills = order.get("fills") or []
    fee = 0.0
    qty = 0.0
    quote = 0.0
    if fills:
        for fill in fills:
            q = float(fill.get("qty", 0.0) or 0.0)
            p = float(fill.get("price", 0.0) or 0.0)
            fee += float(fill.get("commission", 0.0) or 0.0)
            qty += q
            quote += q * p
    else:
        qty = float(order.get("executedQty", 0.0) or 0.0)
        quote = float(order.get("cummulativeQuoteQty", 0.0) or 0.0)

    avg_price = quote / qty if qty > 0 else 0.0
    return {
        "qty": qty,
        "quote": quote,
        "avg_price": avg_price,
        "fee": fee,
        "order_id": order.get("orderId"),
        "status": order.get("status", "UNKNOWN"),
        "side": side,
        "raw": order,
    }


def _find_candidate_symbols(mod, dfs: Dict[str, pd.DataFrame], t, btc_ret: pd.Series, used_symbols: set, cooldown: Dict[str, int], bar_i: int):
    candidates = []
    for sym, df in dfs.items():
        if sym in used_symbols:
            continue
        if cooldown.get(sym, -1) >= bar_i:
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
    return candidates


def _interval_seconds(interval_alias: str) -> int:
    alias = interval_alias.strip().lower()
    if alias.endswith("m"):
        return int(alias[:-1]) * 60
    if alias.endswith("h"):
        return int(alias[:-1]) * 60 * 60
    if alias.endswith("d"):
        return int(alias[:-1]) * 24 * 60 * 60
    if alias.endswith("w"):
        return int(alias[:-1]) * 7 * 24 * 60 * 60
    if alias.endswith("s"):
        return int(alias[:-1])
    raise ValueError(f"Unsupported interval alias: {interval_alias}")


def _watch_orders(client: BinanceTestnetClient, symbols: List[str]):
    status = []
    for sym in symbols:
        try:
            open_orders = client._request("GET", "/api/v3/openOrders", {"symbol": sym}, signed=True)
            if not isinstance(open_orders, list):
                continue
            for o in open_orders:
                status.append({
                    "symbol": sym,
                    "orderId": o.get("orderId"),
                    "side": o.get("side"),
                    "type": o.get("type"),
                    "origQty": float(o.get("origQty", 0.0) or 0.0),
                    "price": o.get("price", "0"),
                    "executedQty": float(o.get("executedQty", 0.0) or 0.0),
                    "status": o.get("status"),
                })
        except Exception:
            continue
    return status


def _load_realtime_data(mod, lookback_days: int):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    start_ms, end_ms = mod.ms(start), mod.ms(end)

    dfs = {}
    for sym in mod.SYMBOLS:
        try:
            dfs[sym] = mod.fetch_klines(sym, start_ms, end_ms, mod.INTERVAL)
        except Exception as exc:
            print(f"[skip] {sym}: {exc}")
    return dfs


def _closest_closed_bars(dfs: Dict[str, pd.DataFrame], interval_alias: str):
    interval_seconds = _interval_seconds(interval_alias)
    now = pd.Timestamp.now(timezone.utc)
    anchor = pd.to_datetime(int(now.timestamp()) // interval_seconds * interval_seconds, unit="s", utc=True)
    closed = {}
    for sym, df in dfs.items():
        # Binance kline row index is the candle open time. Keep only closed bars.
        df2 = df[df.index < anchor].copy()
        if len(df2) > 1:
            closed[sym] = df2
    return closed


def run_live_trading(
    mod,
    client: BinanceTestnetClient,
    initial_capital_usdt: float,
    lookback_days: int,
    poll_seconds: int,
    dry_run: bool,
    out_dir: Path,
    once: bool,
    monitor_interval_seconds: int = 300,
):
    equity = float(initial_capital_usdt)
    equity_curve = []
    trades: List[Dict[str, object]] = []
    positions = []
    symbol_cooldown = {}
    last_monitor = 0.0

    skip_symbols = {"USDCUSDT"} if getattr(mod, "REMOVE_STABLECOIN", True) else set()

    last_t = None
    bar_i = 0
    try:
        while True:
            start = datetime.now(timezone.utc)
            dfs = _load_realtime_data(mod, lookback_days)
            if not dfs or "BTCUSDT" not in dfs:
                print("[warn] BTCUSDT unavailable, skip this cycle")
                time.sleep(poll_seconds)
                continue

            dfs = _closest_closed_bars(dfs, mod.INTERVAL)
            # features
            for sym in list(dfs.keys()):
                dfs[sym] = mod.compute_features(dfs[sym])
                if sym in skip_symbols:
                    dfs.pop(sym)

            if len(dfs) == 0:
                time.sleep(poll_seconds)
                continue

            # common closed bar index
            common = None
            for df in dfs.values():
                common = df.index if common is None else common.intersection(df.index)
            if common is None or len(common) == 0:
                print("[warn] no common timestamps")
                time.sleep(poll_seconds)
                continue

            t = common.max()
            if last_t is not None and t == last_t:
                # no new bar yet
                time.sleep(1)
                continue
            last_t = t
            bar_i += 1

            btc_ret = dfs["BTCUSDT"]["ret24h"]
            updated_positions = []

            # manage exits
            for pos in positions:
                sym = pos["sym"]
                df = dfs.get(sym)
                if df is None or t not in df.index:
                    updated_positions.append(pos)
                    continue

                reason = None
                row = df.loc[t]
                price = float(row["close"])
                hold_bars = bar_i - pos["entry_idx"]
                atrv = row["atr"]

                if pd.isna(atrv):
                    if hold_bars >= mod.MAX_HOLD_BARS:
                        reason = "atr_na_timeout"
                    else:
                        updated_positions.append(pos)
                        continue
                else:
                    pos["peak_px"] = max(pos["peak_px"], price)

                    stop = pos["entry_px"] - mod.STOP_ATR * float(atrv)
                    trail = pos["peak_px"] - mod.TRAIL_ATR * float(atrv)
                    take_profit = pos["entry_px"] + mod.TAKE_PROFIT_ATR * float(atrv)
                    protect = hold_bars < mod.MIN_HOLD_BARS
                    reason = None
                    if (price >= take_profit) or (hold_bars >= mod.MAX_HOLD_BARS) or ((not protect) and ((price < stop) or (price < trail))):
                        reason = "take_profit_or_stop_or_timeout"

                if reason in ("atr_na_timeout", "take_profit_or_stop_or_timeout"):
                    if dry_run:
                        exit_qty = pos["qty"]
                        exit_px = price
                        fee = 0.0
                        quote = exit_qty * exit_px
                        gross = quote - (exit_qty * pos["entry_px"])
                        net = gross - fee
                    else:
                        try:
                            order = client.market_sell(sym, pos["qty"])
                        except Exception as exc:
                            print(f"[warn] sell failed {sym}: {exc}")
                            updated_positions.append(pos)
                            continue
                        stats = _extract_order_stats(order, "SELL")
                        if float(stats["qty"]) <= 0:
                            updated_positions.append(pos)
                            continue

                        exit_qty = float(stats["qty"])
                        exit_px = float(stats["avg_price"])
                        fee = float(stats["fee"])
                        gross = exit_qty * (exit_px - pos["entry_px"])
                        net = gross - fee

                    equity += net
                    trades.append({
                        "sym": sym,
                        "entry_t": pos["entry_t"],
                        "exit_t": t,
                        "entry_px": pos["entry_px"],
                        "exit_px": exit_px,
                        "qty": exit_qty,
                        "gross_pnl": gross,
                        "fee": fee,
                        "net_pnl": net,
                        "hold_bars": hold_bars,
                        "entry_score": pos["entry_score"],
                        "mode": "live_testnet" if not dry_run else "live_testnet_dry",
                        "reason": reason,
                    })
                    symbol_cooldown[sym] = bar_i + mod.SYMBOL_COOLDOWN_BARS
                else:
                    updated_positions.append(pos)

            positions = updated_positions

            # generate entries
            if len(positions) < mod.MAX_OPEN_POSITIONS:
                used_symbols = {p["sym"] for p in positions}
                candidates = _find_candidate_symbols(mod, dfs, t, btc_ret, used_symbols | skip_symbols, symbol_cooldown, bar_i)

                candidates.sort(key=lambda x: x[0], reverse=True)
                open_slots = mod.MAX_OPEN_POSITIONS - len(positions)
                for score, pick_sym, _, open_px in candidates[: min(mod.TOP_MAX_CANDIDATES, open_slots)]:
                    if len(positions) >= mod.MAX_OPEN_POSITIONS:
                        break
                    if equity <= 0:
                        break

                    capital_base = equity / mod.MAX_OPEN_POSITIONS
                    score_norm = min(1.0, max(0.0, (score - mod.TOP_SCORE_MIN) / max(mod.TOP_SCORE_MIN, 1e-12)))
                    alloc = capital_base * (0.45 + 0.55 * score_norm)
                    alloc = min(alloc, equity)
                    if alloc <= 0:
                        continue

                    if dry_run:
                        exit_cost = 0.0
                        qty = (alloc / float(open_px)) if open_px > 0 else 0.0
                        if qty <= 0:
                            continue
                        entry_px = float(open_px)
                        entry_t = t
                    else:
                        try:
                            order = client.market_buy(pick_sym, alloc)
                        except Exception as exc:
                            print(f"[warn] buy failed {pick_sym}: {exc}")
                            continue
                        stats = _extract_order_stats(order, "BUY")
                        qty = float(stats["qty"])
                        if qty <= 0:
                            continue
                        entry_px = float(stats["avg_price"])
                        entry_t = t
                        exit_cost = float(stats["fee"])

                        # subtract all spent quote from equity.
                        equity -= float(order.get("cummulativeQuoteQty", 0.0) or 0.0)
                        if equity < 0:
                            equity = 0.0

                    positions.append({
                        "sym": pick_sym,
                        "entry_px": entry_px,
                        "entry_t": entry_t,
                        "entry_idx": bar_i,
                        "entry_score": score,
                        "peak_px": entry_px,
                        "qty": qty,
                    })
                    if not dry_run:
                        break

                    # dry-run budget simulation
                    equity -= alloc

            equity_curve.append((t, equity, len(positions)))
            print(f"[{t}] equity={equity:.2f}, open_pos={len(positions)}, trades={len(trades)}")

            now_mon = time.time()
            if now_mon - last_monitor >= monitor_interval_seconds:
                open_orders = _watch_orders(client, mod.SYMBOLS[:10])
                print(f"[orders] open_orders={len(open_orders)}")
                for row in open_orders[:20]:
                    print(f"  [open] {row['symbol']} {row['orderId']} {row['side']} {row['origQty']}@{row['price']} status={row['status']} exec={row['executedQty']}")
                last_monitor = now_mon

            if once:
                break

            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            sleep = max(1, poll_seconds - elapsed)
            time.sleep(sleep)
    finally:
        out_dir.mkdir(parents=True, exist_ok=True)
        ec = pd.DataFrame(equity_curve, columns=["t", "equity_usdt", "open_positions"]).set_index("t")
        trades_df = pd.DataFrame(trades)

        ec_path = out_dir / "live_equity_curve.csv"
        trades_path = out_dir / "live_trades.csv"
        ec.to_csv(ec_path)
        trades_df.to_csv(trades_path, index=False)

        if len(trades_df) == 0:
            win_rate = float("nan")
            avg_trade = 0.0
            net = 0.0
            mdd = float("nan")
        else:
            win_rate = (trades_df["net_pnl"] > 0).mean()
            avg_trade = float(trades_df["net_pnl"].mean())
            net = float(trades_df["net_pnl"].sum())
            mdd = float((ec["equity_usdt"] / ec["equity_usdt"].cummax() - 1.0).min()) if len(ec) else float("nan")

        print("\n--- Live Testnet Run Summary ---")
        print(f"Final equity (USDT): {equity:.2f}")
        print(f"Net PnL (USDT):      {net:.2f}")
        print(f"Trades:              {len(trades)}")
        print(f"Win rate:            {win_rate:.2%}" if not np.isnan(win_rate) else "Win rate:            N/A")
        print(f"Avg trade (USDT):    {avg_trade:.2f}")
        print(f"Max drawdown:        {mdd:.2%}" if not np.isnan(mdd) else "Max drawdown:        N/A")
        print(f"Saved: {ec_path}, {trades_path}")


def main():
    parser = argparse.ArgumentParser(description="Run CS-VR Breakout on Binance Spot testnet")
    parser.add_argument("--api-key", default=os.getenv("BINANCE_TESTNET_API_KEY"), help="Binance testnet API key")
    parser.add_argument("--api-secret", default=os.getenv("BINANCE_TESTNET_API_SECRET"), help="Binance testnet API secret")
    parser.add_argument("--base-url", default="https://testnet.binance.vision", help="Binance testnet base URL")
    parser.add_argument("--initial-capital-usdt", type=float, default=100.0, help="Paper trading notional budget per run")
    parser.add_argument("--lookback-days", type=int, default=None, help="fetch lookback days for features")
    parser.add_argument("--poll-seconds", type=int, default=120, help="sleep interval between checks")
    parser.add_argument("--out-dir", default="/workspaces/tradingbot", help="output directory")
    parser.add_argument("--dry-run", action="store_true", help="simulate without sending live testnet orders")
    parser.add_argument("--once", action="store_true", help="run only once (one bar) for quick validation")
    parser.add_argument("--monitor-interval", type=int, default=300, help="seconds between order-book watch logs")
    args = parser.parse_args()

    mod = _load_strategy_module()
    if args.lookback_days is not None:
        lookback_days = args.lookback_days
    else:
        lookback_days = getattr(mod, "LOOKBACK_DAYS", 90)

    if not args.dry_run and (not args.api_key or not args.api_secret):
        raise SystemExit("API key/secret is required unless --dry-run is enabled.")

    client = BinanceTestnetClient(args.api_key or "", args.api_secret or "", base_url=args.base_url)
    if not args.dry_run:
        acct = client.get_account()
        usdt = next((b for b in acct.get("balances", []) if b.get("asset") == "USDT"), None)
        available = float(usdt.get("free", 0.0) if usdt is not None else 0.0)
        print(f"Connected account USDT free: {available:.2f}")

    run_live_trading(
        mod=mod,
        client=client,
        initial_capital_usdt=args.initial_capital_usdt,
        lookback_days=lookback_days,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
        out_dir=Path(args.out_dir),
        once=args.once,
        monitor_interval_seconds=args.monitor_interval,
    )


if __name__ == "__main__":
    main()
