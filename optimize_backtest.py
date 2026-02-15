#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable

import pandas as pd
import importlib.util


def load_main_module():
    spec = importlib.util.spec_from_file_location("bt", "/workspaces/tradingbot/main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_data(mod) -> Dict[str, pd.DataFrame]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=mod.LOOKBACK_DAYS)
    start_ms, end_ms = mod.ms(start), mod.ms(end)

    dfs = {}
    for sym in mod.SYMBOLS:
        try:
            dfs[sym] = mod.fetch_klines(sym, start_ms, end_ms, mod.INTERVAL)
            print(f"Fetched {sym}: {len(dfs[sym])}")
        except Exception as exc:
            print(f"Skip {sym}: {exc}")
    if "BTCUSDT" not in dfs:
        raise RuntimeError("BTCUSDT not fetched; cannot run RS benchmark")
    return dfs


def evaluate_config(mod, base_dfs: Dict[str, pd.DataFrame], cfg: Dict[str, float]):
    for k, v in cfg.items():
        setattr(mod, k, v)

    dfs = {k: v.copy() for k, v in base_dfs.items()}
    res = mod.backtest(dfs)

    return {
        "cfg": dict(cfg),
        "net_pnl_usdt": res["net_pnl_usdt"],
        "trades": int(res["trades"]),
        "win_rate": None if pd.isna(res["win_rate"]) else float(res["win_rate"]),
        "final_equity_usdt": res["final_equity_usdt"],
        "mdd": None if pd.isna(res["mdd"]) else float(res["mdd"]),
    }


def better(a: Dict, b: Dict) -> bool:
    if b is None:
        return True
    if a["net_pnl_usdt"] != b["net_pnl_usdt"]:
        return a["net_pnl_usdt"] > b["net_pnl_usdt"]
    a_mdd = -1e9 if a["mdd"] is None else a["mdd"]
    b_mdd = -1e9 if b["mdd"] is None else b["mdd"]
    if a_mdd != b_mdd:
        return a_mdd > b_mdd
    a_win = -1.0 if a["win_rate"] is None else a["win_rate"]
    b_win = -1.0 if b["win_rate"] is None else b["win_rate"]
    return a_win > b_win


def gen_cases() -> Iterable[Dict[str, float]]:
    hh_candidates = [10, 12]
    rs_candidates = [36, 48]
    vol_candidates = [0.75, 0.85]
    score_min_candidates = [0.04, 0.06, 0.08]
    top_k_candidates = [1, 2]
    tp_candidates = [1.2, 1.4, 1.6]
    hold_candidates = [1, 2]
    cooldown_candidates = [4, 6]
    mom_candidates = [0.25, 0.50]
    open_candidates = [1, 2]
    for hh in hh_candidates:
        for rs in rs_candidates:
            for v in vol_candidates:
                for ts in score_min_candidates:
                    for tk in top_k_candidates:
                        for tp in tp_candidates:
                            for hold in hold_candidates:
                                for cd in cooldown_candidates:
                                    for mom in mom_candidates:
                                        for open_pos in open_candidates:
                                            yield {
                                                "HH_LOOKBACK": hh,
                                                "RS_LOOKBACK_BARS": rs,
                                                "VOL_REGIME_MIN_RATIO": v,
                                                "TOP_SCORE_MIN": ts,
                                                "TOP_MAX_CANDIDATES": tk,
                                                "TAKE_PROFIT_ATR": tp,
                                                "MIN_HOLD_BARS": hold,
                                                "SYMBOL_COOLDOWN_BARS": cd,
                                                "SCORE_MOM_WEIGHT": mom,
                                                "MAX_OPEN_POSITIONS": open_pos,
                                            }


def main():
    parser = argparse.ArgumentParser(description="Brute-force small parameter sweep for trading strategy")
    parser.add_argument("--top-k", type=int, default=10, help="print top-k results")
    parser.add_argument("--max-cases", type=int, default=None, help="optional limit on cases for quick checks")
    parser.add_argument("--lookback-days", type=int, default=None, help="override LOOKBACK_DAYS")
    args = parser.parse_args()

    mod = load_main_module()
    if args.lookback_days is not None:
        mod.LOOKBACK_DAYS = args.lookback_days

    dfs = fetch_data(mod)
    best = None
    results = []
    i = 0

    for cfg in gen_cases():
        i += 1
        if args.max_cases is not None and i > args.max_cases:
            break
        row = evaluate_config(mod, dfs, cfg)
        results.append(row)
        if better(row, best):
            best = row
        print(
            f"[{i}] net={row['net_pnl_usdt']:.2f}, "
            f"mdd={row['mdd'] if row['mdd'] is not None else 'nan'}, "
            f"win={row['win_rate'] if row['win_rate'] is not None else 'nan'}, "
            f"trades={row['trades']}, cfg={cfg}"
        )

    results = sorted(results, key=lambda r: (r["net_pnl_usdt"], r["mdd"] or -1e9, r["win_rate"] or -1.0), reverse=True)
    print(f"\\nBest[{len(results)}]: {results[0] if results else None}")
    print(f"Top {min(args.top_k, len(results))}:")
    for row in results[:args.top_k]:
        print(row)


if __name__ == "__main__":
    main()
