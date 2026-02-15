# tradingbot

## Paper trading (simulation) 실행

- 핵심 전략은 `main.py` 기반으로 유지하고, 모의거래 검증은 `paper_trader.py`로 실행합니다.

```bash
python3 paper_trader.py
```

- 기본 설정으로 90일 데이터, 1bps 슬리피지, 기본 수수료(`main.py`의 `FEE_RATE`)로 시뮬레이션합니다.

옵션 예시:

```bash
python3 paper_trader.py --lookback-days 90 --slippage-bps 2.5 --entry-fee 0.001 --exit-fee 0.001
```

실행 결과:
- `paper_equity_curve.csv`
- `paper_trades.csv`
