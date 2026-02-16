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

## Binance Testnet 실시간 모의거래 (5분봉)

`main.py`의 `INTERVAL`을 `5m`으로 설정해두었습니다.

```bash
python3 live_testnet_trader.py --dry-run --poll-seconds 60 --monitor-interval 30
```

자동 감시 에이전트(백그라운드 재시작 포함) 실행:

```bash
./run_live_testnet_5m_agent.sh
```

에이전트 로그:
- `logs/live_testnet_5m_agent.log`
- 실행 중 중복 실행 방지를 위해 PID는 `logs/live_testnet_5m_agent.pid`에 기록됩니다.
