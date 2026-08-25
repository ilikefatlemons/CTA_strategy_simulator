"""
Phase 2 entry point: run the MACD entry / ATR TP-SL exit / cooldown re-entry
strategy on a single symbol (NVDA) end-to-end and report results.
"""

import pandas as pd

from src.engine.backtest import Strategy, run_backtest
from src.rules.entry import MACDGoldenCross
from src.rules.exit import ATRTakeProfitStopLoss
from src.rules.reentry import CooldownReentry

DATA_PATH = "data/00-美股ETF历史/raw/NVDA_5m.csv"
TRADE_LOG_PATH = "data/phase2_trades.csv"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    strategy = Strategy(
        entry_rule=MACDGoldenCross(),
        exit_rule=ATRTakeProfitStopLoss(),
        reentry_rule=CooldownReentry(),
    )

    result = run_backtest(df, strategy)

    n_trades = len(result.trades)
    wins = [t.return_pct for t in result.trades if t.pnl > 0]
    losses = [t.return_pct for t in result.trades if t.pnl <= 0]
    win_rate = len(wins) / n_trades if n_trades else 0.0
    mean_win_pct = sum(wins) / len(wins) if wins else 0.0
    mean_loss_pct = sum(losses) / len(losses) if losses else 0.0

    print(f"trades: {n_trades}  wins: {len(wins)}  win_rate: {win_rate:.2%}")
    print(f"mean win: {mean_win_pct:.2%}  mean loss: {mean_loss_pct:.2%}")
    print(
        f"equity start: {result.equity_curve.iloc[0]:.2f}  "
        f"end: {result.equity_curve.iloc[-1]:.2f}  "
        f"return: {result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1:.2%}"
    )

    trade_rows = [
        {
            "side": t.side.name,
            "entry_bar_idx": t.entry_bar_idx,
            "entry_price": t.entry_price,
            "exit_bar_idx": t.exit_bar_idx,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
        }
        for t in result.trades
    ]
    trade_log = pd.DataFrame(trade_rows)
    trade_log.to_csv(TRADE_LOG_PATH, index=False)
    print(f"trade log written to {TRADE_LOG_PATH}")


if __name__ == "__main__":
    main()
