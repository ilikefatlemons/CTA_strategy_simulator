"""
回调入场趋势策略 - Week 1-3: 单ticker完整跑通开仓->部分止盈->剩余仓位跟踪
止盈->平仓的全流程回测, 并打开前端(5m/15m/30m/2h四格 + 权益/Sharpe面板)。

先只做NVDA一个ticker（多品种/逆波动率组合层留到Week 6，见文档第六节）。
用全量历史而不是5周窗口 - 2h bias本身71%时间是neutral，级联触发频率低，
5周窗口样本量太小(实测0~4条腿)，全量数据在NVDA上大约能跑出100+批次。
"""

import pandas as pd

from src.engine.pullback_backtest import run_pullback_backtest
from src.performance.sharpe import sharpe_ratio
from src.viz.chart import show_pullback_backtest

SYMBOL = "NVDA"


def main():
    df = pd.read_csv(f"data/raw/{SYMBOL}_5m.csv", parse_dates=["timestamp"])

    result = run_pullback_backtest(df)

    total_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1
    sharpe = sharpe_ratio(result.equity_curve)

    reason_counts: dict[str, int] = {}
    for t in result.trades:
        reason_counts[t.reason] = reason_counts.get(t.reason, 0) + 1

    print(f"[{SYMBOL}] pullback strategy, {len(result.trades)} trade legs, {reason_counts}")
    print(f"Return {total_return:.2%}, Sharpe {sharpe:.2f}")

    show_pullback_backtest(SYMBOL, df, result.trades, result.equity_curve, sharpe, total_return)


if __name__ == "__main__":
    main()
