#!/usr/bin/env python3
"""Unit test for BacktestEngine fixes"""

import pandas as pd
import numpy as np
from backtest_engine import BacktestEngine


def test_simple_buy_sell():
    # 构造简单的价格序列
    dates = pd.date_range('2025-01-01', periods=10)
    close = [100, 102, 104, 103, 105, 107, 106, 108, 110, 109]
    df = pd.DataFrame({'Close': close, 'Open': close, 'High': close, 'Low': close, 'Volume': [100]*10}, index=dates)

    # 信号：第2天买入，第8天卖出
    signals = [
        {'date': dates[1], 'type': 'BUY', 'price': close[1], 'reason': 'test buy'},
        {'date': dates[7], 'type': 'SELL', 'price': close[7], 'reason': 'test sell'}
    ]

    engine = BacktestEngine(initial_capital=10000, commission_rate=0.001)
    result = engine.run_backtest(df, signals)

    perf = result['performance']

    # 验证最终组合价值为非负数且初始值正常
    assert perf['initial_value'] >= 0
    assert perf['final_value'] >= 0

    # 年化收益应该是可计算的（不是复数或NaN）
    assert isinstance(perf['annualized_return'], float)
    assert not (isinstance(perf['annualized_return'], complex))
    assert not (np.isnan(perf['annualized_return']))

    # 至少有一次买入和卖出交易
    assert perf['total_trades'] >= 2


if __name__ == '__main__':
    test_simple_buy_sell()
    print('test_simple_buy_sell passed')
