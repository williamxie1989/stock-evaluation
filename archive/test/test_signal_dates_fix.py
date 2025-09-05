#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import StockAnalyzer
import json
from datetime import datetime

# 创建测试信号数据
signals = [
    {
        'date': datetime.strptime('2025-09-01', '%Y-%m-%d'),
        'type': 'BUY',
        'price': 59.88,
        'reason': 'AI分析建议buy'
    },
    {
        'date': datetime.strptime('2025-09-02', '%Y-%m-%d'),
        'type': 'SELL',
        'price': 59.88,
        'reason': 'AI分析建议sell'
    },
    {
        'date': datetime.strptime('2025-09-03', '%Y-%m-%d'),
        'type': 'HOLD',
        'price': 59.88,
        'reason': 'AI分析建议hold'
    },
    {
        'date': datetime.strptime('2025-09-04', '%Y-%m-%d'),
        'type': 'SELL',
        'price': 59.88,
        'reason': 'AI分析建议sell'
    }
]

# 创建模拟的技术数据
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

technical_data = pd.DataFrame({
    'Close': [59.88, 58.51, 58.66, 57.78, 57.50],
    'RSI': [39.13, 40.25, 41.30, 42.15, 43.20],
    'MACD': [0.06, 0.05, 0.04, 0.03, 0.02],
    'MA20': [59.10, 59.12, 59.15, 59.18, 59.20],
    'MA50': [58.50, 58.55, 58.60, 58.65, 58.70]
})

# 创建模拟的市场数据
market_data = pd.DataFrame({
    'Close': [11.70, 11.71, 11.72, 11.73, 11.74]
})

# 创建模拟的股票信息
stock_info = {
    'symbol': '601318.SS',
    'longName': '中国平安'
}

# 使用本地逻辑分析
analyzer = StockAnalyzer()
result = analyzer._analyze_with_local_logic(technical_data, market_data, stock_info, signals)

# 打印结果
print("=== 分析报告 ===")
print(result["ai_analysis"])

# 验证信号提取
print("\n=== 提取的交易信号 ===")
extracted_signals = analyzer._extract_signals_from_text(result["ai_analysis"])
for signal in extracted_signals:
    print(f"{signal['date']} - {signal['type']} @ {signal['price']}")