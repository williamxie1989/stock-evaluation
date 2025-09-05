import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import StockAnalyzer

# 创建模拟数据
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

# 创建技术数据DataFrame
technical_data = pd.DataFrame({
    'Close': prices,
    'Volume': np.random.randint(1000000, 5000000, 100),
    'MA5': prices + np.random.randn(100) * 0.5,
    'MA20': prices + np.random.randn(100) * 0.8,
    'MA60': prices + np.random.randn(100) * 1.2,
    'MACD': np.random.randn(100) * 0.5,
    'MACD_Signal': np.random.randn(100) * 0.3,
    'RSI': 50 + np.random.randn(100) * 10,
    'Upper_Band': prices + np.random.randn(100) * 2,
    'Lower_Band': prices - np.random.randn(100) * 2
}, index=dates)

technical_data['MA20'] = technical_data['Close'].rolling(window=20).mean()
technical_data['MA50'] = technical_data['Close'].rolling(window=50).mean()
technical_data['MACD'] = technical_data['Close'].ewm(span=12).mean() - technical_data['Close'].ewm(span=26).mean()
technical_data['MACD_Signal'] = technical_data['MACD'].ewm(span=9).mean()
technical_data['RSI'] = 50 + np.random.randn(100) * 10  # 简化RSI计算

# 创建市场数据
market_data = pd.DataFrame({
    'Close': 3000 + np.cumsum(np.random.randn(100) * 2),
}, index=dates)

# 创建股票信息
stock_info = {
    'symbol': 'TEST.STOCK',
    'longName': '测试股票'
}

# 创建交易信号
signals = [
    {'date': str(dates[-30]), 'type': 'BUY', 'price': technical_data['Close'].iloc[-30], 'reason': 'MA金叉'},
    {'date': str(dates[-20]), 'type': 'SELL', 'price': technical_data['Close'].iloc[-20], 'reason': 'RSI超买'},
    {'date': str(dates[-10]), 'type': 'HOLD', 'price': technical_data['Close'].iloc[-10], 'reason': '震荡整理'}
]

# 创建StockAnalyzer实例并测试本地逻辑
analyzer = StockAnalyzer()
result = analyzer._analyze_with_local_logic(technical_data, market_data, stock_info, signals)

print("本地逻辑分析结果:")
print(result)

# 检查交易信号的日期是否正确
if isinstance(result, dict) and 'signals' in result:
    print("\n交易信号:")
    for signal in result['signals']:
        print(f"日期: {signal['date']}, 类型: {signal['type']}, 价格: {signal['price']:.2f}, 原因: {signal['reason']}")
        
    # 检查信号日期是否与输入信号日期一致
    input_dates = [s['date'] for s in signals]
    output_dates = [s['date'] for s in result['signals']]
    
    if input_dates == output_dates:
        print("\n交易信号日期正确，与输入信号日期一致")
    else:
        print(f"\n交易信号日期不正确，输入日期: {input_dates}, 输出日期: {output_dates}")
        
else:
    print("\n分析失败或返回格式不正确")