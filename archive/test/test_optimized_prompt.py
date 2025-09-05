import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from main import StockAnalyzer

# 创建模拟数据
# 模拟股票数据
stock_data = pd.DataFrame({
    'Date': pd.date_range(end=datetime.now(), periods=30),
    'Open': np.random.rand(30) * 10 + 40,
    'High': np.random.rand(30) * 10 + 41,
    'Low': np.random.rand(30) * 10 + 39,
    'Close': np.random.rand(30) * 10 + 40,
    'Volume': np.random.rand(30) * 1000000
})
stock_data.set_index('Date', inplace=True)

# 模拟大盘数据
market_data = pd.DataFrame({
    'Date': pd.date_range(end=datetime.now(), periods=30),
    'Open': np.random.rand(30) * 1000 + 3000,
    'High': np.random.rand(30) * 1000 + 3100,
    'Low': np.random.rand(30) * 1000 + 2900,
    'Close': np.random.rand(30) * 1000 + 3000,
    'Volume': np.random.rand(30) * 10000000
})
market_data.set_index('Date', inplace=True)

# 模拟股票信息
stock_info = {
    'symbol': '600036.SS',
    'longName': '招商银行'
}

# 模拟交易信号
signals = [
    {
        'date': datetime.now() - timedelta(days=3),
        'type': 'BUY',
        'price': 42.5,
        'reason': 'RSI从超卖区回升，可能是买入机会'
    },
    {
        'date': datetime.now() - timedelta(days=1),
        'type': 'SELL',
        'price': 44.8,
        'reason': 'RSI进入超买区，可能是卖出机会'
    }
]

# 创建分析器实例
analyzer = StockAnalyzer()

# 计算技术指标
technical_data = analyzer.calculate_technical_indicators(stock_data)

# 调用AI分析方法
ai_analysis = analyzer.analyze_with_ai(technical_data, market_data, stock_info, signals)

# 打印结果
print("AI分析结果:")
print(ai_analysis)

# 验证结果是否包含关键信息
required_elements = [
    '趋势分析',
    '技术指标解读',
    '交易信号评估',
    '市场环境分析',
    '投资建议',
    '风险提示'
]

print("\n验证结果:")
for element in required_elements:
    if element in ai_analysis:
        print(f"✓ 包含 {element}")
    else:
        print(f"✗ 缺少 {element}")