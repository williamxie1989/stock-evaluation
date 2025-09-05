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
    {'date': str(dates[-3]), 'type': 'BUY', 'price': technical_data['Close'].iloc[-3], 'reason': 'MA金叉'},
    {'date': str(dates[-2]), 'type': 'SELL', 'price': technical_data['Close'].iloc[-2], 'reason': 'RSI超买'},
    {'date': str(dates[-1]), 'type': 'HOLD', 'price': technical_data['Close'].iloc[-1], 'reason': '震荡整理'}
]

# 创建StockAnalyzer实例并测试本地逻辑
analyzer = StockAnalyzer()
result = analyzer._analyze_with_local_logic(technical_data, market_data, stock_info, signals)

print("本地逻辑分析结果:")
print(result)

# 检查返回结果是否符合预期
if isinstance(result, dict) and 'ai_analysis' in result:
    print("\n分析成功，返回了结构化结果")
    print(f"数据源: {result.get('data_source', 'N/A')}")
    print(f"最新价格: {result.get('latest_price', 'N/A')}")
    
    # 检查AI分析报告是否包含关键信息
    ai_analysis = result.get('ai_analysis', '')
    required_elements = [
        '股票代码', '股票名称', '最近5个交易日收盘价', '近期价格趋势',
        '最新交易信号', '最近10个信号统计', '最新技术指标', '市场环境',
        '趋势分析', '技术指标解读', '交易信号评估', '投资建议', '风险提示'
    ]
    
    missing_elements = []
    for element in required_elements:
        if element not in ai_analysis:
            missing_elements.append(element)
    
    if not missing_elements:
        print("AI分析报告格式正确，包含所有必要信息")
    else:
        print(f"AI分析报告缺少以下关键元素: {missing_elements}")
        
    # 检查是否包含具体的技术指标
    tech_indicators = ['RSI', 'MACD', '20日均线', '50日均线']
    missing_indicators = []
    for indicator in tech_indicators:
        if indicator not in ai_analysis:
            missing_indicators.append(indicator)
    
    if not missing_indicators:
        print("技术指标部分完整，包含RSI、MACD、均线等关键指标")
    else:
        print(f"技术指标部分缺少: {missing_indicators}")
        
    # 检查是否包含市场环境信息
    market_elements = ['大盘指数', '行业平均表现']
    missing_market = []
    for element in market_elements:
        if element not in ai_analysis:
            missing_market.append(element)
    
    if not missing_market:
        print("市场环境分析部分完整，包含大盘指数和行业表现")
    else:
        print(f"市场环境分析部分缺少: {missing_market}")
        
else:
    print("\n分析失败或返回格式不正确")