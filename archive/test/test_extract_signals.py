import re
from datetime import datetime

class StockAnalyzer:
    def _extract_signals_from_text(self, text):
        """从文本中提取交易信号"""
        signals = []
        
        # 使用更精确的模式匹配信号，包括日期、类型和价格
        signal_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)\s*-\s*(买入|卖出|持有|BUY|SELL|HOLD)\s*[@￥]\s*(\d+\.\d+)'
        
        # 查找所有匹配的信号
        matches = re.findall(signal_pattern, text)
        for date_str, signal_type, price_str in matches:
            try:
                # 解析日期
                date = datetime.strptime(date_str, '%Y年%m月%d日')
                
                # 解析价格
                price = float(price_str)
                
                # 标准化信号类型
                signal_type_mapping = {
                    '买入': 'BUY',
                    'BUY': 'BUY',
                    '卖出': 'SELL',
                    'SELL': 'SELL',
                    '持有': 'HOLD',
                    '观望': 'HOLD',
                    'HOLD': 'HOLD'
                }
                normalized_type = signal_type_mapping.get(signal_type, 'HOLD')
                
                signals.append({
                    'date': date,
                    'type': normalized_type,
                    'price': price,
                    'reason': f'AI分析建议{normalized_type.lower()}'
                })
            except ValueError:
                # 如果日期或价格解析失败，跳过这个信号
                continue
        
        # 如果没有找到任何信号，尝试使用旧的方法
        if not signals:
            # 定义关键词模式
            buy_patterns = [r'买入', r'购入', r'增持', r'买进', r'投资建议.*买入']
            sell_patterns = [r'卖出', r'抛售', r'减持', r'卖出', r'投资建议.*卖出']
            hold_patterns = [r'持有', r'观望', r'持有观望', r'投资建议.*持有']
            
            # 查找买入信号
            for pattern in buy_patterns:
                if re.search(pattern, text):
                    # 尝试提取买入价格
                    price = 0
                    # 查找"买入价"或"目标价"后的价格
                    buy_price_patterns = [r'买入价[为是]\s*(\d+\.\d+)', r'目标价[为是]\s*(\d+\.\d+)', r'买入[目标][价为]\s*(\d+\.\d+)']
                    for price_pattern in buy_price_patterns:
                        match = re.search(price_pattern, text)
                        if match:
                            price = float(match.group(1))
                            break
                    
                    # 如果没找到特定买入价，则查找文本中的第一个价格
                    if price == 0:
                        price_pattern = r'(\d+\.\d+)'
                        prices = re.findall(price_pattern, text)
                        price = float(prices[0]) if prices else 0
                    
                    # 尝试从文本中提取日期，如果找不到则使用当前日期
                    date = datetime.now()
                    date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)'
                    date_match = re.search(date_pattern, text)
                    if date_match:
                        try:
                            date_str = date_match.group(1)
                            date = datetime.strptime(date_str, '%Y年%m月%d日')
                        except ValueError:
                            pass  # 如果日期解析失败，使用当前日期
                    
                    signals.append({
                        'date': date,
                        'type': 'BUY',
                        'price': price,
                        'reason': f'AI分析建议{pattern}'
                    })
            
            # 查找卖出信号
            for pattern in sell_patterns:
                if re.search(pattern, text):
                    # 尝试提取卖出价格
                    price = 0
                    # 查找"卖出价"或"目标价"后的价格
                    sell_price_patterns = [r'卖出价[为是]\s*(\d+\.\d+)', r'目标价[为是]\s*(\d+\.\d+)', r'卖出[目标][价为]\s*(\d+\.\d+)']
                    for price_pattern in sell_price_patterns:
                        match = re.search(price_pattern, text)
                        if match:
                            price = float(match.group(1))
                            break
                    
                    # 如果没找到特定卖出价，则查找文本中的第一个价格
                    if price == 0:
                        price_pattern = r'(\d+\.\d+)'
                        prices = re.findall(price_pattern, text)
                        price = float(prices[0]) if prices else 0
                    
                    # 尝试从文本中提取日期，如果找不到则使用当前日期
                    date = datetime.now()
                    date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)'
                    date_match = re.search(date_pattern, text)
                    if date_match:
                        try:
                            date_str = date_match.group(1)
                            date = datetime.strptime(date_str, '%Y年%m月%d日')
                        except ValueError:
                            pass  # 如果日期解析失败，使用当前日期
                    
                    signals.append({
                        'date': date,
                        'type': 'SELL',
                        'price': price,
                        'reason': f'AI分析建议{pattern}'
                    })
            
            # 查找持有信号
            for pattern in hold_patterns:
                if re.search(pattern, text):
                    # 持有信号通常不涉及具体价格，但可以提取当前价格或文本中的第一个价格
                    price_pattern = r'(\d+\.\d+)'
                    prices = re.findall(price_pattern, text)
                    price = float(prices[0]) if prices else 0
                    
                    # 尝试从文本中提取日期，如果找不到则使用当前日期
                    date = datetime.now()
                    date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)'
                    date_match = re.search(date_pattern, text)
                    if date_match:
                        try:
                            date_str = date_match.group(1)
                            date = datetime.strptime(date_str, '%Y年%m月%d日')
                        except ValueError:
                            pass  # 如果日期解析失败，使用当前日期
                    
                    signals.append({
                        'date': date,
                        'type': 'HOLD',
                        'price': price,
                        'reason': f'AI分析建议{pattern}'
                    })
        
        # 去重：如果存在相同日期和类型的信号，只保留一个
        unique_signals = []
        seen_signals = set()
        for signal in signals:
            signal_key = (signal['date'], signal['type'])
            if signal_key not in seen_signals:
                seen_signals.add(signal_key)
                unique_signals.append(signal)
        
        return unique_signals

# 测试数据
ai_analysis_text = """
股票代码：TEST.STOCK
股票名称：测试股票
收盘价：95.50
趋势：震荡偏强

技术指标：
- MACD：金叉
- RSI：55
- 布林带：中轨支撑

市场数据：
- 行业板块：科技
- 市场情绪：积极

综合分析：
该股处于震荡偏强格局，MACD出现金叉，RSI处于中性区域，短期内有上涨潜力。

交易信号：
2023年03月15日 - 买入 @ 96.50
AI分析建议买入

2023年03月25日 - 卖出 @ 98.20
AI分析建议卖出

2023年04月05日 - 持有 @ 97.80
AI分析建议持有
"""

# 创建分析器实例并提取信号
analyzer = StockAnalyzer()
extracted_signals = analyzer._extract_signals_from_text(ai_analysis_text)

# 打印提取到的信号
print("提取到的交易信号：")
for signal in extracted_signals:
    print(f"{signal['date'].strftime('%Y-%m-%d')} - {signal['type']} @ {signal['price']}")
    print(f"  {signal['reason']}")
    print()

# 验证信号日期是否正确
expected_dates = ['2023-03-15', '2023-03-25', '2023-04-05']
actual_dates = [signal['date'].strftime('%Y-%m-%d') for signal in extracted_signals]

print("验证结果：")
if set(expected_dates) == set(actual_dates):
    print("交易信号日期正确，与输入信号日期一致。")
else:
    print(f"交易信号日期不正确。期望: {expected_dates}, 实际: {actual_dates}")