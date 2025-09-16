from market_selector_service import MarketSelectorService
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

selector = MarketSelectorService()

# 测试单只股票的评分计算
symbol = '688001.SH'
print(f"=== 测试股票 {symbol} 的评分计算 ===")

# 获取股票数据
stock_data = selector._get_stock_analysis_data(symbol)
if stock_data:
    print(f"股票名称: {stock_data['name']}")
    print(f"市场: {stock_data['market']}")
    print(f"板块: {stock_data['board_type']}")
    print(f"30天涨幅: {stock_data.get('price_change_30d', 0):.2f}%")
    
    # 计算成交量比率
    recent_prices = stock_data.get('recent_prices', [])
    if recent_prices and len(recent_prices) >= 5:
        recent_volume = sum([p['volume'] for p in recent_prices[:5]]) / 5
        avg_volume = sum([p['volume'] for p in recent_prices]) / len(recent_prices)
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        print(f"成交量比率: {volume_ratio:.3f}")
    
    # 计算评分
    criteria = {
        'min_market_cap': 10,
        'max_pe_ratio': 100,
        'min_volume_ratio': 0.5,
        'technical_score_threshold': 0.3
    }
    
    score = selector._calculate_stock_score(stock_data, criteria)
    print(f"\n计算得分: {score}")
    print(f"阈值要求: {criteria['technical_score_threshold']}")
    print(f"是否通过筛选: {score >= criteria['technical_score_threshold']}")
    
    # 手动计算各项得分
    print("\n=== 详细得分分解 ===")
    base_score = 50.0
    print(f"基础分: {base_score}")
    
    # 价格趋势得分
    price_change_30d = stock_data.get('price_change_30d', 0)
    price_score = 0
    if price_change_30d > 10:
        price_score = 30
    elif price_change_30d > 0:
        price_score = 20
    elif price_change_30d > -10:
        price_score = 10
    print(f"价格趋势得分: {price_score} (30天涨幅: {price_change_30d:.2f}%)")
    
    # 成交量得分
    volume_score = 0
    if recent_prices and len(recent_prices) >= 5:
        if volume_ratio >= 2.0:
            volume_score = 20
        elif volume_ratio >= 1.5:
            volume_score = 15
        elif volume_ratio >= 1.2:
            volume_score = 10
        elif volume_ratio >= 0.8:
            volume_score = 8
        elif volume_ratio >= 0.5:
            volume_score = 5
    print(f"成交量得分: {volume_score} (比率: {volume_ratio:.3f})")
    
    # 市场板块得分
    market_score = 0
    if stock_data.get('market') == 'SH' and stock_data.get('board_type') == '科创板':
        market_score = 10
    print(f"市场板块得分: {market_score}")
    
    # 数据完整性得分
    data_score = 0
    if stock_data.get('recent_prices'):
        data_score += 5
    if stock_data.get('realtime_quote'):
        data_score += 5
    print(f"数据完整性得分: {data_score}")
    
    total_calculated = base_score + price_score + volume_score + market_score + data_score
    print(f"\n手动计算总分: {total_calculated}")
    print(f"函数返回总分: {score}")
else:
    print("无法获取股票数据")