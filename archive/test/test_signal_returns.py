from evaluation import StockEvaluation


# 创建StockEvaluation实例
evaluator = StockEvaluation()

# 模拟API返回的信号数据
signals = [
    {
        "date": "2025-05-15",
        "type": "SELL",
        "price": 44.92,
        "reason": "RSI进入超买区，可能是卖出机会"
    },
    {
        "date": "2025-06-06",
        "type": "BUY",
        "price": 44.47,
        "reason": "5日均线上穿20日均线（金叉）"
    }
]

# 模拟最新价格
latest_price = 42.9

# 调用calculate_signal_returns方法
returns = evaluator.calculate_signal_returns(signals, latest_price)

# 打印结果
print("信号收益计算结果:")
for ret in returns:
    print(f"日期: {ret['date']}, 类型: {ret['type']}, 价格: {ret['price']}, 当前价格: {ret['current_price']}, 收益率: {ret['return_rate']:.2f}%")