import json
from evaluation import StockEvaluation

# 创建StockEvaluation实例
evaluator = StockEvaluation()

# 模拟API返回的数据结构
results = [
    {
        "stock": {"symbol": "600036.SS", "name": "招商银行"},
        "success": True,
        "response_time": 1.23,
        "ai_timeout": False,
        "signals_count": 10,
        "latest_price": 0,  # 这个值在旧代码中是0
        "data": {  # 实际数据在data字段中
            "signals": [
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
            ],
            "latest_price": 42.9  # 实际的最新价格
        }
    }
]

# 调用generate_evaluation_report方法中的收益率计算逻辑
all_returns = []
for result in results:
    if result["success"]:
        # 修复后的代码逻辑
        data = result.get("data", {})
        signals = data.get("signals", [])
        latest_price = data.get("latest_price", 0)
        returns = evaluator.calculate_signal_returns(signals, latest_price)
        all_returns.extend(returns)

print("修复后的收益率计算结果:")
print(f"总信号收益计算数: {len(all_returns)}")
for ret in all_returns:
    print(f"日期: {ret['date']}, 类型: {ret['type']}, 价格: {ret['price']}, 当前价格: {ret['current_price']}, 收益率: {ret['return_rate']:.2f}%")