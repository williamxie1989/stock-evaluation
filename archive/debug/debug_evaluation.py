import json
from evaluation.evaluation import StockEvaluation
# 由于不清楚 StockEvaluation 具体所在模块，暂时假设从 stock_evaluation 模块导入该类
try:
    from evaluation.evaluation import StockEvaluation
except ImportError:
    print("警告：无法从 evaluation.evaluation 模块导入 StockEvaluation 类，请检查模块路径和类名是否正确。")
    StockEvaluation = None



# 创建StockEvaluation实例
# 由于不清楚 StockEvaluation 具体所在模块，暂时假设它在 stock_evaluation 模块中
evaluator = None
if StockEvaluation is not None:
    try:
        evaluator = StockEvaluation()
    except Exception as e:
        print(f"错误：无法创建 StockEvaluation 实例: {e}")
        evaluator = None
else:
    print("错误：StockEvaluation 类未正确导入，无法创建实例")

# 模拟API返回的数据
api_response = {
    "success": True,
    "stock_info": {
        "symbol": "600036.SS",
        "name": "600036.SS"
    },
    "technical_data": {
        "Open": [42.98, 42.89, 41.98, 43.44, 42.9],
        "High": [43.1, 43.0, 43.5, 43.5, 43.1],
        "Low": [42.8, 41.9, 41.9, 42.8, 42.7],
        "Close": [42.9, 41.98, 43.44, 42.9, 42.9],
        "Amount": [209129392.0, 179828800.0, 227685120.0, 170998144.0, 137614528.0],
        "Volume": [4861000.0, 4215200.0, 5264800.0, 3967400.0, 3210800.0]
    },
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
        },
        {
            "date": "2025-06-11",
            "type": "SELL",
            "price": 45.31,
            "reason": "价格触及布林带上轨，可能出现回调"
        },
        {
            "date": "2025-07-08",
            "type": "BUY",
            "price": 47.62,
            "reason": "MACD金叉，市场可能转强"
        },
        {
            "date": "2025-07-11",
            "type": "SELL",
            "price": 45.64,
            "reason": "MACD死叉，市场可能转弱"
        },
        {
            "date": "2025-07-28",
            "type": "BUY",
            "price": 44.44,
            "reason": "RSI从超卖区回升，可能是买入机会"
        },
        {
            "date": "2025-08-11",
            "type": "SELL",
            "price": 44.1,
            "reason": "MACD死叉，市场可能转弱"
        },
        {
            "date": "2025-08-15",
            "type": "BUY",
            "price": 43.3,
            "reason": "价格触及布林带下轨，可能出现反弹"
        },
        {
            "date": "2025-08-27",
            "type": "SELL",
            "price": 43.0,
            "reason": "MACD死叉，市场可能转弱"
        },
        {
            "date": "2025-09-01",
            "type": "BUY",
            "price": 41.98,
            "reason": "RSI从超卖区回升，可能是买入机会"
        }
    ],
    "ai_analysis": "基于技术分析的股票评估报告：\n\n股票代码: 600036.SS, 股票名称: 600036.SS\n\n最近5个交易日收盘价: 42.98, 42.89, 41.98, 43.44, 42.90\n近期价格呈下降趋势。\n\n最新交易信号: N/A - RSI从超卖区回升，可能是买入机会\n最近10个信号统计: 买入(0), 卖出(0), 持有(0)\n\n最新技术指标:\n  收盘价: 42.90\n  RSI: 42.11\n\n\n综合分析与投资建议：\n\n1. 趋势分析：\n   近期价格呈下降趋势。\n\n2. 技术指标解读：\n   - RSI指标显示当前市场处于正常状态，没有明显的超买或超卖信号。\n   - \n   - \n\n3. 交易信号评估：\n   最新交易信号为N/A，建议RSI从超卖区回升，可能是买入机会。\n\n4. 投资建议：\n   持有 - 综合技术指标分析，当前股票处于震荡状态，建议继续持有观察。\n\n5. 风险提示：\n   - 市场风险：股价可能因市场波动而大幅变化\n   - 技术风险：技术指标存在滞后性，可能产生误导信号\n   - 流动性风险：在市场剧烈波动时可能难以及时买卖\n\n请谨慎决策，投资有风险，入市需谨慎。",
    "latest_price": 42.9,
    "data_source": "tdx_file"
}

# 模拟evaluate_stocks方法中的处理逻辑
result = {
    "stock": {"symbol": "600036.SS", "name": "招商银行"},
    "success": True,
    "response_time": 1.23,
    "ai_timeout": False,
    "signals_count": len(api_response["signals"]),
    "latest_price": api_response["latest_price"],
    "signals": api_response["signals"]
}

# 调试：打印传入calculate_signal_returns的参数
print("传入calculate_signal_returns的参数:")
print(f"signals: {result.get('signals', [])}")
print(f"latest_price: {result.get('latest_price', 0)}")

# 调用calculate_signal_returns方法
signals = result.get("signals", [])
latest_price = result.get("latest_price", 0)
if evaluator is not None:
    returns = evaluator.calculate_signal_returns(signals, latest_price)
else:
    returns = []

# 打印结果
print("\n信号收益计算结果:")
for ret in returns:
    print(f"日期: {ret['date']}, 类型: {ret['type']}, 价格: {ret['price']}, 当前价格: {ret['current_price']}, 收益率: {ret['return_rate']:.2f}%")

# 模拟generate_evaluation_report中的处理逻辑
all_returns = []
if result["success"]:
    # 确保result中有signals键
    signals = result.get("signals", [])
    latest_price = result.get("latest_price", 0)
    print(f"\n在generate_evaluation_report中调用calculate_signal_returns前:")
    print(f"signals类型: {type(signals)}, 长度: {len(signals)}")
    print(f"latest_price类型: {type(latest_price)}, 值: {latest_price}")
    if evaluator is not None:
        returns = evaluator.calculate_signal_returns(signals, latest_price)
    else:
        returns = []
    print(f"calculate_signal_returns返回的收益数量: {len(returns)}")
    all_returns.extend(returns)

print(f"\n最终all_returns长度: {len(all_returns)}")