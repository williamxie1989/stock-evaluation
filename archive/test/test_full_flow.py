import json
from main import StockAnalyzer

class MockStockAnalyzer(StockAnalyzer):
    def analyze_stock(self, stock_symbol):
        """模拟analyze_stock方法，直接返回包含AI分析文本的数据"""
        # 模拟AI分析文本，包含不同日期的交易信号
        ai_analysis_text = """
你是一位专业的股票分析师，请根据以下股票数据提供简洁而全面的分析和投资建议：

股票代码: TEST.STOCK, 股票名称: 测试股票
最近5个交易日收盘价: 90.25, 91.30, 92.80, 93.50, 94.20
近期价格趋势: 上升

最新交易信号: BUY - 5日均线上穿20日均线（金叉）
最近10个信号统计: 买入(1), 卖出(1), 持有(8)

技术指标: RSI=62.50, MACD=1.20, 20日均线=89.50, 50日均线=85.20
收盘价: 94.20

市场环境: 大盘指数=2997.05 (最近5天趋势: 下降), 行业平均表现=N/A

请按以下要点分析：
1. 趋势分析：短期和长期趋势，结合市场环境
2. 技术指标解读：关键指标及其对股价的影响
3. 交易信号评估：最近信号的可靠性
4. 市场环境分析：大盘走势和行业表现对股票的影响
5. 投资建议：明确的买入、卖出或持有建议，给出价格目标和止损位
6. 风险提示：可能面临的风险

根据对测试股票(TEST.STOCK)的综合技术分析，得出以下结论：

1. 趋势分析：近期价格呈上升趋势。
2. 技术指标解读：RSI指标显示当前市场处于正常状态，没有明显的超买或超卖信号。 MACD指标显示死叉信号，表明市场可能进入空头行情。 均线系统显示多头排 列，市场处于上升趋势。
3. 交易信号评估：最近一个信号为BUY，原因为5日均线上穿20日均线（金叉）。近10个信号统计：买入1次，卖出1次，持有8次。
4. 市场环境分析：当前大盘指数为2997.05 (最近5天趋势: 下降)。行业平均表现暂无数据。
5. 投资建议：买入 - 综合技术指标分析，当前股票处于上升趋势，建议逢低买入。
6. 风险提示：投资者应关注市场波动风险，合理控制仓位，设定止损位。

历史交易信号：
2023年03月15日 - 买入 @ 92.30
2023年03月25日 - 卖出 @ 95.40
2023年04月05日 - 持有 @ 94.81
"""
        
        # 使用修复后的方法提取信号
        signals = self._extract_signals_from_text(ai_analysis_text)
        
        # 返回结果
        return {
            "stock": {
                "symbol": stock_symbol,
                "longName": "测试股票"
            },
            "ai_analysis": ai_analysis_text,
            "latest_price": 94.20,
            "data_source": "local_logic",
            "signals": [self._serialize_signal(signal) for signal in signals]
        }

# 创建分析器实例并测试
analyzer = MockStockAnalyzer()
result = analyzer.analyze_stock("TEST.STOCK")

# 打印结果
print("分析结果：")
print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

# 特别关注交易信号部分
print("\n交易信号：")
for signal in result["signals"]:
    print(f"{signal['date']} - {signal['type']} @ {signal['price']}")