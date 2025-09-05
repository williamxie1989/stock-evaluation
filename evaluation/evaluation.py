import requests
import json
import time
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class StockEvaluation:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        # 股票池
        self.stock_pool = [
            {"symbol": "600036.SS", "name": "招商银行"},
            {"symbol": "601318.SS", "name": "中国平安"},
            {"symbol": "600519.SS", "name": "贵州茅台"},
            {"symbol": "000858.SZ", "name": "五粮液"},
            {"symbol": "000333.SZ", "name": "美的集团"},
            {"symbol": "000001.SS", "name": "上证指数"},
            {"symbol": "399001.SZ", "name": "深证成指"},
            {"symbol": "000300.SS", "name": "沪深300"},
            {"symbol": "600030.SS", "name": "中信证券"},
            {"symbol": "600837.SS", "name": "海通证券"},
            {"symbol": "600104.SS", "name": "上汽集团"},
            {"symbol": "601166.SS", "name": "兴业银行"},
            {"symbol": "600016.SS", "name": "民生银行"},
            {"symbol": "601398.SS", "name": "工商银行"},
            {"symbol": "601939.SS", "name": "建设银行"},
            {"symbol": "601328.SS", "name": "交通银行"},
            {"symbol": "601988.SS", "name": "中国银行"},
            {"symbol": "600000.SS", "name": "浦发银行"},
            {"symbol": "601169.SS", "name": "北京银行"},
            {"symbol": "601818.SS", "name": "光大银行"}
        ]
    
    def analyze_single_stock(self, stock):
        """分析单支股票"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/analyze",
                json={"symbol": stock["symbol"]},
                timeout=35
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success", False):
                    result = data.get("data", {})
                    signals = result.get("signals", [])
                    ai_analysis = result.get("ai_analysis", "")
                    latest_price = result.get("latest_price", 0)
                    
                    # 检查是否使用了AI分析
                    ai_timeout = 'AI分析超时' in ai_analysis or '回退到本地逻辑' in ai_analysis
                    
                    # 处理本地逻辑返回的结构化结果
                    if 'data_source' in result and result['data_source'] == 'local_logic':
                        # 本地逻辑返回的结果结构与AI分析不同，需要适配
                        ai_analysis = result.get('ai_analysis', '')
                        
                    return {
                        "stock": stock,
                        "success": True,
                        "response_time": response_time,
                        "ai_timeout": ai_timeout,
                        "signals_count": len(signals),
                        "latest_price": latest_price,
                        "signals": signals,  # 添加顶层signals字段
                        "data": result,  # 保留原始数据
                        "error": None
                    }
                else:
                    return {
                        "stock": stock,
                        "success": False,
                        "response_time": response_time,
                        "ai_timeout": False,
                        "signals_count": 0,
                        "latest_price": 0,
                        "error": data.get("error", "未知错误")
                    }
            else:
                return {
                    "stock": stock,
                    "success": False,
                    "response_time": response_time,
                    "ai_timeout": False,
                    "signals_count": 0,
                    "latest_price": 0,
                    "error": f"HTTP {response.status_code}"
                }
        except requests.exceptions.Timeout:
            return {
                "stock": stock,
                "success": False,
                "response_time": 35,  # 超时时间
                "ai_timeout": False,
                "signals_count": 0,
                "latest_price": 0,
                "error": "请求超时"
            }
        except Exception as e:
            return {
                "stock": stock,
                "success": False,
                "response_time": 0,
                "ai_timeout": False,
                "signals_count": 0,
                "latest_price": 0,
                "error": str(e)
            }
    
    def evaluate_stocks(self, num_stocks=15, max_workers=5):
        """评估股票分析效果"""
        # 随机选择股票
        selected_stocks = random.sample(self.stock_pool, min(num_stocks, len(self.stock_pool)))
        print(f"开始评估 {len(selected_stocks)} 支股票")
        
        results = []
        futures = []
        
        # 并发分析股票
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务，每次提交后等待40秒
            for i, stock in enumerate(selected_stocks):
                future = executor.submit(self.analyze_single_stock, stock)
                futures.append((future, stock))
                
                # 除了最后一个股票，其他股票分析之间等待40秒
                if i < len(selected_stocks) - 1:
                    print(f"等待40秒后继续分析下一支股票...")
                    time.sleep(40)
            
            # 收集结果
            for future, stock in futures:
                result = future.result()
                results.append(result)
                
                if result["success"]:
                    print(f"{stock['symbol']} ({stock['name']}): 成功，耗时 {result['response_time']:.2f} 秒，信号数 {result['signals_count']}")
                else:
                    print(f"{stock['symbol']} ({stock['name']}): 失败，错误: {result['error']}")
        
        return results
    
    def calculate_signal_returns(self, signals, latest_price):
        """计算信号收益"""
        if not signals or latest_price <= 0:
            return []
        
        returns = []
        for signal in signals:
            signal_price = signal.get("price", 0)
            if signal_price > 0 and latest_price > 0:
                # 计算收益率
                if signal["type"] == "BUY":
                    return_rate = (latest_price - signal_price) / signal_price * 100
                else:  # SELL
                    return_rate = (signal_price - latest_price) / signal_price * 100
                returns.append({
                    "date": signal["date"],
                    "type": signal["type"],
                    "price": signal_price,
                    "current_price": latest_price,
                    "return_rate": return_rate,
                    "reason": signal["reason"]
                })
        
        return returns
    
    def generate_evaluation_report(self, results):
        """生成评估报告"""
        total_stocks = len(results)
        successful_analyses = sum(1 for r in results if r["success"])
        failed_analyses = total_stocks - successful_analyses
        ai_timeouts = sum(1 for r in results if r["ai_timeout"])
        
        # 响应时间统计
        response_times = [r["response_time"] for r in results if r["success"]]
        avg_response_time = np.mean(response_times) if response_times else 0
        max_response_time = np.max(response_times) if response_times else 0
        min_response_time = np.min(response_times) if response_times else 0
        
        # 信号统计
        total_signals = sum(r["signals_count"] for r in results)
        avg_signals_per_stock = total_signals / total_stocks if total_stocks > 0 else 0
        
        # 收益率计算
        all_returns = []
        for result in results:
            if result["success"]:
                # 确保result中有data键
                data = result.get("data", {})
                signals = data.get("signals", [])
                latest_price = data.get("latest_price", 0)
                returns = self.calculate_signal_returns(signals, latest_price)
                all_returns.extend(returns)
        
        avg_return_rate = np.mean([r["return_rate"] for r in all_returns]) if all_returns else 0
        positive_returns = sum(1 for r in all_returns if r["return_rate"] > 0)
        negative_returns = len(all_returns) - positive_returns if all_returns else 0
        
        # 计算胜率
        win_rate = f"{positive_returns/len(all_returns)*100:.1f}% (仅当有收益计算时)" if len(all_returns) > 0 else "N/A (无收益计算)"
        
        # 生成报告
        report = f"""
股票分析效果评估报告
====================
评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. 分析成功率
   - 总股票数: {total_stocks}
   - 成功分析: {successful_analyses}
   - 失败分析: {failed_analyses}
   - 成功率: {successful_analyses/total_stocks*100:.1f}%

2. AI分析性能
   - AI超时次数: {ai_timeouts}
   - 平均响应时间: {avg_response_time:.2f} 秒
   - 最长响应时间: {max_response_time:.2f} 秒
   - 最短响应时间: {min_response_time:.2f} 秒

3. 信号统计
   - 总信号数: {total_signals}
   - 平均每支股票信号数: {avg_signals_per_stock:.1f}

4. 信号收益分析
   - 总信号收益计算数: {len(all_returns)}
   - 平均收益率: {avg_return_rate:.2f}%
   - 正收益信号数: {positive_returns}
   - 负收益信号数: {negative_returns}
   - 胜率: {win_rate}

5. 详细结果
"""
        
        for result in results:
            stock = result["stock"]
            if result["success"]:
                report += f"   {stock['symbol']} ({stock['name']}): 成功, 信号数: {result['signals_count']}, 耗时: {result['response_time']:.2f}秒"
                if result["ai_timeout"]:
                    report += " (AI超时回退)"
                report += "\n"
            else:
                report += f"   {stock['symbol']} ({stock['name']}): 失败, 错误: {result['error']}\n"
        
        report += "\n6. 优化建议\n"
        
        # 根据结果生成优化建议
        if failed_analyses > 0:
            report += "   - 存在分析失败的情况，建议检查数据源连接和错误处理机制\n"
        
        if ai_timeouts > 0:
            report += "   - AI分析存在超时情况，建议优化AI模型响应时间或调整超时设置\n"
        
        if avg_response_time > 30:
            report += "   - 平均响应时间较长，建议优化AI模型性能或增加并发处理能力\n"
        
        if len(all_returns) > 0 and positive_returns/len(all_returns) < 0.5:
            report += "   - 信号胜率较低，建议优化信号生成算法\n"
        
        report += "   - 建议定期更新模型和算法以提高分析准确性\n"
        
        return report
    
    def run_evaluation(self, num_stocks=15, max_workers=5):
        """运行完整评估流程"""
        print("开始股票分析效果评估...")
        
        # 执行评估
        results = self.evaluate_stocks(num_stocks, max_workers)
        
        # 生成报告
        report = self.generate_evaluation_report(results)
        
        # 保存报告
        report_filename = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n评估完成，报告已保存到 {report_filename}")
        print("\n报告摘要:")
        print(report[:1000] + "..." if len(report) > 1000 else report)
        
        return report

if __name__ == "__main__":
    evaluator = StockEvaluation()
    # 评估15支股票，最大并发数为5
    report = evaluator.run_evaluation(15, 5)