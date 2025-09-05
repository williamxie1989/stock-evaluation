#!/usr/bin/env python3
"""测试新功能的脚本"""

import json
import requests
from main import StockAnalyzer

def test_new_features():
    """测试新功能"""
    print("=== 测试股票分析系统新功能 ===")
    
    # 创建分析器实例
    analyzer = StockAnalyzer()
    
    # 测试股票分析
    print("\n1. 测试股票分析...")
    result = analyzer.analyze_stock("600036.SS")
    
    if result.get('success', False):
        print("✓ 股票分析成功")
        
        # 检查新功能
        print("\n2. 检查新功能...")
        
        # 检查回测数据
        if 'backtest' in result:
            print("✓ 回测数据存在")
            print(f"   回测结果: {result['backtest']}")
        else:
            print("✗ 回测数据缺失")
        
        # 检查风险报告
        if 'risk_report' in result:
            print("✓ 风险报告存在")
            print(f"   风险报告: {result['risk_report']}")
        else:
            print("✗ 风险报告缺失")
        
        # 检查K线图数据
        if 'chart_data' in result:
            print("✓ K线图数据存在")
            chart_data = result['chart_data']
            print(f"   K线数据点: {len(chart_data.get('kline', []))}")
            print(f"   技术指标: {len(chart_data.get('indicators', {}))}")
            print(f"   交易信号: {len(chart_data.get('signals', []))}")
        else:
            print("✗ K线图数据缺失")
            
    else:
        print("✗ 股票分析失败")
        print(f"错误信息: {result.get('error', '未知错误')}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_new_features()