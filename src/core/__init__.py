"""
股票评估系统 - 核心模块

包含主要的分析引擎、回测引擎和风险管理组件。
"""

# StockAnalyzer 已在第四阶段被归档，使用其他替代模块
# from .analyzer import StockAnalyzer
from .backtest_engine import BacktestEngine  
from .risk_management import RiskManager

__all__ = ['BacktestEngine', 'RiskManager']