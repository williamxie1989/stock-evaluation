"""
股票评估系统 - 主包

统一入口点，提供所有模块的便捷导入。
"""

from .core import *
from .data import *
from .ml import *
from .trading import *
from .services import *

__version__ = '2.0.0'
__all__ = [
    # Core
    'StockAnalyzer', 'BacktestEngine', 'RiskManager',
    # Data
    'EnhancedDataProvider', 'EnhancedRealtimeProvider',
    'OptimizedEnhancedDataProvider', 'UnifiedDatabaseManager', 'DataSyncService',
    # ML
    'EnhancedFeatureGenerator', 'EnhancedPreprocessor',
    'UnifiedModelTrainer', 'EnhancedMLTrainer', 'UnifiedModelValidator',
    # Trading
    'AdvancedSignalGenerator', 'SignalGenerator', 'AdaptiveTradingSystem',
    'BacktestEngine', 'ParameterOptimizationBacktest',
    # Services
    'MarketSelectorService', 'StockListManager',
    'DataRepairService', 'BatchDataRepair'
]