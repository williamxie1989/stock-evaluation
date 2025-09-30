"""
核心工厂模块 - 精简版
提供系统中关键对象的创建工厂
"""

import logging
from typing import Dict, Any, Optional
from .interfaces import DataProviderInterface as DataProvider, TradingSystemInterface as TradingSystem, SignalGeneratorInterface as SignalGenerator
from ..data.providers.akshare_provider import AkshareDataProvider
from ..data.providers.enhanced_provider import EnhancedDataProvider  
from ..data.providers.realtime_provider import EnhancedRealtimeProvider
from ..core.backtest_engine import BacktestEngine
from ..trading.optimization.parameter_optimizer import ParameterOptimizer
from ..trading.signals.signal_generator import SignalGenerator
from ..trading.systems.trading_system import TradingSystem
from ..core.risk_management import RiskManager
from ..ml.models.model_trainer import ModelTrainer
from ..ml.features.feature_generator import FeatureGenerator
from ..ml.evaluation.model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class CoreFactory:
    """核心工厂类 - 负责创建系统中所有关键对象"""
    
    def __init__(self):
        self._providers = {}
        self._systems = {}
        self._generators = {}
        logger.info("CoreFactory 初始化完成")
    
    def create_data_provider(self, provider_type: str = "akshare", **kwargs) -> DataProvider:
        """创建数据提供器"""
        if provider_type in self._providers:
            return self._providers[provider_type]
        
        if provider_type == "akshare":
            provider = AkshareDataProvider(**kwargs)
        elif provider_type == "enhanced":
            provider = EnhancedDataProvider(**kwargs)
        else:
            raise ValueError(f"不支持的数据提供器类型: {provider_type}")
        
        self._providers[provider_type] = provider
        logger.info(f"创建数据提供器: {provider_type}")
        return provider
    
    def create_realtime_provider(self, **kwargs) -> EnhancedRealtimeProvider:
        """创建实时数据提供器"""
        return EnhancedRealtimeProvider(**kwargs)
    
    def create_backtest_engine(self, **kwargs) -> BacktestEngine:
        """创建回测引擎"""
        return BacktestEngine(**kwargs)
    
    def create_parameter_optimizer(self, **kwargs) -> ParameterOptimizer:
        """创建参数优化器"""
        return ParameterOptimizer(**kwargs)
    
    def create_signal_generator(self, **kwargs) -> SignalGenerator:
        """创建信号生成器"""
        if "signal_generator" in self._generators:
            return self._generators["signal_generator"]
        
        generator = SignalGenerator(**kwargs)
        self._generators["signal_generator"] = generator
        logger.info("创建信号生成器")
        return generator
    
    def create_trading_system(self, **kwargs) -> TradingSystem:
        """创建交易系统"""
        if "trading_system" in self._systems:
            return self._systems["trading_system"]
        
        system = TradingSystem(**kwargs)
        self._systems["trading_system"] = system
        logger.info("创建交易系统")
        return system
    
    def create_risk_manager(self, **kwargs) -> RiskManager:
        """创建风险管理器"""
        return RiskManager(**kwargs)
    
    def create_model_trainer(self, **kwargs) -> ModelTrainer:
        """创建模型训练器"""
        return ModelTrainer(**kwargs)
    
    def create_feature_generator(self, **kwargs) -> FeatureGenerator:
        """创建特征生成器"""
        return FeatureGenerator(**kwargs)
    
    def create_model_evaluator(self, **kwargs) -> ModelEvaluator:
        """创建模型评估器"""
        return ModelEvaluator(**kwargs)
    
    def create_complete_system(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建完整的交易系统"""
        logger.info("创建完整交易系统")
        
        # 创建数据提供器
        data_provider = self.create_data_provider(
            config.get("data_provider", "akshare")
        )
        
        # 创建实时数据提供器
        realtime_provider = self.create_realtime_provider()
        
        # 创建回测引擎
        backtest_engine = self.create_backtest_engine()
        
        # 创建参数优化器
        parameter_optimizer = self.create_parameter_optimizer()
        
        # 创建信号生成器
        signal_generator = self.create_signal_generator()
        
        # 创建交易系统
        trading_system = self.create_trading_system()
        
        # 创建风险管理器
        risk_manager = self.create_risk_manager()
        
        # 创建ML组件
        model_trainer = self.create_model_trainer()
        feature_generator = self.create_feature_generator()
        model_evaluator = self.create_model_evaluator()
        
        return {
            "data_provider": data_provider,
            "realtime_provider": realtime_provider,
            "backtest_engine": backtest_engine,
            "parameter_optimizer": parameter_optimizer,
            "signal_generator": signal_generator,
            "trading_system": trading_system,
            "risk_manager": risk_manager,
            "model_trainer": model_trainer,
            "feature_generator": feature_generator,
            "model_evaluator": model_evaluator
        }

# 全局工厂实例
factory = CoreFactory()

def get_factory() -> CoreFactory:
    """获取全局工厂实例"""
    return factory