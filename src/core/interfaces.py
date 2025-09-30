"""
核心接口定义
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, date


# 数据提供者接口
class DataProviderInterface(ABC):
    """数据提供者接口"""
    
    @abstractmethod
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票数据"""
        pass
    
    @abstractmethod
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        """获取实时数据"""
        pass
    
    @abstractmethod
    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        """获取财务数据"""
        pass
    
    @abstractmethod
    def get_market_overview(self) -> Dict[str, Any]:
        """获取市场概览"""
        pass


# 统一数据提供者接口
class UnifiedDataProviderInterface(ABC):
    """统一数据提供者接口 - 整合多个数据源"""
    
    @abstractmethod
    def add_primary_provider(self, provider: 'DataProviderInterface') -> None:
        """添加主要数据提供者"""
        pass
    
    @abstractmethod
    def add_fallback_provider(self, provider: 'DataProviderInterface') -> None:
        """添加备用数据提供者"""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                          quality_threshold: float = 0.8) -> Optional[pd.DataFrame]:
        """获取历史数据，支持质量评估和自动切换"""
        pass
    
    @abstractmethod
    def get_realtime_data(self, symbols: List[str], max_retries: int = 3) -> Optional[Dict[str, Dict[str, Any]]]:
        """获取实时数据，支持失败重试"""
        pass
    
    @abstractmethod
    def validate_data_source(self, provider: 'DataProviderInterface', 
                           test_symbols: List[str] = None) -> Dict[str, Any]:
        """验证数据源可靠性"""
        pass


# 交易信号接口
class SignalGeneratorInterface(ABC):
    """交易信号生成器接口"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def get_signal_strength(self, data: pd.DataFrame, signal_type: str) -> float:
        """获取信号强度"""
        pass


# 交易系统接口
class TradingSystemInterface(ABC):
    """交易系统接口"""
    
    @abstractmethod
    def execute_trade(self, symbol: str, action: str, quantity: int, price: float) -> Dict[str, Any]:
        """执行交易"""
        pass
    
    @abstractmethod
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """平仓"""
        pass
    
    @abstractmethod
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取投资组合摘要"""
        pass


# 风险管理接口
class RiskManagementInterface(ABC):
    """风险管理接口"""
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, capital: float, risk_per_trade: float) -> int:
        """计算仓位大小"""
        pass
    
    @abstractmethod
    def check_risk_limits(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """检查风险限制"""
        pass
    
    @abstractmethod
    def get_stop_loss_price(self, symbol: str, entry_price: float, position_type: str) -> float:
        """获取止损价格"""
        pass


# 特征生成器接口
class FeatureGeneratorInterface(ABC):
    """特征生成器接口"""
    
    @abstractmethod
    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """生成特征"""
        pass
    
    @abstractmethod
    def select_features(self, features: pd.DataFrame, target: pd.Series, **kwargs) -> List[str]:
        """选择特征"""
        pass


# 模型训练器接口
class ModelTrainerInterface(ABC):
    """模型训练器接口"""
    
    @abstractmethod
    def train_model(self, features: pd.DataFrame, target: pd.Series, model_type: str, **kwargs) -> Any:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, model: Any, features: pd.DataFrame) -> Union[pd.Series, np.ndarray]:
        """预测"""
        pass
    
    @abstractmethod
    def evaluate_model(self, model: Any, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """评估模型"""
        pass


# 回测引擎接口
class BacktestEngineInterface(ABC):
    """回测引擎接口"""
    
    @abstractmethod
    def run_backtest(self, data: pd.DataFrame, strategy: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """运行回测"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """获取性能指标"""
        pass


# 参数优化器接口
class ParameterOptimizerInterface(ABC):
    """参数优化器接口"""
    
    @abstractmethod
    def optimize_parameters(self, strategy_class: Any, data: pd.DataFrame, 
                           param_ranges: Dict[str, Tuple], **kwargs) -> Dict[str, Any]:
        """优化参数"""
        pass
    
    @abstractmethod
    def get_best_parameters(self) -> Dict[str, Any]:
        """获取最佳参数"""
        pass


# 模型评估器接口
class ModelEvaluatorInterface(ABC):
    """模型评估器接口"""
    
    @abstractmethod
    def evaluate_classification_model(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
        """评估分类模型"""
        pass
    
    @abstractmethod
    def evaluate_regression_model(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
        """评估回归模型"""
        pass
    
    @abstractmethod
    def cross_validate_model(self, model: Any, features: pd.DataFrame, 
                           target: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """交叉验证模型"""
        pass


# 股票状态过滤器接口
class StockStatusFilterInterface(ABC):
    """股票状态过滤器接口"""
    
    @abstractmethod
    def filter_by_status(self, stocks: List[Dict[str, Any]], status_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """按状态过滤股票"""
        pass
    
    @abstractmethod
    def get_stock_status(self, symbol: str) -> Dict[str, Any]:
        """获取股票状态"""
        pass


# 市场选择服务接口
class MarketSelectorServiceInterface(ABC):
    """市场选择服务接口"""
    
    @abstractmethod
    def select_market(self, criteria: Dict[str, Any]) -> str:
        """选择市场"""
        pass
    
    @abstractmethod
    def get_market_conditions(self, market: str) -> Dict[str, Any]:
        """获取市场条件"""
        pass


# 数据同步服务接口
class DataSyncServiceInterface(ABC):
    """数据同步服务接口"""
    
    @abstractmethod
    def sync_stock_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """同步股票数据"""
        pass
    
    @abstractmethod
    def sync_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        """同步实时数据"""
        pass
    
    @abstractmethod
    def sync_financial_data(self, symbols: List[str]) -> Dict[str, Any]:
        """同步财务数据"""
        pass


# 股票列表管理器接口
class StockListManagerInterface(ABC):
    """股票列表管理器接口"""
    
    @abstractmethod
    def load_stock_list(self, file_path: str) -> List[Dict[str, Any]]:
        """加载股票列表"""
        pass
    
    @abstractmethod
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """获取股票信息"""
        pass
    
    @abstractmethod
    def filter_stocks(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """过滤股票"""
        pass


# 数据修复服务接口
class DataRepairServiceInterface(ABC):
    """数据修复服务接口"""
    
    @abstractmethod
    def detect_data_issues(self, data: pd.DataFrame, symbol: str = 'Unknown') -> Dict[str, Any]:
        """检测数据问题"""
        pass
    
    @abstractmethod
    def repair_missing_values(self, data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """修复缺失值"""
        pass
    
    @abstractmethod
    def comprehensive_repair(self, data: pd.DataFrame, symbol: str = 'Unknown') -> Dict[str, Any]:
        """综合修复"""
        pass


# 批量数据修复接口
class BatchDataRepairInterface(ABC):
    """批量数据修复接口"""
    
    @abstractmethod
    def repair_stock_data_batch(self, stock_data_dict: Dict[str, pd.DataFrame], 
                               repair_methods: Dict[str, str] = None) -> Dict[str, Any]:
        """批量修复股票数据"""
        pass
    
    @abstractmethod
    def repair_market_data(self, market_data: pd.DataFrame, 
                          market_symbol: str = 'Market') -> Dict[str, Any]:
        """修复市场数据"""
        pass


# 特征生成器接口（增强版）
class EnhancedFeaturesInterface(ABC):
    """增强特征生成器接口"""
    
    @abstractmethod
    def generate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成技术指标"""
        pass
    
    @abstractmethod
    def generate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成动量特征"""
        pass
    
    @abstractmethod
    def generate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成波动率特征"""
        pass


# 模型训练器接口（统一版）
class UnifiedModelTrainerInterface(ABC):
    """统一模型训练器接口"""
    
    @abstractmethod
    def train_classification_model(self, features: pd.DataFrame, target: pd.Series, 
                                  model_type: str = 'random_forest', **kwargs) -> Any:
        """训练分类模型"""
        pass
    
    @abstractmethod
    def train_regression_model(self, features: pd.DataFrame, target: pd.Series, 
                              model_type: str = 'random_forest', **kwargs) -> Any:
        """训练回归模型"""
        pass
    
    @abstractmethod
    def predict_with_model(self, model: Any, features: pd.DataFrame, 
                          model_type: str = 'classification') -> Union[pd.Series, np.ndarray]:
        """使用模型预测"""
        pass


# 模型验证器接口（统一版）
class UnifiedModelValidatorInterface(ABC):
    """统一模型验证器接口"""
    
    @abstractmethod
    def validate_classification_model(self, model: Any, features: pd.DataFrame, 
                                    target: pd.Series) -> Dict[str, Any]:
        """验证分类模型"""
        pass
    
    @abstractmethod
    def validate_regression_model(self, model: Any, features: pd.DataFrame, 
                                target: pd.Series) -> Dict[str, Any]:
        """验证回归模型"""
        pass
    
    @abstractmethod
    def validate_trading_strategy(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证交易策略"""
        pass


# 高级信号生成器接口
class AdvancedSignalGeneratorInterface(ABC):
    """高级信号生成器接口"""
    
    @abstractmethod
    def generate_moving_average_signals(self, data: pd.DataFrame, 
                                       short_window: int = 5, long_window: int = 20) -> pd.Series:
        """生成移动平均线信号"""
        pass
    
    @abstractmethod
    def generate_rsi_signals(self, data: pd.DataFrame, window: int = 14, 
                           oversold: int = 30, overbought: int = 70) -> pd.Series:
        """生成RSI信号"""
        pass
    
    @abstractmethod
    def generate_macd_signals(self, data: pd.DataFrame, 
                            fast_period: int = 12, slow_period: int = 26, 
                            signal_period: int = 9) -> pd.Series:
        """生成MACD信号"""
        pass


# 自适应交易系统接口
class AdaptiveTradingSystemInterface(ABC):
    """自适应交易系统接口"""
    
    @abstractmethod
    def analyze_market_state(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析市场状态"""
        pass
    
    @abstractmethod
    def assess_risk_level(self, portfolio: Dict[str, Any], market_state: Dict[str, Any]) -> str:
        """评估风险等级"""
        pass
    
    @abstractmethod
    def adjust_trading_parameters(self, market_state: Dict[str, Any], 
                                risk_level: str) -> Dict[str, Any]:
        """调整交易参数"""
        pass