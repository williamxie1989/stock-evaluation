"""
参数优化和回测系统
实现第四阶段的参数优化和回测功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from backtest_engine import BacktestEngine
from advanced_signal_generator import AdvancedSignalGenerator
from adaptive_trading_system import AdaptiveTradingSystem

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """参数优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    optimization_time: float
    model_type: str

@dataclass
class BacktestResult:
    """回测结果"""
    performance: Dict[str, Any]
    trades: List[Any]
    portfolio_history: List[Any]
    optimization_params: Dict[str, Any]
    backtest_time: float

class ParameterOptimizationBacktest:
    """参数优化和回测系统"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.signal_generator = AdvancedSignalGenerator()
        self.trading_system = AdaptiveTradingSystem(initial_capital)
        self.backtest_engine = BacktestEngine(initial_capital)
        
        # 参数优化配置
        self.param_grids = self._get_param_grids()
        
    def _get_param_grids(self) -> Dict[str, Dict[str, List]]:
        """获取参数网格配置"""
        return {
            'signal_generator': {
                'rsi_period': [10, 14, 20],
                'macd_fast': [8, 12, 26],
                'macd_slow': [21, 26, 34],
                'bb_period': [14, 20, 26],
                'atr_period': [10, 14, 20],
                'confidence_threshold': [0.6, 0.7, 0.8]
            },
            'trading_system': {
                'risk_tolerance': [0.01, 0.02, 0.03],
                'position_size_multiplier': [0.5, 1.0, 1.5],
                'stop_loss_multiplier': [1.5, 2.0, 2.5],
                'take_profit_multiplier': [2.0, 2.5, 3.0],
                'volatility_threshold': [0.1, 0.15, 0.2]
            }
        }
    
    def optimize_signal_generator(self, df: pd.DataFrame, target_col: str = 'target') -> OptimizationResult:
        """优化信号生成器参数"""
        logger.info("开始优化信号生成器参数...")
        
        start_time = datetime.now()
        
        # 生成特征和标签
        X, y = self._prepare_optimization_data(df, target_col)
        
        # 创建参数优化器
        param_grid = self.param_grids['signal_generator']
        
        # 使用网格搜索优化
        best_params = {}
        best_score = 0
        cv_results = []
        
        # 简化优化过程（实际项目中可以使用更复杂的优化算法）
        for rsi_period in param_grid['rsi_period']:
            for confidence_threshold in param_grid['confidence_threshold']:
                # 设置参数
                params = {
                    'rsi_period': rsi_period,
                    'confidence_threshold': confidence_threshold
                }
                
                # 生成信号并评估
                score = self._evaluate_signal_params(df, params)
                
                cv_results.append({
                    'params': params,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"信号生成器优化完成，最佳参数: {best_params}, 最佳得分: {best_score:.4f}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            cv_results=pd.DataFrame(cv_results),
            optimization_time=optimization_time,
            model_type='signal_generator'
        )
    
    def optimize_trading_system(self, df: pd.DataFrame, signals: List[Dict[str, Any]]) -> OptimizationResult:
        """优化交易系统参数"""
        logger.info("开始优化交易系统参数...")
        
        start_time = datetime.now()
        
        param_grid = self.param_grids['trading_system']
        
        # 简化优化过程
        best_params = {}
        best_score = 0
        cv_results = []
        
        for risk_tolerance in param_grid['risk_tolerance']:
            for position_size_multiplier in param_grid['position_size_multiplier']:
                # 设置参数
                params = {
                    'risk_tolerance': risk_tolerance,
                    'position_size_multiplier': position_size_multiplier
                }
                
                # 评估参数
                score = self._evaluate_trading_params(df, signals, params)
                
                cv_results.append({
                    'params': params,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"交易系统优化完成，最佳参数: {best_params}, 最佳得分: {best_score:.4f}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            cv_results=pd.DataFrame(cv_results),
            optimization_time=optimization_time,
            model_type='trading_system'
        )
    
    def run_comprehensive_backtest(self, df: pd.DataFrame, 
                                  optimization_results: Dict[str, OptimizationResult]) -> BacktestResult:
        """运行综合回测"""
        logger.info("开始综合回测...")
        
        start_time = datetime.now()
        
        # 应用优化后的参数
        if 'signal_generator' in optimization_results:
            signal_params = optimization_results['signal_generator'].best_params
            self._apply_signal_generator_params(signal_params)
        
        if 'trading_system' in optimization_results:
            trading_params = optimization_results['trading_system'].best_params
            self._apply_trading_system_params(trading_params)
        
        # 生成优化后的信号
        optimized_signals = self.signal_generator.generate_advanced_signals(df)
        
        # 将Signal对象转换为字典格式以兼容回测引擎
        signals_dict = []
        for signal in optimized_signals:
            signals_dict.append({
                'date': signal.date,
                'type': signal.type,
                'price': signal.price,
                'reason': signal.reason,
                'confidence': signal.confidence
            })
        
        # 运行回测
        backtest_result = self.backtest_engine.run_backtest(df, signals_dict)
        
        backtest_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"综合回测完成，总收益率: {backtest_result['performance']['total_return']:.4f}")
        
        return BacktestResult(
            performance=backtest_result['performance'],
            trades=backtest_result['trades'],
            portfolio_history=backtest_result['portfolio_history'],
            optimization_params={
                'signal_generator': optimization_results.get('signal_generator', {}).best_params,
                'trading_system': optimization_results.get('trading_system', {}).best_params
            },
            backtest_time=backtest_time
        )
    
    def _prepare_optimization_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """准备优化数据"""
        # 简化版本：使用价格变化作为目标
        df_copy = df.copy()
        
        # 计算未来收益率作为目标
        df_copy['future_return'] = df_copy['Close'].pct_change(5).shift(-5)
        
        # 创建二元分类目标（涨跌）
        df_copy[target_col] = (df_copy['future_return'] > 0).astype(int)
        
        # 删除缺失值
        df_copy = df_copy.dropna()
        
        # 生成技术指标特征
        features = self._generate_technical_features(df_copy)
        
        return features, df_copy[target_col]
    
    def _generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成技术指标特征"""
        features = pd.DataFrame(index=df.index)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # 布林带
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        features['bb_upper'] = rolling_mean + (rolling_std * 2)
        features['bb_lower'] = rolling_mean - (rolling_std * 2)
        features['bb_position'] = (df['Close'] - rolling_mean) / (2 * rolling_std)
        
        # 成交量
        features['volume_sma'] = df['Volume'].rolling(window=10).mean()
        
        return features.dropna()
    
    def _evaluate_signal_params(self, df: pd.DataFrame, params: Dict[str, Any]) -> float:
        """评估信号生成器参数"""
        try:
            # 应用参数
            self._apply_signal_generator_params(params)
            
            # 生成信号
            signals = self.signal_generator.generate_advanced_signals(df)
            
            # 简化评估：信号数量和质量
            if not signals:
                return 0.0
            
            # 计算信号质量得分（正确处理Signal对象）
            total_confidence = sum(s.confidence for s in signals)
            avg_confidence = total_confidence / len(signals)
            
            # 信号多样性得分（避免过多相同类型的信号）
            signal_types = [s.type for s in signals]
            unique_types = len(set(signal_types))
            diversity_score = unique_types / len(signal_types) if signal_types else 0
            
            # 综合得分
            score = avg_confidence * 0.7 + diversity_score * 0.3
            
            return score
            
        except Exception as e:
            logger.error(f"评估信号参数时出错: {e}")
            return 0.0
    
    def _evaluate_trading_params(self, df: pd.DataFrame, signals: List[Dict[str, Any]], 
                                params: Dict[str, Any]) -> float:
        """评估交易系统参数"""
        try:
            # 应用参数
            self._apply_trading_system_params(params)
            
            # 运行简化回测
            result = self.backtest_engine.run_backtest(df, signals)
            
            # 计算综合得分
            performance = result['performance']
            
            # 使用夏普比率作为主要指标
            sharpe_score = max(0, performance.get('sharpe_ratio', 0))
            
            # 考虑最大回撤
            max_drawdown = performance.get('max_drawdown', 1)
            drawdown_score = max(0, 1 - max_drawdown)
            
            # 考虑胜率
            win_rate = performance.get('win_rate', 0)
            
            # 综合得分
            score = sharpe_score * 0.5 + drawdown_score * 0.3 + win_rate * 0.2
            
            return score
            
        except Exception as e:
            logger.error(f"评估交易参数时出错: {e}")
            return 0.0
    
    def _apply_signal_generator_params(self, params: Dict[str, Any]):
        """应用信号生成器参数"""
        for key, value in params.items():
            if hasattr(self.signal_generator, key):
                setattr(self.signal_generator, key, value)
    
    def _apply_trading_system_params(self, params: Dict[str, Any]):
        """应用交易系统参数"""
        for key, value in params.items():
            if hasattr(self.trading_system, key):
                setattr(self.trading_system, key, value)
    
    def generate_optimization_report(self, optimization_results: Dict[str, OptimizationResult]) -> str:
        """生成优化报告"""
        report = """
参数优化报告
============

优化结果汇总
------------
"""
        
        for model_type, result in optimization_results.items():
            report += f"\n{model_type.upper()} 优化结果:\n"
            report += f"- 最佳参数: {result.best_params}\n"
            report += f"- 最佳得分: {result.best_score:.4f}\n"
            report += f"- 优化时间: {result.optimization_time:.2f}秒\n"
            report += f"- CV结果数量: {len(result.cv_results)}\n"
        
        return report
    
    def generate_backtest_report(self, backtest_result: BacktestResult) -> str:
        """生成回测报告"""
        performance = backtest_result.performance
        
        report = """
综合回测报告
============

性能指标
--------
"""
        
        report += f"- 初始资金: ¥{performance['initial_value']:,.2f}\n"
        report += f"- 最终资金: ¥{performance['final_value']:,.2f}\n"
        report += f"- 总收益率: {performance['total_return']:.2%}\n"
        report += f"- 年化收益率: {performance['annualized_return']:.2%}\n"
        report += f"- 最大回撤: {performance['max_drawdown']:.2%}\n"
        report += f"- 夏普比率: {performance['sharpe_ratio']:.2f}\n"
        report += f"- 总交易次数: {performance['total_trades']}\n"
        report += f"- 胜率: {performance['win_rate']:.2%}\n"
        
        report += "\n优化参数\n--------\n"
        for model_type, params in backtest_result.optimization_params.items():
            report += f"{model_type}: {params}\n"
        
        report += f"\n回测时间: {backtest_result.backtest_time:.2f}秒\n"
        
        return report

# 测试函数
def test_parameter_optimization_backtest():
    """测试参数优化和回测系统"""
    logger.info("开始测试参数优化和回测系统...")
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=200)
    np.random.seed(42)
    
    # 生成更真实的股票数据
    returns = np.random.normal(0.001, 0.02, 200)
    prices = [100]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    prices = prices[1:]
    
    data = {
        'Open': [p * 0.99 for p in prices],  # 开盘价略低于收盘价
        'High': [p * 1.02 for p in prices],  # 最高价高于收盘价
        'Low': [p * 0.98 for p in prices],   # 最低价低于收盘价
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, 200)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # 创建优化回测系统
    system = ParameterOptimizationBacktest(initial_capital=100000)
    
    # 优化信号生成器
    signal_optimization = system.optimize_signal_generator(df)
    
    # 生成优化后的信号
    optimized_signals = system.signal_generator.generate_advanced_signals(df)
    
    # 优化交易系统
    trading_optimization = system.optimize_trading_system(df, optimized_signals)
    
    # 运行综合回测
    optimization_results = {
        'signal_generator': signal_optimization,
        'trading_system': trading_optimization
    }
    
    backtest_result = system.run_comprehensive_backtest(df, optimization_results)
    
    # 生成报告
    optimization_report = system.generate_optimization_report(optimization_results)
    backtest_report = system.generate_backtest_report(backtest_result)
    
    print(optimization_report)
    print(backtest_report)
    
    logger.info("参数优化和回测系统测试完成")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    test_parameter_optimization_backtest()