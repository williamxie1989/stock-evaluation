"""
统一模型验证模块 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
import os

logger = logging.getLogger(__name__)

class UnifiedModelValidator:
    """统一模型验证器"""
    
    def __init__(self, validation_dir: str = "validation_results"):
        self.validation_dir = validation_dir
        self.validation_results = {}
        self.model_performance_cache = {}
        
        # 创建验证结果目录
        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir)
        
        logger.info(f"UnifiedModelValidator initialized with validation_dir: {validation_dir}")
    
    def validate_classification_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                                    model_name: str = 'classifier') -> Dict[str, Any]:
        """验证分类模型"""
        try:
            if X.empty or y.empty:
                return {'error': '验证数据为空'}
            
            # 数据预处理
            X_clean = X.dropna()
            y_clean = y.loc[X_clean.index]
            
            if len(X_clean) < 50:
                return {'error': '验证数据不足'}
            
            # 预测
            y_pred = model.predict(X_clean)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_clean)
            
            # 基础评估指标
            performance = {
                'accuracy': float(accuracy_score(y_clean, y_pred)),
                'precision': float(precision_score(y_clean, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_clean, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_clean, y_pred, average='weighted', zero_division=0)),
                'validation_samples': len(X_clean),
                'timestamp': datetime.now()
            }
            
            # 交叉验证
            try:
                cv_scores = cross_val_score(model, X_clean, y_clean, cv=5, scoring='accuracy')
                performance['cv_accuracy_mean'] = float(cv_scores.mean())
                performance['cv_accuracy_std'] = float(cv_scores.std())
            except Exception as e:
                logger.warning(f"交叉验证失败: {e}")
                performance['cv_accuracy_mean'] = None
                performance['cv_accuracy_std'] = None
            
            # 时间序列交叉验证
            try:
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores_ts = cross_val_score(model, X_clean, y_clean, cv=tscv, scoring='accuracy')
                performance['cv_accuracy_ts_mean'] = float(cv_scores_ts.mean())
                performance['cv_accuracy_ts_std'] = float(cv_scores_ts.std())
            except Exception as e:
                logger.warning(f"时间序列交叉验证失败: {e}")
                performance['cv_accuracy_ts_mean'] = None
                performance['cv_accuracy_ts_std'] = None
            
            # 保存验证结果
            self.validation_results[model_name] = performance
            self.model_performance_cache[model_name] = performance
            
            # 保存到文件
            result_path = os.path.join(self.validation_dir, f"{model_name}_classification_validation.json")
            self._save_validation_result(performance, result_path)
            
            logger.info(f"分类模型验证完成: {model_name}, 准确率: {performance['accuracy']:.4f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"分类模型验证失败: {e}")
            return {'error': str(e)}
    
    def validate_regression_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                                model_name: str = 'regressor') -> Dict[str, Any]:
        """验证回归模型"""
        try:
            if X.empty or y.empty:
                return {'error': '验证数据为空'}
            
            # 数据预处理
            X_clean = X.dropna()
            y_clean = y.loc[X_clean.index]
            
            if len(X_clean) < 50:
                return {'error': '验证数据不足'}
            
            # 预测
            y_pred = model.predict(X_clean)
            
            # 基础评估指标
            performance = {
                'mse': float(mean_squared_error(y_clean, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_clean, y_pred))),
                'mae': float(mean_absolute_error(y_clean, y_pred)),
                'r2_score': float(r2_score(y_clean, y_pred)),
                'validation_samples': len(X_clean),
                'timestamp': datetime.now()
            }
            
            # 计算相对误差
            mape = np.mean(np.abs((y_clean - y_pred) / y_clean)) * 100
            performance['mape'] = float(mape)
            
            # 交叉验证
            try:
                cv_scores = cross_val_score(model, X_clean, y_clean, cv=5, scoring='r2')
                performance['cv_r2_mean'] = float(cv_scores.mean())
                performance['cv_r2_std'] = float(cv_scores.std())
            except Exception as e:
                logger.warning(f"交叉验证失败: {e}")
                performance['cv_r2_mean'] = None
                performance['cv_r2_std'] = None
            
            # 时间序列交叉验证
            try:
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores_ts = cross_val_score(model, X_clean, y_clean, cv=tscv, scoring='r2')
                performance['cv_r2_ts_mean'] = float(cv_scores_ts.mean())
                performance['cv_r2_ts_std'] = float(cv_scores_ts.std())
            except Exception as e:
                logger.warning(f"时间序列交叉验证失败: {e}")
                performance['cv_r2_ts_mean'] = None
                performance['cv_r2_ts_std'] = None
            
            # 保存验证结果
            self.validation_results[model_name] = performance
            self.model_performance_cache[model_name] = performance
            
            # 保存到文件
            result_path = os.path.join(self.validation_dir, f"{model_name}_regression_validation.json")
            self._save_validation_result(performance, result_path)
            
            logger.info(f"回归模型验证完成: {model_name}, R²: {performance['r2_score']:.4f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"回归模型验证失败: {e}")
            return {'error': str(e)}
    
    def validate_trading_strategy(self, strategy_results: Dict[str, Any], 
                                  model_name: str = 'trading_strategy') -> Dict[str, Any]:
        """验证交易策略"""
        try:
            if not strategy_results or 'trades' not in strategy_results:
                return {'error': '策略结果数据不完整'}
            
            trades = strategy_results['trades']
            returns = strategy_results.get('returns', [])
            
            if not trades and not returns:
                return {'error': '没有交易记录或收益数据'}
            
            performance = {
                'total_trades': len(trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'validation_samples': len(trades),
                'timestamp': datetime.now()
            }
            
            if trades:
                # 计算胜率
                winning_trades = [t for t in trades if t.get('return', 0) > 0]
                losing_trades = [t for t in trades if t.get('return', 0) <= 0]
                
                performance['winning_trades'] = len(winning_trades)
                performance['losing_trades'] = len(losing_trades)
                performance['total_trades'] = len(trades)
                performance['win_rate'] = len(winning_trades) / len(trades) if trades else 0.0
                
                # 计算平均盈亏
                if winning_trades:
                    performance['avg_win'] = np.mean([t.get('return', 0) for t in winning_trades])
                if losing_trades:
                    performance['avg_loss'] = np.mean([t.get('return', 0) for t in losing_trades])
                
                # 计算收益因子
                if performance['avg_loss'] != 0:
                    performance['profit_factor'] = abs(performance['avg_win'] / performance['avg_loss'])
                
                # 计算总收益
                performance['total_return'] = sum(t.get('return', 0) for t in trades)
            
            if returns:
                # 计算夏普比率
                if len(returns) > 1:
                    returns_array = np.array(returns)
                    excess_returns = returns_array - 0.02  # 假设无风险利率为2%
                    if np.std(excess_returns) != 0:
                        performance['sharpe_ratio'] = np.mean(excess_returns) / np.std(excess_returns)
                
                # 计算最大回撤
                if len(returns) > 1:
                    cumulative_returns = np.cumsum(returns)
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdowns = cumulative_returns - running_max
                    performance['max_drawdown'] = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # 保存验证结果
            self.validation_results[model_name] = performance
            
            # 保存到文件
            result_path = os.path.join(self.validation_dir, f"{model_name}_trading_validation.json")
            self._save_validation_result(performance, result_path)
            
            logger.info(f"交易策略验证完成: {model_name}, 胜率: {performance['win_rate']:.4f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"交易策略验证失败: {e}")
            return {'error': str(e)}
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """比较多个模型"""
        try:
            comparison = {
                'models': {},
                'best_model': None,
                'comparison_timestamp': datetime.now()
            }
            
            for model_name in model_names:
                if model_name in self.validation_results:
                    comparison['models'][model_name] = self.validation_results[model_name]
            
            if not comparison['models']:
                return {'error': '没有可用的模型验证结果'}
            
            # 确定最佳模型（简化版本）
            best_score = -float('inf')
            best_model = None
            
            for model_name, performance in comparison['models'].items():
                if 'accuracy' in performance:  # 分类模型
                    score = performance['accuracy']
                elif 'r2_score' in performance:  # 回归模型
                    score = performance['r2_score']
                elif 'win_rate' in performance:  # 交易策略
                    score = performance['win_rate']
                else:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            comparison['best_model'] = best_model
            
            logger.info(f"模型比较完成: 最佳模型 {best_model}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"模型比较失败: {e}")
            return {'error': str(e)}
    
    def generate_validation_report(self, model_name: str) -> Dict[str, Any]:
        """生成验证报告"""
        try:
            if model_name not in self.validation_results:
                return {'error': '模型验证结果不存在'}
            
            performance = self.validation_results[model_name]
            
            report = {
                'model_name': model_name,
                'validation_date': performance.get('timestamp', datetime.now()),
                'performance_summary': {},
                'recommendations': [],
                'risk_assessment': {}
            }
            
            # 分类模型报告
            if 'accuracy' in performance:
                report['performance_summary'] = {
                    'accuracy': performance['accuracy'],
                    'precision': performance['precision'],
                    'recall': performance['recall'],
                    'f1_score': performance['f1_score']
                }
                
                # 建议
                if performance['accuracy'] >= 0.8:
                    report['recommendations'].append('模型表现优秀，可以考虑部署')
                elif performance['accuracy'] >= 0.6:
                    report['recommendations'].append('模型表现一般，建议进一步优化')
                else:
                    report['recommendations'].append('模型表现较差，需要重新训练或调整')
                
                # 风险评估
                report['risk_assessment'] = {
                    'overfitting_risk': 'low' if performance.get('cv_accuracy_std', 0) < 0.05 else 'high',
                    'generalization_ability': 'good' if performance.get('cv_accuracy_mean', 0) > 0.7 else 'poor'
                }
            
            # 回归模型报告
            elif 'r2_score' in performance:
                report['performance_summary'] = {
                    'r2_score': performance['r2_score'],
                    'rmse': performance['rmse'],
                    'mae': performance['mae'],
                    'mape': performance.get('mape', 0)
                }
                
                # 建议
                if performance['r2_score'] >= 0.7:
                    report['recommendations'].append('模型解释能力强，预测效果良好')
                elif performance['r2_score'] >= 0.4:
                    report['recommendations'].append('模型解释能力一般，建议增加特征或调整参数')
                else:
                    report['recommendations'].append('模型解释能力弱，需要重新设计')
                
                # 风险评估
                report['risk_assessment'] = {
                    'prediction_error_risk': 'low' if performance['mape'] < 10 else 'high',
                    'model_stability': 'good' if performance.get('cv_r2_std', 0) < 0.1 else 'poor'
                }
            
            # 交易策略报告
            elif 'win_rate' in performance:
                report['performance_summary'] = {
                    'win_rate': performance['win_rate'],
                    'total_return': performance['total_return'],
                    'sharpe_ratio': performance['sharpe_ratio'],
                    'max_drawdown': performance['max_drawdown']
                }
                
                # 建议
                if performance['win_rate'] >= 0.6 and performance['sharpe_ratio'] > 1.0:
                    report['recommendations'].append('策略表现优秀，可以考虑实盘测试')
                elif performance['win_rate'] >= 0.5:
                    report['recommendations'].append('策略表现一般，建议优化参数')
                else:
                    report['recommendations'].append('策略胜率较低，需要重新设计')
                
                # 风险评估
                report['risk_assessment'] = {
                    'drawdown_risk': 'low' if performance['max_drawdown'] < 0.1 else 'high',
                    'return_stability': 'good' if performance['sharpe_ratio'] > 0.5 else 'poor'
                }
            
            logger.info(f"验证报告生成完成: {model_name}")
            
            return report
            
        except Exception as e:
            logger.error(f"验证报告生成失败: {e}")
            return {'error': str(e)}
    
    def _save_validation_result(self, result: Dict[str, Any], file_path: str):
        """保存验证结果到文件"""
        try:
            # 转换datetime对象为字符串
            result_copy = result.copy()
            for key, value in result_copy.items():
                if isinstance(value, datetime):
                    result_copy[key] = value.isoformat()
            
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_copy, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"保存验证结果失败: {e}")
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """获取验证历史"""
        try:
            history = []
            for model_name, result in self.validation_results.items():
                history_entry = {
                    'model_name': model_name,
                    'validation_date': result.get('timestamp', datetime.now()),
                    'performance_summary': {}
                }
                
                # 提取关键性能指标
                if 'accuracy' in result:
                    history_entry['performance_summary'] = {
                        'accuracy': result['accuracy'],
                        'f1_score': result['f1_score']
                    }
                elif 'r2_score' in result:
                    history_entry['performance_summary'] = {
                        'r2_score': result['r2_score'],
                        'rmse': result['rmse']
                    }
                elif 'win_rate' in result:
                    history_entry['performance_summary'] = {
                        'win_rate': result['win_rate'],
                        'total_return': result['total_return']
                    }
                
                history.append(history_entry)
            
            # 按时间排序
            history.sort(key=lambda x: x['validation_date'], reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"获取验证历史失败: {e}")
            return []
    
    def reset(self):
        """重置验证器"""
        self.validation_results.clear()
        self.model_performance_cache.clear()
        logger.info("统一模型验证器已重置")