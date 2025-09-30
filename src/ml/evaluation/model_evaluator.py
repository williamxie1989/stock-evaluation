"""
模型评估器 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.evaluation_cache = {}
        self.performance_history = []
        logger.info("ModelEvaluator initialized")
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """评估分类模型"""
        try:
            # 基础指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'sample_count': len(y_true)
            }
            
            # 如果有概率预测，计算AUC等指标
            if y_prob is not None and len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['auc'] = auc
                except Exception as e:
                    logger.warning(f"AUC计算失败: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"分类模型评估失败: {e}")
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'sample_count': len(y_true),
                'error': str(e)
            }
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """评估回归模型"""
        try:
            # 基础指标
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # 计算相对误差
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'mape': mape,
                'sample_count': len(y_true)
            }
            
        except Exception as e:
            logger.error(f"回归模型评估失败: {e}")
            return {
                'mse': 0,
                'rmse': 0,
                'mae': 0,
                'r2_score': 0,
                'mape': 0,
                'sample_count': len(y_true),
                'error': str(e)
            }
    
    def evaluate_trading_strategy(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """评估交易策略"""
        try:
            if len(returns) == 0:
                return {'error': '无收益数据'}
            
            # 基础统计
            total_return = np.prod(1 + returns) - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            
            # 风险指标
            downside_returns = returns[returns < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # 夏普比率
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # 索提诺比率
            sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
            
            # 最大回撤
            cumulative_returns = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # 胜率
            win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
            
            # 盈亏比
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            profit_factor = np.sum(positive_returns) / abs(np.sum(negative_returns)) if len(negative_returns) > 0 else float('inf')
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sample_count': len(returns)
            }
            
            # 如果有基准收益，计算相对指标
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                try:
                    # 计算相对收益
                    relative_returns = returns - benchmark_returns
                    
                    # 信息比率
                    tracking_error = np.std(relative_returns) * np.sqrt(252)
                    information_ratio = np.mean(relative_returns) * 252 / tracking_error if tracking_error > 0 else 0
                    
                    # Beta
                    covariance = np.cov(returns, benchmark_returns)[0, 1]
                    benchmark_variance = np.var(benchmark_returns)
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
                    
                    # Alpha
                    alpha = annualized_return - beta * (np.mean(benchmark_returns) * 252)
                    
                    metrics.update({
                        'information_ratio': information_ratio,
                        'beta': beta,
                        'alpha': alpha,
                        'tracking_error': tracking_error
                    })
                except Exception as e:
                    logger.warning(f"基准相关指标计算失败: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"交易策略评估失败: {e}")
            return {
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sample_count': len(returns),
                'error': str(e)
            }
    
    def cross_validate_model(self, X: pd.DataFrame, y: np.ndarray, model, cv_folds: int = 5) -> Dict[str, Any]:
        """交叉验证模型"""
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
            
            # 确定任务类型
            if len(np.unique(y)) <= 10:  # 分类任务
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scoring = 'f1_weighted'
            else:  # 回归任务
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scoring = 'r2'
            
            # 执行交叉验证
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            return {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'cv_folds': cv_folds,
                'scoring': scoring
            }
            
        except Exception as e:
            logger.error(f"交叉验证失败: {e}")
            return {
                'cv_mean': 0,
                'cv_std': 0,
                'cv_scores': [],
                'cv_folds': cv_folds,
                'scoring': 'unknown',
                'error': str(e)
            }
    
    def evaluate_feature_importance(self, model, feature_names: List[str]) -> Dict[str, Any]:
        """评估特征重要性"""
        try:
            importance_dict = {}
            
            # 获取特征重要性
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                # 线性模型的系数
                coef = np.abs(model.coef_)
                if len(coef.shape) > 1:
                    coef = coef.mean(axis=0)  # 多类别情况下取平均
                importance_dict = dict(zip(feature_names, coef))
            else:
                logger.warning("模型不支持特征重要性评估")
                return {'error': '模型不支持特征重要性评估'}
            
            # 排序并返回
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'feature_importance': dict(sorted_importance),
                'top_features': sorted_importance[:10] if len(sorted_importance) > 10 else sorted_importance,
                'total_features': len(feature_names)
            }
            
        except Exception as e:
            logger.error(f"特征重要性评估失败: {e}")
            return {
                'feature_importance': {},
                'top_features': [],
                'total_features': len(feature_names),
                'error': str(e)
            }
    
    def evaluate_model_stability(self, metrics_history: List[Dict[str, float]]) -> Dict[str, Any]:
        """评估模型稳定性"""
        try:
            if len(metrics_history) < 2:
                return {'error': '历史数据不足'}
            
            # 提取关键指标
            df = pd.DataFrame(metrics_history)
            
            stability_metrics = {}
            
            # 计算每个指标的稳定性
            for col in df.columns:
                if col != 'timestamp':
                    values = df[col].values
                    stability_metrics[f'{col}_mean'] = np.mean(values)
                    stability_metrics[f'{col}_std'] = np.std(values)
                    stability_metrics[f'{col}_cv'] = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    stability_metrics[f'{col}_trend'] = np.polyfit(range(len(values)), values, 1)[0]
            
            # 整体稳定性评分
            cv_values = [v for k, v in stability_metrics.items() if k.endswith('_cv') and not np.isnan(v)]
            overall_stability = 1 - np.mean(cv_values) if cv_values else 0
            
            stability_metrics['overall_stability'] = overall_stability
            stability_metrics['evaluation_count'] = len(metrics_history)
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"模型稳定性评估失败: {e}")
            return {
                'overall_stability': 0,
                'evaluation_count': len(metrics_history),
                'error': str(e)
            }
    
    def get_evaluation_summary(self, model_name: str, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """获取评估摘要"""
        return {
            'model_name': model_name,
            'evaluation_time': datetime.now(),
            'metrics': evaluation_results,
            'performance_grade': self._grade_performance(evaluation_results),
            'recommendations': self._generate_recommendations(evaluation_results)
        }
    
    def _grade_performance(self, metrics: Dict[str, Any]) -> str:
        """性能评级"""
        try:
            # 分类模型评级
            if 'accuracy' in metrics:
                accuracy = metrics['accuracy']
                if accuracy >= 0.9:
                    return 'A'
                elif accuracy >= 0.8:
                    return 'B'
                elif accuracy >= 0.7:
                    return 'C'
                else:
                    return 'D'
            
            # 回归模型评级
            elif 'r2_score' in metrics:
                r2 = metrics['r2_score']
                if r2 >= 0.8:
                    return 'A'
                elif r2 >= 0.6:
                    return 'B'
                elif r2 >= 0.4:
                    return 'C'
                else:
                    return 'D'
            
            # 交易策略评级
            elif 'sharpe_ratio' in metrics:
                sharpe = metrics['sharpe_ratio']
                if sharpe >= 2.0:
                    return 'A'
                elif sharpe >= 1.0:
                    return 'B'
                elif sharpe >= 0.5:
                    return 'C'
                else:
                    return 'D'
            
            return 'Unknown'
            
        except Exception as e:
            logger.error(f"性能评级失败: {e}")
            return 'Unknown'
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        try:
            # 分类模型建议
            if 'accuracy' in metrics:
                accuracy = metrics['accuracy']
                if accuracy < 0.8:
                    recommendations.append("考虑增加特征工程或调整模型参数")
                
                if 'recall' in metrics and metrics['recall'] < 0.7:
                    recommendations.append("召回率较低，考虑调整分类阈值")
            
            # 回归模型建议
            elif 'r2_score' in metrics:
                r2 = metrics['r2_score']
                if r2 < 0.6:
                    recommendations.append("模型解释力不足，考虑更复杂的模型")
                
                if 'mape' in metrics and metrics['mape'] > 20:
                    recommendations.append("预测误差较大，考虑特征选择或模型调优")
            
            # 交易策略建议
            elif 'sharpe_ratio' in metrics:
                sharpe = metrics['sharpe_ratio']
                if sharpe < 1.0:
                    recommendations.append("夏普比率较低，考虑优化策略参数")
                
                if 'max_drawdown' in metrics and metrics['max_drawdown'] > 0.2:
                    recommendations.append("最大回撤较大，考虑加强风险控制")
                
                if 'win_rate' in metrics and metrics['win_rate'] < 0.5:
                    recommendations.append("胜率较低，考虑改进信号生成逻辑")
            
            if not recommendations:
                recommendations.append("模型表现良好，建议定期监控和维护")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            return ["评估过程中出现错误，建议重新评估"]
    
    def reset(self):
        """重置评估器"""
        self.evaluation_cache.clear()
        self.performance_history.clear()
        logger.info("模型评估器已重置")