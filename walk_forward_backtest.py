#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward回测模块
实现按时间切片滚动训练与验证，计算胜率、收益、夏普、最大回撤等指标
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle

from db import DatabaseManager
from ml_trainer import MLTrainer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalkForwardBacktest:
    """
    Walk-Forward回测器
    """
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.ml_trainer = MLTrainer(db_manager=self.db_manager)
        
    def run_backtest(self, 
                    symbol: str,
                    period: str = '10d',
                    train_window: int = 60,  # 训练窗口天数
                    test_window: int = 20,   # 测试窗口天数
                    step_size: int = 10,     # 步长天数
                    min_samples: int = 50) -> Dict:
        """
        运行Walk-Forward回测
        
        Args:
            symbol: 股票代码
            period: 预测期间 (5d, 10d, 20d)
            train_window: 训练窗口天数
            test_window: 测试窗口天数
            step_size: 步长天数
            min_samples: 最小样本数
            
        Returns:
            回测结果字典
        """
        logger.info(f"开始Walk-Forward回测: {symbol}, 期间: {period}")
        
        # 获取样本数据
        samples_df = self._load_samples(symbol, period)
        if samples_df.empty:
            logger.error(f"未找到样本数据: {symbol}, {period}")
            return {}
            
        # 按日期排序
        samples_df = samples_df.sort_values('date').reset_index(drop=True)
        logger.info(f"加载样本数据: {len(samples_df)} 条")
        
        # 执行时间切片回测
        results = self._time_series_split_backtest(
            samples_df, train_window, test_window, step_size, min_samples
        )
        
        # 计算回测指标
        metrics = self._calculate_backtest_metrics(results)
        
        # 保存结果到数据库
        self._save_backtest_results(symbol, period, metrics, results)
        
        logger.info(f"回测完成: {symbol}, 总收益率: {metrics.get('total_return', 0):.4f}")
        return {
            'symbol': symbol,
            'period': period,
            'metrics': metrics,
            'details': results
        }
        
    def _load_samples(self, symbol: str, period: str) -> pd.DataFrame:
        """
        加载样本数据
        """
        try:
            query = """
            SELECT * FROM samples 
            WHERE symbol = ? AND period = ?
            ORDER BY date
            """
            
            with self.db_manager.get_conn() as conn:
                df = pd.read_sql_query(query, conn, params=[symbol, period])
                
            if not df.empty:
                # 解析特征数据
                df['features_dict'] = df['features'].apply(json.loads)
                
            return df
            
        except Exception as e:
            logger.error(f"加载样本数据失败: {e}")
            return pd.DataFrame()
            
    def _time_series_split_backtest(self, 
                                   samples_df: pd.DataFrame,
                                   train_window: int,
                                   test_window: int,
                                   step_size: int,
                                   min_samples: int) -> List[Dict]:
        """
        时间序列切片回测
        """
        results = []
        dates = pd.to_datetime(samples_df['date']).dt.date.unique()
        dates = sorted(dates)
        
        logger.info(f"数据日期范围: {dates[0]} 到 {dates[-1]}, 共 {len(dates)} 个交易日")
        
        # 滑动窗口回测
        for i in range(train_window, len(dates) - test_window + 1, step_size):
            train_start_date = dates[max(0, i - train_window)]
            train_end_date = dates[i - 1]
            test_start_date = dates[i]
            test_end_date = dates[min(len(dates) - 1, i + test_window - 1)]
            
            logger.info(f"回测窗口: 训练 {train_start_date} - {train_end_date}, 测试 {test_start_date} - {test_end_date}")
            
            # 准备训练和测试数据
            train_data = samples_df[
                (pd.to_datetime(samples_df['date']).dt.date >= train_start_date) &
                (pd.to_datetime(samples_df['date']).dt.date <= train_end_date)
            ].copy()
            
            test_data = samples_df[
                (pd.to_datetime(samples_df['date']).dt.date >= test_start_date) &
                (pd.to_datetime(samples_df['date']).dt.date <= test_end_date)
            ].copy()
            
            if len(train_data) < min_samples or len(test_data) == 0:
                logger.warning(f"样本数不足，跳过此窗口: 训练 {len(train_data)}, 测试 {len(test_data)}")
                continue
                
            # 训练模型并预测
            fold_result = self._train_and_predict(train_data, test_data, i)
            if fold_result:
                fold_result.update({
                    'train_start': str(train_start_date),
                    'train_end': str(train_end_date),
                    'test_start': str(test_start_date),
                    'test_end': str(test_end_date)
                })
                results.append(fold_result)
                
        logger.info(f"完成 {len(results)} 个回测窗口")
        return results
        
    def _train_and_predict(self, train_data: pd.DataFrame, test_data: pd.DataFrame, fold_idx: int) -> Optional[Dict]:
        """
        训练模型并进行预测
        """
        try:
            # 准备训练数据
            X_train, y_train = self._prepare_features_labels(train_data)
            X_test, y_test = self._prepare_features_labels(test_data)
            
            if X_train.empty or X_test.empty:
                return None
                
            # 确保训练和测试集特征一致
            common_features = list(set(X_train.columns) & set(X_test.columns))
            if not common_features:
                logger.error(f"训练和测试集没有共同特征")
                return None
                
            X_train = X_train[common_features]
            X_test = X_test[common_features]
                
            # 训练模型
            train_result = self.ml_trainer.train_model(X_train, y_train, test_size=0.2, use_grid_search=False)
            model = train_result['model']
            
            # 预测
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            # 计算指标
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
            
            accuracy = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.5
                
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            # 计算收益指标
            returns = self._calculate_returns(test_data, y_pred_proba)
            
            return {
                'fold': fold_idx,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'predictions': y_pred_proba.tolist(),
                'actual_labels': y_test.tolist(),
                'returns': returns
            }
            
        except Exception as e:
            logger.error(f"训练预测失败 fold {fold_idx}: {e}")
            return None
            
    def _prepare_features_labels(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征和标签数据
        """
        try:
            # 构建特征矩阵
            features_list = []
            for _, row in data.iterrows():
                features_dict = row['features_dict']
                features_list.append(features_dict)
                
            if not features_list:
                return pd.DataFrame(), pd.Series()
                
            X = pd.DataFrame(features_list)
            y = data['label'].astype(int)
            
            # 移除常数特征
            constant_features = X.columns[X.nunique() <= 1].tolist()
            if constant_features:
                X = X.drop(columns=constant_features)
                
            # 填充缺失值
            X = X.fillna(X.median())
            
            return X, y
            
        except Exception as e:
            logger.error(f"准备特征标签失败: {e}")
            return pd.DataFrame(), pd.Series()
            
    def _calculate_returns(self, test_data: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """
        计算收益指标
        """
        try:
            # 获取前瞻收益率
            forward_returns = test_data['forward_return'].values
            
            # 按预测概率排序，选择Top 20%
            n_top = max(1, len(predictions) // 5)
            top_indices = np.argsort(predictions)[-n_top:]
            
            # 计算策略收益
            strategy_returns = forward_returns[top_indices]
            
            return {
                'mean_return': float(np.mean(strategy_returns)),
                'total_return': float(np.sum(strategy_returns)),
                'hit_rate': float(np.mean(strategy_returns > 0)),
                'selected_count': len(top_indices),
                'avg_prediction_score': float(np.mean(predictions[top_indices]))
            }
            
        except Exception as e:
            logger.error(f"计算收益失败: {e}")
            return {}
            
    def _calculate_backtest_metrics(self, results: List[Dict]) -> Dict:
        """
        计算整体回测指标
        """
        if not results:
            return {}
            
        try:
            # 聚合指标
            accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
            aucs = [r['auc'] for r in results if 'auc' in r]
            precisions = [r['precision'] for r in results if 'precision' in r]
            recalls = [r['recall'] for r in results if 'recall' in r]
            
            # 收益指标
            period_returns = [r['returns']['total_return'] for r in results if 'returns' in r and r['returns']]
            hit_rates = [r['returns']['hit_rate'] for r in results if 'returns' in r and r['returns']]
            
            # 计算累计收益和回撤
            if period_returns:
                cumulative_returns = np.cumsum(period_returns).tolist()
                max_drawdown = self._calculate_max_drawdown(cumulative_returns)
                
                # 计算夏普比率 (假设无风险利率为0)
                sharpe_ratio = 0
                if len(period_returns) > 1 and np.std(period_returns) > 0:
                    sharpe_ratio = np.mean(period_returns) / np.std(period_returns) * np.sqrt(252)  # 年化
            else:
                cumulative_returns = [0]
                max_drawdown = 0
                sharpe_ratio = 0
                
            return {
                'total_folds': len(results),
                'avg_accuracy': float(np.mean(accuracies)) if accuracies else 0,
                'avg_auc': float(np.mean(aucs)) if aucs else 0.5,
                'avg_precision': float(np.mean(precisions)) if precisions else 0,
                'avg_recall': float(np.mean(recalls)) if recalls else 0,
                'total_return': float(sum(period_returns)) if period_returns else 0,
                'avg_period_return': float(np.mean(period_returns)) if period_returns else 0,
                'return_volatility': float(np.std(period_returns)) if period_returns else 0,
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'avg_hit_rate': float(np.mean(hit_rates)) if hit_rates else 0,
                'win_rate': float(np.mean([r > 0 for r in period_returns])) if period_returns else 0
            }
            
        except Exception as e:
            logger.error(f"计算回测指标失败: {e}")
            return {}
            
    def _calculate_max_drawdown(self, cumulative_returns: List[float]) -> float:
        """
        计算最大回撤
        """
        if not cumulative_returns:
            return 0
            
        peak = cumulative_returns[0]
        max_dd = 0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak != 0 else 0
            max_dd = max(max_dd, drawdown)
            
        return max_dd
        
    def _save_backtest_results(self, symbol: str, period: str, metrics: Dict, details: List[Dict]):
        """
        保存回测结果到数据库
        """
        try:
            # 保存到model_metrics表
            insert_sql = """
            INSERT OR REPLACE INTO model_metrics 
            (model_name, train_start, train_end, test_start, test_end, 
             win_rate, sharpe, max_drawdown, annual_return, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            current_time = datetime.now().isoformat()
            model_name = f"walk_forward_{symbol}_{period}"
            
            with self.db_manager.get_conn() as conn:
                cursor = conn.cursor()
                
                # 保存回测结果
                cursor.execute(insert_sql, [
                    model_name, 
                    details[0]['train_start'] if details else None,
                    details[-1]['train_end'] if details else None,
                    details[0]['test_start'] if details else None,
                    details[-1]['test_end'] if details else None,
                    metrics.get('win_rate', 0),
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('max_drawdown', 0),
                    metrics.get('total_return', 0),
                    current_time
                ])
                
                conn.commit()
                
            logger.info(f"回测结果已保存到数据库: {symbol}, {period}")
            
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")

def main():
    """测试函数"""
    # 初始化
    db_manager = DatabaseManager()
    backtest = WalkForwardBacktest(db_manager)
    
    # 运行回测
    symbol = "600519.SS"  # 贵州茅台
    result = backtest.run_backtest(
        symbol=symbol,
        period='10d',
        train_window=60,
        test_window=20,
        step_size=10
    )
    
    if result:
        print("\n=== 回测结果 ===")
        metrics = result['metrics']
        print(f"股票代码: {result['symbol']}")
        print(f"预测期间: {result['period']}")
        print(f"总回测窗口: {metrics.get('total_folds', 0)}")
        print(f"平均准确率: {metrics.get('avg_accuracy', 0):.4f}")
        print(f"平均AUC: {metrics.get('avg_auc', 0):.4f}")
        print(f"总收益率: {metrics.get('total_return', 0):.4f}")
        print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"最大回撤: {metrics.get('max_drawdown', 0):.4f}")
        print(f"胜率: {metrics.get('win_rate', 0):.4f}")
        print(f"命中率: {metrics.get('avg_hit_rate', 0):.4f}")
    else:
        print("回测失败")

if __name__ == "__main__":
    main()