#!/usr/bin/env python3
"""
模型预测器 - 使用训练好的模型进行股票预测
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from src.data.unified_data_access import UnifiedDataAccessLayer
from src.ml.features.enhanced_features import EnhancedFeatureGenerator
from src.ml.features.feature_selector_optimizer import FeatureSelectorOptimizer

logger = logging.getLogger(__name__)


class StockPredictor:
    """股票预测器 - 使用训练好的模型进行预测"""
    
    def __init__(self, models_dir: str = "models/good"):
        """
        初始化预测器
        
        Args:
            models_dir: 模型文件目录
        """
        self.models_dir = Path(models_dir)
        self.data_access = UnifiedDataAccessLayer()
        self.feature_generator = EnhancedFeatureGenerator()
        
        # 加载模型
        self.regression_model = None
        self.classification_model = None
        self.selected_features = None
        
        self._load_models()
    
    def _load_models(self):
        """加载训练好的模型"""
        try:
            # 加载回归模型
            regression_path = self.models_dir / "xgboost_regression.pkl"
            if regression_path.exists():
                with open(regression_path, 'rb') as f:
                    self.regression_model = pickle.load(f)
                logger.info(f"回归模型加载成功: {regression_path}")
            else:
                logger.warning(f"回归模型文件不存在: {regression_path}")
            
            # 加载分类模型
            classification_path = self.models_dir / "xgboost_classification.pkl"
            if classification_path.exists():
                with open(classification_path, 'rb') as f:
                    self.classification_model = pickle.load(f)
                logger.info(f"分类模型加载成功: {classification_path}")
            else:
                logger.warning(f"分类模型文件不存在: {classification_path}")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _prepare_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据，包括生成特征和特征选择
        
        Args:
            stock_data: 股票原始数据
            
        Returns:
            处理后的特征数据
        """
        # 生成增强特征（与训练时一致）
        features = self.feature_generator.generate_enhanced_features(stock_data)
        
        if features.empty:
            logger.error("特征生成失败，返回空数据")
            return pd.DataFrame()
        
        logger.info(f"生成特征形状: {features.shape}")
        
        # 如果模型期望的是数值特征名称（feature_0, feature_1等），
        # 我们需要进行特征选择来匹配训练时的特征
        if self._needs_feature_selection(features):
            features = self._apply_feature_selection(features)
        
        return features
    
    def _needs_feature_selection(self, features: pd.DataFrame) -> bool:
        """
        检查是否需要进行特征选择
        
        Args:
            features: 生成的特征数据
            
        Returns:
            是否需要特征选择
        """
        if self.regression_model is None:
            return False
        
        # 检查模型期望的特征数量
        if hasattr(self.regression_model.named_steps['regressor'], 'n_features_in_'):
            expected_features = self.regression_model.named_steps['regressor'].n_features_in_
            current_features = features.shape[1]
            
            logger.info(f"模型期望特征数: {expected_features}, 当前特征数: {current_features}")
            
            # 如果期望的特征名称是数值格式，说明训练时使用了特征选择
            if hasattr(self.regression_model, 'feature_names_in_'):
                feature_names = list(self.regression_model.feature_names_in_)
                if any(name.startswith('feature_') for name in feature_names):
                    return True
            
            return current_features != expected_features
        
        return False
    
    def _apply_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        应用特征选择，将特征数量减少到模型期望的数量
        
        Args:
            features: 原始特征数据
            
        Returns:
            选择后的特征数据
        """
        logger.info("应用特征选择...")
        
        try:
            # 获取模型期望的特征数量
            expected_features = self.regression_model.named_steps['regressor'].n_features_in_
            current_features = features.shape[1]
            
            if current_features <= expected_features:
                logger.info(f"当前特征数 {current_features} 小于等于期望特征数 {expected_features}，无需选择")
                return features
            
            # 使用简单的特征选择方法：选择方差最大的前N个特征
            # 在实际应用中，这里应该使用与训练时相同的特征选择方法
            variances = features.var().sort_values(ascending=False)
            selected_features = variances.head(expected_features).index.tolist()
            
            logger.info(f"从 {current_features} 个特征中选择 {len(selected_features)} 个特征")
            logger.info(f"选择的特征: {selected_features}")
            
            return features[selected_features]
            
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            return features
    
    def _rename_features_for_model(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        将特征重命名为模型期望的格式（feature_0, feature_1等）
        
        Args:
            features: 特征数据
            
        Returns:
            重命名后的特征数据
        """
        if features.empty:
            return features
        
        # 检查模型是否期望数值特征名称
        if hasattr(self.regression_model, 'feature_names_in_'):
            expected_names = list(self.regression_model.feature_names_in_)
            if any(name.startswith('feature_') for name in expected_names):
                # 重命名特征为 feature_0, feature_1 等格式
                new_names = {features.columns[i]: f'feature_{i}' 
                           for i in range(len(features.columns))}
                features = features.rename(columns=new_names)
                logger.info(f"特征重命名完成: {list(features.columns)}")
        
        return features
    
    def predict(self, symbol: str, days_back: int = 90) -> Dict[str, Any]:
        """
        对指定股票进行预测
        
        Args:
            symbol: 股票代码
            days_back: 回溯天数
            
        Returns:
            预测结果字典
        """
        try:
            # 获取历史数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"获取 {symbol} 的历史数据 ({start_date.date()} 到 {end_date.date()})")
            stock_data = self.data_access.get_stock_data(symbol, start_date, end_date, auto_sync=False)
            
            if stock_data.empty:
                logger.error(f"无法获取 {symbol} 的数据")
                return {'error': '无法获取股票数据'}
            
            logger.info(f"获取到数据形状: {stock_data.shape}")
            
            # 数据预处理
            stock_data = stock_data.reset_index()
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                if col in stock_data.columns:
                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce').astype(float)
            
            # 生成特征
            features = self._prepare_features(stock_data)
            if features.empty:
                return {'error': '特征生成失败'}
            
            # 重命名特征以匹配模型期望
            features = self._rename_features_for_model(features)
            
            results = {
                'symbol': symbol,
                'data_date': end_date.date(),
                'features_shape': features.shape,
                'predictions': {}
            }
            
            # 回归预测
            if self.regression_model is not None:
                try:
                    regression_pred = self.regression_model.predict(features)
                    
                    results['predictions']['regression'] = {
                        'latest_prediction': float(regression_pred[-1]),
                        'recent_predictions': regression_pred[-10:].tolist(),
                        'prediction_stats': {
                            'mean': float(regression_pred.mean()),
                            'std': float(regression_pred.std()),
                            'min': float(regression_pred.min()),
                            'max': float(regression_pred.max())
                        }
                    }
                    
                    logger.info(f"回归预测完成: 最新预测值={regression_pred[-1]:.6f}")
                    
                except Exception as e:
                    logger.error(f"回归预测失败: {e}")
                    results['predictions']['regression'] = {'error': str(e)}
            
            # 分类预测
            if self.classification_model is not None:
                try:
                    classification_pred = self.classification_model.predict(features)
                    classification_proba = self.classification_model.predict_proba(features)
                    
                    results['predictions']['classification'] = {
                        'latest_prediction': int(classification_pred[-1]),
                        'latest_probability': float(classification_proba[-1, 1]),
                        'recent_predictions': classification_pred[-10:].tolist(),
                        'prediction_stats': {
                            'positive_ratio': float(classification_pred.mean()),
                            'avg_positive_probability': float(classification_proba[:, 1].mean())
                        }
                    }
                    
                    logger.info(f"分类预测完成: 最新预测={classification_pred[-1]}, 概率={classification_proba[-1, 1]:.3f}")
                    
                except Exception as e:
                    logger.error(f"分类预测失败: {e}")
                    results['predictions']['classification'] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"预测过程失败: {e}")
            return {'error': str(e)}
    
    def predict_batch(self, symbols: List[str], days_back: int = 90) -> Dict[str, Any]:
        """
        批量预测多个股票
        
        Args:
            symbols: 股票代码列表
            days_back: 回溯天数
            
        Returns:
            批量预测结果
        """
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"开始预测 {symbol}")
                result = self.predict(symbol, days_back)
                results[symbol] = result
                
            except Exception as e:
                logger.error(f"预测 {symbol} 失败: {e}")
                results[symbol] = {'error': str(e)}
        
        return {
            'batch_predictions': results,
            'summary': {
                'total_symbols': len(symbols),
                'successful_predictions': sum(1 for r in results.values() if 'error' not in r),
                'failed_predictions': sum(1 for r in results.values() if 'error' in r)
            }
        }


def main():
    """测试预测器"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建预测器
    predictor = StockPredictor()
    
    # 测试单个股票预测
    test_symbol = "000001.SZ"
    print(f"\\n{'='*50}")
    print(f"测试预测 {test_symbol}")
    print(f"{'='*50}")
    
    result = predictor.predict(test_symbol, days_back=90)
    
    if 'error' in result:
        print(f"预测失败: {result['error']}")
    else:
        print(f"股票: {result['symbol']}")
        print(f"数据日期: {result['data_date']}")
        print(f"特征形状: {result['features_shape']}")
        
        if 'regression' in result['predictions']:
            reg_pred = result['predictions']['regression']
            if 'error' not in reg_pred:
                print(f"\\n回归预测:")
                print(f"  最新预测值: {reg_pred['latest_prediction']:.6f}")
                print(f"  预测均值: {reg_pred['prediction_stats']['mean']:.6f}")
                print(f"  预测标准差: {reg_pred['prediction_stats']['std']:.6f}")
        
        if 'classification' in result['predictions']:
            cls_pred = result['predictions']['classification']
            if 'error' not in cls_pred:
                print(f"\\n分类预测:")
                print(f"  最新预测: {cls_pred['latest_prediction']}")
                print(f"  上涨概率: {cls_pred['latest_probability']:.3f}")
                print(f"  正类比例: {cls_pred['prediction_stats']['positive_ratio']:.3f}")
    
    # 测试批量预测
    print(f"\\n{'='*50}")
    print("测试批量预测")
    print(f"{'='*50}")
    
    test_symbols = ["000001.SZ", "000002.SZ"]
    batch_result = predictor.predict_batch(test_symbols, days_back=90)
    
    summary = batch_result['summary']
    print(f"批量预测完成:")
    print(f"  总计股票数: {summary['total_symbols']}")
    print(f"  成功预测: {summary['successful_predictions']}")
    print(f"  失败预测: {summary['failed_predictions']}")
    
    for symbol, result in batch_result['batch_predictions'].items():
        if 'error' not in result:
            print(f"\\n{symbol}:")
            if 'regression' in result['predictions'] and 'error' not in result['predictions']['regression']:
                print(f"  回归预测: {result['predictions']['regression']['latest_prediction']:.6f}")
            if 'classification' in result['predictions'] and 'error' not in result['predictions']['classification']:
                print(f"  上涨概率: {result['predictions']['classification']['latest_probability']:.3f}")


if __name__ == "__main__":
    main()