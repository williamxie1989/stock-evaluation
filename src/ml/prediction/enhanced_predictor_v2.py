# -*- coding: utf-8 -*-
"""
增强预测器 V2
支持新模型格式（包含pipeline、calibrator、selected_features等）
向后兼容旧格式模型
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import pickle

from src.data.unified_data_access import UnifiedDataAccessLayer
from src.data.db.unified_database_manager import UnifiedDatabaseManager
from src.ml.features.unified_feature_builder import UnifiedFeatureBuilder
from config.prediction_config import PREDICTION_PERIOD_DAYS, CLS_THRESHOLD

logger = logging.getLogger(__name__)


class EnhancedPredictorV2:
    """
    增强预测器 V2
    
    支持特性：
    1. 加载V2格式模型（带pipeline、calibrator等）
    2. 使用UnifiedFeatureBuilder构建特征
    3. 返回校准后的概率
    4. 向后兼容旧格式模型
    """
    
    def __init__(self, models_dir: str = "models/v2"):
        """
        初始化预测器
        
        Parameters
        ----------
        models_dir : str
            模型文件目录
        """
        self.models_dir = Path(models_dir)
        
        # 初始化数据访问层
        self.data_access = UnifiedDataAccessLayer()
        self.db_manager = UnifiedDatabaseManager()
        
        # 初始化特征构建器
        self.feature_builder = UnifiedFeatureBuilder(
            data_access=self.data_access,
            db_manager=self.db_manager
        )
        
        # 模型容器
        self.models = {
            'classification': None,
            'regression': None
        }
        
        # 加载模型
        self._load_models()
        
        logger.info("EnhancedPredictorV2 初始化完成")
    
    def _load_models(self):
        """加载模型（V2格式优先，向后兼容V1）"""
        # 尝试加载分类模型
        cls_paths = [
            self.models_dir / f'cls_{PREDICTION_PERIOD_DAYS}d_best.pkl',
            self.models_dir / f'cls_{PREDICTION_PERIOD_DAYS}d_lightgbm.pkl',
            self.models_dir / f'cls_{PREDICTION_PERIOD_DAYS}d_xgboost.pkl',
            self.models_dir.parent / 'good' / 'xgboost_classification.pkl'  # 旧格式fallback
        ]
        
        for path in cls_paths:
            if path.exists():
                try:
                    self.models['classification'] = self._load_model_file(path)
                    logger.info(f"分类模型加载成功: {path}")
                    break
                except Exception as e:
                    logger.warning(f"加载分类模型失败 {path}: {e}")
        
        if self.models['classification'] is None:
            logger.warning("未找到分类模型")
        
        # 尝试加载回归模型
        reg_paths = [
            self.models_dir / f'reg_{PREDICTION_PERIOD_DAYS}d_best.pkl',
            self.models_dir / f'reg_{PREDICTION_PERIOD_DAYS}d_lightgbm.pkl',
            self.models_dir / f'reg_{PREDICTION_PERIOD_DAYS}d_xgboost.pkl',
            self.models_dir.parent / 'good' / 'xgboost_regression.pkl'  # 旧格式fallback
        ]
        
        for path in reg_paths:
            if path.exists():
                try:
                    self.models['regression'] = self._load_model_file(path)
                    logger.info(f"回归模型加载成功: {path}")
                    break
                except Exception as e:
                    logger.warning(f"加载回归模型失败 {path}: {e}")
        
        if self.models['regression'] is None:
            logger.warning("未找到回归模型")
    
    def _load_model_file(self, path: Path) -> Dict:
        """
        加载模型文件（自动检测格式）
        
        Parameters
        ----------
        path : Path
            模型文件路径
        
        Returns
        -------
        model_artifact : dict
            V2格式的模型字典（旧格式会自动转换）
        """
        try:
            # 尝试用joblib加载（V2格式）
            model = joblib.load(path)
            
            # 检查是否为V2格式
            if isinstance(model, dict) and 'pipeline' in model:
                logger.info(f"检测到V2格式模型: {path.name}")
                return model
            else:
                # 旧格式（直接是sklearn模型）
                logger.info(f"检测到V1格式模型: {path.name}，自动转换")
                return self._convert_v1_to_v2(model, path)
                
        except Exception as e:
            # 尝试用pickle加载（更旧的格式）
            logger.info(f"尝试用pickle加载: {path.name}")
            with open(path, 'rb') as f:
                model = pickle.load(f)
            return self._convert_v1_to_v2(model, path)
    
    def _convert_v1_to_v2(self, model, path: Path) -> Dict:
        """
        将V1格式模型转换为V2格式
        
        Parameters
        ----------
        model : sklearn model
            旧格式模型
        path : Path
            模型路径
        
        Returns
        -------
        model_artifact : dict
            V2格式字典
        """
        task = 'classification' if 'classification' in path.name.lower() else 'regression'
        
        return {
            'task': task,
            'model_type': 'legacy',
            'pipeline': model,  # 旧模型直接作为pipeline
            'calibrator': None,
            'selected_features': None,
            'metrics': {},
            'threshold': CLS_THRESHOLD if task == 'classification' else None,
            'training_date': 'unknown',
            'is_best': False,
            'config': {},
            'v2_format': False  # 标记为非V2格式
        }
    
    def predict(self, symbol: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        """
        对单只股票进行预测
        
        Parameters
        ----------
        symbol : str
            股票代码
        as_of_date : str, optional
            截止日期（格式: YYYY-MM-DD），None表示最新
        
        Returns
        -------
        result : dict
            预测结果
        """
        try:
            logger.info(f"开始预测: {symbol}, as_of_date={as_of_date}")
            
            # 获取股票数据
            end_date = as_of_date if as_of_date else datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
            
            stock_data = self.data_access.get_stock_data(symbol, start_date, end_date)
            
            if stock_data is None or len(stock_data) == 0:
                return {'error': f'无法获取股票数据: {symbol}'}
            
            logger.info(f"获取数据: {len(stock_data)} 条记录")
            
            # 构建特征
            features_df = self.feature_builder.build_features([symbol], as_of_date)
            
            if features_df is None or len(features_df) == 0:
                return {'error': f'特征构建失败: {symbol}'}
            
            logger.info(f"特征构建完成: {features_df.shape}")
            
            # 准备预测结果
            result = {
                'symbol': symbol,
                'as_of_date': as_of_date or end_date,
                'features_count': features_df.shape[1],
                'predictions': {}
            }
            
            # 分类预测
            if self.models['classification'] is not None:
                cls_result = self._predict_classification(features_df, self.models['classification'])
                result['predictions']['classification'] = cls_result
            
            # 回归预测
            if self.models['regression'] is not None:
                reg_result = self._predict_regression(features_df, self.models['regression'])
                result['predictions']['regression'] = reg_result
            
            logger.info(f"预测完成: {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {symbol}, 错误: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _predict_classification(self, features_df: pd.DataFrame, model_artifact: Dict) -> Dict:
        """
        分类预测
        
        Parameters
        ----------
        features_df : DataFrame
            特征数据
        model_artifact : dict
            模型artifact
        
        Returns
        -------
        result : dict
            分类预测结果
        """
        try:
            pipeline = model_artifact['pipeline']
            calibrator = model_artifact.get('calibrator')
            selected_features = model_artifact.get('selected_features')
            
            # 特征选择（如果有）
            if selected_features:
                # 确保所有选择的特征都存在
                available_features = [f for f in selected_features if f in features_df.columns]
                if len(available_features) < len(selected_features):
                    logger.warning(f"部分特征缺失: {set(selected_features) - set(available_features)}")
                X = features_df[available_features]
            else:
                # 排除非特征列
                exclude_cols = ['symbol', 'date']
                feature_cols = [c for c in features_df.columns if c not in exclude_cols]
                X = features_df[feature_cols]
            
            # 预测概率（未校准）
            proba = pipeline.predict_proba(X)[:, 1]
            pred = pipeline.predict(X)
            
            # 校准概率（如果有calibrator）
            if calibrator is not None:
                proba_calibrated = calibrator.predict_proba(X)[:, 1]
                logger.info("使用校准后的概率")
            else:
                proba_calibrated = proba
                logger.info("未找到校准器，使用原始概率")
            
            # 构建结果
            result = {
                'prediction': int(pred[0]) if len(pred) == 1 else pred.tolist(),
                'probability': float(proba[0]) if len(proba) == 1 else proba.tolist(),
                'probability_calibrated': float(proba_calibrated[0]) if len(proba_calibrated) == 1 else proba_calibrated.tolist(),
                'threshold': model_artifact.get('threshold', CLS_THRESHOLD),
                'signal': 'BUY' if proba_calibrated[0] > model_artifact.get('threshold', CLS_THRESHOLD) else 'HOLD',
                'confidence': float(proba_calibrated[0]),
                'model_type': model_artifact.get('model_type', 'unknown'),
                'is_calibrated': calibrator is not None,
                'metrics': model_artifact.get('metrics', {})
            }
            
            return result
            
        except Exception as e:
            logger.error(f"分类预测失败: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _predict_regression(self, features_df: pd.DataFrame, model_artifact: Dict) -> Dict:
        """
        回归预测
        
        Parameters
        ----------
        features_df : DataFrame
            特征数据
        model_artifact : dict
            模型artifact
        
        Returns
        -------
        result : dict
            回归预测结果
        """
        try:
            pipeline = model_artifact['pipeline']
            selected_features = model_artifact.get('selected_features')
            
            # 特征选择（如果有）
            if selected_features:
                available_features = [f for f in selected_features if f in features_df.columns]
                if len(available_features) < len(selected_features):
                    logger.warning(f"部分特征缺失: {set(selected_features) - set(available_features)}")
                X = features_df[available_features]
            else:
                exclude_cols = ['symbol', 'date']
                feature_cols = [c for c in features_df.columns if c not in exclude_cols]
                X = features_df[feature_cols]
            
            # 预测
            pred = pipeline.predict(X)
            
            # 构建结果
            result = {
                'prediction': float(pred[0]) if len(pred) == 1 else pred.tolist(),
                'prediction_pct': f"{float(pred[0])*100:.2f}%" if len(pred) == 1 else [f"{p*100:.2f}%" for p in pred],
                'model_type': model_artifact.get('model_type', 'unknown'),
                'metrics': model_artifact.get('metrics', {})
            }
            
            return result
            
        except Exception as e:
            logger.error(f"回归预测失败: {e}", exc_info=True)
            return {'error': str(e)}
    
    def predict_batch(self, symbols: List[str], as_of_date: Optional[str] = None) -> Dict[str, Dict]:
        """
        批量预测
        
        Parameters
        ----------
        symbols : List[str]
            股票代码列表
        as_of_date : str, optional
            截止日期
        
        Returns
        -------
        results : dict
            {symbol: prediction_result}
        """
        results = {}
        
        for symbol in symbols:
            try:
                result = self.predict(symbol, as_of_date)
                results[symbol] = result
            except Exception as e:
                logger.error(f"批量预测失败: {symbol}, {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns
        -------
        info : dict
            模型信息
        """
        info = {
            'models_dir': str(self.models_dir),
            'classification_model': None,
            'regression_model': None
        }
        
        # 分类模型信息
        if self.models['classification']:
            cls_model = self.models['classification']
            info['classification_model'] = {
                'task': cls_model.get('task'),
                'model_type': cls_model.get('model_type'),
                'training_date': cls_model.get('training_date'),
                'is_best': cls_model.get('is_best'),
                'threshold': cls_model.get('threshold'),
                'has_calibrator': cls_model.get('calibrator') is not None,
                'metrics': cls_model.get('metrics', {}),
                'v2_format': cls_model.get('v2_format', True)
            }
        
        # 回归模型信息
        if self.models['regression']:
            reg_model = self.models['regression']
            info['regression_model'] = {
                'task': reg_model.get('task'),
                'model_type': reg_model.get('model_type'),
                'training_date': reg_model.get('training_date'),
                'is_best': reg_model.get('is_best'),
                'metrics': reg_model.get('metrics', {}),
                'v2_format': reg_model.get('v2_format', True)
            }
        
        return info
