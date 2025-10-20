#!/usr/bin/env python3
"""
模型预测器 - 使用训练好的模型进行股票预测
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import joblib

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

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
    
    def reload_models(self, new_models_dir: str = None):
        """
        重新加载模型，支持动态切换模型目录
        
        Args:
            new_models_dir: 新的模型目录路径，如果为None则使用当前目录
        """
        if new_models_dir:
            self.models_dir = Path(new_models_dir)
        
        logger.info(f"重新加载模型，目录: {self.models_dir}")
        self._load_models()
    
    def _load_models(self):
        """加载训练好的模型"""
        try:
            # 优先尝试加载新训练的模型格式
            model_files = list(self.models_dir.glob("*.pkl"))
            
            # 查找回归模型
            regression_path = None
            for pattern in ["reg_*_best.pkl", "reg_*_xgboost.pkl", "reg_*_lightgbm.pkl", "xgboost_regression.pkl"]:
                matches = list(self.models_dir.glob(pattern))
                if matches:
                    regression_path = matches[0]
                    break
            
            if regression_path and regression_path.exists():
                try:
                    # 优先使用joblib加载，兼容新格式模型
                    self.regression_model = joblib.load(regression_path)
                except Exception as e:
                    logger.warning(f"joblib加载回归模型失败: {e}")
                    try:
                        # 如果joblib失败，回退到pickle
                        with open(regression_path, 'rb') as f:
                            self.regression_model = pickle.load(f)
                    except Exception as e2:
                        logger.warning(f"pickle加载回归模型失败: {e2}")
                        # 尝试使用更安全的加载方式，忽略模块缺失
                        try:
                            import sys
                            from types import ModuleType
                            
                            # 创建虚拟模块来绕过模块缺失问题
                            class DummyModule(ModuleType):
                                def __getattr__(self, name):
                                    return DummyModule()
                                
                                def __call__(self, *args, **kwargs):
                                    return DummyModule()
                            
                            # 临时添加虚拟模块
                            dummy_modules = ['src.ml.training.enhanced_trainer_v2', 'src.ml.training.toolkit']
                            for module_name in dummy_modules:
                                if module_name not in sys.modules:
                                    sys.modules[module_name] = DummyModule(module_name)
                            
                            with open(regression_path, 'rb') as f:
                                self.regression_model = pickle.load(f)
                            
                            logger.info("使用虚拟模块绕过加载成功")
                        except Exception as e3:
                            logger.error(f"所有加载方式都失败: {e3}")
                            self.regression_model = None
                
                if self.regression_model is not None:
                    logger.info(f"回归模型加载成功: {regression_path}")
            else:
                logger.warning(f"回归模型文件不存在，搜索模式: {pattern}")
            
            # 查找分类模型
            classification_path = None
            for pattern in ["cls_*_best.pkl", "cls_*_xgboost.pkl", "cls_*_lightgbm.pkl", "xgboost_classification.pkl"]:
                matches = list(self.models_dir.glob(pattern))
                if matches:
                    classification_path = matches[0]
                    break
            
            if classification_path and classification_path.exists():
                try:
                    # 优先使用joblib加载，兼容新格式模型
                    self.classification_model = joblib.load(classification_path)
                except Exception as e:
                    logger.warning(f"joblib加载分类模型失败: {e}")
                    try:
                        # 如果joblib失败，回退到pickle
                        with open(classification_path, 'rb') as f:
                            self.classification_model = pickle.load(f)
                    except Exception as e2:
                        logger.warning(f"pickle加载分类模型失败: {e2}")
                        # 尝试使用更安全的加载方式，忽略模块缺失
                        try:
                            import sys
                            from types import ModuleType
                            
                            # 创建虚拟模块来绕过模块缺失问题
                            class DummyModule(ModuleType):
                                def __getattr__(self, name):
                                    return DummyModule()
                                
                                def __call__(self, *args, **kwargs):
                                    return DummyModule()
                            
                            # 临时添加虚拟模块
                            dummy_modules = ['src.ml.training.enhanced_trainer_v2', 'src.ml.training.toolkit']
                            for module_name in dummy_modules:
                                if module_name not in sys.modules:
                                    sys.modules[module_name] = DummyModule(module_name)
                            
                            with open(classification_path, 'rb') as f:
                                self.classification_model = pickle.load(f)
                            
                            logger.info("使用虚拟模块绕过加载成功")
                        except Exception as e3:
                            logger.error(f"所有加载方式都失败: {e3}")
                            self.classification_model = None
                
                if self.classification_model is not None:
                    logger.info(f"分类模型加载成功: {classification_path}")
            else:
                logger.warning(f"分类模型文件不存在，搜索模式: {pattern}")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            # 不抛出异常，允许部分模型加载失败
    
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
        
        # 对于新格式的PreprocessedModel，直接使用其预处理方法
        if (self.regression_model is not None and 
            hasattr(self.regression_model, 'preprocess') and 
            callable(getattr(self.regression_model, 'preprocess'))):
            try:
                features = self.regression_model.preprocess(features)
                logger.info("使用模型内置预处理方法处理特征")
            except Exception as e:
                logger.warning(f"模型内置预处理失败: {e}")
        
        # 如果模型是包含pipeline字段的字典格式
        if isinstance(self.regression_model, dict) and 'pipeline' in self.regression_model:
            try:
                pipeline = self.regression_model['pipeline']
                features = pipeline.transform(features)
                logger.info("使用字典格式模型的pipeline处理特征")
            except Exception as e:
                logger.warning(f"字典格式模型pipeline处理失败: {e}")
        
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
        
        # 处理新格式的字典模型
        if isinstance(self.regression_model, dict):
            # 检查字典模型是否有pipeline和特征列信息
            if 'pipeline' in self.regression_model:
                pipeline = self.regression_model['pipeline']
                if hasattr(pipeline, 'feature_names_in_'):
                    expected_features = list(pipeline.feature_names_in_)
                    current_features = features.shape[1]
                    
                    logger.info(f"字典模型期望特征数: {len(expected_features)}, 当前特征数: {current_features}")
                    
                    # 如果当前特征数不等于期望特征数，需要特征选择
                    return current_features != len(expected_features)
            
            # 检查是否有selected_features字段
            if 'selected_features' in self.regression_model and self.regression_model['selected_features'] is not None:
                selected_features = self.regression_model['selected_features']
                current_features = features.shape[1]
                
                logger.info(f"字典模型选择特征数: {len(selected_features)}, 当前特征数: {current_features}")
                
                return current_features != len(selected_features)
            
            # 如果没有特征选择信息，不需要特征选择
            return False
        
        # 检查旧格式模型期望的特征数量
        if hasattr(self.regression_model, 'named_steps') and 'regressor' in self.regression_model.named_steps:
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
    
    def _apply_feature_selection(self, features: pd.DataFrame, model_type: str = 'regression') -> pd.DataFrame:
        """
        应用特征选择，确保特征数量与模型期望一致
        
        Args:
            features: 原始特征数据
            model_type: 模型类型 ('regression' 或 'classification')
            
        Returns:
            特征选择后的数据
        """
        try:
            if features.empty:
                return features
            
            # 根据模型类型选择目标模型
            target_model = self.regression_model if model_type == 'regression' else self.classification_model
            
            if target_model is None:
                logger.info(f"{model_type} 模型不存在，跳过特征选择")
                return features
            
            # 处理新格式的字典模型
            if isinstance(target_model, dict):
                # 优先使用pipeline的特征列信息
                if 'pipeline' in target_model:
                    pipeline = target_model['pipeline']
                    if hasattr(pipeline, 'feature_names_in_'):
                        expected_features = list(pipeline.feature_names_in_)
                        
                        # 检查特征是否存在
                        available_features = [col for col in expected_features if col in features.columns]
                        missing_features = [col for col in expected_features if col not in features.columns]
                        
                        if missing_features:
                            logger.warning(f"{model_type} pipeline特征选择: 缺失 {len(missing_features)} 个特征: {missing_features}")
                        
                        if available_features:
                            logger.info(f"{model_type} pipeline特征选择: 从 {features.shape[1]} 个特征中选择 {len(available_features)} 个特征")
                            return features[available_features]
                        else:
                            logger.warning(f"{model_type} pipeline特征选择: 没有可用的特征，返回原始特征")
                            return features
                
                # 其次使用selected_features字段
                elif 'selected_features' in target_model and target_model['selected_features'] is not None:
                    selected_features = target_model['selected_features']
                    
                    # 检查特征是否存在
                    available_features = [col for col in selected_features if col in features.columns]
                    missing_features = [col for col in selected_features if col not in features.columns]
                    
                    if missing_features:
                        logger.warning(f"{model_type} 特征选择: 缺失 {len(missing_features)} 个特征: {missing_features}")
                    
                    if available_features:
                        logger.info(f"{model_type} 特征选择: 从 {features.shape[1]} 个特征中选择 {len(available_features)} 个特征")
                        return features[available_features]
                    else:
                        logger.warning(f"{model_type} 特征选择: 没有可用的特征，返回原始特征")
                        return features
                else:
                    logger.info(f"{model_type} 特征选择: 新格式模型但没有特征列信息，返回原始特征")
                    return features
            
            # 获取旧格式模型期望的特征数量
            if hasattr(target_model, 'named_steps') and 'regressor' in target_model.named_steps:
                if hasattr(target_model.named_steps['regressor'], 'n_features_in_'):
                    expected_features = target_model.named_steps['regressor'].n_features_in_
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
            
            return features
            
        except Exception as e:
            logger.error(f"{model_type} 特征选择失败: {e}")
            return features
    
    def _is_v2_model(self, model) -> bool:
        """
        检测是否为V2模型格式
        
        Args:
            model: 模型对象
            
        Returns:
            True表示V2模型，False表示V1模型
        """
        if model is None:
            return False
            
        # 检查字典格式的模型
        if isinstance(model, dict):
            # V2模型通常包含pipeline和task字段
            if 'pipeline' in model:
                pipeline = model['pipeline']
                # 检查pipeline是否有metadata或task属性
                if hasattr(pipeline, 'metadata') and pipeline.metadata:
                    return True
                if hasattr(pipeline, 'task') and pipeline.task:
                    return True
            # 检查是否有model键且包含V2特征
            elif 'model' in model:
                model_obj = model['model']
                if hasattr(model_obj, 'metadata') and model_obj.metadata:
                    return True
                if hasattr(model_obj, 'task') and model_obj.task:
                    return True
        
        # 检查直接模型对象
        elif hasattr(model, 'metadata') and model.metadata:
            return True
        elif hasattr(model, 'task') and model.task:
            return True
            
        return False
    
    def _check_feature_availability(self, features: pd.DataFrame, model, model_type: str) -> tuple:
        """
        检查特征可用性，处理缺失特征
        
        Args:
            features: 生成的特征DataFrame
            model: 模型对象
            model_type: 模型类型（'regression'或'classification'）
            
        Returns:
            (features_available, missing_features, missing_ratio)
        """
        if model is None:
            return False, [], 0.0
            
        # 获取模型期望的特征
        expected_features = []
        
        # V2模型特征检查
        if self._is_v2_model(model):
            if isinstance(model, dict) and 'pipeline' in model:
                pipeline = model['pipeline']
                if hasattr(pipeline, 'feature_names_in_'):
                    expected_features = pipeline.feature_names_in_
                elif hasattr(pipeline, 'feature_names'):
                    expected_features = pipeline.feature_names
            elif hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
            elif hasattr(model, 'feature_names'):
                expected_features = model.feature_names
        # V1模型特征检查
        else:
            # V1模型通常使用feature_0, feature_1等格式
            if hasattr(model, 'n_features_in_'):
                expected_features = [f'feature_{i}' for i in range(model.n_features_in_)]
            elif hasattr(model, 'n_features'):
                expected_features = [f'feature_{i}' for i in range(model.n_features)]
        
        if not expected_features:
            return True, [], 0.0
            
        # 检查缺失特征
        missing_features = [f for f in expected_features if f not in features.columns]
        missing_ratio = len(missing_features) / len(expected_features)
        
        # 如果缺失特征超过20%，认为特征不可用
        features_available = missing_ratio <= 0.2
        
        if missing_features:
            logger.warning(f"{model_type}模型缺失特征: {missing_features[:5]}... (共{len(missing_features)}个, 缺失率{missing_ratio:.2%})")
        
        return features_available, missing_features, missing_ratio
    
    def _calibrate_probabilities(self, probabilities: np.ndarray, model_type: str = 'classification') -> np.ndarray:
        """
        校准预测概率，确保分布合理
        
        Args:
            probabilities: 原始概率数组
            model_type: 模型类型
            
        Returns:
            校准后的概率数组
        """
        if probabilities is None or len(probabilities) == 0:
            return probabilities
            
        calibrated_probs = probabilities.copy()
        
        # 基础概率校准
        if len(probabilities.shape) == 2:
            # 二分类概率校准
            positive_probs = probabilities[:, 1]
            
            # 限制概率范围在[0.01, 0.99]之间
            positive_probs = np.clip(positive_probs, 0.01, 0.99)
            
            # 应用sigmoid校准，增强区分度
            calibrated_positive = 1 / (1 + np.exp(-2 * (positive_probs - 0.5)))
            
            # 保持概率和为1
            calibrated_probs[:, 1] = calibrated_positive
            calibrated_probs[:, 0] = 1 - calibrated_positive
            
            logger.debug(f"概率校准完成: 原始均值={probabilities[:, 1].mean():.3f}, 校准后均值={calibrated_positive.mean():.3f}")
        
        return calibrated_probs
    
    def _fallback_expected_return(self, symbol: str, days_back: int = 30) -> float:
        """
        基于30天价格数据的回归回退逻辑
        
        Args:
            symbol: 股票代码
            days_back: 回溯天数
            
        Returns:
            估计的预期收益率
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # 获取历史价格数据
            logger.info(f"回退逻辑：获取 {symbol} 的 {days_back} 天价格数据")
            stock_data = self.data_access.get_stock_data(symbol, start_date, end_date, auto_sync=False)
            
            if stock_data.empty:
                logger.warning(f"无法获取 {symbol} 的价格数据，使用默认回退值")
                return 0.0
            
            # 确保有足够的数据点
            if len(stock_data) < 10:
                logger.warning(f"{symbol} 数据点不足 ({len(stock_data)} < 10)，使用默认回退值")
                return 0.0
            
            # 提取收盘价
            if 'close' not in stock_data.columns:
                logger.warning(f"{symbol} 数据中缺少收盘价列")
                return 0.0
            
            close_prices = stock_data['close'].dropna()
            if len(close_prices) < 5:
                logger.warning(f"{symbol} 有效收盘价数据不足")
                return 0.0
            
            # 计算简单收益率
            returns = close_prices.pct_change().dropna()
            if len(returns) < 3:
                logger.warning(f"{symbol} 收益率数据不足")
                return 0.0
            
            # 计算预期收益率（使用最近收益率的加权平均）
            recent_returns = returns.tail(min(10, len(returns)))
            
            # 使用指数衰减权重（最近的数据权重更高）
            weights = np.exp(np.linspace(0, 1, len(recent_returns)))
            weights = weights / weights.sum()
            
            expected_return = np.sum(recent_returns * weights)
            
            # 软限幅：限制在[-0.5, 0.5]范围内
            expected_return = np.clip(expected_return, -0.5, 0.5)
            
            # 处理异常值：如果收益率绝对值过大，进行平滑处理
            if abs(expected_return) > 0.2:
                expected_return = expected_return * 0.5  # 减半处理
            
            logger.info(f"回退逻辑：{symbol} 预期收益率估计为 {expected_return:.4f}")
            return float(expected_return)
            
        except Exception as e:
            logger.error(f"回退逻辑执行失败: {e}")
            return 0.0
    
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
        
        # 处理新格式的字典模型
        if isinstance(self.regression_model, dict):
            # 新格式模型已经包含了正确的特征名称，不需要重命名
            return features
        
        # 检查旧格式模型是否期望数值特征名称
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
        对指定股票进行预测，支持V1/V2模型自动适配
        
        Args:
            symbol: 股票代码
            days_back: 回溯天数
            
        Returns:
            预测结果字典，包含预期收益、校准概率和模型版本信息
        """
        try:
            # 检测模型版本（使用回归模型作为基准）
            is_v2_model = self._is_v2_model(self.regression_model)
            logger.info(f"检测到模型版本: {'V2' if is_v2_model else 'V1'}")
            
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
            
            # 处理重复列名，只保留第一个出现的列
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                if col in stock_data.columns:
                    # 如果列名重复，只处理第一个出现的列
                    if stock_data.columns.tolist().count(col) > 1:
                        logger.warning(f"检测到重复列名 '{col}'，将使用第一个出现的列")
                        # 重命名重复列以避免冲突
                        col_indices = [i for i, name in enumerate(stock_data.columns) if name == col]
                        for idx in col_indices[1:]:
                            stock_data.columns.values[idx] = f"{col}_dup_{idx}"
                    
                    # 只处理原始列名
                    if col in stock_data.columns:
                        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce').astype(float)
            
            # 生成特征
            features = self._prepare_features(stock_data)
            if features.empty:
                return {'error': '特征生成失败'}
            
            # 检查特征可用性
            regression_available, regression_missing, regression_missing_ratio = self._check_feature_availability(
                features, self.regression_model, 'regression'
            )
            classification_available, classification_missing, classification_missing_ratio = self._check_feature_availability(
                features, self.classification_model, 'classification'
            )
            
            feature_availability = {
                'regression_available': regression_available,
                'regression_missing_features': regression_missing,
                'regression_missing_ratio': regression_missing_ratio,
                'classification_available': classification_available,
                'classification_missing_features': classification_missing,
                'classification_missing_ratio': classification_missing_ratio,
                'missing_rate': max(regression_missing_ratio, classification_missing_ratio),
                'quality_score': 1.0 - max(regression_missing_ratio, classification_missing_ratio)
            }
            
            if feature_availability['missing_rate'] > 0.5:
                logger.warning(f"特征缺失率过高: {feature_availability['missing_rate']:.2f}")
            
            # 分别对回归和分类模型应用特征选择
            regression_features = self._apply_feature_selection(features.copy(), 'regression')
            classification_features = self._apply_feature_selection(features.copy(), 'classification')
            
            # 重命名特征以匹配模型期望
            regression_features = self._rename_features_for_model(regression_features)
            classification_features = self._rename_features_for_model(classification_features)
            
            results = {
                'symbol': symbol,
                'data_date': end_date.date(),
                'model_version': 'V2' if is_v2_model else 'V1',
                'feature_availability': feature_availability,
                'features_shape': {
                    'original': features.shape,
                    'regression': regression_features.shape,
                    'classification': classification_features.shape
                },
                'predictions': {}
            }
            
            # 回归预测
            expected_return = None
            if self.regression_model is not None:
                try:
                    # 处理新格式的字典模型
                    if isinstance(self.regression_model, dict):
                        if 'pipeline' in self.regression_model:
                            # 旧格式：包含pipeline键
                            pipeline = self.regression_model['pipeline']
                            regression_pred = pipeline.predict(regression_features)
                        elif 'model' in self.regression_model:
                            # 新格式：包含model键
                            model = self.regression_model['model']
                            regression_pred = model.predict(regression_features)
                        else:
                            raise ValueError("回归模型字典格式不支持，缺少'pipeline'或'model'键")
                    # 处理旧格式的直接模型
                    else:
                        regression_pred = self.regression_model.predict(regression_features)
                    
                    expected_return = float(regression_pred[-1])
                    results['predictions']['regression'] = {
                        'latest_prediction': expected_return,
                        'recent_predictions': regression_pred[-10:].tolist(),
                        'prediction_stats': {
                            'mean': float(regression_pred.mean()),
                            'std': float(regression_pred.std()),
                            'min': float(regression_pred.min()),
                            'max': float(regression_pred.max())
                        }
                    }
                    
                    logger.info(f"回归预测完成: 预期收益={expected_return:.6f}")
                    
                except Exception as e:
                    logger.error(f"回归预测失败: {e}")
                    results['predictions']['regression'] = {'error': str(e)}
            
            # 分类预测和概率校准
            calibrated_probability = None
            if self.classification_model is not None:
                try:
                    # 处理新格式的字典模型
                    if isinstance(self.classification_model, dict):
                        if 'pipeline' in self.classification_model:
                            # 旧格式：包含pipeline键
                            pipeline = self.classification_model['pipeline']
                            classification_pred = pipeline.predict(classification_features)
                            classification_proba = pipeline.predict_proba(classification_features)
                        elif 'model' in self.classification_model:
                            # 新格式：包含model键
                            model = self.classification_model['model']
                            classification_pred = model.predict(classification_features)
                            classification_proba = model.predict_proba(classification_features)
                        else:
                            raise ValueError("分类模型字典格式不支持，缺少'pipeline'或'model'键")
                    # 处理旧格式的直接模型
                    else:
                        classification_pred = self.classification_model.predict(classification_features)
                        classification_proba = self.classification_model.predict_proba(classification_features)
                    
                    raw_probability = float(classification_proba[-1, 1])
                    
                    # 概率校准 - 将单个概率值转换为数组格式
                    raw_prob_array = np.array([[1 - raw_probability, raw_probability]])
                    calibrated_probs = self._calibrate_probabilities(raw_prob_array)
                    calibrated_probability = float(calibrated_probs[0, 1])
                    
                    results['predictions']['classification'] = {
                        'latest_prediction': int(classification_pred[-1]),
                        'raw_probability': raw_probability,
                        'calibrated_probability': calibrated_probability,
                        'recent_predictions': classification_pred[-10:].tolist(),
                        'prediction_stats': {
                            'positive_ratio': float(classification_pred.mean()),
                            'avg_positive_probability': float(classification_proba[:, 1].mean())
                        }
                    }
                    
                    logger.info(f"分类预测完成: 原始概率={raw_probability:.3f}, 校准概率={calibrated_probability:.3f}")
                    
                except Exception as e:
                    logger.error(f"分类预测失败: {e}")
                    results['predictions']['classification'] = {'error': str(e)}
            
            # 回归回退逻辑：如果回归预测失败但有分类预测
            if expected_return is None and calibrated_probability is not None:
                try:
                    expected_return = self._fallback_expected_return(symbol, days_back)
                    logger.info(f"使用回归回退逻辑: 预期收益={expected_return:.6f}")
                except Exception as e:
                    logger.warning(f"回归回退逻辑失败: {e}")
            
            # 最终结果组装
            results['final_prediction'] = {
                'expected_return': expected_return,
                'calibrated_probability': calibrated_probability,
                'prediction_quality': {
                    'has_regression': expected_return is not None,
                    'has_classification': calibrated_probability is not None,
                    'feature_quality': feature_availability['quality_score']
                }
            }
            
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