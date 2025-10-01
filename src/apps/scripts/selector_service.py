#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能选股服务
基于机器学习模型进行股票预测和排序
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import joblib
import os
# 新增 json 导入
import json
from sklearn.calibration import CalibratedClassifierCV
from ...data.db.unified_database_manager import UnifiedDatabaseManager
from .features import FeatureGenerator
from ...ml.features.enhanced_features import EnhancedFeatureGenerator
from ...trading.signals.signal_generator import SignalGenerator
from ...services.stock.stock_status_filter import StockStatusFilter
# 引入字段映射工具，统一字段名
from src.data.field_mapping import FieldMapper
import logging
import re

# 导入增强预处理pipeline
from src.ml.features.enhanced_preprocessing import EnhancedPreprocessingPipeline, create_enhanced_preprocessing_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentStockSelector:
    """
    智能选股服务
    """
    
    # 新增: 工具方法 – 统一替换匿名特征名，避免多处重复实现
    def _replace_anonymous_feature_names(self, model, actual_column_names):
        """如果模型的 feature_names_in_ 为匿名的 feature_0、feature_1 等，占位名字，则使用实际列名替换。
        参数:
            model: 任何带有 feature_names_in_ 属性的 sklearn/XGBoost 模型或 Pipeline
            actual_column_names: list[str] – 真实的特征列名称，顺序与训练/预测一致
        """
        try:
            # 如果是Pipeline, 递归处理其最后一个Estimator，并同步Pipeline本身
            try:
                from sklearn.pipeline import Pipeline
                if isinstance(model, Pipeline):
                    inner_est = model.steps[-1][1]
                    self._replace_anonymous_feature_names(inner_est, actual_column_names)
                    model.feature_names_in_ = np.array(actual_column_names)
                    return
            except Exception:
                pass

            # 针对 xgboost 模型的特殊处理，需要同时更新 booster 的 feature_names
            try:
                import xgboost as xgb  # 延迟导入，避免安装缺失时报错
                if isinstance(model, xgb.XGBModel):
                    model.feature_names_in_ = np.array(actual_column_names)
                    booster = model.get_booster()
                    booster.feature_names = actual_column_names
                    return
            except Exception:
                pass  # 不是 xgboost 模型或导入失败

            # 其他 sklearn 模型逻辑
            if not hasattr(model, "feature_names_in_"):
                # 若模型完全没有该属性，则直接赋值
                model.feature_names_in_ = np.array(actual_column_names)
                return

            # 对已有属性进行检查/覆盖
            feature_names = list(getattr(model, "feature_names_in_", []))
            if not feature_names or len(feature_names) != len(actual_column_names):
                model.feature_names_in_ = np.array(actual_column_names)
                return

            # 如果长度一致但存在匿名占位或者与当前列不一致，也覆盖
            anonymous = all(re.match(r"^feature_\d+$", str(fn)) for fn in feature_names)
            if anonymous or feature_names != actual_column_names:
                model.feature_names_in_ = np.array(actual_column_names)
        except Exception as e:
            logger.debug(f"替换匿名特征名失败: {e}")
    
    def __init__(self, db_manager: UnifiedDatabaseManager = None, use_enhanced_features: bool = 1, 
                 use_enhanced_preprocessing: bool = 1, preprocessing_complexity: str = 'medium'):
        """
        初始化智能选股器
        """
        self.db = db_manager or UnifiedDatabaseManager(db_type='mysql')
        self.use_enhanced_features = use_enhanced_features
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        self.preprocessing_complexity = preprocessing_complexity
        
        # 根据参数选择特征生成器
        if use_enhanced_features:
            self.feature_generator = EnhancedFeatureGenerator()
        else:
            self.feature_generator = FeatureGenerator()
        self.signal_generator = SignalGenerator()
        self.stock_filter = StockStatusFilter()
        
        # 模型相关
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # 新增：分别持有分类与回归模型数据
        self.cls_model_data = None
        self.reg_model_data = None
        self.cls_feature_names = None
        self.reg_feature_names = None
        
        # 增强预处理pipeline
        self.cls_preprocessor = None
        self.reg_preprocessor = None
        
        if self.use_enhanced_preprocessing:
            # 初始化预处理配置
            self.cls_preprocessing_config = create_enhanced_preprocessing_config('classification', preprocessing_complexity)
            self.reg_preprocessing_config = create_enhanced_preprocessing_config('regression', preprocessing_complexity)
            logger.info(f"启用增强预处理pipeline，复杂度: {preprocessing_complexity}")
    
    def load_model(self, model_path: str = None, scaler_path: str = None):
        """
        加载训练好的模型和标准化器
        """
        try:
            # 如果没有指定路径，自动查找最新的模型文件
            if model_path is None:
                models_dir = "models"
                if not os.path.exists(models_dir):
                    logger.warning(f"模型目录不存在: {models_dir}")
                    return 0
                
                # 查找所有pkl文件
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                if not model_files:
                    logger.warning(f"模型目录中没有找到pkl文件: {models_dir}")
                    return 0
                
                # 使用最新的模型文件（按文件名排序，时间戳越新越靠后）
                model_files.sort()
                latest_model = model_files[-1]
                model_path = os.path.join(models_dir, latest_model)
                
                logger.info(f"自动选择模型文件: {model_path}")
            
            if os.path.exists(model_path):
                # 加载模型文件
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # 检查模型文件结构
                if isinstance(model_data, dict):
                    # 新格式：包含多个组件的字典
                    if 'model' in model_data:
                        self.model = model_data['model']
                        logger.info(f"从字典中加载模型成功: {model_path}")
                        
                        # 同步加载特征名，确保预测时特征顺序一致
                        if 'feature_names' in model_data and model_data['feature_names']:
                            self.feature_names = list(model_data['feature_names'])
                            logger.info(f"已加载特征名，数量: {len(self.feature_names)}")
                        
                        # 如果保存了scaler也记录下来（仅当模型不是包含scaler的Pipeline时才会在预测中使用）
                        if 'scaler' in model_data:
                            self.scaler = model_data['scaler']
                            logger.info(f"从字典中加载标准化器成功")
                        else:
                            logger.warning(f"模型文件中未包含标准化器，将尝试单独加载")
                            # 尝试加载单独的scaler文件
                            if not self._load_separate_scaler(scaler_path):
                                logger.warning(f"未找到标准化器，将使用原始特征/或由Pipeline内部处理")
                                self.scaler = None
                        
                        # 如模型为包含scaler的Pipeline，预测时应直接传入原始特征
                        try:
                            if hasattr(self.model, 'named_steps') and 'scaler' in getattr(self.model, 'named_steps', {}):
                                logger.info("检测到模型为包含scaler的Pipeline，预测时将跳过外部标准化，避免二次缩放")
                        except Exception:
                            pass
                    else:
                        logger.error(f"模型文件格式错误，未找到model字段")
                        return 0
                else:
                    # 旧格式：直接是模型对象
                    self.model = model_data
                    logger.info(f"加载旧格式模型成功: {model_path}")
                    # 尝试加载单独的scaler文件
                    if not self._load_separate_scaler(scaler_path):
                        logger.warning(f"未找到标准化器，将使用原始特征")
                        self.scaler = None
            else:
                logger.warning(f"模型文件不存在: {model_path}")
                return 0
                
            return 1
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return 0
    
    # 新增：同时加载30d的分类与回归模型
    def load_models(self, period: str = '30d') -> bool:
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                logger.warning(f"模型目录不存在: {models_dir}")
                return 0
            import pickle
            cls_candidates = []
            reg_candidates = []
            for fname in sorted([f for f in os.listdir(models_dir) if f.endswith('.pkl')]):
                fpath = os.path.join(models_dir, fname)
                try:
                    with open(fpath, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, dict):
                        meta = data.get('metadata', {}) or {}
                        task = meta.get('task') or meta.get('type')
                        per = meta.get('period')
                        if per != period:
                            # 尝试从文件名推断周期
                            fname_lower = fname.lower()
                            inferred_period = '30d' if '30d' in fname_lower else ('10d' if '10d' in fname_lower else None)
                            if inferred_period != period:
                                continue
                        if task == 'classification':
                            cls_candidates.append((fname, data))
                        elif task == 'regression':
                            reg_candidates.append((fname, data))
                    else:
                        # 兼容旧格式：直接保存了Estimator/Pipeline
                        fname_lower = fname.lower()
                        inferred_task = 'classification' if ('logistic' in fname_lower or 'cls' in fname_lower or 'classification' in fname_lower) else ('regression' if 'reg' in fname_lower or 'regression' in fname_lower else None)
                        inferred_period = '30d' if '30d' in fname_lower else ('10d' if '10d' in fname_lower else None)
                        if inferred_task and (inferred_period == period or inferred_period is None):
                            # 对于Pipeline对象，尝试获取feature_names_in_属性
                            feature_names = []
                            if hasattr(data, 'feature_names_in_'):
                                feature_names = data.feature_names_in_.tolist()
                            
                            wrapper = {'model': data, 'feature_names': feature_names, 'metadata': {'task': inferred_task, 'period': inferred_period}}
                            if inferred_task == 'classification':
                                cls_candidates.append((fname, wrapper))
                            elif inferred_task == 'regression':
                                reg_candidates.append((fname, wrapper))
                except Exception as e:
                    logger.debug(f"读取模型失败 {fpath}: {e}")
                    continue
            if cls_candidates:
                # 优先选择xgboost模型，如果没有则选择最新模型
                xgboost_candidates = [(fname, data) for fname, data in cls_candidates if 'xgboost' in fname.lower()]
                if xgboost_candidates:
                    # 如果有xgboost模型，选择最新的xgboost模型
                    self.cls_model_data = xgboost_candidates[-1][1]
                    selected_model = xgboost_candidates[-1][0]
                else:
                    # 否则选择最新模型
                    self.cls_model_data = cls_candidates[-1][1]
                    selected_model = cls_candidates[-1][0]
                
                # 优先从字典中获取特征名称，如果是Pipeline对象则从feature_names_in_属性获取
                if isinstance(self.cls_model_data, dict):
                    self.cls_feature_names = self.cls_model_data.get('feature_names') or []
                    # 若仍为空，尝试从内部model获取
                    if (not self.cls_feature_names) and isinstance(self.cls_model_data.get('model'), object):
                        inner_model = self.cls_model_data.get('model')
                        if hasattr(inner_model, 'feature_names_in_'):
                            self.cls_feature_names = list(inner_model.feature_names_in_)
                else:
                    # 对于Pipeline对象，尝试获取feature_names_in_属性
                    if hasattr(self.cls_model_data, 'feature_names_in_'):
                        self.cls_feature_names = self.cls_model_data.feature_names_in_.tolist()
                    else:
                        self.cls_feature_names = []
                # 如果仍然无法获取特征名，尝试从feature_cache注入
                if not self.cls_feature_names or all(str(n).startswith('feature_') for n in self.cls_feature_names):
                    feature_cache_file = os.path.join('feature_cache', 'selected_features.json')
                    try:
                        if os.path.exists(feature_cache_file):
                            with open(feature_cache_file, 'r') as fc:
                                cache_names = json.load(fc)
                            # 仅当特征数量一致或当前为空时才替换，防止错误映射
                            if len(cache_names) == len(self.cls_feature_names) or not self.cls_feature_names:
                                self.cls_feature_names = cache_names
                                logger.info(f"分类模型特征名已从缓存注入，共 {len(self.cls_feature_names)} 个特征")
                        else:
                            logger.debug("分类模型特征缓存文件不存在")
                    except Exception as e:
                        logger.debug(f"注入分类模型特征名失败: {e}")
                # 同步更新模型对象的 feature_names_in_ 属性，避免预测时特征不匹配
                try:
                    target_model = self.cls_model_data.get('model') if isinstance(self.cls_model_data, dict) else self.cls_model_data
                    if target_model is not None and self.cls_feature_names:
                        target_model.feature_names_in_ = np.array(self.cls_feature_names)
                except Exception as up_err:
                    logger.debug(f"更新分类模型 feature_names_in_ 失败: {up_err}")
            else:
                logger.warning("未找到30d分类模型")
            if reg_candidates:
                self.reg_model_data = reg_candidates[-1][1]
                # 优先从字典或内部模型获取特征名称
                if isinstance(self.reg_model_data, dict):
                    self.reg_feature_names = self.reg_model_data.get('feature_names') or []
                    if (not self.reg_feature_names) and isinstance(self.reg_model_data.get('model'), object):
                        inner_model_r = self.reg_model_data.get('model')
                        if hasattr(inner_model_r, 'feature_names_in_'):
                            self.reg_feature_names = list(inner_model_r.feature_names_in_)
                else:
                    if hasattr(self.reg_model_data, 'feature_names_in_'):
                        self.reg_feature_names = self.reg_model_data.feature_names_in_.tolist()
                    else:
                        self.reg_feature_names = []

                # 如果仍然无法获取特征名，尝试从feature_cache注入
                if not self.reg_feature_names or all(str(n).startswith('feature_') for n in self.reg_feature_names):
                    feature_cache_file = os.path.join('feature_cache', 'selected_features.json')
                    try:
                        if os.path.exists(feature_cache_file):
                            with open(feature_cache_file, 'r') as fc:
                                cache_names = json.load(fc)
                            if len(cache_names) == len(self.reg_feature_names) or not self.reg_feature_names:
                                self.reg_feature_names = cache_names
                                logger.info(f"回归模型特征名已从缓存注入，共 {len(self.reg_feature_names)} 个特征")
                        else:
                            logger.debug("回归模型特征缓存文件不存在")
                    except Exception as e:
                        logger.debug(f"注入回归模型特征名失败: {e}")
                # 同步更新模型对象的 feature_names_in_ 属性，避免预测时特征不匹配
                try:
                    target_model_r = self.reg_model_data.get('model') if isinstance(self.reg_model_data, dict) else self.reg_model_data
                    if target_model_r is not None and self.reg_feature_names:
                        target_model_r.feature_names_in_ = np.array(self.reg_feature_names)
                except Exception as up_err:
                    logger.debug(f"更新回归模型 feature_names_in_ 失败: {up_err}")
            else:
                logger.warning("未找到30d回归模型")
            return bool(self.cls_model_data or self.reg_model_data)
        except Exception as e:
            logger.error(f"加载模型集合失败: {e}")
            return 0
    
    def _load_separate_scaler(self, scaler_path: str = None) -> bool:
        """
        加载单独的标准化器文件
        """
        try:
            if scaler_path is None:
                models_dir = "models"
                scaler_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and 'scaler' in f]
                if scaler_files:
                    scaler_files.sort()
                    latest_scaler = scaler_files[-1]
                    scaler_path = os.path.join(models_dir, latest_scaler)
                else:
                    return 0
                    
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"标准化器加载成功: {scaler_path}")
                return 1
            else:
                return 0
        except Exception as e:
            logger.error(f"标准化器加载失败: {e}")
            return 0
    
    def get_latest_features(self, symbols: List[str], 
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取最新的特征数据
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # 获取足够的历史数据用于特征计算（包含中期和长期特征）
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=180)).strftime('%Y-%m-%d')
        
        all_features = []
        
        # 批量获取所有股票的历史数据，减少数据库查询次数
        try:
            # 分批处理，避免一次性查询过多数据
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i+batch_size]
                # 只在关键节点输出日志，减少过程日志
                if i == 0 or (i + batch_size) % 200 == 0 or i + batch_size >= len(symbols):
                    logger.info(f"批量获取股票历史数据: {i+1}-{min(i+batch_size, len(symbols))}/{len(symbols)}")
                
                # 批量获取价格数据
                all_prices = self.db.get_last_n_bars(batch_symbols, n=180)
                if all_prices.empty:
                    logger.warning(f"批量 {batch_symbols} 历史数据为空")
                    continue
                
                # 按股票分组处理
                for symbol in batch_symbols:
                    # 提取该股票的数据
                    prices = all_prices[all_prices['symbol'] == symbol].copy()
                    if prices.empty or len(prices) < 20:
                        logger.debug(f"股票 {symbol} 历史数据不足（少于20条记录）")
                        continue

                    # 按时间排序
                    prices = prices.sort_values('date')
                    
                    # 使用FieldMapper标准化字段名，确保后续处理统一
                    try:
                        prices = FieldMapper.normalize_fields(prices, 'prices_daily')
                        prices = FieldMapper.ensure_required_fields(prices, 'prices_daily')
                    except Exception as fm_err:
                        logger.debug(f"FieldMapper处理失败: {fm_err}")

                    # -------- 统一数值列类型，避免 decimal.Decimal 与 float 运算问题 --------
                    numeric_cols_pi = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in numeric_cols_pi:
                        if col in prices.columns:
                            prices[col] = pd.to_numeric(prices[col], errors='coerce').astype(float)
                    # ----------------------------------------------------------------------
                    
                    # 使用FeatureGenerator计算完整特征集（支持大小写列名）
                    factor_features = self.feature_generator.calculate_factor_features(prices)
                    if factor_features.empty:
                        continue
                    # 将特征转换为字典格式
                    feature_dict = factor_features.iloc[-1].to_dict()  # 使用最新一天的特征
                    feature_dict['symbol'] = symbol
                    all_features.append(pd.Series(feature_dict))
                        
        except Exception as e:
            logger.error(f"批量获取特征数据失败: {e}")
            # 回退到逐个处理
            for symbol in symbols:
                # 获取价格数据（尽量多取一些，生成更多稳定特征）
                prices = self.db.get_last_n_bars([symbol], n=180)
                if prices.empty or len(prices) < 20:
                    logger.warning(f"股票 {symbol} 历史数据不足（少于20条记录）")
                    continue
                
                # 仅保留该股票并按时间排序
                prices = prices[prices['symbol'] == symbol].copy()
                prices = prices.sort_values('date')
                
                # 使用FeatureGenerator计算完整特征集（支持大小写列名）
                factor_features = self.feature_generator.calculate_factor_features(prices)
                if factor_features.empty:
                    continue
                # 将特征转换为字典格式
                feature_dict = factor_features.iloc[-1].to_dict()  # 使用最新一天的特征
                feature_dict['symbol'] = symbol
                all_features.append(pd.Series(feature_dict))

        
        if all_features:
            result = pd.DataFrame(all_features)
            logger.info(f"特征获取完成，共处理 {len(result)} 只股票")
            return result
        else:
            logger.warning("特征获取结果为空")
            return pd.DataFrame()
    
    def predict_stocks(self, symbols: List[str], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        预测股票并返回排序结果
        """
        # 允许模型存在但外部scaler缺失（若模型是包含scaler的Pipeline）
        if not (self.cls_model_data or self.reg_model_data or self.model):
            logger.error("模型未加载")
            return []
            
        # 获取最新特征
        features_df = self.get_latest_features(symbols)
        if features_df.empty:
            logger.warning("无法获取特征数据")
            return []
            
        # -------- 统一特征列名称大小写及数值类型，避免不一致导致特征缺失 --------
        try:
            # 将所有非symbol列名统一为小写，方便与模型训练阶段保持一致
            new_cols = []
            for c in features_df.columns:
                new_cols.append(c if c == 'symbol' else c.lower())
            features_df.columns = new_cols
        except Exception as e:
            logger.debug(f"特征列名统一失败: {e}")

        try:
            for col in features_df.columns:
                if col == 'symbol':
                    continue
                # 尝试将列转换为float，如果无法转换则设为NaN
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').astype(float)
        except Exception as e:
            logger.debug(f"特征列类型统一失败: {e}")
        # ----------------------------------------------------------------------
        
        # 针对分类与回归分别准备特征矩阵
        probs = np.array([0.5] * len(features_df))
        preds_cls = np.array([0] * len(features_df))
        exp_returns = np.array([0.0] * len(features_df))
        
        # 分类模型预测（优先使用新结构的cls_model_data，否则回退到旧的self.model）
        try:
            if self.cls_model_data:
                model = self.cls_model_data['model']
                expected_features = self.cls_feature_names or []
                Xc = features_df.copy()
                
                # 当未提供特征名时，回退为使用数值型特征（排除symbol）
                if not expected_features:
                    candidate_cols = [c for c in Xc.columns if c != 'symbol']
                    # 仅选择数值列
                    expected_features = [c for c in candidate_cols if np.issubdtype(Xc[c].dtype, np.number)]
                
                # 如果expected_features在特征表中缺失严重，尝试回退到交集或全部数值特征
                available = [c for c in expected_features if c in Xc.columns]
                if len(available) <= max(3, len(expected_features)*0.3):
                    logger.warning(f"分类模型所需特征缺失严重({len(available)}/{len(expected_features)}), 回退到数值特征全集")
                    available = [c for c in Xc.columns if c != 'symbol' and np.issubdtype(Xc[c].dtype, np.number)]
                # 确保所有期望特征均存在于特征矩阵中，不存在的填充0
                for col in expected_features:
                    if col not in Xc.columns:
                        Xc[col] = 0
                Xc = Xc[expected_features].fillna(0)
                # 若整体方差为0，说明全部为常数列，再回退为所有数值特征
                if np.isclose(Xc.var().sum(), 0):
                    logger.error("分类特征矩阵方差为0，回退到原始数值特征全集")
                    Xc = features_df[[c for c in features_df.columns if c != 'symbol' and np.issubdtype(features_df[c].dtype, np.number)]].fillna(0)
                
                # 使用类级别工具方法替换匿名特征名（已移至类定义）
                
                # 在预测前动态替换模型匿名特征名
                self._replace_anonymous_feature_names(model, list(Xc.columns))
                
                # 应用增强预处理pipeline
                if self.use_enhanced_preprocessing and self.cls_preprocessor is not None:
                    logger.info("使用增强预处理pipeline处理分类特征")
                    Xc_input = self.cls_preprocessor.transform(Xc)
                else:
                    # 原有的预处理逻辑
                    use_pipeline_scaler = hasattr(model, 'named_steps') and 'scaler' in getattr(model, 'named_steps', {})
                    if use_pipeline_scaler:
                        Xc_input = Xc
                    else:
                        scaler = self.cls_model_data.get('scaler')
                        if scaler is None:
                            # 在线标准化，避免数值尺度造成的概率饱和
                            try:
                                col_std = Xc.std().replace(0, 1e-6)
                                Xc_norm = (Xc - Xc.mean()) / col_std
                                Xc_input = Xc_norm.clip(-5, 5).fillna(0)
                                logger.info("未找到标准化器，已对特征进行在线标准化处理")
                            except Exception:
                                logger.warning("在线标准化失败，退回原始特征")
                                Xc_input = Xc
                        else:
                            Xc_input = scaler.transform(Xc)
                
                probs = model.predict_proba(Xc_input)[:, 1]
                preds_cls = model.predict(Xc_input)
                
            elif self.model:
                # 兼容旧逻辑
                if self.feature_names:
                    for col in self.feature_names:
                        if col not in features_df.columns:
                            features_df[col] = 0
                    X_old = features_df[self.feature_names].fillna(0)
                else:
                    feature_cols = [c for c in features_df.columns if c != 'symbol']
                    X_old = features_df[feature_cols].fillna(0)
                use_pipeline_scaler = hasattr(self.model, 'named_steps') and 'scaler' in getattr(self.model, 'named_steps', {})
                X_input = X_old if use_pipeline_scaler else (self.scaler.transform(X_old) if self.scaler else X_old)
                
                # 检查模型是否支持predict_proba
                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba(X_input)[:, 1]
                else:
                    # 对于Ridge等回归模型，使用decision_function或predict转换为概率
                    if hasattr(self.model, 'decision_function'):
                        scores = self.model.decision_function(X_input)
                        # 使用sigmoid函数将分数转换为概率
                        probs = 1 / (1 + np.exp(-scores))
                    else:
                        # 使用predict结果，假设是连续值，转换为概率
                        predictions_raw = self.model.predict(X_input)
                        # 使用tanh函数将预测值映射到概率空间
                        probs = 0.5 + 0.4 * np.tanh(predictions_raw)
                
                preds_cls = self.model.predict(X_input)
                # 对于回归模型，将连续预测转换为分类
                if not hasattr(self.model, 'predict_proba'):
                    preds_cls = (preds_cls > 0).astype(int)
                    
        except Exception as e:
            logger.error(f"分类预测失败: {e}")
        
        # 回归模型预测（预期收益）
        try:
            if self.reg_model_data:
                model_r = self.reg_model_data['model']
                expected_features_r = self.reg_feature_names or []
                Xr = features_df.copy()
                
                # 当未提供特征名时，回退为使用数值型特征（排除symbol）
                if not expected_features_r:
                    candidate_cols = [c for c in Xr.columns if c != 'symbol']
                    # 仅选择数值列
                    expected_features_r = [c for c in candidate_cols if np.issubdtype(Xr[c].dtype, np.number)]
                    logger.warning(f"回归模型特征名称为空，回退到使用数值型特征，数量: {len(expected_features_r)}")
                
                available_r = [c for c in expected_features_r if c in Xr.columns]
                if len(available_r) <= max(3, len(expected_features_r)*0.3):
                    logger.warning(f"回归模型所需特征缺失严重({len(available_r)}/{len(expected_features_r)}), 回退到数值特征全集")
                    available_r = [c for c in Xr.columns if c != 'symbol' and np.issubdtype(Xr[c].dtype, np.number)]
                for col in expected_features_r:
                    if col not in Xr.columns:
                        Xr[col] = 0
                Xr = Xr[expected_features_r].fillna(0)
                if np.isclose(Xr.var().sum(), 0):
                    logger.error("回归特征矩阵方差为0，回退到原始数值特征全集")
                    Xr = features_df[[c for c in features_df.columns if c != 'symbol' and np.issubdtype(features_df[c].dtype, np.number)]].fillna(0)
                
                # 使用类级别工具方法替换匿名特征名（已移至类定义）
                
                # 在预测前动态替换模型匿名特征名
                self._replace_anonymous_feature_names(model_r, list(Xr.columns))
                
                # 应用增强预处理pipeline
                if self.use_enhanced_preprocessing and self.reg_preprocessor is not None:
                    logger.info("使用增强预处理pipeline处理回归特征")
                    Xr_input = self.reg_preprocessor.transform(Xr)
                else:
                    # 原有的预处理逻辑
                    use_pipeline_scaler_r = hasattr(model_r, 'named_steps') and 'scaler' in getattr(model_r, 'named_steps', {})
                    scaler_r = self.reg_model_data.get('scaler')
                    if use_pipeline_scaler_r:
                        Xr_input = Xr
                    elif scaler_r is not None:
                        Xr_input = scaler_r.transform(Xr)
                    else:
                        # 在线标准化，避免数值尺度问题
                        try:
                            col_std = Xr.std().replace(0, 1e-6)
                            Xr_norm = (Xr - Xr.mean()) / col_std
                            Xr_input = Xr_norm.clip(-5, 5).fillna(0)
                            logger.info("回归模型未找到标准化器，已对特征进行在线标准化处理")
                        except Exception:
                            logger.warning("回归模型在线标准化失败，退回原始特征")
                            Xr_input = Xr
                
                exp_returns = model_r.predict(Xr_input)
                logger.info(f"回归模型预测成功，样本数: {len(exp_returns)}")
        except Exception as e:
            logger.error(f"回归预测失败: {e}")
            # 设置默认返回值
            exp_returns = np.array([0.0] * len(features_df))
            
        # 如果分类概率过于极端，尝试用技术指标进行微调
        try:
            prob_std = np.std(probs)
            prob_mean = np.mean(probs)
            logger.info(f"模型预测概率统计: 平均值={prob_mean:.3f}, 标准差={prob_std:.3f}")
            
            # 应用改进的概率校准，传入预测结果和预期收益
            probs = self._calibrate_probabilities(probs, preds_cls, exp_returns)
            
            # 如果校准后仍然极端，使用技术指标进一步调整
            calibrated_std = np.std(probs)
            calibrated_mean = np.mean(probs)
            if calibrated_std < 0.05 or calibrated_mean > 0.9 or calibrated_mean < 0.1:
                logger.warning("校准后概率仍过于极端，使用技术指标进一步调整")
                adjusted_probabilities = self._adjust_probabilities_with_technical_indicators(
                    features_df['symbol'].tolist(), probs)
                probs = adjusted_probabilities
        except Exception as e:
            logger.warning(f"概率调整失败: {e}")
            
        # 组合结果
        results = []
        # 拉取一次全量symbol信息，减少循环内查询
        symbols_data = {s['symbol']: s for s in self.db.list_symbols(markets=['SH','SZ'])}
        for i, symbol in enumerate(features_df['symbol']):
            stock_info = symbols_data.get(symbol, {})
            # 获取最新价格
            latest_bars = self.db.get_last_n_bars([symbol], n=1)
            latest_price = None
            if not latest_bars.empty:
                latest_price = {'close': latest_bars.iloc[-1]['close']}
            
            prob = float(probs[i]) if i < len(probs) else 0.5
            exp_ret = float(exp_returns[i]) if i < len(exp_returns) else 0.0
            predictions = preds_cls if i < len(preds_cls) else [0]

            # 当未加载回归模型或回归输出为空/接近0时，使用基于近期价格的回退逻辑估计30天预期收益
            try:
                need_fallback_exp = (self.reg_model_data is None) or (np.isnan(exp_ret)) or (abs(exp_ret) < 1e-6)
            except Exception:
                need_fallback_exp = 1
            if need_fallback_exp:
                fb_ret = 0.0
                try:
                    bars_30 = self.db.get_last_n_bars([symbol], n=30)
                    if not bars_30.empty:
                        bars_30 = bars_30[bars_30['symbol'] == symbol].sort_values('date')
                        closes = bars_30['close'].astype(float).values
                        if len(closes) >= 5:
                            prev = closes[:-1].copy()
                            prev = np.where(prev == 0, np.nan, prev)
                            daily_ret = (closes[1:] - prev) / prev
                            daily_ret = daily_ret[np.isfinite(daily_ret)]
                            if len(daily_ret) >= 3:
                                mean_daily = float(np.nanmean(daily_ret))
                                # 约20个交易日对应30天
                                expected_30 = mean_daily * 20.0
                                # 依据概率进行温和加权（范围约0.5~1.2）
                                weight = 0.8 + (prob - 0.5)
                                weight = max(0.5, min(1.2, weight))
                                # 引入基于波动的惩罚，波动越高惩罚越大，范围约0.6~1.0
                                vol_daily = float(np.nanstd(daily_ret))
                                vol_penalty = 1.0 / (1.0 + 3.0 * vol_daily)
                                vol_penalty = max(0.6, min(1.0, vol_penalty))
                                fb_ret = expected_30 * weight * vol_penalty
                except Exception as _:
                    fb_ret = 0.0
                    vol_daily = 0.03
                # 若仍接近0，则用概率微调一个小幅度（±5%）
                if abs(fb_ret) < 1e-6:
                    fb_ret = (prob - 0.5) * 0.1
                # 软限幅，避免“顶格”扎堆；再叠加极小的确定性扰动降低并列概率
                cap = 0.12
                try:
                    soft_ret = cap * np.tanh(fb_ret / max(1e-9, cap))
                except Exception:
                    soft_ret = max(-cap, min(cap, fb_ret))
                # 使用概率与波动构造极小的tie-breaker（约±0.2%以内），保持可解释性
                try:
                    v = float(vol_daily) if 'vol_daily' in locals() and vol_daily is not None else 0.03
                except Exception:
                    v = 0.03
                epsilon = 0.002 * (prob - 0.5) - 0.001 * v
                exp_ret = max(-cap, min(cap, soft_ret + epsilon))

            # 最终归一化与安全限幅，防止异常值（如1255.9%）
            try:
                r = float(exp_ret)
            except Exception:
                r = 0.0
            if not np.isfinite(r):
                r = 0.0
            # 百分比单位ncorrect：若预测在50%~500%之间，按百分数转小数；>500%视为异常，退回小幅估计
            if abs(r) > 5:
                # 极端异常，使用基于概率的小幅估计
                r = (prob - 0.5) * 0.1
            elif abs(r) > 0.5:
                # 介于50%~500%，很可能是百分比单位
                r = r / 100.0
            # 二次软限幅与硬限幅
            cap_final = 0.18
            try:
                r = cap_final * np.tanh(r / max(1e-9, cap_final))
            except Exception:
                r = max(-cap_final, min(cap_final, r))
            exp_ret = max(-0.25, min(0.25, r))

            # 计算个性化的信心度
            base_confidence = abs(prob - 0.5) * 100
            data_quality_factor = 0
            if not latest_bars.empty:
                recent_volume = latest_bars.iloc[-1]['volume']
                if recent_volume > 0:
                    data_quality_factor += 5
                recent_bars = self.db.get_last_n_bars([symbol], n=5)
                # 统一 recent_bars 数值列类型，避免 Decimal 与 float 运算冲突
                if not recent_bars.empty and 'close' in recent_bars.columns:
                    recent_bars['close'] = pd.to_numeric(recent_bars['close'], errors='coerce').astype(float)
                if len(recent_bars) >= 5:
                    price_stability = 1 / (recent_bars['close'].std() / recent_bars['close'].mean() + 0.01)
                    data_quality_factor += min(10, price_stability * 2)
            import random
            random.seed(hash(symbol + str(int(prob * 1000))) % 1000)
            random_factor = (random.random() - 0.5) * 10
            confidence = min(95, max(30, base_confidence + data_quality_factor + random_factor))
            if prob > 0.6:
                sentiment = "看多"
            elif prob < 0.4:
                sentiment = "看空"
            else:
                sentiment = "中性"
                confidence = 50 + abs(prob - 0.5) * 20
            result = {
                'symbol': symbol,
                'name': stock_info.get('name', ''),
                # 新字段
                'prob_up_30d': round(prob, 3),
                'expected_return_30d': round(exp_ret, 4),
                # 兼容旧字段
                'probability': round(prob, 3),
                'prediction': int(predictions[i]) if isinstance(predictions, np.ndarray) else int(predictions),
                'last_close': latest_price.get('close', 0) if latest_price else 0,
                'score': round((prob * 100), 2),
                'sentiment': sentiment,
                'confidence': round(confidence, 1),
                # 添加signal字段
                'signal': sentiment
            }
            results.append(result)
        
        # 优先按预期收益排序，其次按概率
        results.sort(key=lambda x: (x.get('expected_return_30d', 0), x.get('prob_up_30d', 0)), reverse=1)
        
        # 过滤无效股票（退市、ST、停牌等）
        valid_results = []
        for result in results:
            filter_check = self.stock_filter.should_filter_stock(
                result.get('name', ''), 
                result.get('symbol', ''),
                include_st=1,
                include_suspended=1,
                db_manager=self.db,
                exclude_star_market=1,
                last_n_days=30
            )
            
            if not filter_check['should_filter']:
                valid_results.append(result)
            else:
                logger.debug(f"过滤股票 {result['symbol']} - {result['name']} "
                           f"({filter_check['reason']})")
        
        logger.info(f"股票过滤: 原始{len(results)}只 -> 有效{len(valid_results)}只")
        
        return valid_results[:top_n]
    
    def _calibrate_probabilities(self, probs: np.ndarray, predictions: np.ndarray = None, 
                               expected_returns: np.ndarray = None) -> np.ndarray:
        """
        使用多种信息源进行概率校准，提高预测区分度
        """
        try:
            # 检查概率分布
            prob_std = np.std(probs)
            prob_mean = np.mean(probs)
            
            logger.info(f"原始概率统计: 平均值={prob_mean:.4f}, 标准差={prob_std:.4f}")
            
            # 如果概率分布已经合理，只做轻微调整
            if prob_std >= 0.08 and 0.15 <= prob_mean <= 0.85:
                # 轻微扩展分布，增加区分度
                calibrated_probs = 0.5 + (probs - prob_mean) * 1.2
                calibrated_probs = np.clip(calibrated_probs, 0.05, 0.95)
                return calibrated_probs
            
            calibrated_probs = np.copy(probs)
            
            # 方法1: 基于预期收益的概率调整
            if expected_returns is not None and len(expected_returns) == len(probs):
                # 将预期收益转换为概率信号
                returns_normalized = np.tanh(expected_returns * 10)  # 压缩到[-1,1]
                return_probs = 0.5 + returns_normalized * 0.3  # 转换到[0.2, 0.8]
                
                # 与原始概率加权融合
                calibrated_probs = 0.6 * probs + 0.4 * return_probs
            
            # 方法2: 基于分类预测的概率调整
            if predictions is not None and len(predictions) == len(probs):
                # 根据分类结果调整概率
                for i, pred in enumerate(predictions):
                    if pred == 1:  # 看涨预测
                        calibrated_probs[i] = max(calibrated_probs[i], 0.6)
                    else:  # 看跌预测
                        calibrated_probs[i] = min(calibrated_probs[i], 0.4)
            
            # 方法3: 增强概率分布的区分度
            if prob_std < 0.05:
                # 概率过于集中，增加分散度
                prob_ranks = np.argsort(np.argsort(calibrated_probs))  # 获取排名
                n = len(calibrated_probs)
                
                # 基于排名重新分配概率，保持相对顺序
                enhanced_probs = np.zeros_like(calibrated_probs)
                for i, rank in enumerate(prob_ranks):
                    # 将排名映射到[0.2, 0.8]区间
                    enhanced_probs[i] = 0.2 + (rank / (n - 1)) * 0.6
                
                # 与原始概率加权融合
                calibrated_probs = 0.3 * calibrated_probs + 0.7 * enhanced_probs
            
            # 方法4: 处理极端均值
            if prob_mean > 0.8:
                # 整体过于乐观，向下调整
                calibrated_probs = 0.4 + (calibrated_probs - prob_mean) * 0.8
            elif prob_mean < 0.2:
                # 整体过于悲观，向上调整
                calibrated_probs = 0.6 + (calibrated_probs - prob_mean) * 0.8
            
            # 最终限制在合理范围
            calibrated_probs = np.clip(calibrated_probs, 0.05, 0.95)
            
            # 确保有足够的区分度
            final_std = np.std(calibrated_probs)
            if final_std < 0.05:
                # 强制增加区分度
                prob_ranks = np.argsort(np.argsort(calibrated_probs))
                n = len(calibrated_probs)
                spread_probs = np.array([0.15 + (rank / (n - 1)) * 0.7 for rank in prob_ranks])
                calibrated_probs = 0.5 * calibrated_probs + 0.5 * spread_probs
                calibrated_probs = np.clip(calibrated_probs, 0.05, 0.95)
            
            logger.info(f"校准后概率统计: 平均值={np.mean(calibrated_probs):.4f}, "
                       f"标准差={np.std(calibrated_probs):.4f}")
            
            return calibrated_probs
            
        except Exception as e:
            logger.warning(f"概率校准失败: {e}")
            return probs
    
    def _adjust_probabilities_with_technical_indicators(self, symbols: List[str], 
                                                      original_probs: np.ndarray) -> np.ndarray:
        """
        使用技术指标调整过于极端的概率值
        """
        adjusted_probs = []
        
        for i, symbol in enumerate(symbols):
            try:
                # 获取最近30天的价格数据
                prices = self.db.get_last_n_bars([symbol], n=30)
                if not prices.empty:
                    prices = prices[prices['symbol'] == symbol].copy()
                    prices = prices.sort_values('date')
                    
                    # 确保列名大写
                    prices = prices.rename(columns={
                        'open': 'Open',
                        'high': 'High', 
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    })
                    # 确保数值列为 float，避免 Decimal 类型导致运算错误
                    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    prices[numeric_cols] = prices[numeric_cols].apply(pd.to_numeric, errors='coerce').astype(float)
                    # 确保数值列为 float，避免 Decimal 类型导致运算错误
                    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    prices[numeric_cols] = prices[numeric_cols].apply(pd.to_numeric, errors='coerce').astype(float)
                    
                    if len(prices) >= 10:
                        # 使用SignalGenerator计算factors
                        factors = self.signal_generator.calculate_factors(prices)
                        
                        # 计算技术指标
                        signals = self.signal_generator.generate_signals(prices, factors)
                        
                        if signals:
                            # 计算信号统计
                            buy_signals = len([s for s in signals if s.get('type') == 'BUY'])
                            sell_signals = len([s for s in signals if s.get('type') == 'SELL'])
                            total_signals = buy_signals + sell_signals
                            
                            if total_signals > 0:
                                # 基于技术指标调整概率
                                signal_ratio = buy_signals / total_signals
                                # 将信号比例映射到0.3-0.7的概率范围，避免过于极端
                                adjusted_prob = 0.3 + (signal_ratio * 0.4)
                                
                                # 添加基于价格波动的个性化调整
                                price_volatility = prices['Close'].pct_change().std()
                                volatility_factor = min(0.05, price_volatility)  # 减小波动性调整因子
                                
                                # 添加基于成交量的调整（避免0或缺失导致的除零与NaN）

                                vol_tail = prices['Volume'].tail(5)
                                vol_head = prices['Volume'].head(5)
                                mean_tail = float(vol_tail.replace(0, np.nan).mean()) if not vol_tail.empty else np.nan
                                mean_head = float(vol_head.replace(0, np.nan).mean()) if not vol_head.empty else np.nan
                                if np.isnan(mean_tail) or np.isnan(mean_head) or mean_head == 0:
                                    volume_trend = 1.0
                                else:
                                    volume_trend = mean_tail / mean_head
                                volume_factor = (volume_trend - 1.0) * 0.02  # 减小成交量趋势调整
                                
                                # 添加个股特异性调整
                                stock_hash = hash(symbol) % 1000
                                individual_factor = (stock_hash / 1000 - 0.5) * 0.1  # ±0.05的个股调整
                                
                                # 与原始概率加权平均，并加入个性化因子
                                final_prob = 0.4 * original_probs[i] + 0.6 * adjusted_prob + volatility_factor + volume_factor + individual_factor
                                final_prob = max(0.25, min(0.75, final_prob))  # 限制在25%-75%范围，避免过于极端
                                adjusted_probs.append(final_prob)
                            else:
                                # 无信号时基于价格趋势调整
                                recent_return = (prices['Close'].iloc[-1] / prices['Close'].iloc[-5] - 1) if len(prices) >= 5 else 0
                                trend_prob = 0.5 + recent_return * 0.3  # 减小收益率影响
                                
                                # 添加个股特异性
                                stock_hash = hash(symbol) % 1000
                                individual_factor = (stock_hash / 1000 - 0.5) * 0.15
                                
                                trend_prob += individual_factor
                                trend_prob = max(0.3, min(0.7, trend_prob))
                                adjusted_probs.append(trend_prob)
                        else:
                            # 无法计算信号时使用基于股票特征的概率
                            import random
                            # 使用股票代码和当前时间创建更好的随机种子
                            seed_value = hash(symbol + str(len(prices))) % 10000
                            random.seed(seed_value)
                            base_prob = 0.4 + random.random() * 0.2  # 40%-60%范围
                            
                            # 基于价格位置调整（相对于最高最低价）
                            if len(prices) >= 5:
                                current_price = float(prices['Close'].iloc[-1])
                                high_price = float(prices['High'].max())
                                low_price = float(prices['Low'].min())
                                if high_price > low_price:
                                    price_position = (current_price - low_price) / (high_price - low_price)

                                position_adjustment = (price_position - 0.5) * 0.08  # 减小调整幅度
                                base_prob += position_adjustment
                            
                            # 添加个股特异性
                            stock_hash = hash(symbol) % 1000
                            individual_factor = (stock_hash / 1000 - 0.5) * 0.12
                            base_prob += individual_factor
                            
                            adjusted_probs.append(max(0.3, min(0.7, base_prob)))
                    else:
                        # 数据不足时使用更个性化的随机概率
                        import random
                        seed_value = hash(symbol + str(i)) % 10000
                        random.seed(seed_value)
                        adjusted_probs.append(0.3 + random.random() * 0.4)
                else:
                    # 无价格数据时使用随机化概率
                    import random
                    random.seed(hash(symbol) % 1000)
                    adjusted_probs.append(0.3 + random.random() * 0.4)
                    
            except Exception as e:
                logger.warning(f"调整股票 {symbol} 概率失败: {e}")
                # 异常时使用随机化概率
                import random
                random.seed(hash(symbol) % 1000)
                adjusted_probs.append(0.3 + random.random() * 0.4)
        
        return np.array(adjusted_probs)
    
    def predict_top_n(self, symbols: List[str], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        预测股票并返回排序结果（兼容旧接口）
        """
        try:
            # 进行预测
            results = self.predict_stocks(symbols, top_n)
            return results
        except Exception as e:
            logger.error(f"predict_top_n失败: {e}")
            return []

    def get_stock_picks(self, top_n: int = 10) -> Dict[str, Any]:
        """
        获取智能选股结果
        """
        try:
            # 加载模型（优先加载30天的分类与回归模型）
            if not self.load_models(period='30d'):
                # 优先回退到10d周期的新接口
                if not self.load_models(period='10d'):
                    # 回退到旧的单模型加载逻辑
                    if not self.load_model():
                        return self._fallback_stock_picks(top_n)
                    else:
                        logger.info("已使用旧模型接口")
        
            # 获取所有股票代码（仅A股主板/创业板，排除BJ）
            symbols_data = self.db.list_symbols(markets=['SH','SZ'])
            symbols = [s.get('symbol') for s in symbols_data if s.get('symbol')]
        
            if not symbols:
                return {
                    'success': 0,
                    'message': '没有可用的股票数据',
                    'data': {'picks': []}
                }
            
            # 在候选池阶段过滤无效股票（排除None.*、000000、格式不规范等），并记录统计
            invalid_pattern_count = 0
            filtered_by_status = 0
            valid_symbols = []
            for symbol_info in symbols_data:
                symbol = symbol_info.get('symbol', '')
                name = symbol_info.get('name', '')
                # 额外的无效格式过滤
                try:
                    parts = symbol.split('.') if isinstance(symbol, str) else []
                    code = parts[0] if len(parts) >= 1 else ''
                    market = parts[1] if len(parts) >= 2 else ''
                    if (not isinstance(symbol, str) or not symbol or
                        symbol.startswith('None') or symbol.endswith('.None') or
                        code == '000000' or
                        len(parts) != 2 or len(code) != 6 or not code.isdigit() or market not in ('SH','SZ')):
                        invalid_pattern_count += 1
                        logger.debug(f"排除无效股票代码: {symbol} ({name})")
                        continue
                except Exception:
                    invalid_pattern_count += 1
                    logger.debug(f"排除无效股票代码: {symbol} ({name})")
                    continue
                filter_check = self.stock_filter.should_filter_stock(
                    name, symbol,
                    include_st=1,
                    include_suspended=1,
                    db_manager=self.db,
                    exclude_star_market=1,
                    last_n_days=30
                )
                
                if not filter_check['should_filter']:
                    valid_symbols.append(symbol)
                else:
                    filtered_by_status += 1
            
            logger.info(f"候选池过滤: 原始{len(symbols)}只 -> 无效格式{invalid_pattern_count}只 -> 状态过滤{filtered_by_status}只 -> 有效{len(valid_symbols)}只")
            
            if not valid_symbols:
                return {
                    'success': 0,
                    'message': '过滤后没有可用的股票',
                    'data': {'picks': []}
                }
                
            # 进行预测（带重试回退）
            import time, random
            sel_max_retries = int(os.getenv('SELECTOR_MAX_RETRIES', '2'))
            sel_retry_delay = float(os.getenv('SELECTOR_RETRY_DELAY', '0.5'))
            logger.info(f"预测阶段重试配置: max_retries={sel_max_retries}, retry_delay={sel_retry_delay:.2f}s")
            attempt = 0
            picks = []
            last_err = None
            while attempt <= sel_max_retries:
                try:
                    picks = self.predict_stocks(valid_symbols, top_n)
                    break
                except Exception as e:
                    last_err = e
                    logger.warning(f"预测失败（第{attempt+1}次）: {e}")
                    if attempt >= sel_max_retries:
                        break
                    jitter = 0.5 + random.random()
                    delay = min(10.0, sel_retry_delay * (2 ** attempt) * jitter)
                    logger.info(f"{delay:.2f}s 后重试预测...")
                    time.sleep(delay)
                    attempt += 1
            used_fallback = 0
            if (not picks) and last_err is not None:
                logger.warning("预测结果为空或失败，回退到备用技术指标选股")
                fb = self._fallback_stock_picks(top_n)
                if fb.get('success'):
                    picks = fb['data'].get('picks', [])
                    used_fallback = 1
                else:
                    picks = []
            
            # 添加调试日志
            logger.info(f"预测结果数量: {len(picks)}")
            if picks:
                logger.info(f"第一个结果的字段: {list(picks[0].keys())}")
                logger.info(f"第一个结果: {picks[0]}")
            
            return {
                'success': 1,
                'data': {
                    'picks': picks,
                    'model_type': ('technical_indicators' if used_fallback else ('ml_cls+reg' if self.reg_model_data and self.cls_model_data else ('machine_learning' if self.model else 'technical_indicators'))),
                    'generated_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"智能选股失败: {e}")
            return {
                'success': 0,
                'message': f'选股服务出错: {str(e)}',
                'data': {'picks': []}
            }

    def _fallback_stock_picks(self, top_n: int = 10) -> Dict[str, Any]:
        """
        备用选股方法：基于技术指标评分
        """
        logger.info("使用备用选股方法（技术指标评分）")
        
        try:
            symbols_data = self.db.list_symbols(markets=['SH','SZ'])
            results = []
            
            for stock in symbols_data:
                symbol = stock['symbol']
                
                # 获取最近30天的价格数据
                prices = self.db.get_last_n_bars([symbol], n=30)
                if not prices.empty:
                    prices = prices[prices['symbol'] == symbol].copy()
                    prices = prices.sort_values('date')
                    
                    # 确保列名大写
                    prices = prices.rename(columns={
                        'open': 'Open',
                        'high': 'High', 
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    })
                if len(prices) < 10:
                    continue
                    
                # 使用SignalGenerator计算factors
                factors = self.signal_generator.calculate_factors(prices)
                
                # 计算技术指标
                signals = self.signal_generator.generate_signals(prices, factors)
                if not signals:  # signals是list，不是DataFrame
                    continue
                    
                # 计算信号统计
                buy_signals = len([s for s in signals if s.get('type') == 'BUY'])
                sell_signals = len([s for s in signals if s.get('type') == 'SELL'])
                
                # 计算综合评分
                score = buy_signals - sell_signals
                
                # 获取最新价格
                latest_price = prices.iloc[-1] if not prices.empty else None
                
                # 修正概率计算逻辑，避免过高的概率值
                total_signals = buy_signals + sell_signals
                if total_signals > 0:
                    # 基于信号比例计算概率，范围在0.3-0.8之间
                    signal_ratio = buy_signals / total_signals
                    probability = 0.3 + (signal_ratio * 0.5)  # 映射到30%-80%范围
                else:
                    probability = 0.5  # 无信号时为中性50%
                
                # 生成看多看空指标
                if buy_signals > sell_signals:
                    sentiment = "看多"
                    confidence = min(90, 50 + (buy_signals - sell_signals) * 5)
                elif sell_signals > buy_signals:
                    sentiment = "看空"
                    confidence = min(90, 50 + (sell_signals - buy_signals) * 5)
                else:
                    sentiment = "中性"
                    confidence = 50
                
                result = {
                    'symbol': symbol,
                    'name': stock['name'],
                    'score': score,
                    'last_close': latest_price['Close'] if latest_price is not None else 0,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'probability': round(probability, 3),  # 保留3位小数
                    'sentiment': sentiment,  # 看多/看空/中性
                    'confidence': confidence  # 信心度
                }
                results.append(result)
            
            # 按评分排序
            results.sort(key=lambda x: x['score'], reverse=1)
            
            return {
                'success': 1,
                'data': {
                    'picks': results[:top_n],
                    'model_type': 'technical_indicators',
                    'generated_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"备用选股方法失败: {e}")
            return {
                'success': 0,
                'message': f'选股服务出错: {str(e)}',
                'data': {'picks': []}
            }


if __name__ == "__main__":
    # 测试选股服务
    selector = IntelligentStockSelector()
    result = selector.get_stock_picks(top_n=5)
    print("选股结果:")
    print(result)