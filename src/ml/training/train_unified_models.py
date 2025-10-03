# -*- coding: utf-8 -*-
"""
统一模型训练脚本
整合分类和回归模型训练功能
适配现有数据接入和数据库访问方式
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any, Union
import pickle
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import warnings

# 添加项目根目录到路径
# 将项目根目录添加到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.unified_data_access import UnifiedDataAccessLayer
from src.data.db.unified_database_manager import UnifiedDatabaseManager
from src.core.unified_data_access_factory import create_unified_data_access
from src.ml.features.enhanced_features import EnhancedFeatureGenerator
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from src.ml.training.enhanced_ml_trainer import EnhancedMLTrainer
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator

# ---------------- 滚动时间窗交叉验证 ----------------
class RollingWindowSplit(BaseCrossValidator):
    """按滚动时间窗进行交叉验证。例如设 n_splits=5, window=120, horizon=20:
    第1折: 0~119 训练, 120~139 验证
    第2折: 20~139 训练, 140~159 验证
    以此类推。
    参数
    ------
    window: 训练集窗口长度
    horizon: 测试集长度
    step: 窗口滚动步长
    """
    def __init__(self, n_splits: int = 5, window: int = 120, horizon: int = 20, step: int = 20):
        self.n_splits = n_splits
        self.window = window
        self.horizon = horizon
        self.step = step

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        start = 0
        for i in range(self.n_splits):
            train_start = start
            train_end = train_start + self.window  # exclusive
            test_start = train_end
            test_end = test_start + self.horizon
            if test_end > n_samples:
                break
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            yield (train_idx, test_idx)
            start += self.step

    def get_n_splits(self, X=None, y=None, groups=None):
        """根据数据长度动态返回实际可用的折数，避免 \n        当样本不足时 self.n_splits 大于实际生成的折数，导致交叉验证报错"""
        if X is None:
            # 如果无法获知样本数，则保守返回 self.n_splits
            return self.n_splits
        n_samples = len(X)
        # 计算在给定 window/horizon/step 下，最多可以生成多少折
        max_possible = (n_samples - self.window - self.horizon) // self.step + 1
        max_possible = max_possible if max_possible > 0 else 0
        return min(self.n_splits, max_possible)

# ---------- 使用改进的 PurgedKFoldWithEmbargo 统一交叉验证 ----------
from src.ml.evaluation.cv_utils import PurgedKFoldWithEmbargo
# 将 RollingWindowSplit 指向 PurgedKFoldWithEmbargo，以在后续代码统一使用
RollingWindowSplit = PurgedKFoldWithEmbargo

def get_cv(n_splits: int = 5, embargo: int = 60):
    """统一返回交叉验证分割器，便于后续替换实现。"""
    return PurgedKFoldWithEmbargo(n_splits=n_splits, embargo=embargo)

# ---------------- 加权信息系数(IC)评估器 ----------------

import numpy as np

def _recency_weights(length: int, window: int = 120):
    """生成指数衰减权重，使最近 window 个样本权重更高"""
    if length <= 0:
        return np.array([])
    idx = np.arange(length)
    rel = (idx - idx.min()) / max(1, length - 1)  # 归一化到 [0,1]
    tau = max(window, 1)
    w = np.exp(rel * tau / window)
    return w


def information_coefficient(y_true, y_pred, window: int = 120):
    """加权 Pearson IC，使用最近样本加权 (window=120)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n <= 1 or np.allclose(np.std(y_true), 0):
        return 0.0
    w = _recency_weights(n, window)
    w /= w.sum()
    mu_y = np.sum(w * y_true)
    mu_p = np.sum(w * y_pred)
    cov = np.sum(w * (y_true - mu_y) * (y_pred - mu_p))
    var_y = np.sum(w * (y_true - mu_y) ** 2)
    var_p = np.sum(w * (y_pred - mu_p) ** 2)
    denom = np.sqrt(var_y * var_p)
    return float(cov / denom) if denom > 0 else 0.0

# scorer 对象，可直接用于 sklearn 的 cross_val_score / GridSearchCV
ic_scorer = make_scorer(information_coefficient, greater_is_better=True)
from src.ml.evaluation.metrics import sharpe_ratio, information_ratio
sharpe_scorer = make_scorer(sharpe_ratio, greater_is_better=True)
ir_scorer = make_scorer(information_ratio, greater_is_better=True)

# ================= 新增: 加权 MSE 评估器 (Weighted MSE) =================

def weighted_mse(y_true, y_pred, window: int = 120):
    """加权均方误差 (窗口内最近样本权重更高, 越小越好)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0
    w = _recency_weights(n, window)
    w /= w.sum()
    return float(np.sum(w * (y_true - y_pred) ** 2))

# scorer: greater_is_better=False 代表 Optuna/网格搜索将最小化该分数
weighted_mse_scorer = make_scorer(weighted_mse, greater_is_better=False)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('unified_training.log')
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.WARNING)

class UnifiedModelTrainer:
    """统一模型训练器，支持分类和回归任务"""
    
    def __init__(self, enable_feature_selection: bool = True,
                 feature_cache_dir: str = 'feature_cache',
                 reuse_feature_selection: bool = True,
                 refresh_feature_selection: bool = False,
                 feature_selection_n_jobs: int = -1):
        """初始化统一模型训练器
        Args:
            enable_feature_selection: 是否在数据预处理阶段启用特征选择优化器
            feature_cache_dir: 特征缓存目录
            reuse_feature_selection: 是否复用已存在的特征选择结果
            refresh_feature_selection: 是否强制刷新特征选择结果并覆盖缓存
        """
        # 使用现有的统一数据访问层
        self.data_access = create_unified_data_access()
        self.db_manager = self.data_access.db_manager
        self.feature_generator = EnhancedFeatureGenerator()
        # 新增: 特征选择开关
        self.enable_feature_selection = enable_feature_selection
        # 特征选择缓存相关
        self.feature_cache_dir = feature_cache_dir
        os.makedirs(self.feature_cache_dir, exist_ok=True)
        self.reuse_feature_selection = reuse_feature_selection
        self.refresh_feature_selection = refresh_feature_selection
        self.feature_cache_file = os.path.join(self.feature_cache_dir, 'selected_features.json')
        # 并行核心数
        self.feature_selection_n_jobs = feature_selection_n_jobs
    
    def prepare_training_data(self, stock_list: List[str] = None, mode: str = 'both', lookback_days: int = 365, 
                            n_stocks: int = 1000, prediction_period: int = 30) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        准备训练数据 - 添加数据获取限制和快速失败机制
        
        Args:
            mode: 'classification', 'regression', 或 'both'
            lookback_days: 回溯天数
            n_stocks: 使用的股票数量
            prediction_period: 预测周期（天数）
            
        Returns:
            X: 特征数据
            y: 标签数据字典
        """
        logger.info(f"开始准备训练数据 (模式: {mode})...")
        
        # 获取股票列表 - 使用统一数据访问层
        stock_list = self.data_access.get_all_stock_list()
        if stock_list is None or stock_list.empty:
            raise ValueError("无法获取股票列表")
        
        if 'symbol' not in stock_list.columns:
            raise ValueError("股票列表缺少 'symbol' 列，请检查数据访问层是否已生成标准化代码")
        symbols_info = stock_list.set_index('symbol')
        symbols = symbols_info.index.tolist()
        # 随机打乱股票列表，确保覆盖沪深主板及创业板
        np.random.shuffle(symbols)
        logger.info(f"随机打乱后股票总数 {len(symbols)}，将从中抽取 {n_stocks} 只")
        
        # 设置日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        logger.info(f"数据范围: {start_date.date()} 到 {end_date.date()}")
        
        all_features = []
        all_cls_labels = []
        all_reg_labels = []
        failed_stocks = []  # 记录处理失败或数据不足的股票
        collected_count = 0  # 成功收集的股票数量
        insufficient_symbols = []  # 本地数据不足，稍后再尝试 auto_sync 补拉的股票列表

        # ---------------- 第一阶段：仅使用本地数据 ----------------
        for symbol in symbols:
            if collected_count >= n_stocks:
                break

            if collected_count % 10 == 0:
                logger.info(f"已成功收集 {collected_count} 只股票，当前处理 {symbol}")

            try:
                # 先只用本地数据，auto_sync=False
                stock_data = self.data_access.get_stock_data(symbol, start_date, end_date, auto_sync=False)

                # 判定数据量是否足够
                min_required_len = prediction_period + 15
                if stock_data is None or stock_data.empty or len(stock_data) < min_required_len:
                    insufficient_symbols.append(symbol)
                    failed_stocks.append(symbol)
                    continue

                # ---------------- 以下与原处理逻辑一致 ----------------
                stock_data = stock_data.reset_index()
                stock_data['date'] = pd.to_datetime(stock_data['date'])
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in stock_data.columns:
                        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce').astype(float)
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                stock_data = stock_data.dropna(subset=required_cols)

                features_df = self.feature_generator.generate_features(stock_data)
                if features_df.empty:
                    logger.warning(f"{symbol} 生成增强特征失败，尝试基础特征生成器")
                    from src.ml.features.feature_generator import FeatureGenerator
                    basic_generator = FeatureGenerator()
                    features_df = basic_generator.generate_all_features(stock_data)
                    if features_df.empty:
                        logger.warning(f"{symbol} 基础特征生成也失败，跳过")
                        failed_stocks.append(symbol)
                        continue
                if len(features_df.columns) < 10:
                    logger.warning(f"{symbol} 特征数量较少({len(features_df.columns)}个)，尝试生成基础特征")
                    tech_features = basic_generator.generate_technical_features(stock_data)
                    if not tech_features.empty:
                        features_df = tech_features
                        logger.info(f"{symbol} 使用技术指标特征({len(features_df.columns)}个)")

                close_prices = stock_data['close'].values
                reg_labels = []
                cls_labels = []
                for j in range(len(close_prices) - prediction_period):
                    current_price = close_prices[j]
                    future_price = close_prices[j + prediction_period]
                    return_rate = (future_price - current_price) / current_price
                    reg_labels.append(return_rate)
                    cls_labels.append(1 if return_rate > 0.05 else 0)

                aligned_features = features_df.iloc[:-prediction_period].copy()
                if len(aligned_features) != len(reg_labels):
                    min_len = min(len(aligned_features), len(reg_labels))
                    aligned_features = aligned_features.iloc[:min_len]
                    reg_labels = reg_labels[:min_len]
                    cls_labels = cls_labels[:min_len]
                if aligned_features.empty:
                    failed_stocks.append(symbol)
                    continue

                # 行业 One-Hot 和市值特征
                industry = symbols_info.loc[symbol, 'industry'] if 'industry' in symbols_info.columns else None
                market_cap = symbols_info.loc[symbol, 'market_cap'] if 'market_cap' in symbols_info.columns else None
                aligned_features['industry'] = industry
                aligned_features['market_cap_log'] = np.log1p(market_cap) if market_cap is not None else np.nan
                aligned_features['symbol'] = symbol
                all_features.append(aligned_features)
                all_reg_labels.extend(reg_labels)
                all_cls_labels.extend(cls_labels)
                collected_count += 1
            except Exception as e:
                logger.warning(f"处理股票 {symbol} 失败: {e}")
                import traceback
                logger.warning(f"详细错误信息: {traceback.format_exc()}")
                failed_stocks.append(symbol)

        # ---------------- 第二阶段：对不足股票启用 auto_sync 再尝试 ----------------
        if collected_count < n_stocks and insufficient_symbols:
            logger.info(f"本地数据不足，仅收集到 {collected_count} 只股票，尝试对 {len(insufficient_symbols)} 只股票启用 auto_sync 补拉数据")
            for symbol in insufficient_symbols:
                if collected_count >= n_stocks:
                    break
                try:
                    stock_data = self.data_access.get_stock_data(symbol, start_date, end_date, auto_sync=True)
                    if stock_data is None or stock_data.empty or len(stock_data) < prediction_period + 15:
                        continue
                    # 与第一阶段相同的处理流水线 ----------------
                    stock_data = stock_data.reset_index()
                    stock_data['date'] = pd.to_datetime(stock_data['date'])
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        if col in stock_data.columns:
                            stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce').astype(float)
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    stock_data = stock_data.dropna(subset=required_cols)
                    features_df = self.feature_generator.generate_features(stock_data)
                    if features_df.empty:
                        from src.ml.features.feature_generator import FeatureGenerator
                        basic_generator = FeatureGenerator()
                        features_df = basic_generator.generate_all_features(stock_data)
                        if features_df.empty:
                            continue
                    if len(features_df.columns) < 10:
                        from src.ml.features.feature_generator import FeatureGenerator
                        basic_generator = FeatureGenerator()
                        tech_features = basic_generator.generate_technical_features(stock_data)
                        if not tech_features.empty:
                            features_df = tech_features

                    close_prices = stock_data['close'].values
                    reg_labels = []
                    cls_labels = []
                    for j in range(len(close_prices) - prediction_period):
                        current_price = close_prices[j]
                        future_price = close_prices[j + prediction_period]
                        return_rate = (future_price - current_price) / current_price
                        reg_labels.append(return_rate)
                        cls_labels.append(1 if return_rate > 0.05 else 0)
                    aligned_features = features_df.iloc[:-prediction_period].copy()
                    if len(aligned_features) != len(reg_labels):
                        min_len = min(len(aligned_features), len(reg_labels))
                        aligned_features = aligned_features.iloc[:min_len]
                        reg_labels = reg_labels[:min_len]
                        cls_labels = cls_labels[:min_len]
                    if aligned_features.empty:
                        continue
                    aligned_features['symbol'] = symbol
                    all_features.append(aligned_features)
                    all_reg_labels.extend(reg_labels)
                    all_cls_labels.extend(cls_labels)
                    collected_count += 1
                    logger.info(f"auto_sync 成功补充 {symbol}，已收集 {collected_count}/{n_stocks}")
                except Exception as e:
                    logger.warning(f"auto_sync 获取 {symbol} 失败: {e}")
                    import traceback
                    logger.warning(f"详细错误信息: {traceback.format_exc()}")

        # ---------------- 数据汇总与返回 ----------------
        if not all_features:
            raise ValueError("没有生成任何训练数据")

        X_combined = pd.concat(all_features, ignore_index=True)
        y_reg = pd.Series(all_reg_labels, name='return_rate')
        y_cls = pd.Series(all_cls_labels, name='label_cls')

        logger.info(f"最终生成 {len(X_combined)} 个样本，成功股票数: {collected_count}，失败股票数: {len(failed_stocks)}")

        # 数据预处理
        X_processed, y_reg_processed, y_cls_processed = self._preprocess_data(X_combined, y_reg, y_cls)

        y_dict: Dict[str, pd.Series] = {}
        if mode in ['classification', 'both']:
            y_dict['cls'] = y_cls_processed
        if mode in ['regression', 'both']:
            y_dict['reg'] = y_reg_processed

        return X_processed, y_dict

    def _preprocess_data(self, X: pd.DataFrame, y_reg: pd.Series, y_cls: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """数据预处理 - 包含特征选择优化"""
        logger.info("开始数据预处理...")
        
        # 行业One-Hot编码
        if 'industry' in X.columns:
            industry_dummies = pd.get_dummies(X['industry'], prefix='ind', dummy_na=True)
            X = pd.concat([X.drop(columns=['industry']), industry_dummies], axis=1)
        
        # 移除非数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy()
        
        # 处理无穷大和NaN
        X_cleaned = X_numeric.replace([np.inf, -np.inf], np.nan)
        
        # ---------------- 新增: 先按列统计缺失比例，删除缺失过多的列 ----------------
        na_ratio = X_cleaned.isna().mean()
        cols_to_drop = na_ratio[na_ratio > 0.5].index  # 若某列超过50%缺失，则直接删除
        if len(cols_to_drop) > 0:
            logger.info(f"删除缺失过多的列: {list(cols_to_drop)} (阈值50%)")
        X_reduced = X_cleaned.drop(columns=cols_to_drop)
        
        # 数据对齐
        X_imputed = X_reduced.fillna(X_reduced.median())
        
        # 移除仍包含NaN的行（极端情况下中位数为NaN 或全部缺失）
        valid_indices = X_imputed.notna().all(axis=1)
        X_final = X_imputed[valid_indices]
        
        # 使用索引对齐标签数据
        valid_mask = pd.Series(valid_indices, index=X_imputed.index)
        y_reg_final = y_reg[valid_mask]
        y_cls_final = y_cls[valid_mask]
        
        logger.info(f"清洗后样本数量: {len(X_final)}")
        
        # 移除极端异常值（收益率在±50%之外）
        reasonable_returns = (y_reg_final.abs() <= 0.5)
        X_final = X_final[reasonable_returns]
        y_reg_final = y_reg_final[reasonable_returns]
        y_cls_final = y_cls_final[reasonable_returns]
        
        logger.info(f"移除极端值后样本数量: {len(X_final)}")
        logger.info(f"收益率范围: {y_reg_final.min():.3f} 到 {y_reg_final.max():.3f}")
        logger.info(f"收益率均值: {y_reg_final.mean():.6f}, 标准差: {y_reg_final.std():.6f}")
        logger.info(f"正样本比例: {y_cls_final.mean():.3f}")
        
        # 保留可读特征名，避免预测阶段列名不一致问题
        
        # 如果关闭特征选择，直接返回
        if not getattr(self, 'enable_feature_selection', True):
            logger.info("已关闭特征选择优化，直接返回清洗后的全部特征")
            return X_final, y_reg_final, y_cls_final

        # 若启用缓存且存在且不强制刷新，直接复用
        if self.reuse_feature_selection and not self.refresh_feature_selection:
            if os.path.exists(self.feature_cache_file):
                try:
                    with open(self.feature_cache_file, 'r') as f:
                        cached_features = json.load(f)
                    if isinstance(cached_features, list) and all(f in X_final.columns for f in cached_features):
                        logger.info(f"加载缓存特征集合，共 {len(cached_features)} 个特征")
                        X_selected = X_final[cached_features]
                        return X_selected, y_reg_final, y_cls_final
                except Exception as e:
                    logger.warning(f"加载特征缓存失败: {e}")

        # 特征选择优化
        logger.info("开始特征选择优化...")
        try:
            from src.ml.features.feature_selector_optimizer import FeatureSelectorOptimizer
            
            # 为分类和回归任务分别选择特征
            n_features = min(50, X_final.shape[1])  # 选择最多50个特征
            
            # 分类特征选择
            if len(np.unique(y_cls_final)) > 1:  # 确保有多于一个类别
                cls_selector = FeatureSelectorOptimizer(
                    task_type='classification', 
                    target_n_features=n_features,
                    n_jobs=self.feature_selection_n_jobs
                )
                cls_results = cls_selector.optimize_feature_selection(
                    X_final, y_cls_final, method='auto'
                )
                cls_features = cls_results['selected_features']
                logger.info(f"分类任务选择了 {len(cls_features)} 个特征")
            else:
                cls_features = X_final.columns.tolist()
            
            # 回归特征选择
            reg_selector = FeatureSelectorOptimizer(
                task_type='regression', 
                target_n_features=n_features,
                n_jobs=self.feature_selection_n_jobs
            )
            reg_results = reg_selector.optimize_feature_selection(
                X_final, y_reg_final, method='auto'
            )
            reg_features = reg_results['selected_features']
            logger.info(f"回归任务选择了 {len(reg_features)} 个特征")
            
            # 合并两个任务的特征
            all_selected_features = list(set(cls_features + reg_features))
            X_selected = X_final[all_selected_features]
            
            logger.info(f"特征选择优化完成: 从 {X_final.shape[1]} 个特征中选择 {len(all_selected_features)} 个")
            logger.info(f"特征缩减比例: {(1 - len(all_selected_features)/X_final.shape[1])*100:.1f}%")
            # 缓存特征集合
            if self.reuse_feature_selection or self.refresh_feature_selection:
                try:
                    with open(self.feature_cache_file, 'w') as f:
                        json.dump(all_selected_features, f)
                    logger.info(f"特征集合已缓存至 {self.feature_cache_file}")
                except Exception as e:
                    logger.warning(f"写入特征缓存失败: {e}")

            return X_selected, y_reg_final, y_cls_final
            
        except Exception as e:
            logger.warning(f"特征选择优化失败: {e}，使用全部特征")
            return X_final, y_reg_final, y_cls_final
    
    def train_models(self, X: pd.DataFrame, y: Dict[str, pd.Series],
                 mode: str = 'both', use_grid_search: bool = 1,
                 use_optuna: bool = False, optimization_trials: int = 50,
                 cls_models: List[str] = None, reg_models: List[str] = None) -> Dict[str, Any]:
        """
        训练模型（多线程优化版本）
        
        Args:
            X: 特征数据
            y: 标签数据字典
            mode: 'classification', 'regression', 或 'both'
            use_grid_search: 是否使用网格搜索
            
        Returns:
            results: 训练结果字典
        """
        logger.info(f"开始训练模型 (模式: {mode})...")
        
        results = {}
        
        if mode == 'both':
            # 并行训练分类和回归模型
            logger.info("并行训练分类和回归模型...")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 提交分类和回归训练任务
                future_to_task = {}
                
                if 'cls' in y:
                    future_to_task[executor.submit(self._train_classification_models, 
                                                  X, y['cls'], use_grid_search, use_optuna, optimization_trials, cls_models)] = 'classification'
                
                if 'reg' in y:
                    future_to_task[executor.submit(self._train_regression_models, 
                                                  X, y['reg'], use_grid_search, use_optuna, optimization_trials, reg_models)] = 'regression'
                
                # 收集结果
                for future in as_completed(future_to_task):
                    task_type = future_to_task[future]
                    try:
                        task_results = future.result()
                        results.update(task_results)
                        logger.info(f"{task_type} 模型训练完成")
                    except Exception as e:
                        logger.error(f"{task_type} 模型训练异常: {e}")
        
        else:
            # 单独训练分类或回归模型
            if mode == 'classification' and 'cls' in y:
                logger.info("训练分类模型...")
                cls_results = self._train_classification_models(X, y['cls'], use_grid_search, use_optuna, optimization_trials, cls_models)
                results.update(cls_results)
            
            if mode == 'regression' and 'reg' in y:
                logger.info("训练回归模型...")
                reg_results = self._train_regression_models(X, y['reg'], use_grid_search, use_optuna, optimization_trials, reg_models)
                results.update(reg_results)
        
        return results
    
    def _train_single_classification_model(self, trainer, X, y, model_type, use_grid_search, use_optuna, optimization_trials):
        """训练单个分类模型（用于多线程）"""
        try:
            logger.info(f"开始训练 {model_type} 分类模型...")
            # 初始化占位变量，确保后续逻辑在未触发 Optuna/网格搜索时也能安全引用
            best_params = {}
            cv_score = None
            
            # 如果启用Optuna且安装optuna，则进行贝叶斯优化
            if use_optuna:
                try:
                    import optuna
                    # 已在模块顶部统一导入 TimeSeriesSplit 与 cross_val_score，删除局部导入避免作用域问题
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import Pipeline
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    from sklearn.linear_model import LogisticRegression
                    # 根据模型类型选择分类器
                    if model_type == 'xgboost':
                        try:
                            from xgboost import XGBClassifier
                            # 计算类别不平衡比并动态设置 scale_pos_weight
                            pos_cnt = int((y == 1).sum())
                            neg_cnt = int((y == 0).sum())
                            scale_pos_weight = (neg_cnt / pos_cnt) if pos_cnt > 0 else 1.0
                            classifier = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss',
                                                           scale_pos_weight=scale_pos_weight)
                        except ImportError:
                            logger.warning("XGBoost 未安装, 回退使用 trainer 默认实现")
                            raise ImportError
                        def objective(trial):
                            trial_params = {
                                'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 200, 2000),
                                'classifier__max_depth': trial.suggest_int('classifier__max_depth', 3, 12),
                                'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.2, log=True),
                                'classifier__subsample': trial.suggest_float('classifier__subsample', 0.5, 1.0),
                                'classifier__colsample_bytree': trial.suggest_float('classifier__colsample_bytree', 0.5, 1.0),
                                'classifier__min_child_weight': trial.suggest_int('classifier__min_child_weight', 1, 10),
                                'classifier__gamma': trial.suggest_float('classifier__gamma', 0.0, 5.0),
                                'classifier__max_delta_step': trial.suggest_int('classifier__max_delta_step', 0, 10),
                                'classifier__grow_policy': trial.suggest_categorical('classifier__grow_policy', ['depthwise', 'lossguide']),
                                'classifier__reg_lambda': trial.suggest_float('classifier__reg_lambda', 0.0, 10.0),
                                'classifier__reg_alpha': trial.suggest_float('classifier__reg_alpha', 0.0, 10.0)
                            }
                            # 关闭 early stopping 防止交叉验证报缺少验证集错误
                            trial_params_no_es = {k: v for k, v in trial_params.items()}
                            pipe.set_params(**trial_params_no_es)
                            cv = TimeSeriesSplit(n_splits=3)
                            scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                            return scores.mean()  # maximize
                    elif model_type == 'logistic':
                        classifier = LogisticRegression(max_iter=1000, solver='liblinear')
                        def objective(trial):
                            trial_params = {
                                'classifier__C': trial.suggest_float('classifier__C', 0.001, 10.0, log=True),
                                'classifier__penalty': trial.suggest_categorical('classifier__penalty', ['l1', 'l2'])
                            }
                            pipe.set_params(**trial_params)
                            cv = TimeSeriesSplit(n_splits=3)
                            scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                            return scores.mean()
                    else:
                        raise ValueError("不支持的分类模型类型")

                    # 构建包含特征工程的管道: 缺失值填充 → 分位数缩尾 → 跨截面Z分数标准化 → 分类器
                    from sklearn.impute import SimpleImputer
                    from src.ml.features.preprocessing import Winsorizer, CrossSectionZScore
                    pipe = Pipeline([
                            ('imputer', SimpleImputer(strategy='median')),
                            ('winsorizer', Winsorizer()),
                            ('zscore', CrossSectionZScore()),
                            ('classifier', classifier)
                        ])

                    pipe.set_params(classifier__n_estimators=800)

                    study = optuna.create_study(direction='maximize')

                    # ---------- Optuna 早停策略: 指定连续 early_stop_rounds 次未提升则停止 ----------
                    early_stop_rounds = 10  # 可视情况调整
                    def early_stopping_callback(study, trial):
                        """当连续 early_stop_rounds 次 trial 未带来提升时停止搜索"""
                        # 获取当前最佳值
                        best_value = study.best_value
                        # 第一次调用时初始化
                        if 'no_improve_count' not in study.user_attrs:
                            study.set_user_attr('no_improve_count', 0)
                            study.set_user_attr('best_value_snap', best_value)
                            return
                        # 检查是否有提升
                        direction = study.direction
                        improved = (best_value < study.user_attrs['best_value_snap']) if direction.name == 'MINIMIZE' else (best_value > study.user_attrs['best_value_snap'])
                        if improved:
                            study.set_user_attr('no_improve_count', 0)
                            study.set_user_attr('best_value_snap', best_value)
                        else:
                            cnt = study.user_attrs['no_improve_count'] + 1
                            study.set_user_attr('no_improve_count', cnt)
                            if cnt >= early_stop_rounds:
                                logger.info(f"Optuna: 连续 {early_stop_rounds} 次未提升, 提前停止搜索")
                                study.stop()

                    study.optimize(objective, n_trials=optimization_trials, show_progress_bar=False,
                                    callbacks=[early_stopping_callback])

                    best_params = study.best_params
                    logger.info(f"Optuna完成: 最佳参数 {best_params}")

                    # 使用最佳参数先执行交叉验证评估
                    pipe.set_params(**best_params)
                    # 基于时间序列切分再次交叉验证并记录各折指标
                    cv = TimeSeriesSplit(n_splits=3)
                    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

                    # 留出法评估
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                    # 若为 XGBoost 则启用早停，监控验证集 logloss/AUC
                    fit_params = {}
                    # 默认使用原始 DataFrame
                    X_train_in, X_test_in = X_train, X_test
                    if model_type == 'xgboost':
                        # XGBoost 对 eval_set 中的特征名与训练输入的特征名要求保持一致，
                        # 因此将训练/验证特征统一转换为 numpy.ndarray，避免 "data did not contain feature names" 错误。
                        X_train_in = X_train.values if hasattr(X_train, 'values') else X_train
                        X_test_in = X_test.values if hasattr(X_test, 'values') else X_test
                        eval_set = [
                            (X_train_in, y_train.values if hasattr(y_train, 'values') else y_train),
                            (X_test_in, y_test.values if hasattr(y_test, 'values') else y_test)
                        ]
                        # 为最终拟合重新启用早停（不影响 Optuna 交叉验证）
                        pipe.set_params(classifier__early_stopping_rounds=50)
                        # 通过 fit_params 仅传递 eval_set 与 verbose
                        fit_params = {
                            'classifier__eval_set': eval_set,
                            'classifier__verbose': False
                        }
                    pipe.fit(X_train_in, y_train, **fit_params)
                    y_train_pred = pipe.predict(X_train_in)
                    y_test_pred = pipe.predict(X_test_in)

                    # 如果是 XGBoost，记录验证集最佳迭代及 logloss
                    extra_metrics = {}
                    if model_type == 'xgboost':
                        booster = pipe.named_steps['classifier']
                        if hasattr(booster, 'best_iteration'):
                            extra_metrics['best_iteration'] = int(booster.best_iteration)
                        if booster.evals_result_:
                            # XGBoost 将训练集命名为 validation_0，验证集为 validation_1
                            val_logloss = booster.evals_result_.get('validation_1', {}).get('logloss', [])
                            val_auc = booster.evals_result_.get('validation_1', {}).get('auc', [])
                            if val_logloss:
                                extra_metrics['val_logloss'] = float(min(val_logloss))
                            if val_auc:
                                # AUC 越大越好，记录最大值
                                extra_metrics['val_auc'] = float(max(val_auc))
                    result = {
                         'model': pipe,
                         'model_type': model_type,
                         'best_params': best_params,
                         # 记录分类器所有参数（含默认值与优化结果）
                         'all_params': pipe.named_steps['classifier'].get_params(),
                         # 交叉验证指标后续持久化在 metrics 目录下的 CSV

                    'metrics': {
                            'train_accuracy': accuracy_score(y_train, y_train_pred),
                            'test_accuracy': accuracy_score(y_test, y_test_pred),
                            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
                            'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
                            'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
                            'cv_scores': cv_scores.tolist(),
                            'cv_mean': float(cv_scores.mean()),
                            **extra_metrics,
                        },
                    }
                except ImportError:
                    # 回退至原 trainer 逻辑
                    result = trainer.train_single_model(X, y, model_type, 
                                              use_optimization=use_grid_search, 
                                              optimization_trials=optimization_trials)
            else:
                result = trainer.train_single_model(X, y, model_type, 
                                              use_optimization=use_grid_search, 
                                              optimization_trials=optimization_trials)
            
            # 保存模型（与回归模型保持一致，只保存模型对象本身）
            # 获取特征列顺序，确保保存
            feature_names = list(X.columns) if hasattr(X, 'columns') else None
            save_obj = {
                'model': result['model'],
                'feature_names': feature_names,
                'metadata': {
                    'task': 'classification',
                    'model_type': model_type,
                    'feature_config': getattr(self.feature_generator, 'feature_config', None)
                }
            }
            model_name = f"{model_type}_classification.pkl"
            model_path = os.path.join('models', model_name)
            with open(model_path, 'wb') as f:
                pickle.dump(save_obj, f)
            logger.info(f"{model_type} 分类模型已保存到: {model_path}")
            
            # 记录性能指标
            if 'metrics' in result:
                metrics = result['metrics']
                train_acc = metrics.get('train_accuracy', 0)
                test_acc = metrics.get('test_accuracy', 0)
                test_precision = metrics.get('test_precision', 0)
                test_recall = metrics.get('test_recall', 0)
                test_f1 = metrics.get('test_f1', 0)
                logger.info(f"{model_type} 分类模型性能: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}, 精确率={test_precision:.4f}, 召回率={test_recall:.4f}, F1={test_f1:.4f}")
                # 如果包含早停相关指标，额外输出
                best_iter = metrics.get('best_iteration') if isinstance(metrics, dict) else None
                if best_iter is not None:
                    val_logloss = metrics.get('val_logloss', None)
                    val_auc = metrics.get('val_auc', None)
                    logger.info(
                        f"{model_type} 早停结果: best_iteration={best_iter}, "
                        f"val_logloss={val_logloss if val_logloss is not None else 'NA'}, "
                        f"val_auc={val_auc if val_auc is not None else 'NA'}"
                    )
                # ---- 持久化交叉验证各折指标到 CSV ----
                try:
                    from src.ml.evaluation.metrics_io import save_cv_metrics
                    metrics_dir = 'metrics'
                    os.makedirs(metrics_dir, exist_ok=True)
                    csv_path = os.path.join(metrics_dir, f"{model_type}_classification_cv_scores.csv")
                    cv_scores = metrics.get('cv_scores') if isinstance(metrics, dict) else None
                    if cv_scores is not None:
                        save_cv_metrics({'accuracy': cv_scores}, csv_path)
                        logger.info(f"交叉验证指标已保存到: {csv_path}")
                except Exception as err:
                    logger.warning(f"保存交叉验证指标失败: {err}")

            # 记录最佳得分（如果有）
            if 'best_score' in result:
                logger.info(f"{model_type} 最佳得分: {result['best_score']:.4f}")
            
            return model_type, result
            
        except Exception as e:
            logger.error(f"训练 {model_type} 分类模型失败: {e}")
            return model_type, None

    def _train_stacking_regression_model(self, X: pd.DataFrame, y: pd.Series,
                                       use_grid_search: bool = 1,
                                       use_optuna: bool = False, optimization_trials: int = 50):
        """训练 stacking 回归器 (LightGBM + XGBoost + Ridge -> XGBoost 元学习器)"""
        try:
            # 初始化变量，确保在任何返回路径中都有定义
            best_params = {}
            cv_score = None
            
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import Ridge
            from sklearn.ensemble import StackingRegressor
            import numpy as np

            # 检查依赖库
            try:
                from lightgbm import LGBMRegressor
            except ImportError:
                logger.warning("LightGBM 未安装, stacking 回归器跳过")
                return None
            try:
                from xgboost import XGBRegressor
            except ImportError:
                logger.warning("XGBoost 未安装, stacking 回归器跳过")
                return None

            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # 基学习器
            estimators = [
                ('lgbm', LGBMRegressor(objective='huber', random_state=42, n_jobs=-1, verbosity=-1)),
                ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=300, learning_rate=0.05, max_depth=6,
                                      subsample=0.8, colsample_bytree=0.8, n_jobs=-1)),
                ('ridge', Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))]))
            ]

            # 元学习器
            final_estimator = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=300, learning_rate=0.05,
                                           max_depth=4, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)

            # 交叉验证策略
            # 使用统一封装的 get_cv() (PurgedKFoldWithEmbargo) 以确保时序无泄漏且满足 cross_val_predict 要求
            n_splits = min(5, max(2, len(X_train) // 200))  # 至少 2 折，样本足够时最多 5 折
            cv_strategy = get_cv(n_splits=n_splits, embargo=60)

            stacking_model = StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator,
                passthrough=False,
                n_jobs=-1,
                cv=cv_strategy
            )

            # ---------------------------
            # 超参数优化
            # ---------------------------
            best_params = {}
            cv_score = None
            if use_optuna:
                try:
                    import optuna
                    from sklearn.model_selection import cross_val_score

                    def objective(trial):
                        params = {
                            'final_estimator__n_estimators': trial.suggest_int('final_n_estimators', 100, 600),
                            'final_estimator__learning_rate': trial.suggest_float('final_learning_rate', 0.01, 0.3, log=True),
                            'final_estimator__max_depth': trial.suggest_int('final_max_depth', 3, 8),
                            'lgbm__n_estimators': trial.suggest_int('lgb_n_estimators', 100, 600),
                            'xgb__n_estimators': trial.suggest_int('xgb_n_estimators', 100, 600)
                        }
                        stacking_model.set_params(**params)
                        score = cross_val_score(
                            stacking_model, X_train, y_train, cv=cv_strategy, scoring='neg_mean_squared_error', n_jobs=-1
                        ).mean()
                        return score

                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=optimization_trials, n_jobs=1)
                    best_params = study.best_params
                    # 将 Optuna 参数名称映射回 set_params 所需格式
                    param_mapping = {
                        'final_n_estimators': 'final_estimator__n_estimators',
                        'final_learning_rate': 'final_estimator__learning_rate',
                        'final_max_depth': 'final_estimator__max_depth',
                        'lgb_n_estimators': 'lgbm__n_estimators',
                        'xgb_n_estimators': 'xgb__n_estimators'
                    }
                    translated_params = {param_mapping[k]: v for k, v in best_params.items()}
                    stacking_model.set_params(**translated_params)
                    cv_score = study.best_value
                    # 需要最终再 fit 一次
                    stacking_model.fit(X_train, y_train)
                except Exception as e_opt:
                    logger.warning(f"Optuna 优化失败，使用默认参数: {e_opt}")
                    stacking_model.fit(X_train, y_train)
            elif use_grid_search:
                from sklearn.model_selection import GridSearchCV
                param_grid = {
                    'final_estimator__n_estimators': [100, 300, 500],
                    'final_estimator__learning_rate': [0.03, 0.05, 0.1],
                    'final_estimator__max_depth': [3, 4, 6]
                }
                grid = GridSearchCV(
                    estimator=stacking_model,
                    param_grid=param_grid,
                    cv=cv_strategy,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                grid.fit(X_train, y_train)
                stacking_model = grid.best_estimator_
                best_params = grid.best_params_
                cv_score = grid.best_score_
            else:
                stacking_model.fit(X_train, y_train)

            # 评估
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            # 确保模型已拟合（部分情况下 GridSearchCV / Optuna 可能未在完整训练集上 refit）
            try:
                _ = stacking_model.predict(X_train[:5])
            except Exception:
                stacking_model.fit(X_train, y_train)

            # 生成预测值
            y_train_pred = stacking_model.predict(X_train)
            y_pred = stacking_model.predict(X_test)

            # 计算 IC / R2 / MSE 等指标
            train_ic = information_coefficient(y_train, y_train_pred)
            test_ic = information_coefficient(y_test, y_pred)
            metrics = {
                'train_ic': float(train_ic),
                'test_ic': float(test_ic),
                'train_mse': float(mean_squared_error(y_train, y_train_pred)),
                'test_mse': float(mean_squared_error(y_test, y_pred)),
                'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'train_mae': float(mean_absolute_error(y_train, y_train_pred)),
                'test_mae': float(mean_absolute_error(y_test, y_pred)),
                'train_r2': float(r2_score(y_train, y_train_pred)),
                'test_r2': float(r2_score(y_test, y_pred))
            }
            # 追加新评估指标
            try:
                from src.ml.evaluation.metrics import rank_ic, top_decile_spread, hit_rate
                metrics.update({
                    'rank_ic': float(rank_ic(y_test, y_pred)),
                    'top_decile_spread': float(top_decile_spread(y_test, y_pred)),
                    'hit_rate': float(hit_rate(y_test, y_pred))
                })
            except Exception as e_metric:
                logger.warning(f"计算新增指标失败: {e_metric}")

            logger.info(
                f"stacking 回归器 训练集 IC={metrics['train_ic']:.4f}, R²={metrics['train_r2']:.4f}, MSE={metrics['train_mse']:.6f}; "
                f"测试集 IC={metrics['test_ic']:.4f}, R²={metrics['test_r2']:.4f}, MSE={metrics['test_mse']:.6f}")

            # ================ 保存交叉验证各折指标 ================
            try:
                from sklearn.model_selection import cross_validate
                from src.ml.evaluation.metrics_io import save_cv_metrics
                cv = get_cv(n_splits=5, embargo=60)
                scoring = {
                    'ic': ic_scorer,
                    'r2': 'r2',
                    'mse': 'neg_mean_squared_error',
                    'mae': 'neg_mean_absolute_error',
                    'sharpe': sharpe_scorer,
                    'ir': ir_scorer
                }
                cv_results = cross_validate(stacking_model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
                metrics_dict = {
                    'ic': cv_results['test_ic'],
                    'r2': cv_results['test_r2'],
                    'mse': -cv_results['test_mse'],
                    'mae': -cv_results['test_mae'],
                    'sharpe': cv_results['test_sharpe'],
                    'ir': cv_results['test_ir']
                }
                metrics_dir = 'metrics'
                os.makedirs(metrics_dir, exist_ok=True)
                csv_path = os.path.join(metrics_dir, "stacking_regression_cv_scores.csv")
                save_cv_metrics(metrics_dict, csv_path)
                logger.info(f"交叉验证指标已保存到: {csv_path}")
            except Exception as err:
                logger.warning(f"保存交叉验证指标失败: {err}")

            result = {
                'model': stacking_model,
                'model_type': 'stacking',
                'best_params': best_params,
                'cv_score': cv_score,
                'metrics': metrics,
                'feature_names': X.columns.tolist(),
                'predictions': {
                    'X_train': X_train,
                    'y_train': y_train,
                    'y_train_pred': y_train_pred,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_test_pred': y_pred
                }
            }
            return result
        except Exception as e:
            logger.error(f"训练 stacking 回归模型失败: {e}")
            return None

    def _train_classification_models(self, X: pd.DataFrame, y: pd.Series, 
                                   use_grid_search: bool = 1,
                                   use_optuna: bool = False, optimization_trials: int = 50,
                                   cls_models: List[str] = None) -> Dict[str, Any]:
        """训练分类模型（多线程版本）"""
        # 初始化增强版训练器（用于训练单个分类模型）
        trainer = EnhancedMLTrainer(model_dir="models")
        results = {}
        
        # 分类模型类型（可外部指定）
        classification_models = cls_models if cls_models is not None else ['xgboost', 'logistic']  # 默认使用两种模型
        
        logger.info(f"开始并行训练 {len(classification_models)} 个分类模型...")
        
        # 使用多线程并行训练
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交所有训练任务
            future_to_model = {
                executor.submit(self._train_single_classification_model, 
                               trainer, X, y, model_type, use_grid_search, use_optuna, optimization_trials): model_type
                for model_type in classification_models
            }
            
            # 收集结果
            for future in as_completed(future_to_model):
                model_type = future_to_model[future]
                try:
                    model_type, result = future.result()
                    if result is not None:
                        results[model_type] = result
                        logger.info(f"{model_type} 分类模型训练完成")
                except Exception as e:
                    logger.error(f"{model_type} 分类模型训练异常: {e}")
        
        return results
    
    def _train_single_regression_model(self, X, y, model_type, use_grid_search, use_optuna, optimization_trials):
        """训练单个回归模型（用于多线程）"""
        try:
            logger.info(f"开始训练 {model_type} 回归模型...")
            
            if model_type == 'stacking':
                # Stacking 回归模型
                result = self._train_stacking_regression_model(
                    X, y,
                    use_grid_search=use_grid_search,
                    use_optuna=use_optuna,
                    optimization_trials=optimization_trials
                )
            elif model_type in ['ridge', 'lasso', 'elasticnet']:
                # 线性模型
                result = self._train_linear_model(X, y, model_type, use_grid_search)
            else:
                # 树模型
                result = self._train_tree_regression_model(X, y, model_type, use_grid_search, use_optuna, optimization_trials)
            
            if result is None:
                logger.error(f"{model_type} 回归模型训练返回 None，跳过保存与指标记录")
                return model_type, None

            # 保存模型
            feature_names = list(X.columns) if hasattr(X, 'columns') else None
            save_obj = {
                'model': result['model'],
                'feature_names': feature_names,
                'metadata': {
                    'task': 'regression',
                    'model_type': model_type,
                    'feature_config': getattr(self.feature_generator, 'feature_config', None)
                }
            }
            model_name = f"{model_type}_regression.pkl"
            with open(os.path.join('models', model_name), 'wb') as f:
                pickle.dump(save_obj, f)
            logger.info(f"{model_type} 回归模型已保存到: models/{model_name}")
            
            # 记录性能指标
            metrics = result['metrics']
            logger.info(f"{model_type} 性能: IC={metrics.get('test_ic', 0):.4f}, R²={metrics.get('test_r2', 0):.4f}, MSE={metrics.get('test_mse', 0):.6f}")
            
            return model_type, result
            
        except Exception as e:
            logger.error(f"训练 {model_type} 回归模型失败: {e}")
            return model_type, None

    def _train_regression_models(self, X: pd.DataFrame, y: pd.Series, 
                               use_grid_search: bool = 1,
                               use_optuna: bool = False, optimization_trials: int = 50,
                               reg_models: List[str] = None) -> Dict[str, Any]:
        """训练回归模型（多线程版本）"""
        results = {}
        
        # 回归模型类型（可外部指定）
        regression_models = reg_models if reg_models is not None else ['xgboost', 'lightgbm', 'catboost', 'lasso', 'stacking']  # 默认模型 + stacking
        
        logger.info(f"开始并行训练 {len(regression_models)} 个回归模型...")
        
        # 使用多线程并行训练
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交所有训练任务
            future_to_model = {
                executor.submit(self._train_single_regression_model, 
                               X, y, model_type, use_grid_search, use_optuna, optimization_trials): model_type
                for model_type in regression_models
            }
            
            # 收集结果
            for future in as_completed(future_to_model):
                model_type = future_to_model[future]
                try:
                    model_type, result = future.result()
                    if result is not None:
                        results[model_type] = result
                        logger.info(f"{model_type} 回归模型训练完成")
                except Exception as e:
                    logger.error(f"{model_type} 回归模型训练异常: {e}")
        
        return results
    
    def _train_linear_model(self, X: pd.DataFrame, y: pd.Series, model_type: str, 
                          use_grid_search: bool = 1) -> Dict[str, Any]:
        """训练线性回归模型"""
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        logger.info(f"训练{model_type}回归模型: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
        
        # 根据模型类型选择回归器
        if model_type == 'ridge':
            regressor = Ridge(random_state=42)
            param_grid = {'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}
        elif model_type == 'lasso':
            regressor = Lasso(random_state=42, max_iter=10000)
            param_grid = {'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
        elif model_type == 'elasticnet':
            regressor = ElasticNet(random_state=42, max_iter=10000)
            param_grid = {
                'regressor__alpha': [0.001, 0.01, 0.1, 1.0],
                'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        else:
            raise ValueError(f"不支持的线性模型类型: {model_type}")
        
        # 线性模型不支持Optuna优化，仅使用网格搜索
        # 移除树模型的Optuna优化逻辑，这些逻辑应该在 _train_tree_regression_model 中处理

        # 创建管线
        from sklearn.impute import SimpleImputer
        from src.ml.features.preprocessing import Winsorizer, CrossSectionZScore
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('winsor', Winsorizer()),
            ('zscore', CrossSectionZScore()),
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])
        
        if use_grid_search:
            # 网格搜索优化
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=get_cv(n_splits=5, embargo=60), scoring=ic_scorer, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logger.info(f"网格搜索完成: 最佳参数 {best_params}, 最佳得分 {-best_score:.4f}")
            
            # 使用最佳参数重新训练
            pipeline.set_params(**best_params)
        
        pipeline.fit(X_train, y_train)
        
        # 预测
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # 计算评估指标
        train_ic = information_coefficient(y_train, y_train_pred)
        test_ic = information_coefficient(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # 新增金融评估指标 (RankIC, Top Decile Spread, Hit Rate)
        try:
            from src.ml.evaluation.metrics import rank_ic, top_decile_spread, hit_rate
            train_rank_ic = rank_ic(y_train, y_train_pred)
            test_rank_ic = rank_ic(y_test, y_test_pred)
            train_top_decile_spread = top_decile_spread(y_train, y_train_pred)
            test_top_decile_spread = top_decile_spread(y_test, y_test_pred)
            train_hit_rate = hit_rate(y_train, y_train_pred)
            test_hit_rate = hit_rate(y_test, y_test_pred)
        except Exception as err:
            logger.warning(f"计算新增金融指标失败: {err}")
            import numpy as np
            train_rank_ic = test_rank_ic = np.nan
            train_top_decile_spread = test_top_decile_spread = np.nan
            train_hit_rate = test_hit_rate = np.nan
        
        # 获取特征重要性（系数绝对值）
        feature_importance = {}
        if hasattr(pipeline.named_steps['regressor'], 'coef_'):
            coef = pipeline.named_steps['regressor'].coef_
            if len(coef.shape) > 1:
                coef = coef[0]  # 处理多输出情况
            
            feature_names = X.columns.tolist()
            for i, importance in enumerate(np.abs(coef)):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(importance)
        
        # 构建结果
        result = {
            'model': pipeline,
            'model_type': model_type,
            'best_params': best_params if use_grid_search else {},
            'cv_score': best_score if use_grid_search else None,
            # 记录回归器所有参数
            'all_params': pipeline.named_steps['regressor'].get_params(),
             'feature_names': X.columns.tolist(),
            'feature_importance': feature_importance,
            'metrics': {
                'train_ic': train_ic,
                'test_ic': test_ic,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rank_ic': train_rank_ic,
                'test_rank_ic': test_rank_ic,
                'train_top_decile_spread': train_top_decile_spread,
                'test_top_decile_spread': test_top_decile_spread,
                'train_hit_rate': train_hit_rate,
                'test_hit_rate': test_hit_rate
            },
            'predictions': {
                'X_test': X_test,
                'y_test': y_test,
                'y_test_pred': y_test_pred
            }
        }
        
        logger.info(f"{model_type}回归模型训练完成: 训练集IC={train_ic:.4f}, 测试集IC={test_ic:.4f}, 训练集R²={train_r2:.4f}, 测试集R²={test_r2:.4f}")
        logger.info(f"{model_type} 回归模型全部参数: {result['all_params']}")
        logger.info(f"训练集MSE={train_mse:.6f}, 测试集MSE={test_mse:.6f}")
        logger.info(f"{model_type} 回归模型全部参数: {result['all_params']}")

        # ================= 保存交叉验证各折指标 =================
        try:
            from sklearn.model_selection import cross_validate
            from src.ml.evaluation.metrics_io import save_cv_metrics
            cv = get_cv(n_splits=5, embargo=60)
            scoring = {
                'ic': ic_scorer,
                'r2': 'r2',
                'mse': 'neg_mean_squared_error',
                'mae': 'neg_mean_absolute_error',
                'sharpe': sharpe_scorer,
                'ir': ir_scorer
            }
            cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
            metrics_dict = {
                'ic': cv_results['test_ic'],
                'r2': cv_results['test_r2'],
                'mse': -cv_results['test_mse'],  # 取正值
                'mae': -cv_results['test_mae'],
                'sharpe': cv_results['test_sharpe'],
                'ir': cv_results['test_ir']
            }
            metrics_dir = 'metrics'
            os.makedirs(metrics_dir, exist_ok=True)
            csv_path = os.path.join(metrics_dir, f"{model_type}_regression_cv_scores.csv")
            save_cv_metrics(metrics_dict, csv_path)
            logger.info(f"交叉验证指标已保存到: {csv_path}")
        except Exception as err:
            logger.warning(f"保存交叉验证指标失败: {err}")
        
        return result
    
    def _train_tree_regression_model(self, X: pd.DataFrame, y: pd.Series, model_type: str,
                                   use_grid_search: bool = 1, use_optuna: bool = False, optimization_trials: int = 50) -> Dict[str, Any]:
        """训练树回归模型"""
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        logger.info(f"训练{model_type}回归模型: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
        
        # 根据模型类型选择回归器
        best_params = {}
        best_score = None
        
        if model_type == 'lightgbm':
            try:
                from lightgbm import LGBMRegressor
                # 设置 verbosity=-1 以关闭 LightGBM 冗余警告（如 "No further splits with positive gain"）
                regressor = LGBMRegressor(objective='huber', random_state=42, n_jobs=-1, verbosity=-1)
                param_grid = {
                    'regressor__n_estimators': [100, 300],
                    'regressor__learning_rate': [0.03, 0.1],
                    'regressor__max_depth': [-1, 6, 10],
                    'regressor__subsample': [0.8, 1.0],
                    'regressor__colsample_bytree': [0.8, 1.0],
                    # 始终保持静默
                    'regressor__verbosity': [-1]
                }
            except ImportError:
                logger.error("LightGBM未安装，跳过LightGBM训练")
                return self._empty_regression_result(model_type, X)
        elif model_type == 'catboost':
            try:
                from catboost import CatBoostRegressor
                regressor = CatBoostRegressor(
                    loss_function='Huber:delta=1',
                    depth=6,
                    verbose=False,
                    random_state=42
                )
                param_grid = {
                    'regressor__iterations': [300, 600],
                    'regressor__learning_rate': [0.03, 0.1],
                    'regressor__depth': [4, 6, 8]
                }
            except ImportError:
                logger.error("CatBoost未安装，跳过CatBoost训练")
                return self._empty_regression_result(model_type, X)
        elif model_type == 'randomforest':
            regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                'regressor__n_estimators': [50, 100],
                'regressor__max_depth': [10, 20, None],
                'regressor__min_samples_split': [2, 5],
                'regressor__min_samples_leaf': [1, 2]
            }
        elif model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
                regressor = XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
                param_grid = {
                    'regressor__n_estimators': [50, 100],
                    'regressor__max_depth': [3, 6, 9],
                    'regressor__learning_rate': [0.01, 0.1],
                    'regressor__subsample': [0.8, 1.0],
                    'regressor__colsample_bytree': [0.8, 1.0]
                }
                # 若使用 Optuna，则降级为网格搜索，避免因未正确实现 objective 导致异常

                # 支持 Optuna 搜索（在下方统一实现）
                
            except ImportError:
                logger.error("XGBoost未安装，跳过XGBoost训练")
                return {
                    'model': None,
                    'model_type': model_type,
                    'best_params': {},
                    'cv_score': None,
                    'feature_names': X.columns.tolist() if hasattr(X, 'columns') else [],
                    'feature_importance': {},
                    'metrics': {
                        'train_mse': 0,
                        'test_mse': 0,
                        'train_mae': 0,
                        'test_mae': 0,
                        'train_r2': 0,
                        'test_r2': 0
                    }
                }
        else:
            raise ValueError(f"不支持的树回归模型类型: {model_type}")
        
        # 创建管线
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])
        
        # ----------------------- 超参数优化 -----------------------
        # 优先使用 Optuna，其次才使用 GridSearchCV。
        if use_optuna:
            try:
                import optuna
                # 已在模块顶部统一导入 TimeSeriesSplit 与 cross_val_score

                def objective(trial):
                    search_params = {}
                    if model_type == 'lightgbm':
                        # 扩展 LightGBM 搜索空间，使其与 XGBoost 优化思路保持一致
                        search_params = {
                            'regressor__n_estimators': trial.suggest_int('regressor__n_estimators', 200, 3000),
                            'regressor__learning_rate': trial.suggest_float('regressor__learning_rate', 0.005, 0.2, log=True),
                            # -1 表示不限制深度，LightGBM 会根据 num_leaves 推断；当为 -1 时，max_depth 不生效
                            'regressor__max_depth': trial.suggest_int('regressor__max_depth', -1, 12),
                            # 与 subsample/colsample_bytree 类似的参数
                            'regressor__subsample': trial.suggest_float('regressor__subsample', 0.5, 1.0),
                            'regressor__colsample_bytree': trial.suggest_float('regressor__colsample_bytree', 0.5, 1.0),
                            # LightGBM 额外的重要正则化与结构参数
                            'regressor__num_leaves': trial.suggest_int('regressor__num_leaves', 31, 256),
                            'regressor__min_child_samples': trial.suggest_int('regressor__min_child_samples', 5, 50),
                            'regressor__reg_alpha': trial.suggest_float('regressor__reg_alpha', 0.0, 10.0),
                            'regressor__reg_lambda': trial.suggest_float('regressor__reg_lambda', 0.0, 10.0),
                            'regressor__max_bin': trial.suggest_int('regressor__max_bin', 200, 500),
                            'regressor__min_split_gain': trial.suggest_float('regressor__min_split_gain', 0.0, 1.0),
                            'regressor__bagging_freq': trial.suggest_int('regressor__bagging_freq', 0, 10)
                        }
                    elif model_type == 'catboost':
                        search_params = {
                            'regressor__iterations': trial.suggest_int('regressor__iterations', 200, 800),
                            'regressor__learning_rate': trial.suggest_float('regressor__learning_rate', 0.01, 0.3, log=True),
                            'regressor__depth': trial.suggest_int('regressor__depth', 4, 10)
                        }
                    elif model_type == 'randomforest':
                        search_params = {
                            'regressor__n_estimators': trial.suggest_int('regressor__n_estimators', 50, 300),
                            'regressor__max_depth': trial.suggest_int('regressor__max_depth', 5, 30),
                            'regressor__min_samples_split': trial.suggest_int('regressor__min_samples_split', 2, 10),
                            'regressor__min_samples_leaf': trial.suggest_int('regressor__min_samples_leaf', 1, 5)
                        }
                    elif model_type == 'xgboost':
                        # 扩展搜索空间，与分类侧保持一致
                        search_params = {
                            'regressor__n_estimators': trial.suggest_int('regressor__n_estimators', 200, 2000),
                            'regressor__max_depth': trial.suggest_int('regressor__max_depth', 3, 12),
                            'regressor__learning_rate': trial.suggest_float('regressor__learning_rate', 0.01, 0.2, log=True),
                            'regressor__subsample': trial.suggest_float('regressor__subsample', 0.5, 1.0),
                            'regressor__colsample_bytree': trial.suggest_float('regressor__colsample_bytree', 0.5, 1.0),
                            'regressor__min_child_weight': trial.suggest_int('regressor__min_child_weight', 1, 10),
                            'regressor__gamma': trial.suggest_float('regressor__gamma', 0.0, 5.0),
                            'regressor__max_delta_step': trial.suggest_int('regressor__max_delta_step', 0, 10),
                            'regressor__grow_policy': trial.suggest_categorical('regressor__grow_policy', ['depthwise', 'lossguide']),
                            'regressor__reg_lambda': trial.suggest_float('regressor__reg_lambda', 0.0, 10.0),
                            'regressor__reg_alpha': trial.suggest_float('regressor__reg_alpha', 0.0, 10.0)
                        }
                    # 设置参数并计算交叉验证分数（负MSE，Optuna目标为最小化MSE）
                    pipeline.set_params(**search_params)
                    tscv = get_cv(n_splits=5, embargo=60)
                    score = cross_val_score(
                        pipeline, X_train, y_train,
                        cv=tscv, scoring=ic_scorer, n_jobs=-1
                    ).mean()
                    return score  # 最大化 IC

                study = optuna.create_study(direction='maximize')

                # ---------------- Optuna 早停：连续 early_stop_rounds 次无提升停止 ----------------
                early_stop_rounds = 10
                def early_stopping_callback(study, trial):
                    best_value = study.best_value
                    if 'no_improve_count' not in study.user_attrs:
                        study.set_user_attr('no_improve_count', 0)
                        study.set_user_attr('best_value_snap', best_value)
                        return
                    tol = 5e-4  # 容忍度
                    improved = (best_value - study.user_attrs['best_value_snap']) > tol
                    if improved:
                        study.set_user_attr('no_improve_count', 0)
                        study.set_user_attr('best_value_snap', best_value)
                    else:
                        cnt = study.user_attrs['no_improve_count'] + 1
                        study.set_user_attr('no_improve_count', cnt)
                        if cnt >= early_stop_rounds:
                            logger.info(f"Optuna: 连续 {early_stop_rounds} 次未提升, 提前停止搜索")
                            study.stop()

                study.optimize(objective, n_trials=optimization_trials, show_progress_bar=False,
                                callbacks=[early_stopping_callback])

                best_params = study.best_params
                best_score = study.best_value
                logger.info(f"Optuna优化完成: 最佳参数 {best_params}, 最佳CV MSE {best_score:.6f}")

                # 使用最佳参数重新训练
                pipeline.set_params(**best_params)
            except ImportError:
                logger.error("Optuna未安装，自动回退到网格搜索")
                use_optuna = False  # 回退

        if use_grid_search and not use_optuna:
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=get_cv(n_splits=5), scoring=ic_scorer, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_
            logger.info(f"网格搜索完成: 最佳参数 {best_params}, 最佳CV MSE {best_score:.6f}")
            # 使用最佳参数重新训练
            pipeline.set_params(**best_params)

        # ----------------------- 最终模型训练 -----------------------
        fit_params = {}
        if model_type in ['xgboost', 'lightgbm']:
            # 为基于树的Boosting模型(XGBoost/LightGBM)启用早停，监控验证集；
            eval_set = [
                (X_train.values if hasattr(X_train, 'values') else X_train,
                 y_train.values if hasattr(y_train, 'values') else y_train),
                (X_test.values if hasattr(X_test, 'values') else X_test,
                 y_test.values if hasattr(y_test, 'values') else y_test)
            ]
            # XGBoost 与 LightGBM 参数命名保持一致使用 regressor__early_stopping_rounds
            pipeline.set_params(regressor__early_stopping_rounds=50)
            # LightGBM 某些版本不支持 verbose 关键字，区分处理
            if model_type == 'xgboost':
                fit_params = {
                    'regressor__eval_set': eval_set,
                    'regressor__verbose': False
                }
            else:  # lightgbm
                fit_params = {
                    'regressor__eval_set': eval_set
                }
        pipeline.fit(X_train, y_train, **fit_params)

        # 预测
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # 计算评估指标
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # 新增金融评估指标 (RankIC, Top Decile Spread, Hit Rate)
        try:
            from src.ml.evaluation.metrics import rank_ic, top_decile_spread, hit_rate
            train_rank_ic = rank_ic(y_train, y_train_pred)
            test_rank_ic = rank_ic(y_test, y_test_pred)
            train_top_decile_spread = top_decile_spread(y_train, y_train_pred)
            test_top_decile_spread = top_decile_spread(y_test, y_test_pred)
            train_hit_rate = hit_rate(y_train, y_train_pred)
            test_hit_rate = hit_rate(y_test, y_test_pred)
        except Exception as err:
            logger.warning(f"计算新增金融指标失败: {err}")
            import numpy as np
            train_rank_ic = test_rank_ic = np.nan
            train_top_decile_spread = test_top_decile_spread = np.nan
            train_hit_rate = test_hit_rate = np.nan

        # 特征重要性（如果模型支持）
        feature_importance = {}
        booster = pipeline.named_steps['regressor']
        if hasattr(booster, 'feature_importances_'):
            importances = booster.feature_importances_
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'f{i}' for i in range(len(importances))]
            for i, imp in enumerate(importances):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(imp)

        # 额外指标：针对 XGBoost 记录 best_iteration / val_logloss
        extra_metrics = {}
        if model_type == 'xgboost':
            if hasattr(booster, 'best_iteration'):
                extra_metrics['best_iteration'] = int(booster.best_iteration)
            evals = getattr(booster, 'evals_result_', None)
            if evals and isinstance(evals, dict) and 'validation_0' in evals:
                rmse_list = evals['validation_0'].get('rmse', []) or evals['validation_0'].get('mae', [])
                if rmse_list:
                    extra_metrics['val_rmse'] = float(min(rmse_list))

        # 构建结果
        result = {
            'model': pipeline,
            'model_type': model_type,
            'best_params': best_params,
            'cv_score': best_score,
            # 记录回归器所有参数
            'all_params': pipeline.named_steps['regressor'].get_params(),
            'feature_names': X.columns.tolist() if hasattr(X, 'columns') else [],
            'feature_importance': feature_importance,
            'metrics': {
                'train_ic': information_coefficient(y_train, y_train_pred),
                'test_ic': information_coefficient(y_test, y_test_pred),
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rank_ic': train_rank_ic,
                'test_rank_ic': test_rank_ic,
                'train_top_decile_spread': train_top_decile_spread,
                'test_top_decile_spread': test_top_decile_spread,
                'train_hit_rate': train_hit_rate,
                'test_hit_rate': test_hit_rate
            },
            'predictions': {
                'X_test': X_test,
                'y_test': y_test,
                'y_test_pred': y_test_pred
            }
        }
        # 合并额外指标
        result.update(extra_metrics)

        logger.info(f"{model_type}回归模型训练完成: 训练集IC={information_coefficient(y_train, y_train_pred):.4f}, 测试集IC={information_coefficient(y_test, y_test_pred):.4f}, 训练集R²={train_r2:.4f}, 测试集R²={test_r2:.4f}")
        logger.info(f"训练集MSE={train_mse:.6f}, 测试集MSE={test_mse:.6f}")
        logger.info(f"{model_type} 回归模型全部参数: {result['all_params']}")

        # ================ 保存交叉验证各折指标 ================
        try:
            from sklearn.model_selection import cross_validate
            from src.ml.evaluation.metrics_io import save_cv_metrics
            cv = get_cv(n_splits=5, embargo=60)
            scoring = {
                'ic': ic_scorer,
                'r2': 'r2',
                'mse': 'neg_mean_squared_error',
                'mae': 'neg_mean_absolute_error',
                'sharpe': sharpe_scorer,
                'ir': ir_scorer
            }
            cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
            metrics_dict = {
                'ic': cv_results['test_ic'],
                'r2': cv_results['test_r2'],
                'mse': -cv_results['test_mse'],
                'mae': -cv_results['test_mae'],
                'sharpe': cv_results['test_sharpe'],
                'ir': cv_results['test_ir']
            }
            metrics_dir = 'metrics'
            os.makedirs(metrics_dir, exist_ok=True)
            csv_path = os.path.join(metrics_dir, f"{model_type}_regression_cv_scores.csv")
            save_cv_metrics(metrics_dict, csv_path)
            logger.info(f"交叉验证指标已保存到: {csv_path}")
        except Exception as err:
            logger.warning(f"保存交叉验证指标失败: {err}")

        return result

    def nested_cv(self, X: pd.DataFrame, y: pd.Series, mode: str = 'regression',
                 reg_models: List[str] = None, outer_splits: int = 5, inner_splits: int = 3,
                 use_optuna: bool = False, optimization_trials: int = 50) -> Dict[str, Any]:
        """基于 PurgedKFoldWithEmbargo 的 Nested Cross-Validation

        目前仅针对回归任务实现，返回每个模型在外层测试集上的 IC 指标列表与均值/标准差。
        Args:
            X: 特征 DataFrame
            y: 标签 Series
            mode: 当前仅允许 'regression'
            reg_models: 指定需要评估的回归模型列表
            outer_splits: 外层折数 (默认5)
            inner_splits: 内层折数 (当前参数预留，内部调用 _train_single_regression_model 时通过 Optuna/网格搜索实现)
            use_optuna: 是否在内层使用 Optuna
            optimization_trials: Optuna trial 数
        Returns:
            summary: {model_type: {outer_fold_scores: List[float], mean_ic: float, std_ic: float}}
        """
        if mode != 'regression':
            raise NotImplementedError("nested_cv 目前仅支持回归任务")

        if reg_models is None:
            reg_models = ['xgboost', 'lightgbm', 'catboost', 'lasso']

        # 初始化结果容器
        outer_results: Dict[str, List[float]] = {m: [] for m in reg_models}
        outer_cv = get_cv(n_splits=outer_splits)

        logger.info(f"[NestedCV] 开始 Nested CV: outer_splits={outer_splits}, inner_splits={inner_splits}, models={reg_models}")
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
            logger.info(f"[NestedCV] Outer Fold {fold_idx + 1}/{outer_splits}: 训练 {len(train_idx)} 样本, 测试 {len(test_idx)} 样本")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for model_type in reg_models:
                try:
                    # 内层搜索由 _train_single_regression_model 内部完成
                    model_type_key, result = self._train_single_regression_model(
                        X_train, y_train,
                        model_type=model_type,
                        use_grid_search=0 if use_optuna else 1,
                        use_optuna=use_optuna,
                        optimization_trials=optimization_trials
                    )
                    if result is None:
                        logger.warning(f"[NestedCV] {model_type} 在 Outer Fold {fold_idx + 1} 训练失败，跳过")
                        continue
                    trained_model = result['model']
                    y_pred = trained_model.predict(X_test)
                    ic_val = information_coefficient(y_test, y_pred)
                    outer_results[model_type].append(ic_val)
                    logger.info(f"[NestedCV] Outer {fold_idx + 1} - {model_type}: IC={ic_val:.4f}")
                except Exception as err:
                    logger.error(f"[NestedCV] {model_type} 在 Outer Fold {fold_idx + 1} 异常: {err}")

        # 汇总
        summary: Dict[str, Any] = {}
        for model_type, scores in outer_results.items():
            if not scores:
                continue
            summary[model_type] = {
                'outer_fold_scores': scores,
                'mean_ic': float(np.mean(scores)),
                'std_ic': float(np.std(scores))
            }
        logger.info(f"[NestedCV] 完成 Nested CV 评估: {summary}")
        return summary

    def _empty_regression_result(self, model_type: str, X: pd.DataFrame):
        """当缺少依赖时返回占位结果"""
        return {
            'model': None,
            'model_type': model_type,
            'best_params': {},
            'cv_score': None,
            'feature_names': X.columns.tolist() if hasattr(X, 'columns') else [],
            'feature_importance': {},
            'metrics': {
                'train_mse': 0,
                'test_mse': 0,
                'train_mae': 0,
                'test_mae': 0,
                'train_r2': 0,
                'test_r2': 0
            }
        }

    def evaluate_models(self, results: Dict[str, Any]):
        """评估模型性能"""
        logger.info("\n=== 模型评估结果 ===")
        
        # 分类模型评估
        cls_models = [k for k in results.keys() if k in ['logistic', 'randomforest', 'xgboost']]
        if cls_models:
            logger.info("分类模型:")
            best_cls_model = None
            best_cls_score = -float('inf')
            
            for model_type in cls_models:
                result = results[model_type]
                if 'best_score' in result:
                    score = result['best_score']
                    logger.info(f"  {model_type:12s}: 最佳得分 = {score:.4f}")
                    if score > best_cls_score:
                        best_cls_score = score
                        best_cls_model = model_type
            
            if best_cls_model:
                logger.info(f"  最佳分类模型: {best_cls_model} (得分={best_cls_score:.4f})")
        
        # 回归模型评估
        reg_models = [k for k in results.keys() if k in ['ridge', 'lasso', 'elasticnet', 'randomforest', 'xgboost', 'lightgbm', 'catboost']]
        if reg_models:
            logger.info("回归模型:")
            best_reg_model = None
            best_reg_r2 = -float('inf')
            
            for model_type in reg_models:
                result = results[model_type]
                metrics = result.get('metrics', {})
                test_r2 = metrics.get('test_r2', 0)
                test_mse = metrics.get('test_mse', 0)
                
                logger.info(f"  {model_type:12s}: R²={test_r2:8.4f}, MSE={test_mse:10.6f}")
                
                if test_r2 > best_reg_r2:
                    best_reg_r2 = test_r2
                    best_reg_model = model_type
            
            if best_reg_model:
                logger.info(f"  最佳回归模型: {best_reg_model} (R²={best_reg_r2:.4f})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统一模型训练脚本（支持小规模快速验证）")
    parser.add_argument("--lookback", type=int, default=365, help="回溯天数，默认为365")
    parser.add_argument("--n_stocks", type=int, default=50, help="使用的股票数量，默认为50")
    parser.add_argument("--prediction_period", type=int, default=30, help="预测周期（天数），默认为30")
    parser.add_argument("--mode", choices=['classification', 'regression', 'both'], default='both', help="训练模式")
    parser.add_argument("--no_grid_search", action="store_true", help="禁用网格搜索以加快速度")
    parser.add_argument("--use_optuna", action="store_true", help="使用Optuna进行超参数优化（优先级高于网格搜索）")
    # Nested CV 参数
    parser.add_argument("--nested-cv", action="store_true", dest="nested_cv", help="启用 Nested Cross-Validation 评估")
    parser.add_argument("--outer-splits", type=int, default=5, help="Nested CV 外层折数 (默认5)")
    parser.add_argument("--inner-splits", type=int, default=3, help="Nested CV 内层折数 (默认3)")
    parser.add_argument("--optimization_trials", type=int, default=50, help="Optuna/Bayesian优化时的迭代次数 (默认50)")
    # 新增: 关闭特征选择开关
    parser.add_argument("--disable_feature_selection", action="store_true", help="关闭特征选择优化，加快训练速度")
    parser.add_argument("--fs-n-jobs", type=int, default=-1, help="特征选择时并行CPU核心数（-1 表示全部）")
    # 兼容性补充：快速测试及特征缓存控制
    parser.add_argument("--quick-test", action="store_true", help="启用快速验证模式：自动缩小数据规模、禁用耗时操作")
    parser.add_argument("--no-feature-cache", action="store_true", help="不使用特征选择缓存，始终重新计算特征选择（默认：False）")
    parser.add_argument("--refresh-feature-selection", action="store_true", help="强制刷新并覆盖已有的特征选择结果缓存")
    args = parser.parse_args()

    # -------------------- quick-test 参数覆写 --------------------
    if getattr(args, "quick_test", False):
        logger.info("[quick-test] 启用快速验证模式，自动缩小数据规模并关闭耗时功能 …")
        # 缩短回溯窗口与股票数量
        if hasattr(args, "lookback"):
            args.lookback = min(args.lookback, 90)
        if hasattr(args, "lookback_days"):
            args.lookback_days = min(args.lookback_days, 90)
        args.n_stocks = min(args.n_stocks, 200)
        # 禁用网格搜索与 Optuna
        if hasattr(args, "no_grid_search"):
            args.no_grid_search = True
        if hasattr(args, "use_grid_search"):
            args.use_grid_search = 0
        args.use_optuna = 0
        args.optimization_trials = 5
        # 禁用特征选择并限制并行
        args.disable_feature_selection = True
        args.fs_n_jobs = 1
        logger.info(
            f"[quick-test] 覆写后参数: lookback_days={getattr(args,'lookback_days',getattr(args,'lookback',None))}, "
            f"n_stocks={args.n_stocks}, grid_search=off, optuna=off, feature_selection=off, trials={args.optimization_trials}"
        )
    # -------------------------------------------------------------

    logger.info("=== 开始统一模型训练流程 ===")
    logger.info(f"参数: lookback={args.lookback}, n_stocks={args.n_stocks}, prediction_period={args.prediction_period}, mode={args.mode}, grid_search={'off' if args.no_grid_search else 'on'}")

    try:
        trainer = UnifiedModelTrainer(enable_feature_selection=(not args.disable_feature_selection),
                                      reuse_feature_selection=not args.no_feature_cache,
                                      refresh_feature_selection=args.refresh_feature_selection,
                                      feature_selection_n_jobs=args.fs_n_jobs)
        if args.disable_feature_selection:
            logger.info("已通过命令行开关关闭特征选择优化")
            
        # 准备训练数据（同时生成分类和回归标签）
        X, y = trainer.prepare_training_data(mode=args.mode, lookback_days=args.lookback, n_stocks=args.n_stocks, prediction_period=args.prediction_period)
        
        if X is None or not y:
            logger.error("数据准备失败")
            return
        
        logger.info(f"最终训练数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
        
        # 若启用 Nested CV，则先进行 Nested CV 评估
        if getattr(args, "nested_cv", False):
            logger.info("启用 Nested Cross-Validation 评估模式 …")
            nested_summary = trainer.nested_cv(
                X, y['reg'] if isinstance(y, dict) else y,
                mode='regression',
                outer_splits=args.outer_splits,
                inner_splits=args.inner_splits,
                use_optuna=args.use_optuna,
                optimization_trials=args.optimization_trials
            )
            logger.info(f"Nested CV 评估结果: {nested_summary}")
        # 训练所有模型
        results = trainer.train_models(
            X,
            y,
            mode=args.mode,
            use_grid_search=(0 if args.no_grid_search else 1) if not args.use_optuna else 0,
            use_optuna=1 if args.use_optuna else 0,
            optimization_trials=args.optimization_trials
        )
        
        # 评估模型
        trainer.evaluate_models(results)
        
        logger.info("统一模型训练完成!")
        
    except Exception as e:
        logger.error(f"训练流程失败: {e}", exc_info=1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="统一模型训练脚本，可通过命令行参数灵活指定分类/回归模型及训练相关配置")

    # 数据准备参数
    parser.add_argument('--lookback-days', type=int, default=365,
                        help='回溯天数（默认：365）')
    parser.add_argument('--n-stocks', type=int, default=1000,
                        help='参与训练的股票数量上限（默认：1000）')
    parser.add_argument('--prediction-period', type=int, default=30,
                        help='标签预测周期，单位：天（默认：30）')

    # 任务模式
    parser.add_argument('--mode', default='both', choices=['classification', 'regression', 'both'],
                        help="训练模式：classification / regression / both（默认：both）")

    # 模型列表
    parser.add_argument('--cls-models', type=str, default=None,
                        help='要训练的分类模型列表，逗号分隔。例如 "xgboost,logistic"；若为空使用默认')
    parser.add_argument('--reg-models', type=str, default=None,
                        help='要训练的回归模型列表，逗号分隔。例如 "lasso,ridge"；若为空使用默认')

    # 训练优化相关
    parser.add_argument('--nested-cv', action='store_true', dest='nested_cv', help='启用 Nested Cross-Validation 评估')
    parser.add_argument('--outer-splits', type=int, default=5, help='Nested CV 外层折数 (默认5)')
    parser.add_argument('--inner-splits', type=int, default=3, help='Nested CV 内层折数 (默认3)')
    # 训练优化相关
    parser.add_argument('--use-grid-search', type=int, default=1, choices=[0, 1],
                        help='是否启用网格搜索（0/1，默认：1）')
    # 支持两种写法 --use-optuna / --use_optuna，统一映射到 use_optuna
    parser.add_argument('--use-optuna', action='store_true', dest='use_optuna',
                        help='是否使用 Optuna 进行超参优化（默认：False）')
    parser.add_argument('--use_optuna', action='store_true', dest='use_optuna',
                        help='(兼容) 等同于 --use-optuna')
    parser.add_argument('--optimization-trials', type=int, default=50,
                        help='Optuna/Bayesian 优化的 trial 次数（默认：50）')

    # 特征工程
    parser.add_argument('--disable-feature-selection', action='store_true',
                        help='关闭特征选择优化（默认开启）'),
    parser.add_argument('--fs-n-jobs', type=int, default=-1,
                        help='特征选择时用于并行的CPU核心数（-1 表示全部）')
    # 快速测试模式
    parser.add_argument('--quick-test', action='store_true',
                        help='启用快速验证模式：自动缩小数据规模、禁用耗时操作')
    parser.add_argument('--no-feature-cache', action='store_true',
                         help='不使用特征选择缓存，始终重新计算特征选择（默认：False）')
    parser.add_argument('--refresh-feature-selection', action='store_true',
                        help='强制刷新并覆盖已有的特征选择结果缓存')
    args = parser.parse_args()

    # -------------------- quick-test 参数覆写 --------------------
    if getattr(args, 'quick_test', False):
        logger.info('[quick-test] 启用快速验证模式，自动缩小数据规模并关闭耗时功能 …')
        args.lookback_days = min(args.lookback_days, 380)
        args.n_stocks = min(args.n_stocks, 600)
        args.use_grid_search = 0
        args.n_estimators = 600
        args.use_optuna = 0
        args.optimization_trials = 5
        args.disable_feature_selection = True
        args.fs_n_jobs = 1
    # -------------------------------------------------------------

    # 解析模型列表
    cls_models = [m.strip() for m in args.cls_models.split(',')] if args.cls_models else None
    reg_models = [m.strip() for m in args.reg_models.split(',')] if args.reg_models else None

    # 初始化 Trainer
    trainer = UnifiedModelTrainer(enable_feature_selection=not args.disable_feature_selection,
                                  reuse_feature_selection=not args.no_feature_cache,
                                  refresh_feature_selection=args.refresh_feature_selection,
                                  feature_selection_n_jobs=args.fs_n_jobs)

    # 准备数据
    X, y = trainer.prepare_training_data(
        mode=args.mode,
        lookback_days=args.lookback_days,
        n_stocks=args.n_stocks,
        prediction_period=args.prediction_period
    )

    # 训练模型
    trainer.train_models(
        X, y,
        mode=args.mode,
        use_grid_search=args.use_grid_search,
        use_optuna=args.use_optuna,
        optimization_trials=args.optimization_trials,
        cls_models=cls_models,
        reg_models=reg_models
    )
