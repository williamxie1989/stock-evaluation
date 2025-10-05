#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择优化器
整合多种特征选择方法，筛选最具预测力的特征
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, 
    mutual_info_classif, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureSelectorOptimizer:
    """特征选择优化器 - 筛选最具预测力的特征"""
    
    def __init__(self, task_type: str = 'classification', target_n_features: int = 30, n_jobs: int = -1,
                 importance_model: str = 'rf'):
        """
        初始化特征选择优化器
        
        Args:
            task_type: 任务类型 ('classification' 或 'regression')
            target_n_features: 目标特征数量
            importance_model: 计算特征重要性的模型 (rf / xgb / lgb / auto)
        """
        self.task_type = task_type
        self.target_n_features = target_n_features
        self.selected_features_ = []
        self.feature_scores_ = {}
        self.feature_importance_ = {}
        self.cv_scores_ = {}
        # 并行度控制 (-1 使用所有核心)
        self.n_jobs = n_jobs
        self.importance_model = importance_model
        
    def optimize_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 method: str = 'ensemble', 
                                 cv_folds: int = 5,
                                 random_state: int = 42) -> Dict[str, Any]:
        """
        优化特征选择
        
        Args:
            X: 特征数据
            y: 目标变量
            method: 选择方法 ('ensemble', 'importance', 'stability', 'auto')
            cv_folds: 交叉验证折数
            random_state: 随机种子
            
        Returns:
            特征选择结果字典
        """
        logger.info(f"开始特征选择优化，方法: {method}, 目标特征数: {self.target_n_features}")
        
        # 数据预处理
        X_clean = self._preprocess_features(X)
        
        if method == 'ensemble':
            return self._ensemble_selection(X_clean, y, cv_folds, random_state)
        elif method == 'importance':
            return self._importance_based_selection(X_clean, y, cv_folds, random_state)
        elif method == 'stability':
            return self._stability_selection(X_clean, y, cv_folds, random_state)
        elif method == 'auto':
            return self._auto_selection(X_clean, y, cv_folds, random_state)
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """预处理特征数据"""
        # 确保所有列都是数值类型
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("没有数值类型的特征列")
        
        X = X[numeric_cols].copy()
        
        # 移除常数特征
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            logger.info(f"移除 {len(constant_features)} 个常数特征")
            X = X.drop(columns=constant_features)
        
        # 移除高度相关特征（相关系数 > 0.95）
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > 0.95)]
            if to_drop:
                logger.info(f"移除 {len(to_drop)} 个高度相关特征")
                X = X.drop(columns=to_drop)
        
        # 处理缺失值 - 使用数值列的中位数
        if len(X.columns) > 0:
            X = X.fillna(X.median())
        
        return X
    
    def _ensemble_selection(self, X: pd.DataFrame, y: pd.Series, 
                           cv_folds: int, random_state: int) -> Dict[str, Any]:
        """集成方法特征选择"""
        feature_scores = {}
        
        # 1. 单变量特征选择
        logger.info("执行单变量特征选择...")
        if self.task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k='all')
        else:
            selector = SelectKBest(score_func=f_regression, k='all')
        
        selector.fit(X, y)
        univariate_scores = dict(zip(X.columns, selector.scores_))
        
        # 标准化分数
        max_score = max(univariate_scores.values()) if univariate_scores else 1
        for feature, score in univariate_scores.items():
            feature_scores[feature] = feature_scores.get(feature, 0) + (score / max_score)
        
        # 2. 互信息特征选择
        logger.info("执行互信息特征选择...")
        if self.task_type == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=random_state)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=random_state)
        
        mi_dict = dict(zip(X.columns, mi_scores))
        max_mi = max(mi_dict.values()) if mi_dict else 1
        for feature, score in mi_dict.items():
            feature_scores[feature] = feature_scores.get(feature, 0) + (score / max_mi)
        
        # 3. 基于模型的特征重要性
        logger.info("执行基于模型的特征重要性分析...")
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=self.n_jobs)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=self.n_jobs)
        
        model.fit(X, y)
        importances = model.feature_importances_
        importance_dict = dict(zip(X.columns, importances))
        max_importance = max(importance_dict.values()) if importance_dict else 1
        for feature, importance in importance_dict.items():
            feature_scores[feature] = feature_scores.get(feature, 0) + (importance / max_importance)
        
        # 4. 递归特征消除
        logger.info("执行递归特征消除...")
        if self.task_type == 'classification':
            estimator = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            estimator = Ridge(random_state=random_state)
        
        rfe = RFE(estimator, n_features_to_select=self.target_n_features)
        rfe.fit(X, y)
        
        for i, feature in enumerate(X.columns):
            if rfe.support_[i]:
                # 排名越小越好，所以用倒数
                rfe_score = 1.0 / rfe.ranking_[i]
                feature_scores[feature] = feature_scores.get(feature, 0) + rfe_score
        
        # 综合评分并选择特征
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:self.target_n_features]]
        
        # 交叉验证验证
        cv_score = self._validate_features(X[selected_features], y, cv_folds, random_state)
        
        self.selected_features_ = selected_features
        self.feature_scores_ = dict(sorted_features)
        self.feature_importance_ = importance_dict
        self.cv_scores_ = {'ensemble': cv_score}
        
        return {
            'selected_features': selected_features,
            'feature_scores': dict(sorted_features),
            'cv_score': cv_score,
            'method': 'ensemble',
            'n_original_features': X.shape[1],
            'n_selected_features': len(selected_features)
        }
    
    def _importance_based_selection(self, X: pd.DataFrame, y: pd.Series,
                                    cv_folds: int, random_state: int) -> Dict[str, Any]:
        """基于特征重要性的选择"""
        logger.info("执行基于特征重要性的特征选择...")
        model_type = self.importance_model
        if model_type == 'auto':
            # 简易规则：树模型任务多用 xgb，线性任务 rf
            model_type = 'xgb' if (XGBClassifier is not None) else 'rf'

        model = None
        # 训练模型获取特征重要性
        if model_type == 'rf':
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=self.n_jobs)
            else:
                model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=self.n_jobs)
        elif model_type == 'xgb' and (XGBClassifier is not None):
            if self.task_type == 'classification':
                model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                                      n_jobs=self.n_jobs, random_state=random_state, verbosity=0)
            else:
                model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                                     n_jobs=self.n_jobs, random_state=random_state, verbosity=0)
        elif model_type == 'lgb' and (LGBMClassifier is not None):
            if self.task_type == 'classification':
                model = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1,
                                       n_jobs=self.n_jobs, random_state=random_state)
            else:
                model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=-1,
                                      n_jobs=self.n_jobs, random_state=random_state)
        else:
            logger.warning(f"importance_model={model_type} 无法使用，回退随机森林")
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=self.n_jobs)
            else:
                model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=self.n_jobs)

        model.fit(X, y)
        # XGBoost / LGB 统一获取 feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            booster = model.get_booster()
            # ‘gain’ 更稳定
            score_dict = booster.get_score(importance_type='gain')
            importances = np.array([score_dict.get(f, 0.0) for f in booster.feature_names])

        # 按重要性排序
        importance_dict = dict(zip(X.columns, importances))
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前N个特征
        selected_features = [f[0] for f in sorted_features[:self.target_n_features]]
        
        # 交叉验证验证
        cv_score = self._validate_features(X[selected_features], y, cv_folds, random_state)
        
        self.selected_features_ = selected_features
        self.feature_importance_ = importance_dict
        self.cv_scores_ = {'importance': cv_score}
        
        return {
            'selected_features': selected_features,
            'feature_importance': dict(sorted_features),
            'cv_score': cv_score,
            'method': 'importance',
            'n_original_features': X.shape[1],
            'n_selected_features': len(selected_features)
        }
    
    def _stability_selection(self, X: pd.DataFrame, y: pd.Series,
                           cv_folds: int, random_state: int) -> Dict[str, Any]:
        """稳定性选择"""
        logger.info("执行稳定性特征选择...")
        
        feature_stability = {}
        n_iterations = 10
        
        # 多次随机抽样训练，观察特征选择的稳定性
        for i in range(n_iterations):
            # 随机抽样80%的数据
            sample_size = int(0.8 * len(X))
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            
            # 训练模型获取特征重要性
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=random_state + i, n_jobs=1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=random_state + i, n_jobs=1)
            
            model.fit(X_sample, y_sample)
            # 使用 permutation_importance 替代内置 feature_importances_，可减少模型偏置且支持一致接口
            perm_result = permutation_importance(
                model, X_sample, y_sample,
                n_repeats=5,
                random_state=random_state + i,
                n_jobs=self.n_jobs
            )
            importances = perm_result.importances_mean
            
            # 记录每个特征的重要性
            for feature, importance in zip(X.columns, importances):
                if feature not in feature_stability:
                    feature_stability[feature] = []
                feature_stability[feature].append(importance)
            
            logger.info(f"[Stability] Iteration {i+1}/{n_iterations} completed.")
        
        # 计算稳定性得分（平均值 / 标准差）
        stability_scores = {}
        for feature, scores in feature_stability.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            stability_scores[feature] = mean_score / (std_score + 1e-8)  # 加小常数避免除零
        
        # 按稳定性得分排序
        sorted_features = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:self.target_n_features]]
        
        logger.info("[Stability] All iterations complete. Starting CV validation...")
        # 交叉验证验证
        cv_score = self._validate_features(X[selected_features], y, cv_folds, random_state)
        
        self.selected_features_ = selected_features
        self.feature_scores_ = dict(sorted_features)
        self.cv_scores_ = {'stability': cv_score}
        
        return {
            'selected_features': selected_features,
            'stability_scores': dict(sorted_features),
            'cv_score': cv_score,
            'method': 'stability',
            'n_original_features': X.shape[1],
            'n_selected_features': len(selected_features)
        }
    
    def _auto_selection(self, X: pd.DataFrame, y: pd.Series,
                       cv_folds: int, random_state: int) -> Dict[str, Any]:
        """自动选择最佳方法和特征数量"""
        logger.info("执行自动特征选择...")
        # 保存调用前的目标特征数，以便方法结束后恢复
        original_target = self.target_n_features
        
        best_results = None
        best_score = -np.inf
        
        # 尝试不同的方法和特征数量
        methods = ['ensemble', 'importance', 'stability']
        feature_counts = [20, 30, 50, 70, 100]
        importance_models = ['rf', 'xgb', 'lgb']
        
        combos = []
        for m in methods:
            for n in feature_counts:
                if m == 'importance':
                    for im in importance_models:
                        combos.append((m, n, im))
                else:
                    combos.append((m, n, None))

        def _evaluate_combo(method: str, n_features: int, imp_model: str):
            """评估单个 (method, n_features[, importance_model]) 组合的性能"""
            tmp_selector = FeatureSelectorOptimizer(
                task_type=self.task_type,
                target_n_features=n_features,
                n_jobs=self.n_jobs,
                importance_model=imp_model or self.importance_model
            )
            if method == 'ensemble':
                return tmp_selector._ensemble_selection(X, y, cv_folds, random_state)
            elif method == 'importance':
                return tmp_selector._importance_based_selection(X, y, cv_folds, random_state)
            elif method == 'stability':
                return tmp_selector._stability_selection(X, y, cv_folds, random_state)
            else:
                raise ValueError(f"不支持的方法: {method}")

        if Parallel is not None and self.n_jobs != 1:
            logger.info(f"使用并行计算 auto 网格搜索 (n_jobs={self.n_jobs}) ...")
            results_list = Parallel(n_jobs=self.n_jobs)(
                delayed(_evaluate_combo)(m, n, im) for m, n, im in combos
            )
        else:
            logger.info("使用串行计算 auto 网格搜索 ...")
            results_list = [_evaluate_combo(m, n, im) for m, n, im in combos]

        for res in results_list:
            current_score = res['cv_score']
            method = res['method']
            n_features = res['n_selected_features']
            if current_score > best_score:
                best_score = current_score
                best_results = res.copy()
                best_results['best_method'] = method
                best_results['best_n_features'] = n_features
        
        # 恢复原始目标特征数
        self.target_n_features = original_target
        
        if best_results is None:
            # 如果所有方法都失败，使用简单的相关性选择
            logger.warning("所有高级方法失败，使用简单相关性选择")
            return self._simple_correlation_selection(X, y)
        
        return best_results
    
    def _simple_correlation_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """简单的相关性选择作为备选"""
        correlations = []
        for col in X.columns:
            try:
                corr = np.corrcoef(X[col].values, y.values)[0, 1]
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))
            except:
                continue
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected_features = [col for col, _ in correlations[:self.target_n_features]]
        
        return {
            'selected_features': selected_features,
            'feature_scores': dict(correlations),
            'cv_score': 0.0,
            'method': 'correlation',
            'n_original_features': X.shape[1],
            'n_selected_features': len(selected_features)
        }
    
    def _validate_features(self, X_selected: pd.DataFrame, y: pd.Series, 
                          cv_folds: int, random_state: int) -> float:
        """验证选择的特征"""
        try:
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scores = cross_val_score(model, X_selected, y, cv=cv, scoring='roc_auc', n_jobs=1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scores = cross_val_score(model, X_selected, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=1)
                scores = -scores  # 转换为正数，越小越好
            
            logger.info(f"交叉验证完成，平均得分={float(np.mean(scores)):.6f}")
            return float(np.mean(scores))
        
        except Exception as e:
            logger.warning(f"特征验证失败: {e}")
            return 0.0
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """绘制特征重要性图表"""
        if not self.feature_scores_:
            logger.warning("没有特征重要性数据")
            return
        
        # 获取top N特征
        top_features = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('特征重要性得分')
        plt.title(f'Top {top_n} 特征重要性')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_selection_report(self) -> Dict[str, Any]:
        """获取特征选择报告"""
        return {
            'task_type': self.task_type,
            'target_n_features': self.target_n_features,
            'selected_features': self.selected_features_,
            'feature_scores': self.feature_scores_,
            'feature_importance': self.feature_importance_,
            'cv_scores': self.cv_scores_,
            'selection_summary': {
                'total_original_features': len(self.feature_scores_) if self.feature_scores_ else 0,
                'selected_features': len(self.selected_features_),
                'reduction_ratio': len(self.selected_features_) / len(self.feature_scores_) if self.feature_scores_ else 0
            }
        }

# 测试函数
def test_feature_selector_optimizer():
    """测试特征选择优化器"""
    logger.info("开始测试特征选择优化器...")
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 122  # 模拟增强特征生成器的输出
    
    # 生成特征数据
    X = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) + np.random.randn() * 0.1 
        for i in range(n_features)
    })
    
    # 添加一些有预测力的特征
    X['important_feature_1'] = np.random.randn(n_samples) * 2
    X['important_feature_2'] = np.random.randn(n_samples) * 1.5
    X['important_feature_3'] = np.random.randn(n_samples) * 1.2
    
    # 生成目标变量（与重要特征相关）
    y = (X['important_feature_1'] * 0.5 + 
         X['important_feature_2'] * 0.3 + 
         X['important_feature_3'] * 0.2 + 
         np.random.randn(n_samples) * 0.1)
    
    # 转换为分类问题
    y_cls = (y > np.median(y)).astype(int)
    
    # 测试分类特征选择
    logger.info("测试分类特征选择...")
    selector_cls = FeatureSelectorOptimizer(task_type='classification', target_n_features=30)
    results_cls = selector_cls.optimize_feature_selection(X, y_cls, method='auto')
    
    logger.info(f"分类特征选择结果: {results_cls['n_selected_features']} 个特征")
    logger.info(f"CV得分: {results_cls['cv_score']:.4f}")
    logger.info(f"Top 5 特征: {results_cls['selected_features'][:5]}")
    
    # 测试回归特征选择
    logger.info("测试回归特征选择...")
    selector_reg = FeatureSelectorOptimizer(task_type='regression', target_n_features=30)
    results_reg = selector_reg.optimize_feature_selection(X, y, method='auto')
    
    logger.info(f"回归特征选择结果: {results_reg['n_selected_features']} 个特征")
    logger.info(f"CV得分: {results_reg['cv_score']:.4f}")
    logger.info(f"Top 5 特征: {results_reg['selected_features'][:5]}")
    
    return {
        'classification': results_cls,
        'regression': results_reg
    }

if __name__ == "__main__":
    test_feature_selector_optimizer()