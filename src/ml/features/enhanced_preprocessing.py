#!/usr/bin/env python3
"""
增强的特征选择和预处理pipeline
包含特征选择、异常值处理、特征工程等功能
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutlierRemover(BaseEstimator, TransformerMixin):
    """异常值处理器"""
    
    def __init__(self, method='iqr', threshold=3.0):
        self.method = method
        self.threshold = threshold
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        """学习异常值边界"""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col in X_df.columns:
            if X_df[col].dtype in ['int64', 'float64']:
                if self.method == 'iqr':
                    Q1 = X_df[col].quantile(0.25)
                    Q3 = X_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                elif self.method == 'zscore':
                    mean = X_df[col].mean()
                    std = X_df[col].std()
                    lower = mean - self.threshold * std
                    upper = mean + self.threshold * std
                else:  # percentile
                    lower = X_df[col].quantile(0.01)
                    upper = X_df[col].quantile(0.99)
                
                self.bounds_[col] = (lower, upper)
        
        return self
    
    def transform(self, X):
        """处理异常值"""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, (lower, upper) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower, upper)
        
        return X_df.values if not isinstance(X, pd.DataFrame) else X_df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """特征工程器"""
    
    def __init__(self, create_interactions=1, create_ratios=1, create_polynomials=0):
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.create_polynomials = create_polynomials
        self.feature_names_ = []
        self.original_features_ = []
        
    def fit(self, X, y=None):
        """学习特征名称"""
        if isinstance(X, pd.DataFrame):
            self.original_features_ = list(X.columns)
        else:
            self.original_features_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        return self
    
    def transform(self, X):
        """创建新特征"""
        X_df = pd.DataFrame(X, columns=self.original_features_) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # 原始特征
        new_features = X_df.copy()
        
        # 创建交互特征
        if self.create_interactions and len(X_df.columns) > 1:
            numeric_cols = X_df.select_dtypes(include=[np.number]).columns[:10]  # 限制数量
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    new_features[f'{col1}_x_{col2}'] = X_df[col1] * X_df[col2]
        
        # 创建比率特征
        if self.create_ratios and len(X_df.columns) > 1:
            numeric_cols = X_df.select_dtypes(include=[np.number]).columns[:8]  # 限制数量
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    # 避免除零
                    denominator = X_df[col2].replace(0, np.nan)
                    ratio = X_df[col1] / denominator
                    if not ratio.isna().all():
                        new_features[f'{col1}_div_{col2}'] = ratio.fillna(0)
        
        # 创建多项式特征（谨慎使用）
        if self.create_polynomials:
            numeric_cols = X_df.select_dtypes(include=[np.number]).columns[:5]  # 严格限制
            for col in numeric_cols:
                new_features[f'{col}_squared'] = X_df[col] ** 2
                new_features[f'{col}_sqrt'] = np.sqrt(np.abs(X_df[col]))
        
        # 处理无穷值和NaN
        new_features = new_features.replace([np.inf, -np.inf], np.nan)
        new_features = new_features.fillna(0)
        
        self.feature_names_ = list(new_features.columns)
        
        return new_features

class SmartFeatureSelector(BaseEstimator, TransformerMixin):
    """智能特征选择器"""
    
    def __init__(self, task_type='classification', n_features=50, selection_methods=['univariate', 'rfe']):
        self.task_type = task_type
        self.n_features = n_features
        self.selection_methods = selection_methods
        self.selected_features_ = []
        self.feature_scores_ = {}
        
    def fit(self, X, y):
        """选择最佳特征"""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # 移除常数特征和高度相关特征
        X_cleaned = self._remove_constant_features(X_df)
        X_cleaned = self._remove_highly_correlated_features(X_cleaned)
        
        feature_scores = {}
        
        # 单变量特征选择
        if 'univariate' in self.selection_methods:
            if self.task_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k=min(self.n_features, X_cleaned.shape[1]))
            else:
                selector = SelectKBest(score_func=f_regression, k=min(self.n_features, X_cleaned.shape[1]))
            
            selector.fit(X_cleaned, y)
            scores = selector.scores_
            for i, feature in enumerate(X_cleaned.columns):
                feature_scores[feature] = feature_scores.get(feature, 0) + scores[i]
        
        # 互信息特征选择
        if 'mutual_info' in self.selection_methods:
            if self.task_type == 'classification':
                mi_scores = mutual_info_classif(X_cleaned, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_cleaned, y, random_state=42)
            
            for i, feature in enumerate(X_cleaned.columns):
                feature_scores[feature] = feature_scores.get(feature, 0) + mi_scores[i]
        
        # 递归特征消除
        if 'rfe' in self.selection_methods:
            if self.task_type == 'classification':
                estimator = LogisticRegression(random_state=42, max_iter=1000)
            else:
                estimator = Ridge(random_state=42)
            
            rfe = RFE(estimator, n_features_to_select=min(self.n_features, X_cleaned.shape[1]))
            rfe.fit(X_cleaned, y)
            
            for i, feature in enumerate(X_cleaned.columns):
                if rfe.support_[i]:
                    feature_scores[feature] = feature_scores.get(feature, 0) + rfe.ranking_[i]
        
        # 基于树的特征重要性
        if 'tree_importance' in self.selection_methods:
            if self.task_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            
            estimator.fit(X_cleaned, y)
            importances = estimator.feature_importances_
            
            for i, feature in enumerate(X_cleaned.columns):
                feature_scores[feature] = feature_scores.get(feature, 0) + importances[i]
        
        # 选择最佳特征
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=1)
        self.selected_features_ = [f[0] for f in sorted_features[:self.n_features]]
        self.feature_scores_ = dict(sorted_features)
        
        logger.info(f"特征选择完成: 从 {X_cleaned.shape[1]} 个特征中选择了 {len(self.selected_features_)} 个")
        
        return self
    
    def transform(self, X):
        """应用特征选择"""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # 确保所有选择的特征都存在
        available_features = [f for f in self.selected_features_ if f in X_df.columns]
        if len(available_features) < len(self.selected_features_):
            logger.warning(f"部分特征不可用: {len(self.selected_features_) - len(available_features)} 个特征缺失")
        
        return X_df[available_features]
    
    def _remove_constant_features(self, X):
        """移除常数特征"""
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            logger.info(f"移除 {len(constant_features)} 个常数特征")
            X = X.drop(columns=constant_features)
        
        return X
    
    def _remove_highly_correlated_features(self, X, threshold=0.95):
        """移除高度相关的特征"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return X
        
        corr_matrix = X[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        if to_drop:
            logger.info(f"移除 {len(to_drop)} 个高度相关特征")
            X = X.drop(columns=to_drop)
        
        return X

class EnhancedPreprocessingPipeline:
    """增强的预处理pipeline"""
    
    def __init__(self, task_type='classification', config=None):
        self.task_type = task_type
        self.config = config or self._get_default_config()
        self.pipeline = None
        self.feature_names_ = []
        
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'outlier_removal': {
                'enabled': 1,
                'method': 'iqr',  # 'iqr', 'zscore', 'percentile'
                'threshold': 3.0
            },
            'feature_engineering': {
                'enabled': 1,
                'create_interactions': 1,
                'create_ratios': 1,
                'create_polynomials': 0
            },
            'feature_selection': {
                'enabled': 1,
                'n_features': 50,
                'methods': ['univariate', 'mutual_info', 'tree_importance']
            },
            'scaling': {
                'method': 'robust',  # 'standard', 'robust', 'minmax'
            }
        }
    
    def build_pipeline(self):
        """构建预处理pipeline"""
        steps = []
        
        # 异常值处理
        if self.config['outlier_removal']['enabled']:
            outlier_remover = OutlierRemover(
                method=self.config['outlier_removal']['method'],
                threshold=self.config['outlier_removal']['threshold']
            )
            steps.append(('outlier_removal', outlier_remover))
        
        # 特征工程
        if self.config['feature_engineering']['enabled']:
            feature_engineer = FeatureEngineer(
                create_interactions=self.config['feature_engineering'].get('create_interactions', 1),
                create_ratios=self.config['feature_engineering'].get('create_ratios', 1),
                create_polynomials=self.config['feature_engineering'].get('create_polynomials', 0)
            )
            steps.append(('feature_engineering', feature_engineer))
        
        # 特征选择
        if self.config['feature_selection']['enabled']:
            feature_selector = SmartFeatureSelector(
                task_type=self.task_type,
                n_features=self.config['feature_selection']['n_features'],
                selection_methods=self.config['feature_selection']['methods']
            )
            steps.append(('feature_selection', feature_selector))
        
        # 特征缩放
        scaling_method = self.config['scaling']['method']
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        steps.append(('scaler', scaler))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def fit_transform(self, X, y):
        """拟合并转换数据"""
        if self.pipeline is None:
            self.build_pipeline()
        
        logger.info(f"开始预处理，原始特征数: {X.shape[1]}")
        
        X_transformed = self.pipeline.fit_transform(X, y)
        
        # 获取最终特征名称
        if hasattr(self.pipeline.named_steps.get('feature_selection'), 'selected_features_'):
            self.feature_names_ = self.pipeline.named_steps['feature_selection'].selected_features_
        elif hasattr(self.pipeline.named_steps.get('feature_engineering'), 'feature_names_'):
            self.feature_names_ = self.pipeline.named_steps['feature_engineering'].feature_names_
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X_transformed.shape[1])]
        
        logger.info(f"预处理完成，最终特征数: {X_transformed.shape[1]}")
        
        return X_transformed
    
    def transform(self, X):
        """转换数据"""
        if self.pipeline is None:
            raise ValueError("Pipeline未拟合，请先调用fit_transform")
        
        return self.pipeline.transform(X)
    
    def get_feature_importance_report(self):
        """获取特征重要性报告"""
        if not hasattr(self.pipeline.named_steps.get('feature_selection'), 'feature_scores_'):
            return "特征选择器未启用或未拟合"
        
        feature_scores = self.pipeline.named_steps['feature_selection'].feature_scores_
        selected_features = self.pipeline.named_steps['feature_selection'].selected_features_
        
        report = "特征重要性报告:\n"
        report += "=" * 50 + "\n"
        
        for i, feature in enumerate(selected_features[:20]):  # 显示前20个
            score = feature_scores.get(feature, 0)
            report += f"{i+1:2d}. {feature:<30} {score:.4f}\n"
        
        return report

def create_enhanced_preprocessing_config(task_type='classification', complexity='medium'):
    """创建增强预处理配置"""
    
    configs = {
        'simple': {
            'outlier_removal': {'enabled': 1, 'method': 'iqr', 'threshold': 3.0},
            'feature_engineering': {'enabled': 0, 'create_interactions': 0, 'create_ratios': 0, 'create_polynomials': 0},
            'feature_selection': {'enabled': 1, 'n_features': 30, 'methods': ['univariate']},
            'scaling': {'method': 'standard'}
        },
        'medium': {
            'outlier_removal': {'enabled': 1, 'method': 'iqr', 'threshold': 3.0},
            'feature_engineering': {'enabled': 1, 'create_interactions': 1, 'create_ratios': 0, 'create_polynomials': 0},
            'feature_selection': {'enabled': 1, 'n_features': 50, 'methods': ['univariate', 'mutual_info']},
            'scaling': {'method': 'robust'}
        },
        'complex': {
            'outlier_removal': {'enabled': 1, 'method': 'iqr', 'threshold': 3.0},
            'feature_engineering': {'enabled': 1, 'create_interactions': 1, 'create_ratios': 1, 'create_polynomials': 0},
            'feature_selection': {'enabled': 1, 'n_features': 80, 'methods': ['univariate', 'mutual_info', 'tree_importance']},
            'scaling': {'method': 'robust'}
        }
    }
    
    return configs.get(complexity, configs['medium'])

# 为向后兼容创建别名
EnhancedPreprocessor = EnhancedPreprocessingPipeline

if __name__ == "__main__":
    # 测试预处理pipeline
    from sklearn.datasets import make_classification, make_regression
    
    # 测试分类
    print("测试分类预处理...")
    X_cls, y_cls = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    
    config = create_enhanced_preprocessing_config('classification', 'medium')
    preprocessor = EnhancedPreprocessingPipeline('classification', config)
    
    X_cls_transformed = preprocessor.fit_transform(X_cls, y_cls)
    print(f"分类数据: {X_cls.shape} -> {X_cls_transformed.shape}")
    print(preprocessor.get_feature_importance_report())
    
    # 测试回归
    print("\n测试回归预处理...")
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    
    config = create_enhanced_preprocessing_config('regression', 'medium')
    preprocessor = EnhancedPreprocessingPipeline('regression', config)
    
    X_reg_transformed = preprocessor.fit_transform(X_reg, y_reg)
    print(f"回归数据: {X_reg.shape} -> {X_reg_transformed.shape}")
    print(preprocessor.get_feature_importance_report())