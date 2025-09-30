"""
统一模型训练器 - 集成特征解析、多模型训练和评估
"""

import logging
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pickle

logger = logging.getLogger(__name__)

class UnifiedTrainer:
    """统一模型训练器 - 支持特征解析、多模型并行训练和评估"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models = {}
        self.model_performance = {}
        
        # 创建模型目录
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        logger.info(f"UnifiedTrainer initialized with model_dir: {model_dir}")
    
    def parse_features_from_samples(self, samples_df: pd.DataFrame) -> pd.DataFrame:
        """
        从样本数据中解析features列的JSON数据
        
        Args:
            samples_df: 包含features列的样本数据
            
        Returns:
            解析后的特征DataFrame
        """
        if samples_df.empty or 'features' not in samples_df.columns:
            logger.error("样本数据为空或缺少features列")
            return pd.DataFrame()
        
        try:
            # 解析features列的JSON数据
            feature_data = []
            symbols = []
            dates = []
            labels = []
            forward_returns = []
            
            for idx, row in samples_df.iterrows():
                try:
                    # 解析JSON特征
                    if isinstance(row['features'], str):
                        features = json.loads(row['features'])
                    else:
                        features = row['features']
                    
                    # 展平嵌套的字典结构
                    flat_features = self._flatten_dict(features)
                    feature_data.append(flat_features)
                    
                    # 保存其他信息
                    symbols.append(row['symbol'])
                    dates.append(row['date'])
                    labels.append(row['label'])
                    forward_returns.append(row['forward_return'])
                    
                except Exception as e:
                    logger.warning(f"解析features失败 (行 {idx}): {e}")
                    continue
            
            if not feature_data:
                logger.error("没有成功解析任何特征数据")
                return pd.DataFrame()
            
            # 创建特征DataFrame
            features_df = pd.DataFrame(feature_data)
            
            # 添加其他列
            features_df['symbol'] = symbols
            features_df['date'] = dates
            features_df['label'] = labels
            features_df['forward_return'] = forward_returns
            
            logger.info(f"成功解析 {len(features_df)} 个样本的特征数据，特征维度: {features_df.shape[1]}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"解析features数据失败: {e}")
            return pd.DataFrame()
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """展平嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def prepare_training_data(self, features_df: pd.DataFrame, 
                            target_type: str = 'both') -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        准备训练数据
        
        Args:
            features_df: 特征数据
            target_type: 'classification', 'regression', 或 'both'
            
        Returns:
            (特征矩阵, 目标字典)
        """
        if features_df.empty:
            return pd.DataFrame(), {}
        
        # 只保留数值类型的特征列
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 移除不需要的列
        exclude_cols = ['symbol', 'date', 'label', 'forward_return']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not feature_cols:
            logger.error("没有找到有效的数值特征")
            return pd.DataFrame(), {}
        
        X = features_df[feature_cols].copy()
        
        # 处理缺失值和异常值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 移除常数特征
        constant_features = X.columns[X.std() == 0].tolist()
        if constant_features:
            logger.info(f"移除常数特征: {constant_features}")
            X = X.drop(columns=constant_features)
            feature_cols = [c for c in feature_cols if c not in constant_features]
        
        # 准备目标变量
        y_dict = {}
        
        if target_type in ['classification', 'both']:
            y_cls = features_df['label'].copy()
            y_dict['classification'] = y_cls
            logger.info(f"分类标签分布: {y_cls.value_counts().to_dict()}")
        
        if target_type in ['regression', 'both']:
            y_reg = features_df['forward_return'].copy()
            y_dict['regression'] = y_reg
            logger.info(f"回归目标统计: mean={y_reg.mean():.4f}, std={y_reg.std():.4f}")
        
        logger.info(f"训练数据准备完成: {X.shape[1]} 个特征, {X.shape[0]} 个样本")
        
        return X, y_dict
    
    def train_classification_models(self, X: pd.DataFrame, y: pd.Series,
                                  use_grid_search: bool = True) -> Dict[str, Any]:
        """训练多个分类模型并选择最佳模型"""
        try:
            if X.empty or y.empty:
                return {'error': '训练数据为空'}
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"分类训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
            
            # 定义要训练的模型
            models = {
                'logistic_regression': self._create_logistic_pipeline(),
                'random_forest': self._create_random_forest_classifier(),
                'xgboost': self._create_xgboost_classifier() if self._check_xgboost() else None
            }
            
            # 移除不可用的模型
            models = {k: v for k, v in models.items() if v is not None}
            
            results = {}
            
            # 并行训练多个模型
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_model = {
                    executor.submit(self._train_single_classifier, 
                                  X_train, y_train, X_test, y_test, 
                                  model_name, model, use_grid_search): model_name
                    for model_name, model in models.items()
                }
                
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        if result and 'error' not in result:
                            results[model_name] = result
                            logger.info(f"{model_name} 分类模型训练完成")
                    except Exception as e:
                        logger.error(f"{model_name} 分类模型训练失败: {e}")
            
            if not results:
                return {'error': '所有分类模型训练失败'}
            
            # 选择最佳模型（基于AUC分数）
            best_model = max(results.keys(), key=lambda k: results[k]['performance']['auc'])
            best_result = results[best_model]
            
            logger.info(f"选择最佳分类模型: {best_model}, AUC: {best_result['performance']['auc']:.4f}")
            
            return {
                'best_model': best_model,
                'best_result': best_result,
                'all_results': results
            }
            
        except Exception as e:
            logger.error(f"分类模型训练失败: {e}")
            return {'error': str(e)}
    
    def train_regression_models(self, X: pd.DataFrame, y: pd.Series,
                              use_grid_search: bool = True) -> Dict[str, Any]:
        """训练多个回归模型并选择最佳模型"""
        try:
            if X.empty or y.empty:
                return {'error': '训练数据为空'}
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            logger.info(f"回归训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
            
            # 定义要训练的模型
            models = {
                'ridge': self._create_ridge_pipeline(),
                'lasso': self._create_lasso_pipeline(),
                'elasticnet': self._create_elasticnet_pipeline(),
                'random_forest': self._create_random_forest_regressor(),
                'xgboost': self._create_xgboost_regressor() if self._check_xgboost() else None
            }
            
            # 移除不可用的模型
            models = {k: v for k, v in models.items() if v is not None}
            
            results = {}
            
            # 并行训练多个模型
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_model = {
                    executor.submit(self._train_single_regressor, 
                                  X_train, y_train, X_test, y_test, 
                                  model_name, model, use_grid_search): model_name
                    for model_name, model in models.items()
                }
                
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        if result and 'error' not in result:
                            results[model_name] = result
                            logger.info(f"{model_name} 回归模型训练完成")
                    except Exception as e:
                        logger.error(f"{model_name} 回归模型训练失败: {e}")
            
            if not results:
                return {'error': '所有回归模型训练失败'}
            
            # 选择最佳模型（基于R²分数）
            best_model = max(results.keys(), key=lambda k: results[k]['performance']['r2'])
            best_result = results[best_model]
            
            logger.info(f"选择最佳回归模型: {best_model}, R²: {best_result['performance']['r2']:.4f}")
            
            return {
                'best_model': best_model,
                'best_result': best_result,
                'all_results': results
            }
            
        except Exception as e:
            logger.error(f"回归模型训练失败: {e}")
            return {'error': str(e)}
    
    def _train_single_classifier(self, X_train, y_train, X_test, y_test, 
                                 model_name: str, model, use_grid_search: bool):
        """训练单个分类模型"""
        try:
            logger.info(f"开始训练 {model_name} 分类模型...")
            
            if use_grid_search and hasattr(self, f'_get_{model_name}_params'):
                # 使用网格搜索
                param_grid = getattr(self, f'_get_{model_name}_params')()
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='roc_auc', 
                    n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                logger.info(f"{model_name} 最佳参数: {grid_search.best_params_}")
            else:
                # 直接训练
                model.fit(X_train, y_train)
                best_model = model
            
            # 评估模型
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            performance = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
            }
            
            if y_pred_proba is not None:
                performance['auc'] = float(roc_auc_score(y_test, y_pred_proba))
            
            # 交叉验证
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
            performance['cv_accuracy_mean'] = float(cv_scores.mean())
            performance['cv_accuracy_std'] = float(cv_scores.std())
            
            # 保存模型
            model_path = os.path.join(self.model_dir, f"{model_name}_classifier.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            logger.info(f"{model_name} 分类模型训练完成，准确率: {performance['accuracy']:.4f}")
            
            return {
                'model_name': model_name,
                'model': best_model,
                'performance': performance,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"训练 {model_name} 分类模型失败: {e}")
            return {'error': str(e)}
    
    def _train_single_regressor(self, X_train, y_train, X_test, y_test, 
                                model_name: str, model, use_grid_search: bool):
        """训练单个回归模型"""
        try:
            logger.info(f"开始训练 {model_name} 回归模型...")
            
            if use_grid_search and hasattr(self, f'_get_{model_name}_params'):
                # 使用网格搜索
                param_grid = getattr(self, f'_get_{model_name}_params')()
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='r2', 
                    n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                logger.info(f"{model_name} 最佳参数: {grid_search.best_params_}")
            else:
                # 直接训练
                model.fit(X_train, y_train)
                best_model = model
            
            # 评估模型
            y_pred = best_model.predict(X_test)
            
            performance = {
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }
            
            # 交叉验证
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
            performance['cv_r2_mean'] = float(cv_scores.mean())
            performance['cv_r2_std'] = float(cv_scores.std())
            
            # 保存模型
            model_path = os.path.join(self.model_dir, f"{model_name}_regressor.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            logger.info(f"{model_name} 回归模型训练完成，R²: {performance['r2']:.4f}")
            
            return {
                'model_name': model_name,
                'model': best_model,
                'performance': performance,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"训练 {model_name} 回归模型失败: {e}")
            return {'error': str(e)}
    
    # 模型创建方法
    def _create_logistic_pipeline(self):
        """创建逻辑回归管线"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
    
    def _create_random_forest_classifier(self):
        """创建随机森林分类器"""
        return RandomForestClassifier(
            n_estimators=100, max_depth=10, 
            random_state=42, n_jobs=-1
        )
    
    def _create_xgboost_classifier(self):
        """创建XGBoost分类器"""
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=100, max_depth=6,
                learning_rate=0.1, random_state=42
            )
        except ImportError:
            logger.warning("XGBoost未安装，跳过XGBoost模型")
            return None
    
    def _create_ridge_pipeline(self):
        """创建岭回归管线"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ])
    
    def _create_lasso_pipeline(self):
        """创建Lasso回归管线"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Lasso(alpha=0.1))
        ])
    
    def _create_elasticnet_pipeline(self):
        """创建ElasticNet回归管线"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5))
        ])
    
    def _create_random_forest_regressor(self):
        """创建随机森林回归器"""
        return RandomForestRegressor(
            n_estimators=100, max_depth=10,
            random_state=42, n_jobs=-1
        )
    
    def _create_xgboost_regressor(self):
        """创建XGBoost回归器"""
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=100, max_depth=6,
                learning_rate=0.1, random_state=42, n_jobs=-1
            )
        except ImportError:
            logger.warning("XGBoost未安装，跳过XGBoost回归模型")
            return None
    
    # 参数网格方法
    def _get_logistic_regression_params(self):
        """逻辑回归参数网格"""
        return {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        }
    
    def _get_random_forest_classifier_params(self):
        """随机森林分类器参数网格"""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    
    def _get_xgboost_classifier_params(self):
        """XGBoost分类器参数网格"""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    
    def _get_ridge_params(self):
        """岭回归参数网格"""
        return {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0]
        }
    
    def _get_lasso_params(self):
        """Lasso回归参数网格"""
        return {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0]
        }
    
    def _get_elasticnet_params(self):
        """ElasticNet回归参数网格"""
        return {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0],
            'regressor__l1_ratio': [0.1, 0.5, 0.7, 0.9]
        }
    
    def _get_random_forest_regressor_params(self):
        """随机森林回归器参数网格"""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    
    def _get_xgboost_regressor_params(self):
        """XGBoost回归器参数网格"""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    
    def _check_xgboost(self):
        """检查XGBoost是否可用"""
        try:
            import xgboost
            return True
        except ImportError:
            return False

# 使用示例和测试函数
def main():
    """主函数 - 用于测试"""
    import sys
    sys.path.append('/Users/xieyongliang/stock-evaluation')
    
    from src.data.db.unified_database_manager import UnifiedDatabaseManager
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 初始化数据库管理器
        db_manager = UnifiedDatabaseManager()
        
        # 查询样本数据
        query = "SELECT * FROM samples WHERE period = '10d' ORDER BY date DESC LIMIT 100"
        results = db_manager.execute_query(query)
        
        if not results:
            logger.error("没有找到样本数据")
            return
        
        # 转换为DataFrame
        samples_df = pd.DataFrame(results)
        logger.info(f"加载了 {len(samples_df)} 个样本数据")
        
        # 初始化训练器
        trainer = UnifiedTrainer()
        
        # 解析特征
        features_df = trainer.parse_features_from_samples(samples_df)
        if features_df.empty:
            logger.error("特征解析失败")
            return
        
        # 准备训练数据
        X, y_dict = trainer.prepare_training_data(features_df, target_type='both')
        if X.empty or not y_dict:
            logger.error("训练数据准备失败")
            return
        
        # 训练分类模型
        if 'classification' in y_dict:
            logger.info("开始训练分类模型...")
            cls_results = trainer.train_classification_models(X, y_dict['classification'])
            
            if 'best_model' in cls_results:
                logger.info(f"最佳分类模型: {cls_results['best_model']}")
                logger.info(f"最佳AUC: {cls_results['best_result']['performance']['auc']:.4f}")
        
        # 训练回归模型
        if 'regression' in y_dict:
            logger.info("开始训练回归模型...")
            reg_results = trainer.train_regression_models(X, y_dict['regression'])
            
            if 'best_model' in reg_results:
                logger.info(f"最佳回归模型: {reg_results['best_model']}")
                logger.info(f"最佳R²: {reg_results['best_result']['performance']['r2']:.4f}")
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()