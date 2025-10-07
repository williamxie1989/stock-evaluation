"""
模型训练器 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models = {}
        self.model_performance = {}
        
        # 创建模型目录
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        logger.info(f"ModelTrainer initialized with model_dir: {model_dir}")
    
    def train_classification_model(self, X: pd.DataFrame, y: pd.Series, 
                                 model_type: str = 'random_forest', 
                                 model_name: str = 'classifier',
                                 **kwargs) -> Dict[str, Any]:
        """训练分类模型"""
        try:
            if X.empty or y.empty:
                return {'error': '训练数据为空'}
            
            # 数据预处理
            X_clean = X.dropna()
            y_clean = y.loc[X_clean.index]
            
            if len(X_clean) < 100:
                return {'error': '训练数据不足'}
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
            )
            
            # 选择模型
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', 10),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'logistic_regression':
                model = LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
            else:
                return {'error': f'不支持的模型类型: {model_type}'}
            
            # 训练模型
            logger.info(f"训练分类模型: {model_type}")
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            performance = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_clean, y_clean, cv=5, scoring='accuracy')
            performance['cv_accuracy_mean'] = float(cv_scores.mean())
            performance['cv_accuracy_std'] = float(cv_scores.std())
            
            # 保存模型
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            
            # 保存模型信息
            self.models[model_name] = model
            self.model_performance[model_name] = performance
            
            logger.info(f"分类模型训练完成: {model_name}, 准确率: {performance['accuracy']:.4f}")
            
            return {
                'model_name': model_name,
                'model_type': model_type,
                'performance': performance,
                'model_path': model_path,
                'feature_importance': self._get_feature_importance(model, X.columns)
            }
            
        except Exception as e:
            logger.error(f"分类模型训练失败: {e}")
            return {'error': str(e)}
    
    def train_regression_model(self, X: pd.DataFrame, y: pd.Series, 
                              model_type: str = 'random_forest', 
                              model_name: str = 'regressor',
                              **kwargs) -> Dict[str, Any]:
        """训练回归模型"""
        try:
            if X.empty or y.empty:
                return {'error': '训练数据为空'}
            
            # 数据预处理
            X_clean = X.dropna()
            y_clean = y.loc[X_clean.index]
            
            if len(X_clean) < 100:
                return {'error': '训练数据不足'}
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42
            )
            
            # 选择模型
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', 10),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'linear_regression':
                model = LinearRegression()
            else:
                return {'error': f'不支持的模型类型: {model_type}'}
            
            # 训练模型
            logger.info(f"训练回归模型: {model_type}")
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            
            performance = {
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2_score': float(r2_score(y_test, y_pred)),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_clean, y_clean, cv=5, scoring='r2')
            performance['cv_r2_mean'] = float(cv_scores.mean())
            performance['cv_r2_std'] = float(cv_scores.std())
            
            # 保存模型
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            
            # 保存模型信息
            self.models[model_name] = model
            self.model_performance[model_name] = performance
            
            logger.info(f"回归模型训练完成: {model_name}, R²: {performance['r2_score']:.4f}")
            
            return {
                'model_name': model_name,
                'model_type': model_type,
                'performance': performance,
                'model_path': model_path,
                'feature_importance': self._get_feature_importance(model, X.columns)
            }
            
        except Exception as e:
            logger.error(f"回归模型训练失败: {e}")
            return {'error': str(e)}
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """加载模型"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                self.models[model_name] = model
                logger.info(f"模型加载成功: {model_name}")
                return model
            else:
                logger.warning(f"模型文件不存在: {model_path}")
                return None
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return None
    
    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """从指定路径加载训练好的模型（分类模型），并进行概率预测"""
        model = self.load_model(model_name)
        if model is None:
            logger.error(f"模型 {model_name} 不存在")
            return np.array([])
        
        # 检查模型是否支持 predict_proba
        if not hasattr(model, 'predict_proba'):
            logger.warning(f"模型 {model_name} 不支持概率预测")
            # 尝试常规预测并返回
            try:
                return model.predict(X)
            except:
                return np.array([])
        
        # 清理数据
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].dtype == 'object':
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        X_clean = X_clean.fillna(0.0)
        
        try:
            return model.predict_proba(X_clean)
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return np.array([])
            
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """从指定路径加载训练好的模型，并进行预测（支持回归和分类）"""
        model = self.load_model(model_name)
        if model is None:
            logger.error(f"模型 {model_name} 不存在")
            return np.array([])
        
        # 清理数据
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].dtype == 'object':
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        X_clean = X_clean.fillna(0.0)
        
        try:
            return model.predict(X_clean)
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return np.array([])
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """获取特征重要性"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = {}
                for name, importance in zip(feature_names, model.feature_importances_):
                    importance_dict[name] = float(importance)
                return importance_dict
            else:
                return {}
        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return {}
    
    def get_model_performance(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型性能"""
        return self.model_performance.get(model_name)
    
    def list_models(self) -> List[str]:
        """列出所有模型"""
        return list(self.models.keys())
    
    def delete_model(self, model_name: str) -> bool:
        """删除模型"""
        try:
            # 从内存中删除
            if model_name in self.models:
                del self.models[model_name]
            
            # 从性能记录中删除
            if model_name in self.model_performance:
                del self.model_performance[model_name]
            
            # 删除模型文件
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            if os.path.exists(model_path):
                os.remove(model_path)
            
            logger.info(f"模型删除成功: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"模型删除失败: {e}")
            return False
    
    def reset(self):
        """重置模型训练器"""
        self.models.clear()
        self.model_performance.clear()
        logger.info("模型训练器已重置")