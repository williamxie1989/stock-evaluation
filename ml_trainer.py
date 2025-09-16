#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习训练管线
StandardScaler + LogisticRegression，支持超参配置、模型持久化、特征重要性输出
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

from db import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTrainer:
    """
    机器学习训练器
    支持逻辑回归模型训练、验证和持久化
    """
    
    def __init__(self, db_manager: DatabaseManager, model_dir: str = "models"):
        self.db_manager = db_manager
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # 默认超参数
        self.default_params = {
            'logistic__C': [0.01, 0.1, 1.0, 10.0],
            'logistic__penalty': ['l1', 'l2'],
            'logistic__solver': ['liblinear'],
            'logistic__max_iter': [1000]
        }
        # 添加回归超参数网格
        self.default_regression_params = {
            'ridge__alpha': [0.1, 1.0, 10.0, 100.0]
        }
        
    def load_samples_from_db(self, symbols: List[str] = None, 
                           period: str = '10d',
                           start_date: str = None,
                           end_date: str = None) -> pd.DataFrame:
        """
        从数据库加载训练样本
        
        Args:
            symbols: 股票代码列表，None表示所有股票
            period: 预测期数，如'5d', '10d', '20d'
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含特征和标签的DataFrame
        """
        try:
            query = """
            SELECT symbol, date, period, label, forward_return, features
            FROM samples
            WHERE period = ?
            """
            params = [period]
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
                
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            query += " ORDER BY symbol, date"
            
            with self.db_manager.get_conn() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
            if df.empty:
                logger.warning("未找到匹配的样本数据")
                return pd.DataFrame()
                
            # 解析特征JSON
            features_list = []
            for _, row in df.iterrows():
                try:
                    features = json.loads(row['features'])  # JSON解析
                    features['symbol'] = row['symbol']
                    features['date'] = row['date']
                    features['label'] = row['label']
                    features['forward_return'] = row['forward_return']
                    features_list.append(features)
                except Exception as e:
                    logger.warning(f"解析特征失败: {e}")
                    continue
                    
            if not features_list:
                logger.warning("没有有效的特征数据")
                return pd.DataFrame()
                
            result_df = pd.DataFrame(features_list)
            logger.info(f"加载样本数据: {len(result_df)} 条, 期数: {period}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"加载样本数据失败: {e}")
            return pd.DataFrame()
    
    def prepare_features_and_target(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        按目标列准备特征与目标（支持分类与回归）
        """
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=float), []
        exclude_cols = ['symbol', 'date', 'label', 'forward_return']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        X = X.fillna(X.median())
        constant_features = X.columns[X.std() == 0].tolist()
        if constant_features:
            logger.info(f"移除常数特征: {constant_features}")
            X = X.drop(columns=constant_features)
            feature_cols = [c for c in feature_cols if c not in constant_features]
        logger.info(f"特征准备完成: {X.shape[1]} 个特征, {len(y)} 个样本, 目标列: {target_col}")
        if target_col == 'label':
            logger.info(f"标签分布: {y.value_counts().to_dict()}")
        else:
            logger.info(f"目标统计: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
        return X, y, feature_cols
    
    def create_pipeline(self, params: Dict = None) -> Pipeline:
        """
        创建训练管线
        
        Args:
            params: 模型参数
            
        Returns:
            sklearn Pipeline
        """
        if params is None:
            params = {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000}
            
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(
                random_state=42,
                class_weight='balanced',  # 处理类别不平衡
                **params
            ))
        ])
        
        return pipeline
    
    def create_regression_pipeline(self, params: Dict = None) -> Pipeline:
        """创建回归训练管线: StandardScaler + Ridge"""
        if params is None:
            params = {'alpha': 1.0}
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(**params))
        ])
        return pipeline
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2,
                   use_grid_search: bool = True,
                   cv_folds: int = 5) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
            test_size: 测试集比例
            use_grid_search: 是否使用网格搜索
            cv_folds: 交叉验证折数
            
        Returns:
            训练结果字典
        """
        try:
            if X.empty or len(y) == 0:
                raise ValueError("特征或标签数据为空")
                
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            logger.info(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
            
            # 创建基础管线
            base_pipeline = self.create_pipeline()
            
            if use_grid_search:
                # 网格搜索最优参数
                logger.info("开始网格搜索...")
                
                # 使用时间序列交叉验证
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                
                grid_search = GridSearchCV(
                    base_pipeline,
                    self.default_params,
                    cv=tscv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                logger.info(f"最优参数: {best_params}")
                logger.info(f"最优CV得分: {grid_search.best_score_:.4f}")
                
            else:
                # 使用默认参数训练
                best_model = base_pipeline
                best_model.fit(X_train, y_train)
                best_params = {}
                
            # 预测
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            y_train_proba = best_model.predict_proba(X_train)[:, 1]
            y_test_proba = best_model.predict_proba(X_test)[:, 1]
            
            # 计算指标
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
            
            logger.info(f"训练集AUC: {train_auc:.4f}")
            logger.info(f"测试集AUC: {test_auc:.4f}")
            
            # 特征重要性
            feature_importance = self._get_feature_importance(best_model, X.columns)
            
            # 构建结果
            result = {
                'model': best_model,
                'best_params': best_params,
                'feature_names': X.columns.tolist(),
                'feature_importance': feature_importance,
                'metrics': {
                    'train_auc': train_auc,
                    'test_auc': test_auc,
                    'train_accuracy': (y_train_pred == y_train).mean(),
                    'test_accuracy': (y_test_pred == y_test).mean()
                },
                'predictions': {
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_test_pred': y_test_pred,
                    'y_test_proba': y_test_proba
                },
                'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise
    
    def _get_feature_importance(self, model: Pipeline, feature_names: List[str]) -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            
        Returns:
            特征重要性字典
        """
        try:
            # 获取逻辑回归的系数
            coefficients = model.named_steps['logistic'].coef_[0]
            
            # 计算特征重要性（系数绝对值）
            importance = {name: abs(coef) for name, coef in zip(feature_names, coefficients)}
            
            # 按重要性排序
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return importance
            
        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return {}
    
    def save_model(self, model_result: Dict[str, Any], 
                  model_name: str,
                  metadata: Dict = None) -> str:
        """
        保存模型到文件
        
        Args:
            model_result: 训练结果
            model_name: 模型名称
            metadata: 额外元数据
            
        Returns:
            保存的文件路径
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_file = self.model_dir / f"{model_name}_{timestamp}.pkl"
            
            # 准备保存数据 - 保存完整的pipeline包含scaler
            pipeline = model_result['model']
            save_data = {
                'model': pipeline,  # 保存完整的pipeline（包含scaler和模型）
                'scaler': pipeline.named_steps['scaler'] if hasattr(pipeline, 'named_steps') and 'scaler' in pipeline.named_steps else None,
                'feature_names': model_result['feature_names'],
                'feature_importance': model_result['feature_importance'],
                'best_params': model_result['best_params'],
                'metrics': model_result['metrics'],
                'classification_report': model_result['classification_report'],
                'confusion_matrix': model_result['confusion_matrix'],
                'created_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            # 保存模型
            with open(model_file, 'wb') as f:
                pickle.dump(save_data, f)
                
            logger.info(f"模型已保存: {model_file}")
            
            # 保存模型信息到数据库
            self._save_model_info_to_db(model_name, str(model_file), save_data)
            
            return str(model_file)
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise
    
    def _save_model_info_to_db(self, model_name: str, model_path: str, model_data: Dict):
        """
        保存模型信息到数据库
        """
        try:
            # 创建模型信息表
            with self.db_manager.get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_info (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        model_path TEXT NOT NULL,
                        feature_count INTEGER,
                        train_auc REAL,
                        test_auc REAL,
                        best_params TEXT,
                        feature_importance TEXT,
                        created_at TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                
                # 插入模型信息
                conn.execute("""
                    INSERT INTO model_info 
                    (model_name, model_path, feature_count, train_auc, test_auc, 
                     best_params, feature_importance, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    model_path,
                    len(model_data['feature_names']),
                    (model_data['metrics'].get('train_auc') if isinstance(model_data.get('metrics'), dict) else None),
                    (model_data['metrics'].get('test_auc') if isinstance(model_data.get('metrics'), dict) else None),
                    json.dumps(model_data.get('best_params', {})),
                    json.dumps(model_data.get('feature_importance', {})),
                    model_data['created_at'],
                    json.dumps(model_data.get('metadata', {}))
                ))
                
                conn.commit()
                
            logger.info("模型信息已保存到数据库")
            
        except Exception as e:
            logger.error(f"保存模型信息到数据库失败: {e}")
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        从文件加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            模型数据字典
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            logger.info(f"模型已加载: {model_path}")
            return model_data
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def predict(self, model_data: Dict[str, Any], X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        使用模型进行预测
        
        Args:
            model_data: 模型数据
            X: 特征矩阵
            
        Returns:
            预测结果字典
        """
        try:
            model = model_data['model']
            
            # 确保特征顺序一致
            expected_features = model_data['feature_names']
            X_aligned = X[expected_features]
            
            # 预测
            predictions = model.predict(X_aligned)
            probabilities = model.predict_proba(X_aligned)[:, 1]
            
            return {
                'predictions': predictions,
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise

def main():
    """测试训练功能"""
    try:
        db_manager = DatabaseManager()
        trainer = MLTrainer(db_manager)
        logger.info("开始训练30天模型（分类与回归）")
        # 加载30天样本
        df = trainer.load_samples_from_db(period='30d')
        if df.empty:
            logger.error("没有可用的训练数据")
            return
        # 分类任务
        X_cls, y_cls, feature_names_cls = trainer.prepare_features_and_target(df, target_col='label')
        if X_cls.empty:
            logger.error("分类特征数据为空")
            return
        cls_result = trainer.train_model(X_cls, y_cls, use_grid_search=True)
        cls_model_path = trainer.save_model(
            cls_result,
            'logreg_cls_30d',
            metadata={'period': '30d', 'task': 'classification'}
        )
        logger.info(f"分类模型已保存: {cls_model_path}")
        # 回归任务（预测forward_return）
        X_reg, y_reg, feature_names_reg = trainer.prepare_features_and_target(df, target_col='forward_return')
        if not X_reg.empty:
            reg_result = trainer.train_regression_model(X_reg, y_reg, use_grid_search=True)
            reg_model_path = trainer.save_model(
                reg_result,
                'ridge_reg_30d',
                metadata={'period': '30d', 'task': 'regression', 'target': 'forward_return'}
            )
            logger.info(f"回归模型已保存: {reg_model_path}")
        else:
            logger.warning("回归特征数据为空，跳过回归训练")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise

if __name__ == '__main__':
    main()