"""
增强ML训练器 - MySQL版本
使用UnifiedDatabaseManager进行数据库操作
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# XGBoost支持 - 可选
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import joblib
import json
import os
from src.data.db.unified_database_manager import UnifiedDatabaseManager

logger = logging.getLogger(__name__)


class EnhancedMLTrainer:
    """增强ML训练器 - 支持多种模型和特征工程"""
    
    def __init__(self, db_type: str = "mysql", model_dir: str = "models"):
        # 创建模型保存目录
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.db_manager = UnifiedDatabaseManager(db_type=db_type)
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            },
            'linear_regression': {},
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
        }
        logger.info(f"EnhancedMLTrainer initialized with db_type: {db_type}")
    
    def load_training_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载训练数据"""
        try:
            query = """
                SELECT 
                    date, open, high, low, close, volume,
                    ma5, ma10, ma20, ma30, ma60,
                    rsi, macd, boll_upper, boll_middle, boll_lower,
                    turnover_rate, pe_ratio, pb_ratio, market_cap
                FROM stock_features 
                WHERE symbol = %s AND date BETWEEN %s AND %s
                ORDER BY date ASC
            """
            result = self.db_manager.execute_query(query, (symbol, start_date, end_date))
            if result:
                df = pd.DataFrame(result)
                df['date'] = pd.to_datetime(df['date'])
                return df
            return None
        except Exception as e:
            logger.error(f"加载训练数据失败 {symbol}: {e}")
            return None
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """准备特征"""
        try:
            # 基础价格特征
            data['price_change'] = data['close'].pct_change()
            data['volume_change'] = data['volume'].pct_change()
            data['high_low_ratio'] = data['high'] / data['low']
            data['open_close_ratio'] = data['open'] / data['close'].shift(1)
            
            # 技术指标特征
            data['ma5_ratio'] = data['close'] / data['ma5']
            data['ma10_ratio'] = data['close'] / data['ma10']
            data['ma20_ratio'] = data['close'] / data['ma20']
            data['rsi_normalized'] = data['rsi'] / 100
            
            # 估值特征
            data['pe_normalized'] = np.log1p(data['pe_ratio'])
            data['pb_normalized'] = np.log1p(data['pb_ratio'])
            data['market_cap_log'] = np.log1p(data['market_cap'])
            
            # 目标变量：未来5日收益率
            data['target'] = data['close'].shift(-5) / data['close'] - 1
            
            # 选择特征列
            feature_cols = [
                'price_change', 'volume_change', 'high_low_ratio', 'open_close_ratio',
                'ma5_ratio', 'ma10_ratio', 'ma20_ratio', 'rsi_normalized',
                'pe_normalized', 'pb_normalized', 'market_cap_log',
                'turnover_rate', 'volume'
            ]
            
            # 清理数据
            data_clean = data.dropna()
            
            if len(data_clean) < 50:  # 最少需要50个样本
                logger.warning(f"数据量太少: {len(data_clean)}")
                return pd.DataFrame(), []
            
            return data_clean[feature_cols], feature_cols
            
        except Exception as e:
            logger.error(f"准备特征失败: {e}")
            return pd.DataFrame(), []
    
    def train_model(self, symbol: str, model_type: str = 'random_forest', 
                   start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """训练模型"""
        try:
            # 默认日期范围
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            logger.info(f"开始训练 {symbol} 的 {model_type} 模型，日期范围: {start_date} 到 {end_date}")
            
            # 加载数据
            data = self.load_training_data(symbol, start_date, end_date)
            if data is None or data.empty:
                logger.error(f"无法加载 {symbol} 的训练数据")
                return {'success': False, 'error': '无法加载训练数据'}
            
            # 准备特征
            features_df, feature_names = self.prepare_features(data)
            if features_df.empty:
                logger.error(f"特征准备失败 {symbol}")
                return {'success': False, 'error': '特征准备失败'}
            
            # 分割数据
            X = features_df.drop('target', axis=1)
            y = features_df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 训练模型
            if model_type not in self.model_configs:
                logger.error(f"不支持的模型类型: {model_type}")
                return {'success': False, 'error': f'不支持的模型类型: {model_type}'}
            
            model_config = self.model_configs[model_type]
            
            if model_type == 'random_forest':
                model = RandomForestRegressor(**model_config)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(**model_config)
            elif model_type == 'linear_regression':
                model = LinearRegression(**model_config)
            elif model_type == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    logger.error("XGBoost未安装，无法训练xgboost模型")
                    return {'success': False, 'error': 'XGBoost未安装'}
                model = XGBRegressor(**model_config)
            else:
                return {'success': False, 'error': f'未知的模型类型: {model_type}'}
            
            # 训练
            model.fit(X_train_scaled, y_train)
            
            # 预测
            y_pred = model.predict(X_test_scaled)
            
            # 评估
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 保存模型和标准化器
            model_key = f"{symbol}_{model_type}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_names = feature_names
            
            # 保存到数据库
            self.save_model_to_db(symbol, model_type, model, scaler, feature_names, {
                'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
                'train_samples': len(X_train), 'test_samples': len(X_test)
            })
            
            logger.info(f"模型训练完成 {symbol}_{model_type}: R²={r2:.4f}, RMSE={rmse:.4f}")
            
            return {
                'success': True,
                'symbol': symbol,
                'model_type': model_type,
                'metrics': {
                    'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
                    'train_samples': len(X_train), 'test_samples': len(X_test)
                },
                'feature_names': feature_names,
                'feature_importance': self.get_feature_importance(model, feature_names)
            }
            
        except Exception as e:
            logger.error(f"训练模型失败 {symbol}_{model_type}: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """获取特征重要性"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = {}
                for name, importance in zip(feature_names, model.feature_importances_):
                    importance_dict[name] = float(importance)
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return {}
        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return {}
    
    def predict(self, symbol: str, model_type: str, features: Dict[str, float]) -> Optional[float]:
        """预测"""
        try:
            model_key = f"{symbol}_{model_type}"
            if model_key not in self.models or model_key not in self.scalers:
                logger.error(f"模型未找到: {model_key}")
                return None
            
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            # 准备特征向量
            feature_vector = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                else:
                    feature_vector.append(0.0)  # 默认值
            
            # 标准化和预测
            feature_scaled = scaler.transform([feature_vector])
            prediction = model.predict(feature_scaled)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"预测失败 {symbol}_{model_type}: {e}")
            return None
    
    def save_model_to_db(self, symbol: str, model_type: str, model, scaler, feature_names: List[str], metrics: Dict[str, Any]) -> bool:
        """保存模型到数据库"""
        try:
            # 序列化模型和标准化器
            model_blob = joblib.dumps(model)
            scaler_blob = joblib.dumps(scaler)
            
            # 保存到数据库
            query = """
                INSERT INTO ml_models (symbol, model_type, model_data, scaler_data, 
                                     feature_names, metrics, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                model_data = VALUES(model_data),
                scaler_data = VALUES(scaler_data),
                feature_names = VALUES(feature_names),
                metrics = VALUES(metrics),
                created_at = VALUES(created_at)
            """
            
            self.db_manager.execute_update(query, (
                symbol, model_type, model_blob, scaler_blob,
                json.dumps(feature_names), json.dumps(metrics),
                datetime.now()
            ))
            
            logger.info(f"模型保存到数据库成功: {symbol}_{model_type}")
            return True
            
        except Exception as e:
            logger.error(f"保存模型到数据库失败: {e}")
            return False
    
    def load_model_from_db(self, symbol: str, model_type: str) -> bool:
        """从数据库加载模型"""
        try:
            query = """
                SELECT model_data, scaler_data, feature_names, metrics
                FROM ml_models 
                WHERE symbol = %s AND model_type = %s
                ORDER BY created_at DESC 
                LIMIT 1
            """
            
            result = self.db_manager.execute_query(query, (symbol, model_type))
            if not result:
                logger.warning(f"数据库中未找到模型: {symbol}_{model_type}")
                return False
            
            row = result[0]
            model = joblib.loads(row['model_data'])
            scaler = joblib.loads(row['scaler_data'])
            feature_names = json.loads(row['feature_names'])
            
            model_key = f"{symbol}_{model_type}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_names = feature_names
            
            logger.info(f"从数据库加载模型成功: {symbol}_{model_type}")
            return True
            
        except Exception as e:
            logger.error(f"从数据库加载模型失败: {e}")
            return False
    
    def get_model_performance(self, symbol: str, model_type: str) -> Optional[Dict[str, Any]]:
        """获取模型性能"""
        try:
            query = """
                SELECT metrics, created_at
                FROM ml_models 
                WHERE symbol = %s AND model_type = %s
                ORDER BY created_at DESC 
                LIMIT 1
            """
            
            result = self.db_manager.execute_query(query, (symbol, model_type))
            if result:
                row = result[0]
                return {
                    'metrics': json.loads(row['metrics']),
                    'created_at': row['created_at'].isoformat() if hasattr(row['created_at'], 'isoformat') else str(row['created_at'])
                }
            return None
            
        except Exception as e:
            logger.error(f"获取模型性能失败: {e}")
            return None
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """获取所有模型列表"""
        try:
            query = """
                SELECT symbol, model_type, created_at
                FROM ml_models 
                ORDER BY created_at DESC
            """
            
            result = self.db_manager.execute_query(query)
            return [dict(row) for row in result] if result else []
            
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []

    def train_single_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'logistic',
                           use_optimization: bool = True, optimization_trials: int = 10) -> Dict[str, Any]:
        """使用传入的特征数据训练单个分类模型。

        Args:
            X: 特征 DataFrame。
            y: 标签 Series。
            model_type: 模型类型，支持 'logistic' 或 'xgboost'。
            use_optimization: 是否进行超参数优化。
            optimization_trials: 优化搜索次数/网格规模。
        Returns:
            包含模型、指标、最佳参数等信息的字典。
        """
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        logger.info(f"train_single_model -> model_type={model_type}, opt={use_optimization}")

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )

        best_params = {}
        best_score = None

        if model_type == 'logistic':
            classifier = LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs')
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', classifier)
            ])
            if use_optimization:
                param_grid = {
                    'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'clf__penalty': ['l2']
                }
                grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid.fit(X_train, y_train)
                pipeline = grid.best_estimator_
                best_params = grid.best_params_
                best_score = grid.best_score_
            else:
                pipeline.fit(X_train, y_train)
            model = pipeline

        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError('XGBoost 未安装，无法训练 xgboost 分类模型')

            # 处理类别不平衡，计算 scale_pos_weight
            pos_cnt = (y_train == 1).sum()
            neg_cnt = (y_train == 0).sum()
            scale_pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0

            base_params = dict(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                n_jobs=-1,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
            )
            classifier = XGBClassifier(**base_params)

            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)

            if use_optimization:
                param_grid = {
                    'clf__n_estimators': [200, 400],
                    'clf__learning_rate': [0.03, 0.05, 0.1],
                    'clf__max_depth': [3, 6, 9],
                    'clf__subsample': [0.7, 0.8, 1.0],
                    'clf__colsample_bytree': [0.7, 0.8, 1.0],
                    'clf__gamma': [0, 1],
                    'clf__reg_alpha': [0, 0.1, 0.5],
                    'clf__reg_lambda': [1, 5, 10],
                }
                grid = GridSearchCV(
                    classifier,
                    param_grid,
                    cv=tscv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0,
                )
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                best_params = grid.best_params_
                best_score = grid.best_score_
            else:
                classifier.fit(X_train, y_train)
                model = classifier

            # 交叉验证每折评估并记录日志
            fold_metrics = []
            for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
                model_fold = XGBClassifier(**model.get_params())
                model_fold.fit(X_tr, y_tr)
                y_val_pred = model_fold.predict(X_val)
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                fm = {
                    'fold': fold + 1,
                    'accuracy': accuracy_score(y_val, y_val_pred),
                    'precision': precision_score(y_val, y_val_pred, zero_division=0),
                    'recall': recall_score(y_val, y_val_pred, zero_division=0),
                    'f1': f1_score(y_val, y_val_pred, zero_division=0),
                }
                fold_metrics.append(fm)
                logger.info(
                    f"XGBoost 时序CV 第{fm['fold']}折: acc={fm['accuracy']:.4f}, "
                    f"precision={fm['precision']:.4f}, recall={fm['recall']:.4f}, f1={fm['f1']:.4f}"
                )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, zero_division=0)
        }

        result = {
            'model': model,
            'metrics': metrics
        }
        if best_params:
            result['best_params'] = best_params
        if best_score is not None:
            result['best_score'] = best_score

        logger.info(f"{model_type} 训练完成: 测试准确率={metrics['test_accuracy']:.4f}")
        return result


    def train_single_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'logistic',
                           use_optimization: bool = True, optimization_trials: int = 10) -> Dict[str, Any]:
        """使用传入的特征数据训练单个分类模型。

        Args:
            X: 特征 DataFrame。
            y: 标签 Series。
            model_type: 模型类型，支持 'logistic' 或 'xgboost'。
            use_optimization: 是否进行超参数优化。
            optimization_trials: 优化搜索次数/网格规模。
        Returns:
            包含模型、指标、最佳参数等信息的字典。
        """
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        logger.info(f"train_single_model -> model_type={model_type}, opt={use_optimization}")

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )

        best_params = {}
        best_score = None

        if model_type == 'logistic':
            classifier = LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs')
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', classifier)
            ])
            if use_optimization:
                param_grid = {
                    'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'clf__penalty': ['l2']
                }
                grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid.fit(X_train, y_train)
                pipeline = grid.best_estimator_
                best_params = grid.best_params_
                best_score = grid.best_score_
            else:
                pipeline.fit(X_train, y_train)
            model = pipeline

        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError('XGBoost 未安装，无法训练 xgboost 分类模型')

            # 处理类别不平衡，计算 scale_pos_weight
            pos_cnt = (y_train == 1).sum()
            neg_cnt = (y_train == 0).sum()
            scale_pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0

            base_params = dict(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                n_jobs=-1,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
            )
            classifier = XGBClassifier(**base_params)

            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)

            if use_optimization:
                param_grid = {
                    'clf__n_estimators': [200, 400],
                    'clf__learning_rate': [0.03, 0.05, 0.1],
                    'clf__max_depth': [3, 6, 9],
                    'clf__subsample': [0.7, 0.8, 1.0],
                    'clf__colsample_bytree': [0.7, 0.8, 1.0],
                    'gamma': [0, 1],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [1, 5, 10],
                }
                grid = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=tscv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0,
                )
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                best_params = grid.best_params_
                best_score = grid.best_score_
            else:
                pipeline.fit(X_train, y_train)
                model = pipeline
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, zero_division=0)
        }

        result = {
            'model': model,
            'metrics': metrics
        }
        if best_params:
            result['best_params'] = best_params
        if best_score is not None:
            result['best_score'] = best_score

        logger.info(f"{model_type} 训练完成: 测试准确率={metrics['test_accuracy']:.4f}")
        return result
