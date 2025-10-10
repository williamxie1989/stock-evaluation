"""
特征生成器 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """特征生成器"""
    
    def __init__(self):
        self.feature_cache = {}
        self.feature_importance = {}
        logger.info("FeatureGenerator initialized")
    
    def generate_technical_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """生成基础技术指标特征"""
        try:
            # 标准化列名为小写，并将复权列重命名为标准列名
            price_data = price_data.copy()
            price_data.columns = [str(c).lower() for c in price_data.columns]
            rename_map = {}
            for std_col, candidates in {
                "open": ["open", "open_hfq", "open_qfq"],
                "close": ["close", "close_hfq", "close_qfq"],
                "high": ["high", "high_hfq", "high_qfq"],
                "low": ["low", "low_hfq", "low_qfq"],
            }.items():
                if std_col not in price_data.columns:
                    for cand in candidates:
                        if cand in price_data.columns:
                            rename_map[cand] = std_col
                            break
            if rename_map:
                price_data = price_data.rename(columns=rename_map)

            if len(price_data) < 20:  # 降低要求到20天
                logger.warning("数据不足，无法生成完整的技术指标特征")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=price_data.index)
            
            # 基础价格特征
            if 'close' in price_data.columns:
                features['price'] = price_data['close']
                features['price_change'] = price_data['close'].pct_change()
                features['price_ma5'] = price_data['close'].rolling(window=5).mean()
                features['price_ma10'] = price_data['close'].rolling(window=10).mean()
                features['price_ma20'] = price_data['close'].rolling(window=20).mean()
                features['price_ma60'] = price_data['close'].rolling(window=60).mean()
                
                # 价格相对均线位置
                features['price_to_ma5'] = price_data['close'] / features['price_ma5']
                features['price_to_ma20'] = price_data['close'] / features['price_ma20']
                
            # 成交量特征
            if 'volume' in price_data.columns:
                features['volume'] = price_data['volume']
                features['volume_ma5'] = price_data['volume'].rolling(window=5).mean()
                features['volume_ma20'] = price_data['volume'].rolling(window=20).mean()
                features['volume_ratio'] = price_data['volume'] / features['volume_ma20']
            
            # 波动率特征
            if 'close' in price_data.columns:
                returns = price_data['close'].pct_change().dropna()
                features['volatility_5d'] = returns.rolling(window=5).std() * np.sqrt(252)
                features['volatility_20d'] = returns.rolling(window=20).std() * np.sqrt(252)
            
            # 价格范围特征
            if all(col in price_data.columns for col in ['high', 'low', 'close']):
                features['price_range'] = (price_data['high'] - price_data['low']) / price_data['close']
                features['price_position'] = (price_data['close'] - price_data['low']) / (price_data['high'] - price_data['low'] + 1e-8)
            
            # 清除包含NaN的行
            features = features.dropna()
            
            logger.info(f"生成技术指标特征: {len(features.columns)} 个特征, {len(features)} 个样本")
            return features
            
        except Exception as e:
            logger.error(f"技术指标特征生成失败: {e}")
            return pd.DataFrame()
    
    def generate_momentum_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """生成动量特征"""
        try:
            # 标准化列名并重命名
            price_data = price_data.copy()
            price_data.columns = [str(c).lower() for c in price_data.columns]
            rename_map = {}
            for std_col, candidates in {
                "open": ["open", "open_hfq", "open_qfq"],
                "close": ["close", "close_hfq", "close_qfq"],
                "high": ["high", "high_hfq", "high_qfq"],
                "low": ["low", "low_hfq", "low_qfq"],
            }.items():
                if std_col not in price_data.columns:
                    for cand in candidates:
                        if cand in price_data.columns:
                            rename_map[cand] = std_col
                            break
            if rename_map:
                price_data = price_data.rename(columns=rename_map)

            if len(price_data) < 15:  # 降低要求到15天
                logger.warning("数据不足，无法生成完整的动量特征")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=price_data.index)
            
            if 'close' in price_data.columns:
                # RSI
                features['rsi_14'] = self._calculate_rsi(price_data['close'], period=14)
                
                # MACD
                ema12 = price_data['close'].ewm(span=12).mean()
                ema26 = price_data['close'].ewm(span=26).mean()
                features['macd'] = ema12 - ema26
                features['macd_signal'] = features['macd'].ewm(span=9).mean()
                features['macd_histogram'] = features['macd'] - features['macd_signal']
                
                # 价格动量
                features['momentum_5d'] = price_data['close'] / price_data['close'].shift(5) - 1
                features['momentum_10d'] = price_data['close'] / price_data['close'].shift(10) - 1
                features['momentum_20d'] = price_data['close'] / price_data['close'].shift(20) - 1
                
                # ROC (Rate of Change)
                features['roc_10d'] = price_data['close'].pct_change(periods=10) * 100
                
            # 清除包含NaN的行
            features = features.dropna()
            
            logger.info(f"生成动量特征: {len(features.columns)} 个特征, {len(features)} 个样本")
            return features
            
        except Exception as e:
            logger.error(f"动量特征生成失败: {e}")
            return pd.DataFrame()
    
    def generate_volume_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """生成成交量特征"""
        try:
            # 标准化列名并重命名
            price_data = price_data.copy()
            price_data.columns = [str(c).lower() for c in price_data.columns]
            rename_map = {}
            for std_col, candidates in {
                "open": ["open", "open_hfq", "open_qfq"],
                "close": ["close", "close_hfq", "close_qfq"],
                "high": ["high", "high_hfq", "high_qfq"],
                "low": ["low", "low_hfq", "low_qfq"],
            }.items():
                if std_col not in price_data.columns:
                    for cand in candidates:
                        if cand in price_data.columns:
                            rename_map[cand] = std_col
                            break
            if rename_map:
                price_data = price_data.rename(columns=rename_map)

            if len(price_data) < 15:  # 降低要求到15天
                logger.warning("数据不足，无法生成完整的成交量特征")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=price_data.index)
            
            if 'volume' in price_data.columns and 'close' in price_data.columns:
                # OBV (On Balance Volume)
                features['obv'] = self._calculate_obv(price_data)
                
                # VWAP (Volume Weighted Average Price)
                features['vwap'] = self._calculate_vwap(price_data)
                
                # 成交量变化率
                features['volume_change'] = price_data['volume'].pct_change()
                features['volume_roc_5d'] = price_data['volume'].pct_change(periods=5)
                
                # 量价关系
                price_change = price_data['close'].pct_change()
                volume_change = price_data['volume'].pct_change()
                features['price_volume_correlation'] = price_change.rolling(window=10).corr(volume_change)
                
                # 成交量异常检测
                volume_ma20 = price_data['volume'].rolling(window=20).mean()
                features['volume_anomaly'] = (price_data['volume'] - volume_ma20) / volume_ma20
            
            # 清除包含NaN的行
            features = features.dropna()
            
            logger.info(f"生成成交量特征: {len(features.columns)} 个特征, {len(features)} 个样本")
            return features
            
        except Exception as e:
            logger.error(f"成交量特征生成失败: {e}")
            return pd.DataFrame()
    
    def generate_volatility_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """生成波动率特征"""
        try:
            # 标准化列名并重命名
            price_data = price_data.copy()
            price_data.columns = [str(c).lower() for c in price_data.columns]
            rename_map = {}
            for std_col, candidates in {
                "open": ["open", "open_hfq", "open_qfq"],
                "close": ["close", "close_hfq", "close_qfq"],
                "high": ["high", "high_hfq", "high_qfq"],
                "low": ["low", "low_hfq", "low_qfq"],
            }.items():
                if std_col not in price_data.columns:
                    for cand in candidates:
                        if cand in price_data.columns:
                            rename_map[cand] = std_col
                            break
            if rename_map:
                price_data = price_data.rename(columns=rename_map)

            if len(price_data) < 15:  # 降低要求到15天
                logger.warning("数据不足，无法生成完整的波动率特征")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=price_data.index)
            
            if 'close' in price_data.columns:
                returns = price_data['close'].pct_change().dropna()
                
                # ATR (Average True Range)
                if all(col in price_data.columns for col in ['high', 'low', 'close']):
                    features['atr_14'] = self._calculate_atr(price_data, period=14)
                
                # 布林带
                ma20 = price_data['close'].rolling(window=20).mean()
                std20 = price_data['close'].rolling(window=20).std()
                features['bb_upper'] = ma20 + 2 * std20
                features['bb_lower'] = ma20 - 2 * std20
                features['bb_position'] = (price_data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-8)
                
                # 历史波动率
                features['volatility_5d'] = returns.rolling(window=5).std() * np.sqrt(252)
                features['volatility_10d'] = returns.rolling(window=10).std() * np.sqrt(252)
                features['volatility_20d'] = returns.rolling(window=20).std() * np.sqrt(252)
                
                # 波动率变化
                features['volatility_change'] = features['volatility_10d'].pct_change()
                
                # 价格范围
                if all(col in price_data.columns for col in ['high', 'low']):
                    features['daily_range'] = (price_data['high'] - price_data['low']) / price_data['close']
            
            # 清除包含NaN的行
            features = features.dropna()
            
            logger.info(f"生成波动率特征: {len(features.columns)} 个特征, {len(features)} 个样本")
            return features
            
        except Exception as e:
            logger.error(f"波动率特征生成失败: {e}")
            return pd.DataFrame()
    
    def _compute_hash(self, df: pd.DataFrame) -> int:
        """计算数据哈希值，用于缓存键"""
        try:
            subset_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            if not subset_cols:
                return hash(df.shape)
            hashed = pd.util.hash_pandas_object(df[subset_cols], index=True)
            return int(hashed.sum())
        except Exception:
            # 兜底
            return hash(df.values.tobytes())
    
    def generate_all_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """生成所有特征，带缓存"""
        try:
            # 缓存检查
            data_hash = self._compute_hash(price_data)
            cache_key = ("all_features", data_hash)
            if cache_key in self.feature_cache:
                logger.debug("命中特征缓存")
                return self.feature_cache[cache_key].copy()
            if len(price_data) < 20:  # 降低要求到20天
                logger.warning("数据不足，无法生成完整的特征集")
                return pd.DataFrame()
            
            # 生成各类特征
            technical_features = self.generate_technical_features(price_data)
            volume_features = self.generate_volume_features(price_data)
            volatility_features = self.generate_volatility_features(price_data)
            
            # 合并
            result = pd.concat([technical_features, volume_features, volatility_features], axis=1)
            # 去除重复列
            result = result.loc[:, ~result.columns.duplicated()]
            # 智能缺失值处理：只删除缺失值比例过高的行和列
            # 删除缺失值比例超过80%的列
            result = result.loc[:, result.isnull().mean() < 0.8]
            # 删除缺失值比例超过50%的行
            result = result.dropna(thresh=len(result.columns) * 0.5)
            
            # 写入缓存
            self.feature_cache[cache_key] = result.copy()
            
            logger.info(f"生成所有特征: {len(result.columns)} 个特征, {len(result)} 个样本")
            return result
            
        except Exception as e:
            logger.error(f"特征生成失败: {e}")
            return pd.DataFrame()
    
    def select_features(self, features: pd.DataFrame, target: pd.Series, 
                       method: str = 'correlation', top_k: int = 20) -> List[str]:
        """特征选择"""
        try:
            if features.empty or target.empty:
                return []
            
            if method == 'correlation':
                # 相关性选择
                correlations = []
                for col in features.columns:
                    corr = np.corrcoef(features[col].values, target.values)[0, 1]
                    if not np.isnan(corr):
                        correlations.append((col, abs(corr)))
                
                # 按相关性排序
                correlations.sort(key=lambda x: x[1], reverse=True)
                selected_features = [col for col, _ in correlations[:top_k]]
                
            elif method == 'mutual_info':
                # 互信息选择
                from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
                
                if len(np.unique(target)) <= 10:  # 分类任务
                    mi_scores = mutual_info_classif(features, target, random_state=42)
                else:  # 回归任务
                    mi_scores = mutual_info_regression(features, target, random_state=42)
                
                feature_scores = list(zip(features.columns, mi_scores))
                feature_scores.sort(key=lambda x: x[1], reverse=True)
                selected_features = [col for col, _ in feature_scores[:top_k]]
                
            else:
                logger.warning(f"未知的特征选择方法: {method}")
                return list(features.columns)
            
            logger.info(f"特征选择完成: 从 {len(features.columns)} 个特征中选择 {len(selected_features)} 个")
            return selected_features
            
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            return list(features.columns)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"RSI计算失败: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_obv(self, price_data: pd.DataFrame) -> pd.Series:
        """计算OBV"""
        try:
            if 'close' not in price_data.columns or 'volume' not in price_data.columns:
                return pd.Series(index=price_data.index, dtype=float)
            
            obv = pd.Series(index=price_data.index, dtype=float)
            obv.iloc[0] = 0
            
            for i in range(1, len(price_data)):
                if price_data['close'].iloc[i] > price_data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + price_data['volume'].iloc[i]
                elif price_data['close'].iloc[i] < price_data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - price_data['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        except Exception as e:
            logger.error(f"OBV计算失败: {e}")
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_vwap(self, price_data: pd.DataFrame) -> pd.Series:
        """计算VWAP"""
        try:
            if 'close' not in price_data.columns or 'volume' not in price_data.columns:
                return pd.Series(index=price_data.index, dtype=float)
            
            typical_price = price_data['close']  # 简化版本，只使用收盘价
            vwap = (typical_price * price_data['volume']).cumsum() / price_data['volume'].cumsum()
            return vwap
        except Exception as e:
            logger.error(f"VWAP计算失败: {e}")
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        try:
            if not all(col in price_data.columns for col in ['high', 'low', 'close']):
                return pd.Series(index=price_data.index, dtype=float)
            
            high_low = price_data['high'] - price_data['low']
            high_close = np.abs(price_data['high'] - price_data['close'].shift())
            low_close = np.abs(price_data['low'] - price_data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            
            return atr
        except Exception as e:
            logger.error(f"ATR计算失败: {e}")
            return pd.Series(index=price_data.index, dtype=float)
    
    def get_feature_info(self, features: pd.DataFrame) -> Dict[str, Any]:
        """获取特征信息"""
        try:
            info = {
                'feature_count': len(features.columns),
                'sample_count': len(features),
                'features': list(features.columns),
                'data_types': features.dtypes.to_dict(),
                'missing_values': features.isnull().sum().to_dict(),
                'feature_statistics': {}
            }
            
            # 计算每个特征的统计信息
            for col in features.columns:
                col_data = features[col].dropna()
                if len(col_data) > 0:
                    info['feature_statistics'][col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median())
                    }
            
            return info
            
        except Exception as e:
            logger.error(f"获取特征信息失败: {e}")
            return {'error': str(e)}
    
    def reset(self):
        """重置特征生成器"""
        self.feature_cache.clear()
        self.feature_importance.clear()
        logger.info("特征生成器已重置")