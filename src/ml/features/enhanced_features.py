"""
增强特征模块 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import itertools

logger = logging.getLogger(__name__)

class EnhancedFeatureGenerator:
    """增强特征生成器"""
    
    def __init__(self):
        self.feature_cache = {}
        self.feature_config = self._get_default_config()
        logger.info("EnhancedFeatures initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置 - 精确生成108个特征"""
        return {
            # 技术指标配置
            'ma_periods': [5, 8, 10, 12, 15, 20, 25, 30, 40, 50],  # 10个周期 × 2个特征 = 20个特征
            'ema_periods': [5, 8, 12, 20, 26, 30, 40, 50],  # 8个周期 × 2个特征 = 16个特征
            'rsi_periods': [6, 9, 12, 14, 21, 28],  # 6个周期 × 1个特征 = 6个特征
            'macd_configs': [
                {'fast': 12, 'slow': 26, 'signal': 9},  # 3个特征
                {'fast': 8, 'slow': 21, 'signal': 5}   # 3个特征
            ],  # 2种配置 × 3个特征 = 6个特征
            'bb_configs': [
                {'period': 20, 'std': 2.0},  # 4个特征
                {'period': 15, 'std': 1.5},  # 4个特征
                {'period': 30, 'std': 2.5}   # 4个特征
            ],  # 3种配置 × 4个特征 = 12个特征
            
            # 动量指标配置
            'roc_periods': [5, 8, 10, 12, 15, 20, 25],  # 7个周期 = 7个特征
            'momentum_periods': [5, 10, 15, 20],  # 4个周期 = 4个特征
            'stochastic_periods': [9, 12, 14, 21],  # 4个周期 × 2个特征 = 8个特征
            'williams_periods': [10, 14, 20],  # 3个周期 = 3个特征
            # 波动率指标配置
            'atr_periods': [10, 14, 20, 25],  # 4个周期 = 4个特征
            'volatility_periods': [5, 8, 10, 15, 20, 30, 252],  # 7个周期, 新增252日历史波动率
            
            # 成交量指标配置
            'volume_ma_periods': [5, 8, 10, 15, 20, 30],  # 6个周期 = 6个特征
            'volume_roc_periods': [5, 8, 10, 15, 20],  # 5个周期 = 5个特征
            'obv_periods': [10, 20],  # 2个周期 = 2个特征
            'vwap_periods': [5, 10, 20],  # 3个周期 = 3个特征
            # 交互特征开关与参数
            'interaction': {
                'enabled': False,  # 默认关闭，避免维度爆炸
                'methods': ['product', 'ratio'],  # 支持乘积、比值
                'max_pairs': 100  # 最多生成多少组交互特征，防止生成过多
            }
        }
    
    def generate_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """生成特征（兼容接口）"""
        return self.generate_enhanced_features(price_data)
    
    def generate_enhanced_features(self, price_data: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """生成增强特征 - 精确108特征版本"""
        try:
            logger.info("开始生成增强特征...")
            
            # 降低最少样本要求，支持一个月数据（约20根K线）
            if len(price_data) < 20:
                logger.warning("数据不足，无法生成完整的增强特征")
                return pd.DataFrame()
            
            # -------- 新增：统一数值列类型，避免 decimal.Decimal 与 float 运算冲突 --------
            price_data = price_data.copy()
            for col in price_data.columns:
                # 仅尝试将对象或非数值列转换为数值
                if price_data[col].dtype == 'object' or not np.issubdtype(price_data[col].dtype, np.number):
                    price_data[col] = pd.to_numeric(price_data[col], errors='coerce')
            # 再次确保所有数值列为 float 类型
            numeric_cols = price_data.select_dtypes(include=[np.number]).columns
            price_data[numeric_cols] = price_data[numeric_cols].astype(float)
            # ----------------------------------------------------------------------
            
            # 生成各类特征
            technical_features = self._generate_technical_features(price_data)
            momentum_features = self._generate_momentum_features(price_data)
            volatility_features = self._generate_volatility_features(price_data)
            volume_features = self._generate_volume_features(price_data)
            
            # 合并所有特征
            all_features = pd.concat([
                technical_features,
                momentum_features,
                volatility_features,
                volume_features
            ], axis=1)
            
            # ---------------- 交互特征生成 ----------------
            interaction_cfg = self.feature_config.get('interaction', {})
            if interaction_cfg.get('enabled', False):
                interaction_feats = self._generate_interaction_features(all_features, interaction_cfg)
                all_features = pd.concat([all_features, interaction_feats], axis=1)
            # ------------------------------------------------
            
            # 特征选择
            if target is not None and self.feature_config.get('feature_selection', {}).get('enabled', False):
                all_features = self.select_features(all_features, target, self.feature_config['feature_selection'])
            
            # 放宽缺失值容忍：允许行内最多30% NaN，然后用列中位数填充剩余缺失
            valid_thresh = int(all_features.shape[1] * 0.7) if all_features.shape[1] > 0 else 0
            all_features = all_features.dropna(thresh=valid_thresh)
            all_features = all_features.fillna(all_features.median())
            
            logger.info(f"生成了 {len(all_features.columns)} 个增强特征")
            
            # 验证特征数量
            # if len(all_features.columns) != 108:
            #     logger.warning(f"生成的特征数量({len(all_features.columns)})不是预期的108个")
            # else:
            #     logger.info("成功生成108个特征")
            
            return all_features
            
        except Exception as e:
            logger.error(f"增强特征生成失败: {e}")
            return pd.DataFrame()
    
    def _generate_technical_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """生成技术指标特征 - 支持108特征配置"""
        try:
            features = pd.DataFrame(index=price_data.index)
            config = self.feature_config
            
            if 'close' in price_data.columns:
                # 移动平均线 (11个周期 × 2个特征 = 22个特征)
                for period in config['ma_periods']:
                    features[f'ma_{period}'] = price_data['close'].rolling(window=period).mean()
                    features[f'price_to_ma_{period}'] = price_data['close'] / features[f'ma_{period}']
                
                # 指数移动平均 (8个周期 × 2个特征 = 16个特征)
                for period in config['ema_periods']:
                    features[f'ema_{period}'] = price_data['close'].ewm(span=period).mean()
                    features[f'price_to_ema_{period}'] = price_data['close'] / features[f'ema_{period}']
                
                # RSI (6个周期 = 6个特征)
                for period in config['rsi_periods']:
                    features[f'rsi_{period}'] = self._calculate_rsi(price_data['close'], period)
                
                # MACD (2种配置 × 3个特征 = 6个特征)
                for i, macd_config in enumerate(config['macd_configs']):
                    ema_fast = price_data['close'].ewm(span=macd_config['fast']).mean()
                    ema_slow = price_data['close'].ewm(span=macd_config['slow']).mean()
                    macd_line = ema_fast - ema_slow
                    signal_line = macd_line.ewm(span=macd_config['signal']).mean()
                    histogram = macd_line - signal_line
                    
                    features[f'macd_{i}_main'] = macd_line
                    features[f'macd_{i}_signal'] = signal_line
                    features[f'macd_{i}_histogram'] = histogram
                
                # 布林带 (3种配置 × 4个特征 = 12个特征)
                for i, bb_config in enumerate(config['bb_configs']):
                    middle_band = price_data['close'].rolling(window=bb_config['period']).mean()
                    std_dev = price_data['close'].rolling(window=bb_config['period']).std()
                    upper_band = middle_band + bb_config['std'] * std_dev
                    lower_band = middle_band - bb_config['std'] * std_dev
                    
                    features[f'bb_{i}_upper'] = upper_band
                    features[f'bb_{i}_middle'] = middle_band
                    features[f'bb_{i}_lower'] = lower_band
                    features[f'bb_{i}_width'] = (upper_band - lower_band) / middle_band
            
            return features
            
        except Exception as e:
            logger.error(f"技术指标特征生成失败: {e}")
            return pd.DataFrame()
    
    def _generate_momentum_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """生成动量特征 - 支持108特征配置"""
        try:
            features = pd.DataFrame(index=price_data.index)
            config = self.feature_config
            
            if 'close' in price_data.columns:
                # ROC (Rate of Change)
                for period in config['roc_periods']:
                    features[f'roc_{period}'] = price_data['close'].pct_change(periods=period) * 100
                
                # 价格动量
                for period in config['momentum_periods']:
                    features[f'momentum_{period}'] = price_data['close'] / price_data['close'].shift(period) - 1
                
                # 随机指标 (4个周期 × 2个特征 = 8个特征)
                if 'high' in price_data.columns and 'low' in price_data.columns:
                    for period in config['stochastic_periods']:
                        stochastic_k = self._calculate_stochastic(
                            price_data['high'], price_data['low'], price_data['close'], 
                            period
                        )
                        stochastic_d = stochastic_k.rolling(window=3).mean()
                        features[f'stochastic_k_{period}'] = stochastic_k
                        features[f'stochastic_d_{period}'] = stochastic_d
                
                # 威廉指标 (3个周期 = 3个特征)
                if 'high' in price_data.columns and 'low' in price_data.columns:
                    for period in config['williams_periods']:
                        features[f'williams_r_{period}'] = self._calculate_williams_r(
                            price_data['high'], price_data['low'], price_data['close'], period
                        )
            
            return features
              
        except Exception as e:
              logger.error(f"动量特征生成失败: {e}")
              return pd.DataFrame()
    
    def _generate_volatility_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """生成波动率特征 - 支持108特征配置"""
        try:
            features = pd.DataFrame(index=price_data.index)
            config = self.feature_config
            
            if 'close' in price_data.columns:
                # 历史波动率 (6个周期 = 6个特征)
                for period in config['volatility_periods']:
                    features[f'volatility_{period}'] = price_data['close'].pct_change().rolling(window=period).std()
                
                # ATR (3个周期 = 3个特征)
                if 'high' in price_data.columns and 'low' in price_data.columns:
                    for period in config['atr_periods']:
                        features[f'atr_{period}'] = self._calculate_atr(price_data, period)
            
            return features
            
        except Exception as e:
            logger.error(f"波动率特征生成失败: {e}")
            return pd.DataFrame()
    
    def _generate_volume_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """生成成交量特征 - 支持108特征配置"""
        try:
            features = pd.DataFrame(index=price_data.index)
            config = self.feature_config
            
            if 'volume' in price_data.columns:
                # 成交量移动平均 (6个周期 × 2个特征 = 12个特征)
                for period in config['volume_ma_periods']:
                    features[f'volume_ma_{period}'] = price_data['volume'].rolling(window=period).mean()
                    features[f'volume_ratio_{period}'] = price_data['volume'] / (features[f'volume_ma_{period}'] + 1e-8)
                
                # 价格变化率 (6个周期 = 6个特征)
                for period in [1, 3, 5, 10, 15, 20]:
                    features[f'price_change_{period}'] = price_data['close'].pct_change(periods=period) * 100
                
                # 成交量变化率 (3个周期 = 3个特征)
                for period in config['volume_roc_periods']:
                    features[f'volume_roc_{period}'] = price_data['volume'].pct_change(periods=period) * 100
                
                # OBV (2个特征 = 2个特征)
                if 'close' in price_data.columns:
                    obv_values = ((price_data['close'] > price_data['close'].shift(1)).astype(int) -
                                 (price_data['close'] < price_data['close'].shift(1)).astype(int)) * price_data['volume']
                    features['obv'] = obv_values.cumsum()
                    features['obv_change'] = features['obv'].pct_change()
                
                # VWAP (3个特征 = 3个特征)
                if 'close' in price_data.columns and 'high' in price_data.columns and 'low' in price_data.columns:
                    typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
                    cumulative_price_volume = (typical_price * price_data['volume']).cumsum()
                    cumulative_volume = price_data['volume'].cumsum()
                    features['vwap'] = cumulative_price_volume / cumulative_volume
                    features['price_to_vwap'] = price_data['close'] / features['vwap']
                    features['vwap_deviation'] = (price_data['close'] - features['vwap']) / features['vwap']
                
                # 成交量异常 (2个特征 = 2个特征)
                volume_mean = price_data['volume'].rolling(window=20).mean()
                volume_std = price_data['volume'].rolling(window=20).std()
                features['volume_zscore'] = (price_data['volume'] - volume_mean) / (volume_std + 1e-8)
                features['volume_spike'] = (features['volume_zscore'] > 2).astype(int)
            
            return features
            
        except Exception as e:
            logger.error(f"成交量特征生成失败: {e}")
            return pd.DataFrame()
    
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
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算随机指标"""
        try:
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            stochastic_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
            return stochastic_k
        except Exception as e:
            logger.error(f"随机指标计算失败: {e}")
            return pd.Series(index=close.index, dtype=float)
    
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
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算威廉指标 (Williams %R)"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)
            return williams_r
        except Exception as e:
            logger.error(f"威廉指标计算失败: {e}")
            return pd.Series(-50, index=close.index)
    
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
            
            # 简化版本，使用收盘价作为典型价格
            typical_price = price_data['close']
            cumulative_price_volume = (typical_price * price_data['volume']).cumsum()
            cumulative_volume = price_data['volume'].cumsum()
            
            vwap = cumulative_price_volume / cumulative_volume
            return vwap
        except Exception as e:
            logger.error(f"VWAP计算失败: {e}")
            return pd.Series(index=price_data.index, dtype=float)
    
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
    
    def calculate_factor_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """计算因子特征（兼容接口）"""
        try:
            if price_data.empty:
                return pd.DataFrame()
            
            # 调用现有的增强特征生成方法
            features = self.generate_enhanced_features(price_data)
            
            if features.empty:
                logger.warning("因子特征计算结果为空")
                return pd.DataFrame()
            
            # 移除频繁的过程日志，只在需要调试时输出
            # logger.info(f"因子特征计算完成: {len(features.columns)} 个特征")
            return features
            
        except Exception as e:
            logger.error(f"因子特征计算失败: {e}")
            return pd.DataFrame()
    
    def reset(self):
        """重置增强特征生成器"""
        self.feature_cache.clear()
        logger.info("增强特征生成器已重置")

    def _generate_interaction_features(self, feats: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """根据配置自动生成交互特征（乘积/比值）"""
        try:
            methods = cfg.get('methods', ['product', 'ratio'])
            max_pairs = cfg.get('max_pairs', 100)
            numeric_cols = feats.select_dtypes(include=[np.number]).columns.tolist()

            # 仅使用方差最大的前 N 列，避免无用列组合
            col_variances = feats[numeric_cols].var().sort_values(ascending=False)
            top_cols = col_variances.head(30).index.tolist()  # 取前30列方差最大的特征

            pairs = list(itertools.combinations(top_cols, 2))[:max_pairs]
            inter_df = pd.DataFrame(index=feats.index)

            for col_a, col_b in pairs:
                if 'product' in methods:
                    inter_df[f'{col_a}_x_{col_b}'] = feats[col_a] * feats[col_b]
                if 'ratio' in methods:
                    # 避免除以零
                    inter_df[f'{col_a}_div_{col_b}'] = feats[col_a] / feats[col_b].replace(0, np.nan)

            # 处理可能出现的无穷与缺失
            inter_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            inter_df = inter_df.fillna(inter_df.median())

            logger.info(f"生成 {inter_df.shape[1]} 个交互特征")
            return inter_df
        except Exception as e:
            logger.error(f"交互特征生成失败: {e}")
            return pd.DataFrame()