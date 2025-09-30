"""
优化的增强数据提供器 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataQuality:
    """数据质量指标"""
    completeness: float  # 完整性
    accuracy: float    # 准确性
    timeliness: float  # 及时性
    consistency: float # 一致性
    overall_score: float # 综合评分

class OptimizedEnhancedDataProvider:
    """优化的增强数据提供器"""
    
    def __init__(self, primary_provider=None, fallback_providers=None):
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or []
        self.cache = {}  # 数据缓存
        self.quality_cache = {}  # 质量缓存
        logger.info("OptimizedEnhancedDataProvider initialized")
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取股票历史数据"""
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        # 检查缓存
        if cache_key in self.cache:
            logger.info(f"从缓存获取数据: {symbol}")
            return self.cache[cache_key].copy()
        
        # 尝试主要数据源
        if self.primary_provider:
            try:
                data = self.primary_provider.get_stock_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    # 评估数据质量
                    quality = self._assess_data_quality(data)
                    if quality.overall_score >= 0.8:
                        self.cache[cache_key] = data.copy()
                        self.quality_cache[cache_key] = quality
                        logger.info(f"从主要数据源获取数据成功: {symbol}, 质量评分: {quality.overall_score:.2f}")
                        return data
            except Exception as e:
                logger.warning(f"主要数据源获取失败: {symbol}, 错误: {e}")
        
        # 尝试备用数据源
        for provider in self.fallback_providers:
            try:
                data = provider.get_stock_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    quality = self._assess_data_quality(data)
                    if quality.overall_score >= 0.6:
                        self.cache[cache_key] = data.copy()
                        self.quality_cache[cache_key] = quality
                        logger.info(f"从备用数据源获取数据成功: {symbol}, 质量评分: {quality.overall_score:.2f}")
                        return data
            except Exception as e:
                logger.warning(f"备用数据源获取失败: {symbol}, 错误: {e}")
        
        logger.error(f"所有数据源获取失败: {symbol}")
        return None
    
    def get_realtime_data(self, symbols: List[str]) -> Optional[Dict[str, Dict[str, Any]]]:
        """获取实时数据"""
        cache_key = f"realtime_{','.join(sorted(symbols))}"
        
        # 检查缓存（缓存5分钟）
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data.get('timestamp', datetime.min) < timedelta(minutes=5):
                logger.info(f"从缓存获取实时数据: {len(symbols)} 只股票")
                return cached_data['data']
        
        # 尝试主要数据源
        if self.primary_provider:
            try:
                data = self.primary_provider.get_realtime_data(symbols)
                if data:
                    self.cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    logger.info(f"从主要数据源获取实时数据成功: {len(symbols)} 只股票")
                    return data
            except Exception as e:
                logger.warning(f"主要数据源获取实时数据失败: {e}")
        
        # 尝试备用数据源
        for provider in self.fallback_providers:
            try:
                data = provider.get_realtime_data(symbols)
                if data:
                    self.cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    logger.info(f"从备用数据源获取实时数据成功: {len(symbols)} 只股票")
                    return data
            except Exception as e:
                logger.warning(f"备用数据源获取实时数据失败: {e}")
        
        logger.error("所有数据源获取实时数据失败")
        return None
    
    def get_financial_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取财务数据"""
        cache_key = f"financial_{symbol}"
        
        # 检查缓存
        if cache_key in self.cache:
            logger.info(f"从缓存获取财务数据: {symbol}")
            return self.cache[cache_key].copy()
        
        # 尝试主要数据源
        if self.primary_provider:
            try:
                data = self.primary_provider.get_financial_data(symbol)
                if data:
                    self.cache[cache_key] = data.copy()
                    logger.info(f"从主要数据源获取财务数据成功: {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"主要数据源获取财务数据失败: {symbol}, 错误: {e}")
        
        # 尝试备用数据源
        for provider in self.fallback_providers:
            try:
                data = provider.get_financial_data(symbol)
                if data:
                    self.cache[cache_key] = data.copy()
                    logger.info(f"从备用数据源获取财务数据成功: {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"备用数据源获取财务数据失败: {symbol}, 错误: {e}")
        
        logger.error(f"所有数据源获取财务数据失败: {symbol}")
        return None
    
    def get_market_overview(self) -> Optional[Dict[str, Any]]:
        """获取市场概览"""
        cache_key = "market_overview"
        
        # 检查缓存（缓存15分钟）
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data.get('timestamp', datetime.min) < timedelta(minutes=15):
                logger.info("从缓存获取市场概览")
                return cached_data['data']
        
        # 尝试主要数据源
        if self.primary_provider:
            try:
                data = self.primary_provider.get_market_overview()
                if data:
                    self.cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    logger.info("从主要数据源获取市场概览成功")
                    return data
            except Exception as e:
                logger.warning(f"主要数据源获取市场概览失败: {e}")
        
        # 尝试备用数据源
        for provider in self.fallback_providers:
            try:
                data = provider.get_market_overview()
                if data:
                    self.cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    logger.info("从备用数据源获取市场概览成功")
                    return data
            except Exception as e:
                logger.warning(f"备用数据源获取市场概览失败: {e}")
        
        logger.error("所有数据源获取市场概览失败")
        return None
    
    def get_sector_performance(self) -> Optional[Dict[str, float]]:
        """获取板块表现"""
        cache_key = "sector_performance"
        
        # 检查缓存（缓存30分钟）
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data.get('timestamp', datetime.min) < timedelta(minutes=30):
                logger.info("从缓存获取板块表现")
                return cached_data['data']
        
        # 尝试主要数据源
        if self.primary_provider:
            try:
                data = self.primary_provider.get_sector_performance()
                if data:
                    self.cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    logger.info("从主要数据源获取板块表现成功")
                    return data
            except Exception as e:
                logger.warning(f"主要数据源获取板块表现失败: {e}")
        
        # 尝试备用数据源
        for provider in self.fallback_providers:
            try:
                data = provider.get_sector_performance()
                if data:
                    self.cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    logger.info("从备用数据源获取板块表现成功")
                    return data
            except Exception as e:
                logger.warning(f"备用数据源获取板块表现失败: {e}")
        
        logger.error("所有数据源获取板块表现失败")
        return None
    
    def get_all_stock_list(self) -> Optional[List[Dict[str, Any]]]:
        """获取全市场股票列表"""
        cache_key = "all_stock_list"
        
        # 检查缓存（缓存60分钟）
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data.get('timestamp', datetime.min) < timedelta(minutes=60):
                logger.info("从缓存获取全市场股票列表")
                return cached_data['data']
        
        # 尝试主要数据源
        if self.primary_provider:
            try:
                data = self.primary_provider.get_all_stock_list()
                if data:
                    self.cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    logger.info(f"从主要数据源获取全市场股票列表成功: {len(data)} 只股票")
                    return data
            except Exception as e:
                logger.warning(f"主要数据源获取全市场股票列表失败: {e}")
        
        # 尝试备用数据源
        for provider in self.fallback_providers:
            try:
                data = provider.get_all_stock_list()
                if data:
                    self.cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    logger.info(f"从备用数据源获取全市场股票列表成功: {len(data)} 只股票")
                    return data
            except Exception as e:
                logger.warning(f"备用数据源获取全市场股票列表失败: {e}")
        
        logger.error("所有数据源获取全市场股票列表失败")
        return None
    
    def _assess_data_quality(self, data: pd.DataFrame) -> DataQuality:
        """评估数据质量"""
        try:
            # 完整性检查
            total_points = len(data)
            missing_points = data.isnull().sum().sum()
            completeness = 1 - (missing_points / (total_points * len(data.columns)))
            
            # 准确性检查（检查异常值）
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outlier_count = 0
            total_numeric_points = 0
            
            for col in numeric_cols:
                if col in ['open', 'high', 'low', 'close']:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                    outlier_count += outliers
                    total_numeric_points += len(data[col])
            
            accuracy = 1 - (outlier_count / total_numeric_points) if total_numeric_points > 0 else 1
            
            # 及时性检查（检查最新数据日期）
            if 'date' in data.columns:
                latest_date = pd.to_datetime(data['date'].max())
                days_diff = (datetime.now() - latest_date).days
                timeliness = max(0, 1 - days_diff / 30)  # 30天内的数据认为是及时的
            else:
                timeliness = 0.8
            
            # 一致性检查（检查价格逻辑）
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                price_logic_errors = 0
                total_checks = len(data)
                
                # 检查最高价是否低于收盘价
                price_logic_errors += (data['high'] < data['close']).sum()
                # 检查最低价是否高于收盘价
                price_logic_errors += (data['low'] > data['close']).sum()
                # 检查开盘价是否在高低价范围内
                price_logic_errors += ((data['open'] < data['low']) | (data['open'] > data['high'])).sum()
                
                consistency = 1 - (price_logic_errors / (total_checks * 3))
            else:
                consistency = 0.9
            
            # 综合评分
            overall_score = np.mean([completeness, accuracy, timeliness, consistency])
            
            return DataQuality(
                completeness=completeness,
                accuracy=accuracy,
                timeliness=timeliness,
                consistency=consistency,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"数据质量评估失败: {e}")
            return DataQuality(completeness=0, accuracy=0, timeliness=0, consistency=0, overall_score=0)
    
    def get_data_quality(self, symbol: str, start_date: str, end_date: str) -> Optional[DataQuality]:
        """获取数据质量"""
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        if cache_key in self.quality_cache:
            return self.quality_cache[cache_key]
        
        # 获取数据并评估质量
        data = self.get_stock_data(symbol, start_date, end_date)
        if data is not None:
            quality = self._assess_data_quality(data)
            self.quality_cache[cache_key] = quality
            return quality
        
        return None
    
    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
        self.quality_cache.clear()
        logger.info("缓存已清除")
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            if self.primary_provider:
                return self.primary_provider.test_connection()
            
            for provider in self.fallback_providers:
                if provider.test_connection():
                    return True
            
            return False
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False