"""
增强实时数据提供器 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from ..providers.realtime_provider import EnhancedRealtimeProvider

logger = logging.getLogger(__name__)

class EnhancedRealtimeProviderV2(EnhancedRealtimeProvider):
    """增强实时数据提供器V2 - 继承自基础实时提供器"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        logger.info(f"EnhancedRealtimeProviderV2 initialized with max_retries={max_retries}, retry_delay={retry_delay}s")
    
    def get_realtime_data_enhanced(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取增强的实时数据"""
        basic_data = self.get_realtime_data(symbol)
        if basic_data is None:
            return None
        
        # 添加增强功能
        enhanced_data = basic_data.copy()
        
        # 添加时间戳
        enhanced_data['timestamp'] = datetime.now()
        
        # 添加数据质量指标
        enhanced_data['data_quality'] = self._assess_data_quality(basic_data)
        
        # 添加市场状态
        enhanced_data['market_status'] = self._get_market_status()
        
        logger.info(f"获取增强实时数据: {symbol}, 质量: {enhanced_data['data_quality']}")
        return enhanced_data
    
    def get_batch_realtime_data_enhanced(self, symbols: List[str]) -> Dict[str, Any]:
        """批量获取增强的实时数据"""
        results = {}
        for symbol in symbols:
            try:
                data = self.get_realtime_data_enhanced(symbol)
                if data is not None:
                    results[symbol] = data
            except Exception as e:
                logger.error(f"获取增强实时数据失败 {symbol}: {e}")
        
        logger.info(f"批量获取增强实时数据完成: {len(results)}/{len(symbols)} 成功")
        return results
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> str:
        """评估数据质量"""
        try:
            # 检查价格合理性
            price = data.get('price', 0)
            if price <= 0 or price > 10000:  # 价格异常
                return "poor"
            
            # 检查成交量
            volume = data.get('volume', 0)
            if volume < 0:  # 成交量为负
                return "poor"
            
            # 检查涨跌幅
            change = abs(data.get('change', 0))
            if change > 20:  # 涨跌幅超过20%
                return "questionable"
            
            return "good"
        except Exception:
            return "unknown"
    
    def _get_market_status(self) -> str:
        """获取市场状态"""
        current_time = datetime.now()
        
        # 简化的市场状态判断
        if current_time.weekday() >= 5:  # 周末
            return "closed"
        
        # 交易时间判断（简化版）
        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=15, minute=0, second=0, microsecond=0)
        
        if market_open <= current_time <= market_close:
            return "open"
        else:
            return "closed"
    
    def get_market_overview(self) -> Optional[Dict[str, Any]]:
        """获取市场概览"""
        try:
            # 获取主要指数
            indices = ['sh000001', 'sz399001', 'sz399006']  # 上证指数、深证成指、创业板指
            
            overview = {
                'timestamp': datetime.now(),
                'indices': {}
            }
            
            for index in indices:
                data = self.get_realtime_data(index)
                if data:
                    overview['indices'][index] = {
                        'price': data['price'],
                        'change': data['change'],
                        'volume': data['volume']
                    }
            
            # 计算市场情绪（简化版）
            if overview['indices']:
                changes = [info['change'] for info in overview['indices'].values()]
                avg_change = np.mean(changes)
                
                if avg_change > 1:
                    sentiment = "bullish"
                elif avg_change < -1:
                    sentiment = "bearish"
                else:
                    sentiment = "neutral"
                
                overview['market_sentiment'] = sentiment
                overview['average_change'] = avg_change
            
            logger.info(f"获取市场概览完成，情绪: {overview.get('market_sentiment', 'unknown')}")
            return overview
            
        except Exception as e:
            logger.error(f"获取市场概览失败: {e}")
            return None
    
    def get_sector_performance(self) -> Optional[Dict[str, Any]]:
        """获取板块表现"""
        try:
            # 主要板块代码（简化版）
            sectors = {
                'technology': 'BK0737',  # 科技板块
                'finance': 'BK0475',     # 金融板块
                'healthcare': 'BK0727',  # 医疗板块
                'consumer': 'BK0738',    # 消费板块
                'energy': 'BK0739'       # 能源板块
            }
            
            performance = {
                'timestamp': datetime.now(),
                'sectors': {}
            }
            
            for sector_name, sector_code in sectors.items():
                data = self.get_realtime_data(sector_code)
                if data:
                    performance['sectors'][sector_name] = {
                        'change': data['change'],
                        'volume': data['volume'],
                        'price': data['price']
                    }
            
            # 找出表现最好和最差的板块
            if performance['sectors']:
                changes = [(name, info['change']) for name, info in performance['sectors'].items()]
                changes.sort(key=lambda x: x[1])
                
                performance['best_performer'] = changes[-1][0] if changes else None
                performance['worst_performer'] = changes[0][0] if changes else None
            
            logger.info(f"获取板块表现完成，最佳: {performance.get('best_performer')}, 最差: {performance.get('worst_performer')}")
            return performance
            
        except Exception as e:
            logger.error(f"获取板块表现失败: {e}")
            return None