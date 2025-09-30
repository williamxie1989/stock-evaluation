"""
股票状态过滤器 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class StockStatusFilter:
    """股票状态过滤器"""
    
    def __init__(self):
        self.trading_status_cache = {}  # 交易状态缓存
        self.quality_cache = {}  # 质量缓存
        logger.info("StockStatusFilter initialized")
    
    def filter_tradable_stocks(self, symbols: List[str], current_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """过滤可交易股票"""
        tradable_stocks = []
        
        for symbol in symbols:
            if symbol in current_data:
                data = current_data[symbol]
                if self._is_tradable(symbol, data):
                    tradable_stocks.append(symbol)
        
        logger.info(f"可交易股票过滤: 从 {len(symbols)} 只中筛选出 {len(tradable_stocks)} 只可交易股票")
        return tradable_stocks
    
    def _is_tradable(self, symbol: str, data: Dict[str, Any]) -> bool:
        """检查股票是否可交易"""
        try:
            # 检查价格
            current_price = data.get('current_price', 0)
            if current_price <= 0:
                logger.debug(f"{symbol} 价格无效: {current_price}")
                return False
            
            # 检查涨跌幅
            change_pct = data.get('change_pct', 0)
            if abs(change_pct) > 0.11:  # 涨跌幅超过11%
                logger.debug(f"{symbol} 涨跌幅异常: {change_pct:.2%}")
                return False
            
            # 检查成交量
            volume = data.get('volume', 0)
            if volume <= 0:
                logger.debug(f"{symbol} 无成交量: {volume}")
                return False
            
            # 检查是否停牌
            if data.get('is_suspended', False):
                logger.debug(f"{symbol} 已停牌")
                return False
            
            # 检查是否退市
            if data.get('is_delisted', False):
                logger.debug(f"{symbol} 已退市")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查 {symbol} 交易状态失败: {e}")
            return False
    
    def filter_quality_stocks(self, symbols: List[str], financial_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """过滤优质股票"""
        quality_stocks = []
        
        for symbol in symbols:
            if symbol in financial_data:
                data = financial_data[symbol]
                if self._is_quality_stock(symbol, data):
                    quality_stocks.append(symbol)
        
        logger.info(f"优质股票过滤: 从 {len(symbols)} 只中筛选出 {len(quality_stocks)} 只优质股票")
        return quality_stocks
    
    def _is_quality_stock(self, symbol: str, financial_data: Dict[str, Any]) -> bool:
        """检查是否为优质股票"""
        try:
            # 检查市值
            market_cap = financial_data.get('market_cap', 0)
            if market_cap < 5e8:  # 市值小于5亿
                logger.debug(f"{symbol} 市值过小: {market_cap}")
                return False
            
            # 检查市盈率
            pe_ratio = financial_data.get('pe_ratio', 0)
            if pe_ratio <= 0 or pe_ratio > 100:  # 市盈率异常
                logger.debug(f"{symbol} 市盈率异常: {pe_ratio}")
                return False
            
            # 检查市净率
            pb_ratio = financial_data.get('pb_ratio', 0)
            if pb_ratio <= 0 or pb_ratio > 10:  # 市净率异常
                logger.debug(f"{symbol} 市净率异常: {pb_ratio}")
                return False
            
            # 检查ROE
            roe = financial_data.get('roe', 0)
            if roe < 0.05:  # ROE小于5%
                logger.debug(f"{symbol} ROE过低: {roe:.2%}")
                return False
            
            # 检查负债率
            debt_ratio = financial_data.get('debt_ratio', 1)
            if debt_ratio > 0.8:  # 负债率超过80%
                logger.debug(f"{symbol} 负债率过高: {debt_ratio:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查 {symbol} 质量失败: {e}")
            return False
    
    def filter_by_technical(self, symbols: List[str], technical_data: Dict[str, pd.DataFrame]) -> List[str]:
        """基于技术指标过滤"""
        filtered_stocks = []
        
        for symbol in symbols:
            if symbol in technical_data:
                data = technical_data[symbol]
                if self._pass_technical_filters(symbol, data):
                    filtered_stocks.append(symbol)
        
        logger.info(f"技术指标过滤: 从 {len(symbols)} 只中筛选出 {len(filtered_stocks)} 只符合技术条件的股票")
        return filtered_stocks
    
    def _pass_technical_filters(self, symbol: str, data: pd.DataFrame) -> bool:
        """通过技术指标过滤"""
        try:
            if len(data) < 20:  # 数据不足
                return False
            
            # 获取最新数据
            latest = data.iloc[-1]
            
            # 检查价格趋势（20日均线）
            if 'ma20' in data.columns:
                ma20 = latest['ma20']
                current_price = latest['close']
                if current_price < ma20 * 0.95:  # 价格低于20日均线5%以上
                    logger.debug(f"{symbol} 价格低于20日均线")
                    return False
            
            # 检查成交量
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
                current_volume = latest['volume']
                if current_volume < avg_volume * 0.5:  # 成交量低于20日均值50%
                    logger.debug(f"{symbol} 成交量过低")
                    return False
            
            # 检查波动率
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # 年化波动率
                if volatility > 0.6:  # 波动率超过60%
                    logger.debug(f"{symbol} 波动率过高: {volatility:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"技术指标检查 {symbol} 失败: {e}")
            return False
    
    def get_stock_status_summary(self, symbols: List[str], current_data: Dict[str, Dict[str, Any]],
                                financial_data: Dict[str, Dict[str, Any]] = None,
                                technical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """获取股票状态摘要"""
        # 可交易股票
        tradable_stocks = self.filter_tradable_stocks(symbols, current_data)
        
        # 优质股票
        quality_stocks = []
        if financial_data:
            quality_stocks = self.filter_quality_stocks(tradable_stocks, financial_data)
        
        # 技术过滤
        technical_stocks = []
        if technical_data:
            technical_stocks = self.filter_by_technical(tradable_stocks, technical_data)
        
        # 综合筛选
        final_stocks = tradable_stocks
        if quality_stocks:
            final_stocks = [s for s in final_stocks if s in quality_stocks]
        if technical_stocks:
            final_stocks = [s for s in final_stocks if s in technical_stocks]
        
        return {
            'total_stocks': len(symbols),
            'tradable_stocks': len(tradable_stocks),
            'quality_stocks': len(quality_stocks),
            'technical_stocks': len(technical_stocks),
            'final_stocks': len(final_stocks),
            'final_symbols': final_stocks,
            'filter_summary': {
                'tradable_filter_passed': len(tradable_stocks),
                'quality_filter_passed': len(quality_stocks),
                'technical_filter_passed': len(technical_stocks),
                'overall_passed': len(final_stocks),
                'filter_efficiency': len(final_stocks) / len(symbols) if symbols else 0
            },
            'timestamp': datetime.now()
        }
    
    def should_include(self, symbol: str, current_data: Dict[str, Any] = None) -> bool:
        """简化接口：判断股票是否应该被包含在选股结果中"""
        try:
            # 如果没有提供当前数据，默认返回True（包含）
            if current_data is None:
                return True
            
            # 使用现有的交易状态检查逻辑
            return self._is_tradable(symbol, current_data)
            
        except Exception as e:
            logger.error(f"判断 {symbol} 是否应该包含失败: {e}")
            return False

    def should_filter_stock(self, name: str, symbol: str, include_st: int = 0, 
                            include_suspended: int = 0, db_manager=None, 
                            exclude_star_market: int = 0, last_n_days: int = 30) -> Dict[str, Any]:
        """
        判断股票是否应该被过滤
        
        Args:
            name: 股票名称
            symbol: 股票代码
            include_st: 是否包含ST股票 (1=包含, 0=排除)
            include_suspended: 是否包含停牌股票 (1=包含, 0=排除)
            db_manager: 数据库管理器
            exclude_star_market: 是否排除科创板 (1=排除, 0=包含)
            last_n_days: 检查最近N天的数据
            
        Returns:
            Dict[str, Any]: 包含过滤结果和原因的字典
                - should_filter: 是否应该过滤 (True=过滤, False=不过滤)
                - reason: 过滤原因
        """
        try:
            # 检查股票名称中的ST标记
            if not include_st and ('ST' in name.upper() or '*ST' in name.upper()):
                return {
                    'should_filter': True,
                    'reason': f'ST股票: {name}'
                }
            
            # 检查科创板
            if exclude_star_market and symbol.startswith('688'):
                return {
                    'should_filter': True,
                    'reason': f'科创板股票: {symbol}'
                }
            
            # 如果有数据库管理器，检查更多状态
            if db_manager:
                try:
                    # 获取最近的价格数据
                    prices = db_manager.get_last_n_bars([symbol], n=last_n_days)
                    
                    if prices.empty:
                        return {
                            'should_filter': True,
                            'reason': f'无价格数据: {symbol}'
                        }
                    
                    # 检查最新价格
                    latest_price = prices.iloc[-1]
                    current_price = latest_price.get('close', 0)
                    
                    if current_price <= 0:
                        return {
                            'should_filter': True,
                            'reason': f'价格无效: {current_price}'
                        }
                    
                    # 检查是否长期停牌（最近N天无有效数据）
                    valid_days = len(prices[prices['close'] > 0])
                    if valid_days < last_n_days * 0.5:  # 少于50%的天数有有效数据
                        if not include_suspended:
                            return {
                                'should_filter': True,
                                'reason': f'长期数据缺失，可能停牌: {symbol}'
                            }
                    
                    # 检查涨跌停异常
                    price_changes = prices['close'].pct_change().dropna()
                    if len(price_changes) > 0:
                        max_change = abs(price_changes).max()
                        if max_change > 0.11:  # 单日涨跌幅超过11%
                            return {
                                'should_filter': True,
                                'reason': f'价格异常波动: {max_change:.2%}'
                            }
                    
                    # 检查成交量异常
                    volumes = prices['volume'].dropna()
                    if len(volumes) > 0:
                        avg_volume = volumes.mean()
                        if avg_volume <= 0:
                            return {
                                'should_filter': True,
                                'reason': f'无成交量: {symbol}'
                            }
                    
                except Exception as e:
                    logger.warning(f"数据库检查 {symbol} 失败: {e}")
                    # 数据库检查失败时，不过滤股票，继续其他检查
            
            # 通过所有检查，不过滤
            return {
                'should_filter': False,
                'reason': '通过筛选'
            }
            
        except Exception as e:
            logger.error(f"股票过滤检查失败 {symbol}: {e}")
            # 发生异常时，默认不过滤，避免误杀
            return {
                'should_filter': False,
                'reason': f'检查异常: {str(e)}'
            }

    def get_trading_status(self, symbol: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取交易状态详情"""
        try:
            data = current_data.get(symbol, {})
            
            return {
                'symbol': symbol,
                'is_tradable': self._is_tradable(symbol, data),
                'current_price': data.get('current_price', 0),
                'change_pct': data.get('change_pct', 0),
                'volume': data.get('volume', 0),
                'is_suspended': data.get('is_suspended', False),
                'is_delisted': data.get('is_delisted', False),
                'trading_warnings': self._get_trading_warnings(symbol, data),
                'last_updated': datetime.now()
            }
        except Exception as e:
            logger.error(f"获取 {symbol} 交易状态失败: {e}")
            return {
                'symbol': symbol,
                'is_tradable': False,
                'error': str(e),
                'last_updated': datetime.now()
            }
    
    def _get_trading_warnings(self, symbol: str, data: Dict[str, Any]) -> List[str]:
        """获取交易警告"""
        warnings = []
        
        try:
            # 价格警告
            current_price = data.get('current_price', 0)
            if current_price <= 0:
                warnings.append("价格无效")
            elif current_price < 1:  # 价格低于1元
                warnings.append("低价股风险")
            
            # 涨跌幅警告
            change_pct = data.get('change_pct', 0)
            if abs(change_pct) > 0.095:  # 接近涨跌停
                warnings.append("涨跌幅过大")
            
            # 成交量警告
            volume = data.get('volume', 0)
            if volume <= 0:
                warnings.append("无成交量")
            
            # 状态警告
            if data.get('is_suspended', False):
                warnings.append("股票停牌")
            
            if data.get('is_delisted', False):
                warnings.append("股票退市")
            
        except Exception as e:
            logger.error(f"获取 {symbol} 交易警告失败: {e}")
            warnings.append("数据异常")
        
        return warnings