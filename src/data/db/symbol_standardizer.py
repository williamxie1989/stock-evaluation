#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票代码标准化器
提供统一的股票代码格式标准化功能，确保所有数据访问层使用一致的symbol格式
"""

import re
import logging
from typing import Optional, List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class SymbolStandardizer:
    """股票代码标准化器 - 单例模式"""
    
    _instance: Optional['SymbolStandardizer'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SymbolStandardizer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # A股代码前缀规则
        self.SH_PREFIXES = ['60', '688', '689', '787', '789']  # 沪市
        self.SZ_PREFIXES = ['00', '30', '39']  # 深市
        
        # 需要过滤的代码前缀
        self.EXCLUDED_PREFIXES = ['88']  # 指数、基金等
        
        # 市场后缀映射
        self.MARKET_SUFFIXES = {
            'SH': '.SH',
            'SZ': '.SZ',
            'SS': '.SH'  # 处理.SS后缀（某些数据源使用）
        }
        
        self._initialized = True
        logger.info("SymbolStandardizer initialized")
    
    @classmethod
    def get_instance(cls) -> 'SymbolStandardizer':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def standardize_symbol(self, symbol: str) -> str:
        """
        标准化股票代码，确保格式为: 数字代码 + .SH/.SZ 后缀
        
        Args:
            symbol: 原始股票代码，可能格式如：
                    - 600000
                    - 600000.SH
                    - 600000.SS
                    - SH600000
                    - sh600000
                    
        Returns:
            标准化后的股票代码，如：600000.SH
            
        Raises:
            ValueError: 如果股票代码格式无效
        """
        if not symbol:
            raise ValueError("股票代码不能为空")
        
        try:
            # 清理和预处理
            symbol = str(symbol).strip().upper()
            
            # 移除空格和特殊字符
            symbol = re.sub(r'\s+', '', symbol)
            
            # 提取数字部分和后缀
            numeric_part, suffix = self._extract_parts(symbol)
            
            # 验证数字部分
            if not numeric_part or not numeric_part.isdigit():
                raise ValueError(f"股票代码必须包含数字部分: {symbol}")
            
            # 检查是否为6位数字
            if len(numeric_part) != 6:
                raise ValueError(f"股票代码必须是6位数字: {symbol}")
            
            # 检查是否需要过滤
            if self._should_exclude(numeric_part):
                raise ValueError(f"股票代码被过滤规则排除: {symbol}")
            
            # 确定市场后缀
            market_suffix = self._determine_market_suffix(numeric_part, suffix)
            
            # 构建标准化代码
            standardized = f"{numeric_part}{market_suffix}"
            
            logger.debug(f"Symbol standardized: {symbol} -> {standardized}")
            return standardized
            
        except Exception as e:
            logger.error(f"股票代码标准化失败: {symbol}, 错误: {e}")
            raise ValueError(f"无效的股票代码格式: {symbol}")
    
    def _extract_parts(self, symbol: str) -> tuple[str, Optional[str]]:
        """提取数字部分和后缀"""
        # 匹配模式：数字 + 可选后缀
        match = re.match(r'^(\d{6})(?:\.([A-Z]{2}))?$', symbol)
        if match:
            numeric_part = match.group(1)
            suffix = match.group(2) if match.group(2) else None
            return numeric_part, suffix
        
        # 匹配模式：市场前缀 + 数字
        match = re.match(r'^([A-Z]{2})(\d{6})$', symbol)
        if match:
            market_prefix = match.group(1)
            numeric_part = match.group(2)
            suffix_map = {'SH': 'SH', 'SZ': 'SZ'}
            suffix = suffix_map.get(market_prefix)
            return numeric_part, suffix
        
        # 纯数字
        if symbol.isdigit() and len(symbol) == 6:
            return symbol, None
        
        raise ValueError(f"无法解析股票代码格式: {symbol}")
    
    def _should_exclude(self, numeric_part: str) -> bool:
        """检查是否应该过滤该股票代码"""
        for prefix in self.EXCLUDED_PREFIXES:
            if numeric_part.startswith(prefix):
                return True
        return False
    
    def _determine_market_suffix(self, numeric_part: str, suffix: Optional[str]) -> str:
        """确定市场后缀"""
        # 如果已有有效后缀，进行标准化
        if suffix:
            if suffix in self.MARKET_SUFFIXES:
                return self.MARKET_SUFFIXES[suffix]
            elif suffix in ['SH', 'SZ']:
                return f".{suffix}"
        
        # 根据数字前缀判断市场
        for prefix in self.SH_PREFIXES:
            if numeric_part.startswith(prefix):
                return '.SH'
        
        for prefix in self.SZ_PREFIXES:
            if numeric_part.startswith(prefix):
                return '.SZ'
        
        # 默认规则：60/68开头为沪市，其他为深市
        if numeric_part.startswith(('60', '68')):
            return '.SH'
        else:
            return '.SZ'
    
    def standardize_symbols(self, symbols: List[str]) -> List[str]:
        """批量标准化股票代码"""
        standardized = []
        errors = []
        
        for symbol in symbols:
            try:
                std_symbol = self.standardize_symbol(symbol)
                standardized.append(std_symbol)
            except ValueError as e:
                errors.append(f"{symbol}: {e}")
                logger.warning(f"跳过无效股票代码: {e}")
        
        if errors:
            logger.warning(f"批量标准化完成，跳过了 {len(errors)} 个无效代码")
        
        return standardized
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """检查股票代码是否有效"""
        try:
            self.standardize_symbol(symbol)
            return True
        except ValueError:
            return False
    
    def extract_numeric_code(self, symbol: str) -> str:
        """提取数字代码部分"""
        standardized = self.standardize_symbol(symbol)
        return standardized.split('.')[0]
    
    def get_market_from_symbol(self, symbol: str) -> str:
        """从股票代码获取市场信息"""
        standardized = self.standardize_symbol(symbol)
        return standardized.split('.')[1]
    
    def remove_suffix(self, symbol: str) -> str:
        """移除后缀，返回纯数字代码"""
        try:
            standardized = self.standardize_symbol(symbol)
            return standardized.split('.')[0]
        except ValueError:
            # 如果无法标准化，尝试移除已知后缀
            for suffix in ['.SH', '.SZ', '.SS']:
                if symbol.upper().endswith(suffix):
                    return symbol[:-len(suffix)]
            return symbol
    
    def add_suffix(self, numeric_code: str, market: str) -> str:
        """为数字代码添加市场后缀"""
        market = market.upper()
        if market in ['SH', 'SZ']:
            return f"{numeric_code}.{market}"
        elif market == 'SS':
            return f"{numeric_code}.SH"
        else:
            raise ValueError(f"无效的市场标识: {market}")
    
    def filter_valid_symbols(self, symbols: List[str]) -> List[str]:
        """过滤有效的股票代码"""
        valid = []
        for symbol in symbols:
            if self.is_valid_symbol(symbol):
                valid.append(symbol)
        return valid
    
    @lru_cache(maxsize=1000)
    def cached_standardize(self, symbol: str) -> str:
        """带缓存的标准化方法"""
        return self.standardize_symbol(symbol)
    
    def clear_cache(self):
        """清除缓存"""
        self.cached_standardize.cache_clear()
        logger.info("Symbol standardizer cache cleared")
    
    def get_standardization_stats(self) -> Dict[str, Any]:
        """获取标准化统计信息"""
        cache_info = self.cached_standardize.cache_info()
        return {
            'cache_hits': cache_info.hits,
            'cache_misses': cache_info.misses,
            'cache_size': cache_info.currsize,
            'max_cache_size': cache_info.maxsize
        }


# 全局实例
def get_symbol_standardizer() -> SymbolStandardizer:
    """获取全局股票代码标准化器实例"""
    return SymbolStandardizer.get_instance()


def standardize_symbol(symbol: str) -> str:
    """
    快捷函数：标准化单个股票代码
    
    Args:
        symbol: 原始股票代码
        
    Returns:
        标准化后的股票代码
    """
    return get_symbol_standardizer().standardize_symbol(symbol)


def standardize_symbols(symbols: List[str]) -> List[str]:
    """
    快捷函数：批量标准化股票代码
    
    Args:
        symbols: 原始股票代码列表
        
    Returns:
        标准化后的股票代码列表
    """
    return get_symbol_standardizer().standardize_symbols(symbols)