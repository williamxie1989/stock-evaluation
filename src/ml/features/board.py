# -*- coding: utf-8 -*-
"""
板块特征工程模块
根据股票代码推断板块类型（创业板/科创板/主板/北交所等）
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)


class BoardFeatureGenerator:
    """
    板块特征生成器
    根据股票代码规则推断所属板块
    """
    
    # 板块识别规则
    BOARD_RULES = {
        '科创板': lambda code: code.startswith('688'),
        '创业板': lambda code: code.startswith('300'),
        '北交所': lambda code: code.startswith('43') or code.startswith('83') or code.startswith('87'),
        '深主板': lambda code: code.startswith('000') or code.startswith('001'),
        '沪主板': lambda code: code.startswith('60') and not code.startswith('688'),
        'B股': lambda code: code.startswith('200') or code.startswith('900'),
    }
    
    def __init__(self):
        """初始化板块特征生成器"""
        pass
    
    def infer_board(self, symbol: str) -> str:
        """
        根据股票代码推断板块
        
        Args:
            symbol: 股票代码 (6位数字)
        
        Returns:
            板块名称
        """
        if pd.isna(symbol) or symbol == '':
            return 'Unknown'
        
        # 提取纯数字代码（去除前缀如 sh, sz 等）
        code = self._extract_code(symbol)
        
        # 按规则匹配
        for board_name, rule_func in self.BOARD_RULES.items():
            if rule_func(code):
                return board_name
        
        # 默认分类
        return 'Other'
    
    def _extract_code(self, symbol: str) -> str:
        """
        提取纯数字代码
        
        Examples:
            'sh600000' -> '600000'
            '600000' -> '600000'
            'SZ000001' -> '000001'
        """
        # 去除常见前缀
        symbol = str(symbol).upper()
        symbol = re.sub(r'^(SH|SZ|BJ)', '', symbol)
        
        # 提取数字
        code = re.findall(r'\d+', symbol)
        
        if code:
            return code[0]
        
        return symbol
    
    def add_board_feature(self, df: pd.DataFrame, symbol_col: str = 'symbol') -> pd.DataFrame:
        """
        为 DataFrame 添加板块特征
        
        Args:
            df: 包含股票代码列的 DataFrame
            symbol_col: 股票代码列名
        
        Returns:
            添加了 'board' 列的 DataFrame
        """
        if symbol_col not in df.columns:
            logger.error(f"DataFrame 必须包含 '{symbol_col}' 列")
            return df
        
        df['board'] = df[symbol_col].apply(self.infer_board)
        
        return df
    
    def get_board_stats(self, symbols: List[str]) -> pd.DataFrame:
        """
        获取板块统计信息
        
        Args:
            symbols: 股票代码列表
        
        Returns:
            DataFrame with columns: board, count, percentage
        """
        boards = [self.infer_board(s) for s in symbols]
        
        stats = pd.Series(boards).value_counts()
        total = len(symbols)
        
        stats_df = pd.DataFrame({
            'board': stats.index,
            'count': stats.values,
            'percentage': stats.values / total * 100
        })
        
        return stats_df
    
    def get_board_mapping(self, symbols: List[str]) -> Dict[str, str]:
        """
        获取股票到板块的映射
        
        Args:
            symbols: 股票代码列表
        
        Returns:
            字典 {symbol: board}
        """
        return {symbol: self.infer_board(symbol) for symbol in symbols}


def infer_board_from_symbol(symbols) -> pd.Series:
    """
    批量推断股票板块
    
    Args:
        symbols: 股票代码（Series 或 list）
    
    Returns:
        板块 Series
    """
    generator = BoardFeatureGenerator()
    
    if isinstance(symbols, pd.Series):
        return symbols.apply(generator.infer_board)
    elif isinstance(symbols, list):
        return pd.Series([generator.infer_board(s) for s in symbols])
    else:
        return generator.infer_board(str(symbols))


def add_board_features(df: pd.DataFrame, symbol_col: str = 'symbol') -> pd.DataFrame:
    """
    为 DataFrame 添加板块特征（便捷函数）
    
    Args:
        df: 包含股票代码的 DataFrame
        symbol_col: 股票代码列名
    
    Returns:
        添加了 'board' 列的 DataFrame
    """
    generator = BoardFeatureGenerator()
    return generator.add_board_feature(df, symbol_col=symbol_col)
