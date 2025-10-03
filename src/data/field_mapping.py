#!/usr/bin/env python3
"""
字段映射工具
用于标准化不同数据源的字段名称和结构
"""

import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class FieldMapper:
    """字段映射器，用于标准化不同数据源的字段名称"""
    
    # 标准字段映射表
    FIELD_MAPPINGS = {
        'prices_daily': {
            # 标准字段 -> 可能的变体
            'date': ['date', 'Date', 'DATE', 'trade_date', 'tradeDate', 'datetime'],
            'open': ['open', 'Open', 'OPEN', 'open_price'],
            'high': ['high', 'High', 'HIGH', 'high_price'],
            'low': ['low', 'Low', 'LOW', 'low_price'],
            'close': ['close', 'Close', 'CLOSE', 'close_price', 'closePrice'],
            'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'turnover', 'amount', '成交量'],
            'symbol': ['symbol', 'Symbol', 'SYMBOL', 'code', 'Code', 'CODE', 'stock_code', 'ts_code', '股票代码']
        }
        ,
        'predictions': {
            'symbol': ['symbol', 'code'],
            'date': ['date', 'prediction_date', 'predict_date'],
            'prob_up_30d': ['prob_up_30d', 'probability', 'prob'],
            'expected_return_30d': ['expected_return_30d', 'expected_return', 'exp_ret'],
            'confidence': ['confidence'],
            'score': ['score'],
            'sentiment': ['sentiment'],
            'prediction': ['prediction', 'pred']
        }
    }
    
    # 必需字段定义
    REQUIRED_FIELDS = {
        'prices_daily': ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol'],
        'predictions': ['symbol', 'date', 'prob_up_30d', 'expected_return_30d']
    }
    
    @classmethod
    def normalize_fields(cls, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        标准化DataFrame的字段名称
        
        Args:
            df: 输入DataFrame
            table_name: 表名
            
        Returns:
            字段标准化后的DataFrame
        """
        if table_name not in cls.FIELD_MAPPINGS:
            logger.warning(f"未知的表名: {table_name}，返回原始数据")
            return df
        
        mappings = cls.FIELD_MAPPINGS[table_name]
        result_df = df.copy()
        
        # 创建字段映射
        column_mapping = {}
        for standard_field, variants in mappings.items():
            for variant in variants:
                if variant in result_df.columns:
                    column_mapping[variant] = standard_field
                    break
        
        # 应用映射
        if column_mapping:
            result_df = result_df.rename(columns=column_mapping)
            # logger.info(f"字段映射应用: {column_mapping}")
        
        return result_df
    
    @classmethod
    def ensure_required_fields(cls, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        确保DataFrame包含所有必需字段
        
        Args:
            df: 输入DataFrame
            table_name: 表名
            
        Returns:
            包含所有必需字段的DataFrame
        """
        if table_name not in cls.REQUIRED_FIELDS:
            logger.warning(f"未知的表名: {table_name}，返回原始数据")
            return df
        
        required_fields = cls.REQUIRED_FIELDS[table_name]
        result_df = df.copy()
        
        # 检查缺失的必需字段
        missing_fields = [field for field in required_fields if field not in result_df.columns]
        
        if missing_fields:
            logger.warning(f"缺失必需字段: {missing_fields}")
            
            # 尝试从现有字段推断缺失字段
            for field in missing_fields:
                if field == 'symbol' and 'code' in result_df.columns:
                    result_df[field] = result_df['code']
                elif field == 'volume' and 'turnover' in result_df.columns:
                    result_df[field] = result_df['turnover']
                elif field == 'date' and isinstance(result_df.index, pd.DatetimeIndex):
                    # 如果索引是日期类型，则使用索引填充 date 字段
                    result_df[field] = result_df.index
                else:
                    # 为缺失字段提供默认值
                    if field in ['open', 'high', 'low', 'close', 'volume']:
                        result_df[field] = 0.0
                    elif field == 'date':
                        result_df[field] = pd.Timestamp.now()
                    elif field == 'symbol':
                        result_df[field] = 'UNKNOWN'
        
        return result_df
    
    @classmethod
    def validate_data_structure(cls, df: pd.DataFrame, table_name: str) -> bool:
        """
        验证数据结构是否符合要求
        
        Args:
            df: 输入DataFrame
            table_name: 表名
            
        Returns:
            是否通过验证
        """
        if table_name not in cls.REQUIRED_FIELDS:
            logger.warning(f"未知的表名: {table_name}，跳过验证")
            return True
        
        required_fields = cls.REQUIRED_FIELDS[table_name]
        
        # 检查必需字段
        missing_fields = [field for field in required_fields if field not in df.columns]
        # 如果缺失 date 字段但索引是日期类型，允许使用索引作为日期列
        if 'date' in missing_fields and isinstance(df.index, pd.DatetimeIndex):
            missing_fields.remove('date')
        if missing_fields:
            logger.error(f"数据结构验证失败 - 缺失必需字段: {missing_fields}")
            return False
        
        # 检查数据类型
        type_checks = {
            'date': lambda x: pd.api.types.is_datetime64_any_dtype(x) or pd.api.types.is_object_dtype(x),
            'open': pd.api.types.is_numeric_dtype,
            'high': pd.api.types.is_numeric_dtype,
            'low': pd.api.types.is_numeric_dtype,
            'close': pd.api.types.is_numeric_dtype,
            'volume': pd.api.types.is_numeric_dtype,
            'symbol': pd.api.types.is_object_dtype
        }
        
        for field in required_fields:
            if field == 'date' and field not in df.columns:
                continue  # date 字段由索引提供
            if field in df.columns:
                series = df[field]
                # 若未定义类型检查规则，默认通过
                checker = type_checks.get(field, None)
                # 尝试将字符串数字列转换为数值
                if field not in ['symbol', 'date'] and series.dtype == 'object':
                    try:
                        series = pd.to_numeric(series)
                    except Exception:
                        pass
                if checker and not checker(series):
                    logger.error(f"数据结构验证失败 - 字段 {field} 类型不符合要求")
                    return False
        
        # 检查是否有数据
        if df.empty:
            logger.error("数据结构验证失败 - 数据为空")
            return False
        
        logger.info(f"表 {table_name} 数据结构验证通过")
        return True
    
    @classmethod
    def get_field_description(cls, table_name: str, field_name: str) -> str:
        """
        获取字段描述
        
        Args:
            table_name: 表名
            field_name: 字段名
            
        Returns:
            字段描述
        """
        descriptions = {
            'prices_daily': {
                'date': '交易日期',
                'open': '开盘价',
                'high': '最高价',
                'low': '最低价',
                'close': '收盘价',
                'volume': '成交量',
                'symbol': '股票代码'
            }
        }
        
        return descriptions.get(table_name, {}).get(field_name, field_name)