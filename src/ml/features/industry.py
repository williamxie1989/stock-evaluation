# -*- coding: utf-8 -*-
"""
行业特征工程模块
从数据库获取行业信息，进行清洗和标准化
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Iterable
import logging
import re

from src.data.db.symbol_standardizer import get_symbol_standardizer, standardize_symbol

logger = logging.getLogger(__name__)


class IndustryFeatureGenerator:
    """
    行业特征生成器
    从 MySQL stocks 表读取行业信息，清洗并提供映射
    """
    
    # 行业名称标准化映射表 (中英文别名合并)
    INDUSTRY_MAPPING = {
        # 科技类
        '计算机': '计算机',
        '电子': '电子',
        '通信': '通信',
        '软件': '计算机',
        '半导体': '电子',
        '通讯': '通信',
        
        # 制造类
        '机械设备': '机械设备',
        '机械': '机械设备',
        '电气设备': '电气设备',
        '电力设备': '电气设备',
        '汽车': '汽车',
        '国防军工': '国防军工',
        '军工': '国防军工',
        
        # 消费类
        '食品饮料': '食品饮料',
        '医药生物': '医药生物',
        '医药': '医药生物',
        '纺织服装': '纺织服装',
        '轻工制造': '轻工制造',
        '商业贸易': '商业贸易',
        '家用电器': '家用电器',
        '家电': '家用电器',
        
        # 金融地产
        '银行': '银行',
        '非银金融': '非银金融',
        '房地产': '房地产',
        
        # 周期类
        '化工': '化工',
        '钢铁': '钢铁',
        '有色金属': '有色金属',
        '采掘': '采掘',
        '建筑材料': '建筑材料',
        '建材': '建筑材料',
        '建筑装饰': '建筑装饰',
        '建筑': '建筑装饰',
        
        # 公用事业
        '公用事业': '公用事业',
        '电力': '公用事业',
        '环保': '环保',
        '交通运输': '交通运输',
        '运输': '交通运输',
        
        # 其他
        '传媒': '传媒',
        '农林牧渔': '农林牧渔',
        '休闲服务': '休闲服务',
        '综合': '综合',
    }
    
    def __init__(self, db_manager, min_frequency: float = 0.005):
        """
        初始化行业特征生成器
        
        Args:
            db_manager: 数据库管理器对象
            min_frequency: 最小行业频率阈值，低于此值的行业归为 'Other'
        """
        self.db = db_manager
        self.min_frequency = min_frequency
        self.industry_cache = {}  # 缓存: {symbol: industry}
        self.industry_stats = {}  # 统计: {industry: count}
        self.symbol_standardizer = get_symbol_standardizer()

    def _normalize_symbol_key(self, symbol: str) -> str:
        """生成用于匹配的统一主键"""
        if symbol is None:
            return ''
        try:
            std = standardize_symbol(symbol)
            return std.split('.')[0]
        except Exception:
            digits = re.findall(r'\d{3,}', str(symbol))
            return digits[0] if digits else str(symbol)

    def _prepare_query_symbols(self, symbols: Iterable[str]) -> List[str]:
        """准备数据库查询所需的候选symbol列表"""
        candidates = set()
        for symbol in symbols:
            if not symbol:
                continue
            raw = str(symbol).strip()
            candidates.add(raw)
            try:
                std = standardize_symbol(raw)
                candidates.add(std)
                candidates.add(std.split('.')[0])
            except Exception:
                digits = re.findall(r'\d{6}', raw)
                if digits:
                    candidates.add(digits[0])
        return list(candidates)
    
    def fetch_industry_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        从数据库获取行业数据
        
        Args:
            symbols: 股票代码列表 (可选)，None 表示获取全部
        
        Returns:
            DataFrame with columns: symbol, industry
        """
        try:
            if symbols is None or len(symbols) == 0:
                # 获取全部
                query = """
                SELECT symbol, 
                       COALESCE(NULLIF(TRIM(industry), ''), 'Unknown') AS industry
                FROM stocks
                """
                results = self.db.execute_query(query)
            else:
                # 获取指定股票
                query_symbols = self._prepare_query_symbols(symbols)
                if not query_symbols:
                    logger.warning("未获取到行业数据")
                    return pd.DataFrame(columns=['symbol', 'industry'])
                placeholders = ','.join(['?' if self.db.db_type == 'sqlite' else '%s'] * len(query_symbols))
                query = f"""
                SELECT symbol, 
                       COALESCE(NULLIF(TRIM(industry), ''), 'Unknown') AS industry
                FROM stocks
                WHERE symbol IN ({placeholders})
                """
                results = self.db.execute_query(query, tuple(query_symbols))

            df = pd.DataFrame(results) if results else pd.DataFrame(columns=['symbol', 'industry'])
            if not df.empty:
                missing_cols = {'symbol', 'industry'} - set(df.columns)
                for col in missing_cols:
                    df[col] = np.nan
                df = df[['symbol', 'industry']]
            
            if df is None or len(df) == 0:
                logger.warning("未获取到行业数据")
                return pd.DataFrame(columns=['symbol', 'industry'])
            
            # 标准化行业名称
            df['industry'] = df['industry'].apply(self._standardize_industry)
            
            # 更新缓存
            for _, row in df.iterrows():
                cache_symbol = row['symbol']
                industry_value = row['industry']
                self.industry_cache[cache_symbol] = industry_value
                normalized_key = self._normalize_symbol_key(cache_symbol)
                if normalized_key:
                    self.industry_cache[normalized_key] = industry_value
            
            # 统计行业分布
            self._update_industry_stats(df)
            
            logger.info(f"获取了 {len(df)} 只股票的行业数据，共 {len(self.industry_stats)} 个行业")
            
            return df
        
        except Exception as e:
            logger.error(f"获取行业数据失败: {e}")
            return pd.DataFrame(columns=['symbol', 'industry'])
    
    def _standardize_industry(self, raw_industry: str) -> str:
        """
        标准化行业名称
        
        Args:
            raw_industry: 原始行业名称
        
        Returns:
            标准化后的行业名称
        """
        if pd.isna(raw_industry) or raw_industry == '':
            return 'Unknown'
        
        # 去除空格、转小写
        industry = str(raw_industry).strip()
        
        # 查找映射表
        for key, value in self.INDUSTRY_MAPPING.items():
            if key in industry:
                return value
        
        # 如果没有匹配，返回原始值（首字母大写）
        return industry if industry != '' else 'Unknown'
    
    def _update_industry_stats(self, df: pd.DataFrame):
        """更新行业统计信息"""
        self.industry_stats = df['industry'].value_counts().to_dict()
    
    def merge_low_frequency_industries(self, df: pd.DataFrame, total_count: Optional[int] = None) -> pd.DataFrame:
        """
        合并低频行业为 'Other'
        
        Args:
            df: 包含 'industry' 列的 DataFrame
            total_count: 总样本数 (可选)，用于计算频率
        
        Returns:
            处理后的 DataFrame
        """
        if total_count is None:
            total_count = len(df)
        
        # 计算每个行业的频率
        if not self.industry_stats:
            self._update_industry_stats(df)
        
        # 识别低频行业
        low_freq_industries = [
            ind for ind, count in self.industry_stats.items()
            if count / total_count < self.min_frequency
        ]
        
        if low_freq_industries:
            logger.info(f"合并 {len(low_freq_industries)} 个低频行业为 'Other': {low_freq_industries[:5]}...")
            df['industry'] = df['industry'].apply(
                lambda x: 'Other' if x in low_freq_industries else x
            )
        
        return df
    
    def get_industry_for_symbols(self, symbols: List[str], use_cache: bool = True) -> Dict[str, str]:
        """
        获取股票的行业映射
        
        Args:
            symbols: 股票代码列表
            use_cache: 是否使用缓存
        
        Returns:
            字典 {symbol: industry}
        """
        result = {}
        
        if use_cache:
            missing_symbols = []
            for symbol in symbols:
                cache_key_primary = symbol
                cache_key_secondary = self._normalize_symbol_key(symbol)
                if cache_key_primary in self.industry_cache:
                    result[symbol] = self.industry_cache[cache_key_primary]
                elif cache_key_secondary in self.industry_cache:
                    result[symbol] = self.industry_cache[cache_key_secondary]
                else:
                    missing_symbols.append(symbol)
            
            # 如果有缺失，从数据库补充
            if missing_symbols:
                df = self.fetch_industry_data(missing_symbols)
                for _, row in df.iterrows():
                    result[row['symbol']] = row['industry']
        else:
            # 直接从数据库获取
            df = self.fetch_industry_data(symbols)
            for _, row in df.iterrows():
                result[row['symbol']] = row['industry']
        
        # 填充缺失值
        for symbol in symbols:
            if symbol not in result:
                result[symbol] = 'Unknown'
        
        return result
    
    def get_industry_stats(self) -> pd.DataFrame:
        """
        获取行业统计信息
        
        Returns:
            DataFrame with columns: industry, count, percentage
        """
        if not self.industry_stats:
            self.fetch_industry_data()
        
        total = sum(self.industry_stats.values())
        
        stats_df = pd.DataFrame([
            {
                'industry': ind,
                'count': count,
                'percentage': count / total * 100
            }
            for ind, count in sorted(self.industry_stats.items(), key=lambda x: x[1], reverse=True)
        ])
        
        return stats_df


def add_industry_features(df: pd.DataFrame, db_manager, merge_low_freq: bool = True) -> pd.DataFrame:
    """
    为 DataFrame 添加行业特征
    
    Args:
        df: 包含 'symbol' 列的 DataFrame
        db_manager: 数据库管理器
        merge_low_freq: 是否合并低频行业
    
    Returns:
        添加了 'industry' 列的 DataFrame
    """
    if 'symbol' not in df.columns:
        logger.error("DataFrame 必须包含 'symbol' 列")
        return df
    
    generator = IndustryFeatureGenerator(db_manager)

    symbols = df['symbol'].astype(str)
    symbol_keys = symbols.apply(generator._normalize_symbol_key)
    df_with_key = df.copy()
    df_with_key['_symbol_key'] = symbol_keys

    industry_df = generator.fetch_industry_data(symbols.unique().tolist())

    if industry_df.empty:
        df_with_key['industry'] = 'Unknown'
        return df_with_key.drop(columns=['_symbol_key'])

    industry_df['_symbol_key'] = industry_df['symbol'].apply(generator._normalize_symbol_key)
    industry_df = industry_df.drop_duplicates('_symbol_key', keep='first')

    if merge_low_freq:
        total_count = len(symbol_keys.unique())
        industry_df = generator.merge_low_frequency_industries(industry_df, total_count=total_count)

    merged = df_with_key.merge(
        industry_df[['_symbol_key', 'industry']],
        on='_symbol_key',
        how='left'
    )

    merged['industry'].fillna('Unknown', inplace=True)
    return merged.drop(columns=['_symbol_key'])
