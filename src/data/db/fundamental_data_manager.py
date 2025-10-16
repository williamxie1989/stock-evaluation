# -*- coding: utf-8 -*-
"""
基本面数据持久化管理器

功能：
1. 将基本面数据（财务指标、估值指标等）持久化到数据库
2. 避免重复调用AKShare API
3. 支持批量导入和增量更新
4. 提供快速查询接口

设计思路：
- fundamental_data_quarterly: 季度财务数据（营收、利润、ROE等）
- fundamental_data_daily: 日度估值数据（PE、PB、市值等）
"""

import re
import time
from functools import reduce
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta, date
import logging
import akshare as ak
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class FundamentalDataManager:
    """基本面数据持久化管理器"""
    
    def __init__(self, db_manager):
        """
        Parameters
        ----------
        db_manager : UnifiedDatabaseManager
            统一数据库管理器
        """
        self.db_manager = db_manager
        self._ensure_tables()
    
    def _ensure_tables(self):
        """确保基本面数据表存在"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # 1. 季度财务数据表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fundamental_data_quarterly (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL COMMENT '股票代码（带后缀）',
                        report_date DATE NOT NULL COMMENT '财报日期（季度末）',
                        publish_date DATE COMMENT '发布日期',
                        
                        -- 盈利指标
                        revenue DECIMAL(20, 2) COMMENT '营业收入（元）',
                        net_profit DECIMAL(20, 2) COMMENT '净利润（元）',
                        gross_profit_margin DECIMAL(10, 4) COMMENT '毛利率',
                        net_profit_margin DECIMAL(10, 4) COMMENT '净利率',
                        roe DECIMAL(10, 4) COMMENT 'ROE（净资产收益率）',
                        roa DECIMAL(10, 4) COMMENT 'ROA（资产回报率）',
                        eps DECIMAL(10, 4) COMMENT '每股收益',
                        
                        -- 成长指标
                        revenue_yoy DECIMAL(10, 4) COMMENT '营收同比增长率',
                        net_profit_yoy DECIMAL(10, 4) COMMENT '净利润同比增长率',
                        eps_yoy DECIMAL(10, 4) COMMENT 'EPS同比增长率',
                        
                        -- 财务质量指标
                        debt_to_asset DECIMAL(10, 4) COMMENT '资产负债率',
                        current_ratio DECIMAL(10, 4) COMMENT '流动比率',
                        quick_ratio DECIMAL(10, 4) COMMENT '速动比率',
                        operating_cash_flow DECIMAL(20, 2) COMMENT '经营现金流（元）',
                        total_assets DECIMAL(20, 2) COMMENT '总资产（元）',
                        total_equity DECIMAL(20, 2) COMMENT '股东权益（元）',
                        
                        -- 元数据
                        data_source VARCHAR(50) DEFAULT 'akshare' COMMENT '数据源',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        
                        UNIQUE KEY uk_symbol_date (symbol, report_date),
                        INDEX idx_symbol (symbol),
                        INDEX idx_date (report_date)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='季度财务数据表'
                """)
                
                # 2. 日度估值数据表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fundamental_data_daily (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL COMMENT '股票代码（带后缀）',
                        trade_date DATE NOT NULL COMMENT '交易日期',
                        
                        -- 估值指标
                        pe_ttm DECIMAL(10, 4) COMMENT 'PE（TTM）',
                        pb DECIMAL(10, 4) COMMENT 'PB（市净率）',
                        ps_ttm DECIMAL(10, 4) COMMENT 'PS（市销率）',
                        market_cap DECIMAL(20, 2) COMMENT '总市值（元）',
                        circulating_market_cap DECIMAL(20, 2) COMMENT '流通市值（元）',
                        dividend_yield DECIMAL(10, 4) COMMENT '股息率',
                        
                        -- 历史分位数（相对估值）
                        pe_percentile DECIMAL(10, 4) COMMENT 'PE历史分位数',
                        pb_percentile DECIMAL(10, 4) COMMENT 'PB历史分位数',
                        
                        -- 元数据
                        data_source VARCHAR(50) DEFAULT 'akshare' COMMENT '数据源',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        
                        UNIQUE KEY uk_symbol_date (symbol, trade_date),
                        INDEX idx_symbol (symbol),
                        INDEX idx_date (trade_date)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='日度估值数据表'
                """)
                
                # 3. 季度数据日频化结果表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fundamental_data_daily_expanded (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL COMMENT '股票代码（带后缀）',
                        trade_date DATE NOT NULL COMMENT '交易日期',
                        
                        revenue DECIMAL(20, 2) COMMENT '营业收入（元）',
                        net_profit DECIMAL(20, 2) COMMENT '净利润（元）',
                        net_profit_margin DECIMAL(10, 4) COMMENT '净利率',
                        roe DECIMAL(10, 4) COMMENT 'ROE',
                        eps DECIMAL(10, 4) COMMENT '每股收益',
                        revenue_yoy DECIMAL(10, 4) COMMENT '营收同比增长率',
                        net_profit_yoy DECIMAL(10, 4) COMMENT '净利润同比增长率',
                        debt_to_asset DECIMAL(10, 4) COMMENT '资产负债率',
                        current_ratio DECIMAL(10, 4) COMMENT '流动比率',
                        quick_ratio DECIMAL(10, 4) COMMENT '速动比率',
                        pe_ttm DECIMAL(10, 4) COMMENT '市盈率(TTM)',
                        pb DECIMAL(10, 4) COMMENT '市净率',
                        
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        
                        UNIQUE KEY uk_symbol_date_expanded (symbol, trade_date),
                        INDEX idx_symbol_expanded (symbol),
                        INDEX idx_date_expanded (trade_date)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='季度财务数据日频化表'
                """)
                
                cursor.close()
                logger.info("✅ 基本面数据表已就绪")
                
        except Exception as e:
            logger.error(f"创建基本面数据表失败: {e}")
            raise
    
    def save_quarterly_data(self, symbol: str, df: pd.DataFrame) -> int:
        """
        保存季度财务数据（批量插入/更新）
        
        Parameters
        ----------
        symbol : str
            股票代码（带后缀，如 000001.SZ）
        df : pd.DataFrame
            季度财务数据，必须包含 report_date 列
        
        Returns
        -------
        count : int
            成功保存的记录数
        """
        if df is None or df.empty:
            return 0
        
        df = df.copy()
        df = self._clean_financial_dataframe(df, mode='quarterly')
        df = self._normalize_quarterly_dates(df)
        if df.empty:
            return 0
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # 准备插入数据
                records = []
                for _, row in df.iterrows():
                    report_date = self._to_date(row.get('report_date'))
                    publish_date = self._to_date(row.get('publish_date'))
                    record = (
                        symbol,
                        report_date,
                        publish_date,
                        # 盈利指标
                        row.get('revenue'),
                        row.get('net_profit'),
                        row.get('gross_profit_margin'),
                        row.get('net_profit_margin'),
                        row.get('roe'),
                        row.get('roa'),
                        row.get('eps'),
                        # 成长指标
                        row.get('revenue_yoy'),
                        row.get('net_profit_yoy'),
                        row.get('eps_yoy'),
                        # 财务质量
                        row.get('debt_to_asset'),
                        row.get('current_ratio'),
                        row.get('quick_ratio'),
                        row.get('operating_cash_flow'),
                        row.get('total_assets'),
                        row.get('total_equity')
                    )
                    records.append(record)
                
                # 批量插入（ON DUPLICATE KEY UPDATE）
                sql = """
                    INSERT INTO fundamental_data_quarterly (
                        symbol, report_date, publish_date,
                        revenue, net_profit, gross_profit_margin, net_profit_margin,
                        roe, roa, eps,
                        revenue_yoy, net_profit_yoy, eps_yoy,
                        debt_to_asset, current_ratio, quick_ratio,
                        operating_cash_flow, total_assets, total_equity
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON DUPLICATE KEY UPDATE
                        publish_date = VALUES(publish_date),
                        revenue = VALUES(revenue),
                        net_profit = VALUES(net_profit),
                        gross_profit_margin = VALUES(gross_profit_margin),
                        net_profit_margin = VALUES(net_profit_margin),
                        roe = VALUES(roe),
                        roa = VALUES(roa),
                        eps = VALUES(eps),
                        revenue_yoy = VALUES(revenue_yoy),
                        net_profit_yoy = VALUES(net_profit_yoy),
                        eps_yoy = VALUES(eps_yoy),
                        debt_to_asset = VALUES(debt_to_asset),
                        current_ratio = VALUES(current_ratio),
                        quick_ratio = VALUES(quick_ratio),
                        operating_cash_flow = VALUES(operating_cash_flow),
                        total_assets = VALUES(total_assets),
                        total_equity = VALUES(total_equity),
                        updated_at = CURRENT_TIMESTAMP
                """
                
                if records:
                    cursor.executemany(sql, records)
                    conn.commit()
                cursor.close()
                
                logger.info(f"{symbol}: 保存了 {len(records)} 条季度财务数据")
                return len(records)
                
        except Exception as e:
            logger.error(f"{symbol}: 保存季度数据失败 - {e}")
            return 0
    
    def save_daily_valuation_data(self, symbol: str, df: pd.DataFrame) -> int:
        """
        保存日度估值数据（批量插入/更新）
        
        Parameters
        ----------
        symbol : str
            股票代码（带后缀）
        df : pd.DataFrame
            日度估值数据，必须包含 trade_date 列
        
        Returns
        -------
        count : int
            成功保存的记录数
        """
        if df is None or df.empty:
            return 0
        
        df = df.copy()
        df = self._clean_financial_dataframe(df, mode='daily')
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(
                df['trade_date'], errors='coerce'
            ).dt.date
            df = df.dropna(subset=['trade_date'])
        
        if df.empty:
            return 0
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # 准备插入数据
                records = []
                for _, row in df.iterrows():
                    trade_date = self._to_date(row.get('trade_date'))
                    record = (
                        symbol,
                        trade_date,
                        row.get('pe_ttm'),
                        row.get('pb'),
                        row.get('ps_ttm'),
                        row.get('market_cap'),
                        row.get('circulating_market_cap'),
                        row.get('dividend_yield'),
                        row.get('pe_percentile'),
                        row.get('pb_percentile')
                    )
                    records.append(record)
                
                # 批量插入
                sql = """
                    INSERT INTO fundamental_data_daily (
                        symbol, trade_date, pe_ttm, pb, ps_ttm,
                        market_cap, circulating_market_cap, dividend_yield,
                        pe_percentile, pb_percentile
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON DUPLICATE KEY UPDATE
                        pe_ttm = VALUES(pe_ttm),
                        pb = VALUES(pb),
                        ps_ttm = VALUES(ps_ttm),
                        market_cap = VALUES(market_cap),
                        circulating_market_cap = VALUES(circulating_market_cap),
                        dividend_yield = VALUES(dividend_yield),
                        pe_percentile = VALUES(pe_percentile),
                        pb_percentile = VALUES(pb_percentile),
                        updated_at = CURRENT_TIMESTAMP
                """
                
                if records:
                    cursor.executemany(sql, records)
                    conn.commit()
                
                logger.info(f"{symbol}: 保存了 {len(records)} 条日度估值数据")
                return len(records)
            
        except Exception as e:
            logger.error(f"{symbol}: 保存日度估值数据失败 - {e}")
            return 0
    
    def get_quarterly_data(
        self, 
        symbol: str, 
        end_date: datetime,
        lookback_quarters: int = 8
    ) -> Optional[pd.DataFrame]:
        """
        获取季度财务数据（从数据库）
        
        Parameters
        ----------
        symbol : str
            股票代码（带后缀）
        end_date : datetime
            截止日期
        lookback_quarters : int
            回溯季度数
        
        Returns
        -------
        df : pd.DataFrame or None
            季度财务数据
        """
        try:
            # 计算起始日期（往前推2年）
            start_date = end_date - timedelta(days=365 * 2)
            
            with self.db_manager.get_connection() as conn:
                sql = """
                    SELECT * FROM fundamental_data_quarterly
                    WHERE symbol = %s
                      AND report_date <= %s
                      AND report_date >= %s
                    ORDER BY report_date DESC
                    LIMIT %s
                """
                
                df = pd.read_sql(
                    sql, 
                    conn, 
                    params=(symbol, end_date.date(), start_date.date(), lookback_quarters)
                )
                
                if df.empty:
                    return None
                
                return df
            
        except Exception as e:
            logger.debug(f"{symbol}: 读取季度数据失败 - {e}")
            return None
    
    def get_daily_valuation(
        self, 
        symbol: str, 
        trade_date: datetime
    ) -> Optional[pd.Series]:
        """
        获取某日的估值数据
        
        Parameters
        ----------
        symbol : str
            股票代码（带后缀）
        trade_date : datetime
            交易日期
        
        Returns
        -------
        data : pd.Series or None
            估值数据
        """
        try:
            with self.db_manager.get_connection() as conn:
                sql = """
                    SELECT * FROM fundamental_data_daily
                    WHERE symbol = %s
                      AND trade_date = %s
                    LIMIT 1
                """
                
                df = pd.read_sql(sql, conn, params=(symbol, trade_date.date()))
                
                if df.empty:
                    return None
                
                return df.iloc[0]
            
        except Exception as e:
            logger.debug(f"{symbol}@{trade_date.date()}: 读取估值数据失败 - {e}")
            return None
    
    def get_quarterly_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        获取指定区间的全部季度财务数据
        """
        try:
            with self.db_manager.get_connection() as conn:
                sql = """
                    SELECT symbol, report_date, publish_date,
                           revenue, net_profit, gross_profit_margin, net_profit_margin,
                           roe, roa, eps, revenue_yoy, net_profit_yoy,
                           debt_to_asset, current_ratio, quick_ratio,
                           operating_cash_flow, total_assets, total_equity
                    FROM fundamental_data_quarterly
                    WHERE symbol = %s
                      AND report_date BETWEEN %s AND %s
                    ORDER BY COALESCE(publish_date, report_date)
                """
                df = pd.read_sql(
                    sql,
                    conn,
                    params=(
                        symbol,
                        pd.Timestamp(start_date).date(),
                        pd.Timestamp(end_date).date()
                    )
                )
                return df
        except Exception as e:
            logger.debug(f"{symbol}: 区间季度数据读取失败 - {e}")
            return pd.DataFrame()
    
    def get_daily_valuation_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        获取指定区间的日度估值数据
        """
        try:
            with self.db_manager.get_connection() as conn:
                sql = """
                    SELECT symbol, trade_date,
                           pe_ttm, pb
                    FROM fundamental_data_daily
                    WHERE symbol = %s
                      AND trade_date BETWEEN %s AND %s
                    ORDER BY trade_date
                """
                df = pd.read_sql(
                    sql,
                    conn,
                    params=(
                        symbol,
                        pd.Timestamp(start_date).date(),
                        pd.Timestamp(end_date).date()
                    )
                )
                return df
        except Exception as e:
            logger.debug(f"{symbol}: 区间估值数据读取失败 - {e}")
            return pd.DataFrame()
    
    def save_daily_expanded(self, symbol: str, df: pd.DataFrame) -> int:
        """
        保存季报日频化后的数据
        """
        if df is None or df.empty:
            return 0
        
        df = df.copy()
        df['trade_date'] = pd.to_datetime(df['date']).dt.date
        df.drop(columns=['date'], inplace=True, errors='ignore')
        df = df.where(pd.notna(df), None)
        
        columns = [
            'revenue', 'net_profit', 'net_profit_margin', 'roe', 'eps',
            'revenue_yoy_growth', 'net_profit_yoy_growth',
            'debt_to_asset', 'current_ratio', 'quick_ratio',
            'pe_ttm', 'pb'
        ]
        
        records = []
        for _, row in df.iterrows():
            record = (
                symbol,
                row['trade_date'],
                row.get('revenue'),
                row.get('net_profit'),
                row.get('net_profit_margin'),
                row.get('roe'),
                row.get('eps'),
                row.get('revenue_yoy'),
                row.get('net_profit_yoy'),
                row.get('debt_to_asset'),
                row.get('current_ratio'),
                row.get('quick_ratio'),
                row.get('pe_ttm'),
                row.get('pb')
            )
            records.append(record)
        
        if not records:
            return 0
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                sql = """
                    INSERT INTO fundamental_data_daily_expanded (
                        symbol, trade_date,
                        revenue, net_profit, net_profit_margin, roe, eps,
                        revenue_yoy, net_profit_yoy,
                        debt_to_asset, current_ratio, quick_ratio,
                        pe_ttm, pb
                    ) VALUES (
                        %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s
                    ) ON DUPLICATE KEY UPDATE
                        revenue = VALUES(revenue),
                        net_profit = VALUES(net_profit),
                        net_profit_margin = VALUES(net_profit_margin),
                        roe = VALUES(roe),
                        eps = VALUES(eps),
                        revenue_yoy = VALUES(revenue_yoy),
                        net_profit_yoy = VALUES(net_profit_yoy),
                        debt_to_asset = VALUES(debt_to_asset),
                        current_ratio = VALUES(current_ratio),
                        quick_ratio = VALUES(quick_ratio),
                        pe_ttm = VALUES(pe_ttm),
                        pb = VALUES(pb),
                        updated_at = CURRENT_TIMESTAMP
                """
                cursor.executemany(sql, records)
                conn.commit()
                return len(records)
        except Exception as e:
            logger.error(f"{symbol}: 保存日频化基本面数据失败 - {e}")
            return 0
    
    def delete_daily_expanded(self, symbol_variants: List[str]) -> None:
        if not symbol_variants:
            return
        placeholders = ','.join(['%s'] * len(symbol_variants))
        sql = f"DELETE FROM fundamental_data_daily_expanded WHERE symbol IN ({placeholders})"
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, symbol_variants)
                conn.commit()
        except Exception as e:
            logger.warning(f"删除日频化数据失败: {e}")
    
    def get_data_coverage(self, symbols: List[str]) -> pd.DataFrame:
        """
        检查数据覆盖率
        
        Parameters
        ----------
        symbols : list of str
            股票代码列表
        
        Returns
        -------
        df : pd.DataFrame
            数据覆盖统计
        """
        try:
            with self.db_manager.get_connection() as conn:
                # 查询每只股票的数据记录数
                placeholders = ','.join(['%s'] * len(symbols))
                sql = f"""
                    SELECT 
                        symbol,
                        COUNT(*) as quarterly_count,
                        MIN(report_date) as earliest_quarter,
                        MAX(report_date) as latest_quarter
                    FROM fundamental_data_quarterly
                    WHERE symbol IN ({placeholders})
                    GROUP BY symbol
                """
                
                df_quarterly = pd.read_sql(sql, conn, params=symbols)
                
                sql = f"""
                    SELECT 
                        symbol,
                        COUNT(*) as daily_count,
                        MIN(trade_date) as earliest_date,
                        MAX(trade_date) as latest_date
                    FROM fundamental_data_daily
                    WHERE symbol IN ({placeholders})
                    GROUP BY symbol
                """
                
                df_daily = pd.read_sql(sql, conn, params=symbols)
                
                # 合并结果
                df = pd.merge(
                    df_quarterly, 
                    df_daily, 
                    on='symbol', 
                    how='outer', 
                    suffixes=('_quarterly', '_daily')
                )
                
                return df
            
        except Exception as e:
            logger.error(f"查询数据覆盖率失败: {e}")
            return pd.DataFrame()
            if conn:
                conn.close()
    
    def bulk_update_from_akshare(
        self, 
        symbols: List[str], 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        update_quarterly: bool = True,
        update_daily: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        per_symbol_sleep: float = 0.0
    ) -> Dict[str, int]:
        """
        从AKShare批量更新基本面数据
        
        Parameters
        ----------
        symbols : list of str
            股票代码列表（带后缀）
        start_date : str, optional
            起始日期（日度数据用）
        end_date : str, optional
            结束日期（日度数据用）
        update_quarterly : bool
            是否更新季度数据
        update_daily : bool
            是否更新日度数据
        
        Returns
        -------
        stats : dict
            更新统计信息
        """
        stats = {
            'success': 0,
            'failed': 0,
            'quarterly_records': 0,
            'daily_records': 0,
            'failed_symbols': []
        }
        
        for symbol in symbols:
            try:
                symbol_clean = symbol.split('.')[0]
                has_success = False
                
                # 更新季度数据
                if update_quarterly:
                    try:
                        df_quarterly = self._fetch_quarterly_financials(
                            symbol=symbol,
                            max_retries=max_retries,
                            retry_delay=retry_delay
                        )
                        
                        if df_quarterly is not None and not df_quarterly.empty:
                            count = self.save_quarterly_data(symbol, df_quarterly)
                            stats['quarterly_records'] += count
                            has_success = has_success or count > 0
                        else:
                            logger.warning(f"{symbol}: 未获取到季度财务数据")
                    except Exception as e:
                        logger.warning(f"{symbol}: 更新季度数据失败 - {e}")
                
                # 更新日度估值数据
                if update_daily:
                    try:
                        df_daily = self._fetch_daily_valuation_data(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            max_retries=max_retries,
                            retry_delay=retry_delay
                        )
                        
                        if df_daily is not None and not df_daily.empty:
                            count = self.save_daily_valuation_data(symbol, df_daily)
                            stats['daily_records'] += count
                            has_success = has_success or count > 0
                        else:
                            logger.debug(f"{symbol}: 未获取到日度估值数据")
                    except Exception as e:
                        logger.warning(f"{symbol}: 更新日度数据失败 - {e}")
                
                if has_success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['failed_symbols'].append(symbol)
                
            except Exception as e:
                logger.error(f"{symbol}: 批量更新失败 - {e}")
                stats['failed'] += 1
                stats['failed_symbols'].append(symbol)
            
            if per_symbol_sleep > 0:
                time.sleep(per_symbol_sleep)
        
        logger.info(f"批量更新完成: 成功{stats['success']}只, 失败{stats['failed']}只")
        logger.info(f"  - 季度数据: {stats['quarterly_records']}条")
        logger.info(f"  - 日度数据: {stats['daily_records']}条")
        if stats['failed_symbols']:
            logger.warning(
                f"失败股票示例: {stats['failed_symbols'][:20]}"
            )
        
        return stats
    
    def _clean_quarterly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗季度财务数据（将AKShare返回的数据标准化）"""
        # TODO: 根据实际AKShare返回的列名进行映射
        # 这里需要根据具体的API返回格式进行调整
        df_clean = df.copy()
        
        # 列名映射（示例）
        column_mapping = {
            '报告期': 'report_date',
            '公告日期': 'publish_date',
            '营业收入': 'revenue',
            '营业总收入': 'revenue',
            '净利润': 'net_profit',
            '毛利率': 'gross_profit_margin',
            '销售净利率': 'net_profit_margin',
            '净利率': 'net_profit_margin',
            'ROE': 'roe',
            '净资产收益率': 'roe',
            'ROA': 'roa',
            'EPS': 'eps',
            '基本每股收益': 'eps',
            '营收同比': 'revenue_yoy',
            '营业总收入同比增长率': 'revenue_yoy',
            '净利润同比': 'net_profit_yoy',
            '净利润同比增长率': 'net_profit_yoy',
            '资产负债率': 'debt_to_asset',
            '流动比率': 'current_ratio',
            '速动比率': 'quick_ratio',
            '保守速动比率': 'quick_ratio_conservative',
            '每股经营现金流': 'operating_cash_flow_per_share'
        }
        
        df_clean.rename(columns=column_mapping, inplace=True)
        
        # 日期转换
        if 'report_date' in df_clean.columns:
            df_clean['report_date'] = pd.to_datetime(df_clean['report_date'])
        
        df_clean = self._clean_financial_dataframe(df_clean, mode='quarterly')
        df_clean = self._normalize_quarterly_dates(df_clean)
        
        return df_clean
    
    def _fetch_quarterly_financials(
        self,
        symbol: str,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> pd.DataFrame:
        """从 AKShare 获取季度财务数据并清洗"""
        numeric_code = symbol.split('.')[0]
        last_error: Optional[Exception] = None
        
        for attempt in range(1, max_retries + 1):
            try:
                df_raw = ak.stock_financial_abstract_ths(
                    symbol=numeric_code,
                    indicator="按报告期"
                )
                
                if df_raw is None or df_raw.empty:
                    logger.warning(f"{symbol}: 第{attempt}次获取季度数据为空")
                    last_error = None
                else:
                    return self._clean_quarterly_data(df_raw)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{symbol}: 第{attempt}次获取季度数据失败 - {e}"
                )
            time.sleep(max(retry_delay * attempt, 0.0))
        
        if last_error:
            logger.error(f"{symbol}: 无法获取季度财务数据，最后错误: {last_error}")
        return pd.DataFrame()

    def _fetch_daily_valuation_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_retries: int = 2,
        retry_delay: float = 3.0
    ) -> pd.DataFrame:
        """
        通过百度股市通接口获取日度估值数据

        指标覆盖：总市值、市盈率(TTM)、市盈率(静)、市净率、市现率
        返回值包含 trade_date、symbol、market_cap、pe_ttm、pb 等字段
        """
        code = symbol.split('.')[0]
        indicator_map = {
            'market_cap': ('总市值', lambda v: v * 1e8 if v is not None else None),
            'pe_ttm': ('市盈率(TTM)', None),
            'pe_static': ('市盈率(静)', None),
            'pb': ('市净率', None),
            'pcf': ('市现率', None),
        }
        
        period = self._infer_baidu_period(start_date, end_date)
        frames: List[pd.DataFrame] = []
        
        for column, (indicator, transform) in indicator_map.items():
            data = None
            last_error: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    df_raw = ak.stock_zh_valuation_baidu(
                        symbol=code,
                        indicator=indicator,
                        period=period
                    )
                    if df_raw is not None and not df_raw.empty:
                        df_raw = df_raw.rename(columns={'date': 'trade_date', 'value': column})
                        if transform:
                            df_raw[column] = df_raw[column].apply(transform)
                        frames.append(df_raw)
                    else:
                        logger.debug(f"{symbol}: 指标 {indicator} 未返回数据")
                    data = df_raw
                    break
                except Exception as e:
                    last_error = e
                    logger.debug(f"{symbol}: 指标 {indicator} 获取失败 (尝试 {attempt}/{max_retries}) - {e}")
                    time.sleep(max(retry_delay * attempt, 0.0))
            if data is None and last_error:
                logger.info(f"{symbol}: 无法获取指标 {indicator}, 错误: {last_error}")
        
        if not frames:
            return pd.DataFrame()
        
        df_merged = reduce(
            lambda left, right: pd.merge(left, right, on='trade_date', how='outer'),
            frames
        )
        
        df_merged['trade_date'] = pd.to_datetime(df_merged['trade_date'], errors='coerce').dt.date
        df_merged = df_merged.dropna(subset=['trade_date'])
        df_merged = df_merged.sort_values('trade_date').drop_duplicates(subset=['trade_date'], keep='last')
        
        if start_date:
            df_merged = df_merged[df_merged['trade_date'] >= pd.to_datetime(start_date).date()]
        if end_date:
            df_merged = df_merged[df_merged['trade_date'] <= pd.to_datetime(end_date).date()]
        
        df_merged['symbol'] = symbol
        
        # 补齐数据库所需字段
        expected_columns = [
            'trade_date', 'symbol', 'pe_ttm', 'pb', 'ps_ttm', 'market_cap',
            'circulating_market_cap', 'dividend_yield', 'pe_percentile', 'pb_percentile'
        ]
        for col in expected_columns:
            if col not in df_merged.columns:
                df_merged[col] = None
        
        # 将市现率映射到辅助列，暂不入库
        if 'pcf' in df_merged.columns:
            df_merged['pcf'] = df_merged['pcf']
        
        return df_merged[expected_columns + [col for col in df_merged.columns if col not in expected_columns]]

    @staticmethod
    def _infer_baidu_period(start_date: Optional[str], end_date: Optional[str]) -> str:
        """
        根据时间范围推断百度接口需要的 period 参数
        """
        if not start_date or not end_date:
            return '近五年'
        
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            delta_years = (end - start).days / 365.25
            if delta_years <= 1:
                return '近一年'
            if delta_years <= 3:
                return '近三年'
            if delta_years <= 5:
                return '近五年'
            if delta_years <= 10:
                return '近十年'
            return '全部'
        except Exception:
            return '近五年'

    # ----------------------------------------------------------------------------------
    # 数据清洗工具
    # ----------------------------------------------------------------------------------

    def _clean_numeric_value(self, value: Union[str, float, int, None]) -> Optional[float]:
        """
        清洗单个数值，处理中文单位、百分比以及常见占位符
        """
        if value is None:
            return None
        
        if isinstance(value, (int, float, np.number)):
            if np.isnan(value):
                return None
            return float(value)
        
        value_str = str(value).strip()
        if value_str == '':
            return None
        
        normalized = value_str.replace(',', '')
        normalized = normalized.replace('，', '')
        normalized = normalized.replace(' ', '')
        
        # 常见的空值/无效标记
        if normalized in {'--', '-', 'nan', 'NaN', 'None', 'null'}:
            return None
        
        multiplier = 1.0
        percent = False
        negative = False
        
        # 处理括号表示的负数，例如 (1.23亿)
        if normalized.startswith('(') and normalized.endswith(')'):
            negative = True
            normalized = normalized[1:-1]
        
        # 提前处理百分号，避免和单位混淆
        if normalized.endswith('%') or normalized.endswith('％'):
            percent = True
            normalized = normalized[:-1]
        
        # 中文单位转换
        if normalized.endswith('万亿'):
            multiplier = 1e12
            normalized = normalized[:-2]
        elif normalized.endswith('亿'):
            multiplier = 1e8
            normalized = normalized[:-1]
        elif normalized.endswith('万'):
            multiplier = 1e4
            normalized = normalized[:-1]
        elif normalized.endswith('千'):
            multiplier = 1e3
            normalized = normalized[:-1]
        
        # 去除可能残留的中文字符
        normalized = re.sub(r'[^\d\.\-eE+]', '', normalized)
        
        if normalized in {'', '.', '-'}:
            return None
        
        try:
            numeric = float(normalized) * multiplier
            if percent:
                numeric /= 100.0
            if negative:
                numeric = -numeric
            return numeric
        except ValueError:
            return None

    def _clean_financial_dataframe(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        批量清洗DataFrame中的数值列

        Parameters
        ----------
        df : pd.DataFrame
            待清洗的数据
        mode : str
            'quarterly' 或 'daily'
        """
        df_clean = df.copy()
        if df_clean.empty:
            return df_clean
        
        if mode == 'quarterly':
            numeric_columns = [
                'revenue', 'net_profit', 'gross_profit_margin', 'net_profit_margin',
                'roe', 'roa', 'eps', 'revenue_yoy', 'net_profit_yoy', 'eps_yoy',
                'debt_to_asset', 'current_ratio', 'quick_ratio', 
                'operating_cash_flow', 'total_assets', 'total_equity'
            ]
        elif mode == 'daily':
            numeric_columns = [
                'pe_ttm', 'pb', 'ps_ttm', 'market_cap',
                'circulating_market_cap', 'dividend_yield',
                'pe_percentile', 'pb_percentile'
            ]
        else:
            numeric_columns = []
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self._clean_numeric_value)
        
        return df_clean

    def _normalize_quarterly_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保季度数据中的日期列为 datetime/date 类型"""
        df_normalized = df.copy()
        if 'report_date' in df_normalized.columns:
            df_normalized['report_date'] = pd.to_datetime(
                df_normalized['report_date'], errors='coerce'
            ).dt.date
        if 'publish_date' in df_normalized.columns:
            df_normalized['publish_date'] = pd.to_datetime(
                df_normalized['publish_date'], errors='coerce'
            ).dt.date
        df_normalized = df_normalized.dropna(subset=['report_date'])
        return df_normalized

    @staticmethod
    def _to_date(value: Union[datetime, pd.Timestamp, date, str, None]) -> Optional[date]:
        """将各种日期类型转换为 date"""
        if value is None or pd.isna(value):
            return None
        
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        
        if isinstance(value, datetime):
            return value.date()
        
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            return value.to_pydatetime().date()
        
        if isinstance(value, np.datetime64):
            parsed = pd.to_datetime(value, errors='coerce')
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime().date()
        
        if isinstance(value, str) and value != '':
            parsed = pd.to_datetime(value, errors='coerce')
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime().date()
        
        return None
