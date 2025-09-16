#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from db import DatabaseManager
import pandas as pd

def analyze_insufficient_data():
    """分析历史数据不足的股票情况"""
    db = DatabaseManager()
    
    with db.get_conn() as conn:
        # 1. 检查记录数少于30条的股票详情
        print("=== 记录数少于30条的股票详情（前20只） ===")
        insufficient_query = """
        SELECT 
            s.symbol, 
            s.name, 
            s.market, 
            s.board_type, 
            COUNT(p.date) as record_count,
            MIN(p.date) as start_date,
            MAX(p.date) as end_date
        FROM stocks s 
        LEFT JOIN prices_daily p ON s.symbol = p.symbol 
        GROUP BY s.symbol 
        HAVING COUNT(p.date) < 30 
        ORDER BY record_count DESC 
        LIMIT 20
        """
        insufficient_df = pd.read_sql_query(insufficient_query, conn)
        print(insufficient_df)
        
        # 2. 按市场分析数据不足情况
        print("\n=== 各市场数据不足股票统计 ===")
        market_query = """
        SELECT 
            s.market,
            COUNT(*) as total_stocks,
            SUM(CASE WHEN record_count < 30 THEN 1 ELSE 0 END) as insufficient_stocks,
            ROUND(SUM(CASE WHEN record_count < 30 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as insufficient_rate
        FROM (
            SELECT 
                s.symbol, 
                s.market, 
                COUNT(p.date) as record_count
            FROM stocks s 
            LEFT JOIN prices_daily p ON s.symbol = p.symbol 
            GROUP BY s.symbol
        ) s
        GROUP BY s.market
        ORDER BY insufficient_rate DESC
        """
        market_df = pd.read_sql_query(market_query, conn)
        print(market_df)
        
        # 3. 按板块分析数据不足情况
        print("\n=== 各板块数据不足股票统计 ===")
        board_query = """
        SELECT 
            s.board_type,
            COUNT(*) as total_stocks,
            SUM(CASE WHEN record_count < 30 THEN 1 ELSE 0 END) as insufficient_stocks,
            ROUND(SUM(CASE WHEN record_count < 30 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as insufficient_rate
        FROM (
            SELECT 
                s.symbol, 
                s.board_type, 
                COUNT(p.date) as record_count
            FROM stocks s 
            LEFT JOIN prices_daily p ON s.symbol = p.symbol 
            GROUP BY s.symbol
        ) s
        GROUP BY s.board_type
        ORDER BY insufficient_rate DESC
        """
        board_df = pd.read_sql_query(board_query, conn)
        print(board_df)
        
        # 4. 检查完全没有数据的股票
        print("\n=== 完全没有历史数据的股票（前10只） ===")
        no_data_query = """
        SELECT 
            s.symbol, 
            s.name, 
            s.market, 
            s.board_type
        FROM stocks s 
        LEFT JOIN prices_daily p ON s.symbol = p.symbol 
        WHERE p.symbol IS NULL
        LIMIT 10
        """
        no_data_df = pd.read_sql_query(no_data_query, conn)
        print(no_data_df)
        

        
        # 6. 检查最近上市的股票（可能数据不足是正常的）
        print("\n=== 最近上市股票的数据情况 ===")
        recent_query = """
        SELECT 
            s.symbol,
            s.name,
            s.market,
            s.board_type,
            COUNT(p.date) as record_count,
            MIN(p.date) as first_trading_date
        FROM stocks s 
        LEFT JOIN prices_daily p ON s.symbol = p.symbol 
        GROUP BY s.symbol
        HAVING COUNT(p.date) < 30 AND MIN(p.date) >= '2024-01-01'
        ORDER BY first_trading_date DESC
        LIMIT 10
        """
        recent_df = pd.read_sql_query(recent_query, conn)
        print(recent_df)

if __name__ == "__main__":
    analyze_insufficient_data()