#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库完整性分析脚本
分析stock_data.sqlite3中各表的数据情况
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

def analyze_database():
    db_path = '/Users/xieyongliang/stock-evaluation/stock_data.sqlite3'
    
    if not os.path.exists(db_path):
        print(f"数据库文件不存在: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=" * 80)
    print("股票数据库完整性分析报告")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. 获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"\n数据库中共有 {len(tables)} 个表:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # 2. 分析每个表的基本信息
    print("\n" + "=" * 50)
    print("各表数据统计")
    print("=" * 50)
    
    for table_name in [t[0] for t in tables]:
        print(f"\n【{table_name}】")
        
        # 获取表结构
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"  字段数: {len(columns)}")
        print(f"  字段: {', '.join([col[1] for col in columns])}")
        
        # 获取记录数
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  记录数: {count:,}")
        
        # 如果有数据，显示样本
        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            samples = cursor.fetchall()
            print(f"  样本数据:")
            for i, sample in enumerate(samples, 1):
                print(f"    {i}: {sample}")
    
    # 3. 重点分析股票基础信息表
    print("\n" + "=" * 50)
    print("股票基础信息分析")
    print("=" * 50)
    
    if 'stocks' in [t[0] for t in tables]:
        # 按市场统计股票数量
        cursor.execute("""
            SELECT market, COUNT(*) as count 
            FROM stocks 
            GROUP BY market 
            ORDER BY count DESC
        """)
        market_stats = cursor.fetchall()
        print("\n各市场股票数量:")
        total_stocks = 0
        for market, count in market_stats:
            print(f"  {market}: {count:,} 只")
            total_stocks += count
        print(f"  总计: {total_stocks:,} 只")
        
        # 分析股票代码分布
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN symbol LIKE '00%' THEN '深市主板(00)'
                    WHEN symbol LIKE '30%' THEN '创业板(30)'
                    WHEN symbol LIKE '60%' THEN '沪市主板(60)'
                    WHEN symbol LIKE '68%' THEN '科创板(68)'
                    WHEN symbol LIKE '8%' OR symbol LIKE '4%' THEN '北交所(8/4)'
                    ELSE '其他'
                END as code_type,
                COUNT(*) as count
            FROM stocks
            GROUP BY code_type
            ORDER BY count DESC
        """)
        code_stats = cursor.fetchall()
        print("\n按代码前缀分类:")
        for code_type, count in code_stats:
            print(f"  {code_type}: {count:,} 只")
    
    # 4. 分析历史价格数据
    print("\n" + "=" * 50)
    print("历史价格数据分析")
    print("=" * 50)
    
    if 'prices_daily' in [t[0] for t in tables]:
        # 总体统计
        cursor.execute("SELECT COUNT(*) FROM prices_daily")
        total_price_records = cursor.fetchone()[0]
        print(f"\n历史价格记录总数: {total_price_records:,}")
        
        # 按股票统计数据量
        cursor.execute("""
            SELECT symbol, COUNT(*) as days
            FROM prices_daily
            GROUP BY symbol
            ORDER BY days DESC
            LIMIT 10
        """)
        top_stocks = cursor.fetchall()
        print("\n数据最多的10只股票:")
        for symbol, days in top_stocks:
            print(f"  {symbol}: {days:,} 天")
        
        # 统计有历史数据的股票数量
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol) FROM prices_daily
        """)
        stocks_with_data = cursor.fetchone()[0]
        print(f"\n有历史数据的股票数量: {stocks_with_data:,}")
        
        # 数据覆盖率
        if total_stocks > 0:
            coverage_rate = (stocks_with_data / total_stocks) * 100
            print(f"数据覆盖率: {coverage_rate:.2f}%")
        
        # 按市场分析数据覆盖情况
        cursor.execute("""
            SELECT 
                s.market,
                COUNT(DISTINCT s.symbol) as total_stocks,
                COUNT(DISTINCT p.symbol) as stocks_with_data,
                ROUND(COUNT(DISTINCT p.symbol) * 100.0 / COUNT(DISTINCT s.symbol), 2) as coverage_rate
            FROM stocks s
            LEFT JOIN prices_daily p ON s.symbol = p.symbol
            GROUP BY s.market
            ORDER BY coverage_rate DESC
        """)
        market_coverage = cursor.fetchall()
        print("\n各市场数据覆盖情况:")
        for market, total, with_data, rate in market_coverage:
            print(f"  {market}: {with_data}/{total} ({rate}%)")
        
        # 分析数据时间范围
        cursor.execute("""
            SELECT 
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(DISTINCT date) as trading_days
            FROM prices_daily
        """)
        date_range = cursor.fetchone()
        if date_range[0]:
            print(f"\n数据时间范围: {date_range[0]} 至 {date_range[1]}")
            print(f"交易日数量: {date_range[2]:,} 天")
        
        # 分析最近数据情况
        recent_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol)
            FROM prices_daily
            WHERE date >= ?
        """, (recent_date,))
        recent_stocks = cursor.fetchone()[0]
        print(f"\n最近30天有数据的股票: {recent_stocks:,} 只")
    
    # 5. 分析历史数据不足的股票
    print("\n" + "=" * 50)
    print("历史数据不足分析")
    print("=" * 50)
    
    # 统计各股票的历史数据天数
    cursor.execute("""
        SELECT 
            s.symbol,
            s.name,
            s.market,
            COALESCE(p.days, 0) as data_days
        FROM stocks s
        LEFT JOIN (
            SELECT symbol, COUNT(*) as days
            FROM prices_daily
            GROUP BY symbol
        ) p ON s.symbol = p.symbol
        ORDER BY data_days ASC
    """)
    stock_data_stats = cursor.fetchall()
    
    # 按数据天数分组统计
    no_data = sum(1 for s in stock_data_stats if s[3] == 0)
    less_than_30 = sum(1 for s in stock_data_stats if 0 < s[3] < 30)
    less_than_100 = sum(1 for s in stock_data_stats if 30 <= s[3] < 100)
    less_than_250 = sum(1 for s in stock_data_stats if 100 <= s[3] < 250)
    more_than_250 = sum(1 for s in stock_data_stats if s[3] >= 250)
    
    print(f"\n按历史数据天数分组:")
    print(f"  无数据: {no_data:,} 只 ({no_data/len(stock_data_stats)*100:.1f}%)")
    print(f"  <30天: {less_than_30:,} 只 ({less_than_30/len(stock_data_stats)*100:.1f}%)")
    print(f"  30-99天: {less_than_100:,} 只 ({less_than_100/len(stock_data_stats)*100:.1f}%)")
    print(f"  100-249天: {less_than_250:,} 只 ({less_than_250/len(stock_data_stats)*100:.1f}%)")
    print(f"  ≥250天: {more_than_250:,} 只 ({more_than_250/len(stock_data_stats)*100:.1f}%)")
    
    # 显示无数据的股票样本
    print(f"\n无历史数据的股票样本（前20只）:")
    no_data_stocks = [s for s in stock_data_stats if s[3] == 0][:20]
    for symbol, name, market, days in no_data_stocks:
        print(f"  {symbol} ({market}) - {name}")
    
    # 6. 检查其他相关表
    print("\n" + "=" * 50)
    print("其他表数据分析")
    print("=" * 50)
    
    other_tables = ['financial_data', 'technical_indicators', 'market_data']
    for table in other_tables:
        if table in [t[0] for t in tables]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"\n{table}: {count:,} 条记录")
            
            if count > 0:
                cursor.execute(f"SELECT COUNT(DISTINCT symbol) FROM {table}")
                unique_stocks = cursor.fetchone()[0]
                print(f"  涉及股票数: {unique_stocks:,} 只")
    
    conn.close()
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

if __name__ == "__main__":
    analyze_database()