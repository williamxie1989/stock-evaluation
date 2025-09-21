#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整查询数据库各类表项数据
分析股票历史数据情况和数据完整性问题
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

def connect_db():
    """连接数据库"""
    db_path = '/Users/xieyongliang/stock-evaluation/stock_data.sqlite3'
    if not os.path.exists(db_path):
        print(f"数据库文件不存在: {db_path}")
        return None
    return sqlite3.connect(db_path)

def get_table_info(conn, table_name):
    """获取表的详细信息"""
    print(f"\n{'='*60}")
    print(f"表名: {table_name}")
    print(f"{'='*60}")
    
    # 获取表结构
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    print("\n表结构:")
    print("-" * 40)
    for col in columns:
        print(f"  {col[1]} ({col[2]}) - {'主键' if col[5] else ''}{'非空' if col[3] else '可空'}")
    
    # 获取记录总数
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_count = cursor.fetchone()[0]
    print(f"\n记录总数: {total_count:,}")
    
    if total_count == 0:
        print("表为空")
        return
    
    # 获取数据样本
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
    samples = cursor.fetchall()
    
    print("\n数据样本 (前5条):")
    print("-" * 40)
    column_names = [col[1] for col in columns]
    for i, sample in enumerate(samples, 1):
        print(f"记录 {i}:")
        for j, value in enumerate(sample):
            print(f"  {column_names[j]}: {value}")
        print()

def analyze_stocks_table(conn):
    """详细分析stocks表"""
    print(f"\n{'='*80}")
    print("STOCKS表详细分析")
    print(f"{'='*80}")
    
    # 基本统计
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM stocks")
    total_stocks = cursor.fetchone()[0]
    print(f"股票总数: {total_stocks:,}")
    
    # 按市场分组统计
    cursor.execute("""
        SELECT market, COUNT(*) as count 
        FROM stocks 
        GROUP BY market 
        ORDER BY count DESC
    """)
    markets = cursor.fetchall()
    print("\n各市场股票数量:")
    for market, count in markets:
        print(f"  {market}: {count:,} 只")
    
    # 检查是否有status列
    cursor.execute("PRAGMA table_info(stocks)")
    columns_info = cursor.fetchall()
    has_status = any(col[1] == 'status' for col in columns_info)
    
    if has_status:
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM stocks 
            GROUP BY status 
            ORDER BY count DESC
        """)
        statuses = cursor.fetchall()
        print("\n各状态股票数量:")
        for status, count in statuses:
            print(f"  {status}: {count:,} 只")
    else:
        print("\n注意: stocks表中没有status列")
    
    # 检查代码格式
    cursor.execute("""
        SELECT 
            CASE 
                WHEN symbol LIKE '%.SZ' THEN '深交所格式(.SZ)'
                WHEN symbol LIKE '%.SS' THEN '上交所格式(.SS)'
                WHEN symbol LIKE '%.BJ' THEN '北交所格式(.BJ)(已移除)'
                WHEN symbol LIKE '%%%%%' THEN '港股格式(5位数字)'
                WHEN LENGTH(symbol) = 6 AND symbol GLOB '[0-9][0-9][0-9][0-9][0-9][0-9]' THEN '6位数字格式'
                ELSE '其他格式'
            END as format_type,
            COUNT(*) as count
        FROM stocks 
        GROUP BY format_type 
        ORDER BY count DESC
    """)
    formats = cursor.fetchall()
    print("\n股票代码格式分布:")
    for format_type, count in formats:
        print(f"  {format_type}: {count:,} 只")

def analyze_prices_daily_table(conn):
    """详细分析prices_daily表"""
    print(f"\n{'='*80}")
    print("PRICES_DAILY表详细分析")
    print(f"{'='*80}")
    
    cursor = conn.cursor()
    
    # 基本统计
    cursor.execute("SELECT COUNT(*) FROM prices_daily")
    total_records = cursor.fetchone()[0]
    print(f"历史价格记录总数: {total_records:,}")
    
    # 股票数量统计
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM prices_daily")
    unique_stocks = cursor.fetchone()[0]
    print(f"有价格数据的股票数量: {unique_stocks:,}")
    
    # 时间范围
    cursor.execute("SELECT MIN(date), MAX(date) FROM prices_daily")
    min_date, max_date = cursor.fetchone()
    print(f"数据时间范围: {min_date} 至 {max_date}")
    
    # 检查是否有data_source列
    cursor.execute("PRAGMA table_info(prices_daily)")
    columns_info = cursor.fetchall()
    has_data_source = any(col[1] == 'data_source' for col in columns_info)
    
    if has_data_source:
        cursor.execute("""
            SELECT data_source, COUNT(*) as count, COUNT(DISTINCT symbol) as stocks
            FROM prices_daily 
            GROUP BY data_source 
            ORDER BY count DESC
        """)
        sources = cursor.fetchall()
        print("\n各数据源统计:")
        for source, count, stocks in sources:
            source_name = source if source else '未知来源'
            print(f"  {source_name}: {count:,} 条记录, {stocks:,} 只股票")
    else:
        print("\n注意: prices_daily表中没有data_source列")
    
    # 代码格式分析
    cursor.execute("""
        SELECT 
            CASE 
                WHEN symbol LIKE '%.SZ' THEN '深交所格式(.SZ)'
                WHEN symbol LIKE '%.SS' THEN '上交所格式(.SS)'
                WHEN symbol LIKE '%.BJ' THEN '北交所格式(.BJ)(已移除)'
                WHEN symbol LIKE '%%%%%' THEN '港股格式(5位数字)'
                WHEN LENGTH(symbol) = 6 AND symbol GLOB '[0-9][0-9][0-9][0-9][0-9][0-9]' THEN '6位数字格式'
                ELSE '其他格式'
            END as format_type,
            COUNT(DISTINCT symbol) as stocks,
            COUNT(*) as records
        FROM prices_daily 
        GROUP BY format_type 
        ORDER BY records DESC
    """)
    formats = cursor.fetchall()
    print("\n价格数据中的股票代码格式分布:")
    for format_type, stocks, records in formats:
        print(f"  {format_type}: {stocks:,} 只股票, {records:,} 条记录")
    
    # 数据密度分析
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
    
    # 最近数据更新情况
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT symbol) as stocks_1d
        FROM prices_daily 
        WHERE date >= date('now', '-1 day')
    """)
    stocks_1d = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT symbol) as stocks_7d
        FROM prices_daily 
        WHERE date >= date('now', '-7 days')
    """)
    stocks_7d = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT symbol) as stocks_30d
        FROM prices_daily 
        WHERE date >= date('now', '-30 days')
    """)
    stocks_30d = cursor.fetchone()[0]
    
    print(f"\n最近数据更新情况:")
    print(f"  1天内有数据的股票: {stocks_1d:,} 只")
    print(f"  7天内有数据的股票: {stocks_7d:,} 只")
    print(f"  30天内有数据的股票: {stocks_30d:,} 只")

def analyze_data_matching(conn):
    """分析stocks表和prices_daily表的数据匹配情况"""
    print(f"\n{'='*80}")
    print("数据匹配分析")
    print(f"{'='*80}")
    
    cursor = conn.cursor()
    
    # 直接匹配
    cursor.execute("""
        SELECT COUNT(*) 
        FROM stocks s 
        INNER JOIN prices_daily p ON s.symbol = p.symbol
    """)
    direct_match = cursor.fetchone()[0]
    
    # stocks表中有但prices_daily表中没有的
    cursor.execute("""
        SELECT COUNT(DISTINCT s.symbol) 
        FROM stocks s 
        LEFT JOIN prices_daily p ON s.symbol = p.symbol 
        WHERE p.symbol IS NULL
    """)
    stocks_no_prices = cursor.fetchone()[0]
    
    # prices_daily表中有但stocks表中没有的
    cursor.execute("""
        SELECT COUNT(DISTINCT p.symbol) 
        FROM prices_daily p 
        LEFT JOIN stocks s ON p.symbol = s.symbol 
        WHERE s.symbol IS NULL
    """)
    prices_no_stocks = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM stocks")
    total_stocks = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM prices_daily")
    total_price_symbols = cursor.fetchone()[0]
    
    print(f"直接匹配的记录数: {direct_match:,}")
    print(f"stocks表总股票数: {total_stocks:,}")
    print(f"prices_daily表总股票数: {total_price_symbols:,}")
    print(f"stocks表中有但prices_daily表中没有的股票: {stocks_no_prices:,} 只")
    print(f"prices_daily表中有但stocks表中没有的股票: {prices_no_stocks:,} 只")
    
    # 显示一些不匹配的样本
    cursor.execute("""
        SELECT s.symbol, s.name, s.market 
        FROM stocks s 
        LEFT JOIN prices_daily p ON s.symbol = p.symbol 
        WHERE p.symbol IS NULL 
        LIMIT 10
    """)
    no_price_samples = cursor.fetchall()
    
    print("\nstocks表中无价格数据的股票样本:")
    for symbol, name, market in no_price_samples:
        print(f"  {symbol} ({market}) - {name}")
    
    cursor.execute("""
        SELECT p.symbol, COUNT(*) as days 
        FROM prices_daily p 
        LEFT JOIN stocks s ON p.symbol = s.symbol 
        WHERE s.symbol IS NULL 
        GROUP BY p.symbol 
        ORDER BY days DESC 
        LIMIT 10
    """)
    orphan_prices = cursor.fetchall()
    
    print("\nprices_daily表中的孤立价格数据样本:")
    for symbol, days in orphan_prices:
        print(f"  {symbol}: {days:,} 天数据")

def main():
    """主函数"""
    print("开始完整查询数据库各类表项数据...")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    conn = connect_db()
    if not conn:
        return
    
    try:
        # 获取所有表名
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\n数据库中共有 {len(tables)} 个表:")
        for table in tables:
            print(f"  - {table}")
        
        # 详细分析每个表
        for table in tables:
            get_table_info(conn, table)
        
        # 重点分析stocks和prices_daily表
        analyze_stocks_table(conn)
        analyze_prices_daily_table(conn)
        
        # 分析数据匹配情况
        analyze_data_matching(conn)
        
        print(f"\n{'='*80}")
        print("完整数据库查询分析完成")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()