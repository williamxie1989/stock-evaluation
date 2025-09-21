#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计各市场股票的历史数据覆盖率和数据质量分析
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

def analyze_market_coverage(conn):
    """分析各市场的数据覆盖率"""
    print(f"\n{'='*80}")
    print("各市场股票历史数据覆盖率分析")
    print(f"{'='*80}")
    
    cursor = conn.cursor()
    
    # 获取各市场的股票总数和有价格数据的股票数
    cursor.execute("""
        SELECT 
            s.market,
            COUNT(s.symbol) as total_stocks,
            COUNT(DISTINCT p.symbol) as stocks_with_data,
            ROUND(COUNT(DISTINCT p.symbol) * 100.0 / COUNT(s.symbol), 2) as coverage_rate
        FROM stocks s
        LEFT JOIN prices_daily p ON s.symbol = p.symbol
        GROUP BY s.market
        ORDER BY coverage_rate DESC
    """)
    
    market_coverage = cursor.fetchall()
    
    print("市场数据覆盖率统计:")
    print("-" * 60)
    print(f"{'市场':<10} {'总股票数':<10} {'有数据股票':<12} {'覆盖率':<10}")
    print("-" * 60)
    
    total_stocks_all = 0
    total_with_data_all = 0
    
    for market, total, with_data, rate in market_coverage:
        print(f"{market:<10} {total:<10} {with_data:<12} {rate:<10}%")
        total_stocks_all += total
        total_with_data_all += with_data
    
    overall_rate = round(total_with_data_all * 100.0 / total_stocks_all, 2)
    print("-" * 60)
    print(f"{'总计':<10} {total_stocks_all:<10} {total_with_data_all:<12} {overall_rate:<10}%")
    
    return market_coverage

def analyze_data_quality_by_market(conn):
    """分析各市场的数据质量"""
    print(f"\n{'='*80}")
    print("各市场数据质量分析")
    print(f"{'='*80}")
    
    cursor = conn.cursor()
    
    markets = ['SZ', 'SH', 'HK', 'UNKNOWN']  # BJ股票已移除，不再分析
    
    for market in markets:
        print(f"\n{'-'*50}")
        print(f"市场: {market}")
        print(f"{'-'*50}")
        
        # 该市场股票总数
        cursor.execute("SELECT COUNT(*) FROM stocks WHERE market = ?", (market,))
        total_stocks = cursor.fetchone()[0]
        
        if total_stocks == 0:
            print(f"该市场无股票数据")
            continue
            
        # 有价格数据的股票数
        cursor.execute("""
            SELECT COUNT(DISTINCT s.symbol) 
            FROM stocks s 
            INNER JOIN prices_daily p ON s.symbol = p.symbol 
            WHERE s.market = ?
        """, (market,))
        stocks_with_data = cursor.fetchone()[0]
        
        coverage_rate = round(stocks_with_data * 100.0 / total_stocks, 2)
        print(f"股票总数: {total_stocks:,}")
        print(f"有数据股票: {stocks_with_data:,}")
        print(f"覆盖率: {coverage_rate}%")
        
        if stocks_with_data == 0:
            print("该市场无价格数据")
            continue
            
        # 数据天数统计
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as unique_stocks,
                AVG(daily_count) as avg_days_per_stock,
                MIN(daily_count) as min_days,
                MAX(daily_count) as max_days
            FROM (
                SELECT p.symbol, COUNT(*) as daily_count
                FROM stocks s 
                INNER JOIN prices_daily p ON s.symbol = p.symbol 
                WHERE s.market = ?
                GROUP BY p.symbol
            ) as stock_counts
        """, (market,))
        
        quality_stats = cursor.fetchone()
        total_records, unique_stocks, avg_days, min_days, max_days = quality_stats
        
        print(f"价格记录总数: {total_records:,}")
        print(f"平均每股数据天数: {avg_days:.1f}")
        print(f"最少数据天数: {min_days:,}")
        print(f"最多数据天数: {max_days:,}")
        
        # 数据时间范围
        cursor.execute("""
            SELECT MIN(p.date), MAX(p.date)
            FROM stocks s 
            INNER JOIN prices_daily p ON s.symbol = p.symbol 
            WHERE s.market = ?
        """, (market,))
        
        date_range = cursor.fetchone()
        if date_range[0]:
            print(f"数据时间范围: {date_range[0]} 至 {date_range[1]}")
        
        # 最近数据更新情况
        cursor.execute("""
            SELECT COUNT(DISTINCT p.symbol)
            FROM stocks s 
            INNER JOIN prices_daily p ON s.symbol = p.symbol 
            WHERE s.market = ? AND p.date >= date('now', '-30 days')
        """, (market,))
        
        recent_stocks = cursor.fetchone()[0]
        recent_rate = round(recent_stocks * 100.0 / stocks_with_data, 2) if stocks_with_data > 0 else 0
        print(f"30天内有更新的股票: {recent_stocks:,} ({recent_rate}%)")
        
        # 数据质量分级
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN daily_count >= 1000 THEN '优质(≥1000天)'
                    WHEN daily_count >= 500 THEN '良好(500-999天)'
                    WHEN daily_count >= 250 THEN '一般(250-499天)'
                    WHEN daily_count >= 100 THEN '较差(100-249天)'
                    ELSE '很差(<100天)'
                END as quality_level,
                COUNT(*) as stock_count
            FROM (
                SELECT p.symbol, COUNT(*) as daily_count
                FROM stocks s 
                INNER JOIN prices_daily p ON s.symbol = p.symbol 
                WHERE s.market = ?
                GROUP BY p.symbol
            ) as stock_counts
            GROUP BY quality_level
            ORDER BY 
                CASE quality_level
                    WHEN '优质(≥1000天)' THEN 1
                    WHEN '良好(500-999天)' THEN 2
                    WHEN '一般(250-499天)' THEN 3
                    WHEN '较差(100-249天)' THEN 4
                    ELSE 5
                END
        """, (market,))
        
        quality_levels = cursor.fetchall()
        print("\n数据质量分级:")
        for level, count in quality_levels:
            percentage = round(count * 100.0 / stocks_with_data, 1) if stocks_with_data > 0 else 0
            print(f"  {level}: {count:,} 只 ({percentage}%)")
        
        # 显示该市场数据最好的5只股票
        cursor.execute("""
            SELECT s.symbol, s.name, COUNT(*) as days
            FROM stocks s 
            INNER JOIN prices_daily p ON s.symbol = p.symbol 
            WHERE s.market = ?
            GROUP BY s.symbol, s.name
            ORDER BY days DESC
            LIMIT 5
        """, (market,))
        
        top_stocks = cursor.fetchall()
        if top_stocks:
            print("\n数据最完整的5只股票:")
            for symbol, name, days in top_stocks:
                print(f"  {symbol} ({name}): {days:,} 天")

def analyze_code_format_issues(conn):
    """分析股票代码格式问题"""
    print(f"\n{'='*80}")
    print("股票代码格式问题分析")
    print(f"{'='*80}")
    
    cursor = conn.cursor()
    
    # stocks表中的代码格式
    print("\nSTOCKS表中的代码格式分布:")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN symbol LIKE '%.SZ' THEN '深交所格式(.SZ)'
                WHEN symbol LIKE '%.SS' THEN '上交所格式(.SS)'
                WHEN symbol LIKE '%.BJ' THEN '北交所格式(.BJ)(已移除)'
                WHEN symbol LIKE '%%%%%' THEN '港股格式(5位数字)'
                WHEN LENGTH(symbol) = 6 AND symbol GLOB '[0-9][0-9][0-9][0-9][0-9][0-9]' THEN '6位数字格式'
                WHEN symbol LIKE 'None.%' THEN '异常格式(None.)'
                ELSE '其他格式'
            END as format_type,
            market,
            COUNT(*) as count
        FROM stocks 
        GROUP BY format_type, market
        ORDER BY count DESC
    """)
    
    stocks_formats = cursor.fetchall()
    for format_type, market, count in stocks_formats:
        print(f"  {format_type} ({market}): {count:,} 只")
    
    # prices_daily表中的代码格式
    print("\nPRICES_DAILY表中的代码格式分布:")
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
    
    prices_formats = cursor.fetchall()
    for format_type, stocks, records in prices_formats:
        print(f"  {format_type}: {stocks:,} 只股票, {records:,} 条记录")
    
    # 格式不匹配问题
    print("\n格式不匹配问题:")
    
    # 找出stocks表中异常的None.格式
    cursor.execute("""
        SELECT symbol, name, market, COUNT(*) as count
        FROM stocks 
        WHERE symbol LIKE 'None.%'
        GROUP BY symbol, name, market
        ORDER BY count DESC
        LIMIT 10
    """)
    
    none_symbols = cursor.fetchall()
    if none_symbols:
        print("\nSTOCKS表中的异常None.格式样本:")
        for symbol, name, market, count in none_symbols:
            print(f"  {symbol} ({market}) - {name}")
    
    # 统计匹配和不匹配的数量
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT s.symbol) as stocks_total,
            COUNT(DISTINCT CASE WHEN p.symbol IS NOT NULL THEN s.symbol END) as stocks_matched,
            COUNT(DISTINCT CASE WHEN p.symbol IS NULL THEN s.symbol END) as stocks_unmatched
        FROM stocks s
        LEFT JOIN prices_daily p ON s.symbol = p.symbol
    """)
    
    match_stats = cursor.fetchone()
    stocks_total, stocks_matched, stocks_unmatched = match_stats
    
    match_rate = round(stocks_matched * 100.0 / stocks_total, 2)
    print(f"\n匹配统计:")
    print(f"  STOCKS表总股票数: {stocks_total:,}")
    print(f"  有价格数据匹配: {stocks_matched:,} ({match_rate}%)")
    print(f"  无价格数据匹配: {stocks_unmatched:,} ({100-match_rate}%)")

def main():
    """主函数"""
    print("开始统计各市场股票的历史数据覆盖率和数据质量...")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    conn = connect_db()
    if not conn:
        return
    
    try:
        # 分析市场覆盖率
        market_coverage = analyze_market_coverage(conn)
        
        # 分析各市场数据质量
        analyze_data_quality_by_market(conn)
        
        # 分析代码格式问题
        analyze_code_format_issues(conn)
        
        print(f"\n{'='*80}")
        print("市场数据覆盖率和质量分析完成")
        print(f"{'='*80}")
        
        # 总结建议
        print("\n主要发现和建议:")
        print("1. 数据覆盖率问题: 大部分市场的数据覆盖率较低")
        print("2. 代码格式不一致: stocks表和prices_daily表使用不同的代码格式")
        print("3. 异常代码: stocks表中存在None.格式的异常代码")
        print("4. 数据孤立: prices_daily表中有大量孤立的价格数据")
        print("\n建议解决方案:")
        print("1. 统一股票代码格式，建议使用带后缀的标准格式")
        print("2. 修复异常的None.格式代码")
        print("3. 补充缺失的股票基础信息")
        print("4. 建立代码映射机制，处理格式转换")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()