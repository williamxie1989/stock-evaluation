#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/Users/xieyongliang/stock-evaluation')

from db import DatabaseManager
from datetime import datetime, timedelta

def comprehensive_data_analysis():
    """全面分析数据完整性问题"""
    db = DatabaseManager()
    
    with db.get_conn() as conn:
        cursor = conn.cursor()
        
        print("=== 股票数据完整性分析报告 ===")
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "="*60)
        
        # 1. 基础统计
        print("\n1. 基础数据统计")
        cursor.execute("SELECT COUNT(*) FROM stocks")
        total_stocks = cursor.fetchone()[0]
        print(f"   stocks表总股票数: {total_stocks:,}")
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM prices_daily")
        stocks_with_prices = cursor.fetchone()[0]
        print(f"   有价格数据的股票数: {stocks_with_prices:,}")
        
        coverage_rate = (stocks_with_prices / total_stocks * 100) if total_stocks > 0 else 0
        print(f"   数据覆盖率: {coverage_rate:.2f}%")
        
        # 2. 各市场数据分布
        print("\n2. 各市场数据分布")
        cursor.execute("""
            SELECT s.market, s.board_type, 
                   COUNT(*) as total_stocks,
                   COUNT(DISTINCT p.symbol) as stocks_with_data,
                   ROUND(COUNT(DISTINCT p.symbol) * 100.0 / COUNT(*), 2) as coverage_rate
            FROM stocks s
            LEFT JOIN prices_daily p ON s.symbol = p.symbol
            WHERE s.market IN ('SH', 'SZ', 'BJ')
            GROUP BY s.market, s.board_type
            ORDER BY s.market, s.board_type
        """)
        
        market_data = cursor.fetchall()
        print(f"   {'市场':<8} {'板块':<12} {'总数':<8} {'有数据':<8} {'覆盖率':<8}")
        print("   " + "-"*50)
        
        for market, board_type, total, with_data, rate in market_data:
            with_data = with_data or 0
            rate = rate or 0.0
            board_type = board_type or 'N/A'
            print(f"   {market:<8} {board_type:<12} {total:<8} {with_data:<8} {rate:<8.2f}%")
        
        # 3. 数据时间范围分析
        print("\n3. 数据时间范围分析")
        cursor.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM prices_daily")
        min_date, max_date, trading_days = cursor.fetchone()
        print(f"   数据时间范围: {min_date} 到 {max_date}")
        print(f"   交易日总数: {trading_days:,}天")
        
        # 检查最近数据
        recent_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol) 
            FROM prices_daily 
            WHERE date >= ?
        """, (recent_date,))
        recent_stocks = cursor.fetchone()[0]
        print(f"   最近7天有数据的股票: {recent_stocks}只")
        
        # 4. 问题股票分析
        print("\n4. 问题股票分析")
        
        # 603开头股票分析
        cursor.execute("""
            SELECT COUNT(*) 
            FROM stocks 
            WHERE symbol LIKE '603%'
        """)
        total_603 = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(DISTINCT s.symbol)
            FROM stocks s
            LEFT JOIN prices_daily p ON s.symbol = p.symbol
            WHERE s.symbol LIKE '603%' AND p.symbol IS NULL
        """)
        missing_603 = cursor.fetchone()[0]
        
        print(f"   603开头股票总数: {total_603}")
        print(f"   603开头无数据股票: {missing_603}")
        print(f"   603开头缺失率: {(missing_603/total_603*100):.1f}%")
        
        # 5. 数据质量分析
        print("\n5. 数据质量分析")
        cursor.execute("""
            SELECT symbol, COUNT(*) as record_count,
                   MIN(date) as first_date, MAX(date) as last_date
            FROM prices_daily
            GROUP BY symbol
            HAVING COUNT(*) < 30
            ORDER BY record_count
            LIMIT 10
        """)
        
        insufficient_data = cursor.fetchall()
        print(f"   数据不足股票(少于30条记录)示例:")
        for symbol, count, first_date, last_date in insufficient_data:
            print(f"     {symbol}: {count}条记录 ({first_date} 到 {last_date})")
        
        # 6. 建议和解决方案
        print("\n6. 问题诊断和建议")
        print("   " + "="*50)
        
        if coverage_rate < 50:
            print("   🚨 严重问题: 数据覆盖率过低")
            print("   建议:")
            print("     1. 运行完整的数据同步: python data_sync_service.py")
            print("     2. 检查数据源配置和网络连接")
            print("     3. 考虑分批次同步数据，避免网络超时")
        
        if missing_603 > total_603 * 0.8:
            print("   ⚠️  603开头股票数据严重缺失")
            print("   建议:")
            print("     1. 检查股票代码格式是否正确")
            print("     2. 验证数据源是否包含这些股票")
            print("     3. 运行数据修复服务针对性修复")
        
        if recent_stocks < stocks_with_prices * 0.9:
            print("   📅 数据更新不及时")
            print("   建议:")
            print("     1. 设置定时任务自动同步数据")
            print("     2. 检查数据同步服务的运行状态")
        
        print("\n7. 推荐的修复步骤")
        print("   " + "="*50)
        print("   步骤1: 运行数据修复服务")
        print("          python data_repair_service.py")
        print("   步骤2: 批量同步缺失数据")
        print("          python -c \"from data_sync_service import DataSyncService; ds=DataSyncService(); ds.sync_market_data(max_symbols=100)\"")
        print("   步骤3: 验证修复结果")
        print("          python comprehensive_data_analysis.py")
        
if __name__ == "__main__":
    comprehensive_data_analysis()