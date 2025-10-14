"""
批量填充prices_daily表的复权价字段（open_qfq, close_qfq, high_qfq, low_qfq, open_hfq, close_hfq, high_hfq, low_hfq）
依赖AkshareDataProvider.get_stock_data_with_adjust

优化版本：增加数据完整性检查和智能增量更新功能
支持-r参数重新执行，优先处理缺失数据
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import time
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pymysql
from src.data.providers.akshare_provider import AkshareDataProvider

# 读取.env配置
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
MYSQL_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', ''),
    'port': int(os.getenv('DB_PORT', 3306)),
    'charset': 'utf8mb4'
}
TABLE_NAME = 'prices_daily'
SLEEP_BETWEEN_STOCKS = 0.2
DONE_FILE = Path(__file__).parent / 'fill_adjusted_prices.done.txt'
MISSING_DATA_FILE = Path(__file__).parent / 'fill_adjusted_prices.missing_data.txt'

# 需要检查的14个关键字段
KEY_FIELDS = [
    'open', 'high', 'low', 'close', 'volume', 'amount',
    'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq',
    'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq'
]

# 复权价字段（需要确保完整更新的字段）
ADJUST_FIELDS = [
    'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq',
    'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq'
]

def get_main_a_symbols():
    """
    只返回主板A股（.SH排除688/900，.SZ全部），返回格式：['600000.SH', ...]
    """
    provider = AkshareDataProvider()
    df = provider.get_all_stock_list()
    sh_main = df[(df['market']=='SH') & (df['board_type']=='主板')]
    sz_main = df[(df['market']=='SZ') & (df['board_type']!='B股')]
    symbols = list(sh_main['code'].apply(lambda x: f"{x}.SH")) + list(sz_main['code'].apply(lambda x: f"{x}.SZ"))
    return symbols

def get_all_symbols(conn, valid_symbols):
    sql = f"SELECT DISTINCT symbol FROM {TABLE_NAME} WHERE symbol IN (%s)" % (','.join(['%s']*len(valid_symbols)))
    with conn.cursor() as cur:
        cur.execute(sql, valid_symbols)
        return [row[0] for row in cur.fetchall()]

def get_date_range(conn, symbol):
    sql = f"SELECT MIN(date), MAX(date) FROM {TABLE_NAME} WHERE symbol=%s"
    with conn.cursor() as cur:
        cur.execute(sql, (symbol,))
        return cur.fetchone()

def check_data_completeness(conn, symbol, start_date, end_date):
    """
    检查股票数据完整性，返回缺失数据的日期列表
    
    Args:
        conn: 数据库连接
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        list: 缺失数据的日期列表，如果数据完整则返回空列表
    """
    # 构建检查字段的SQL条件
    field_conditions = []
    for field in KEY_FIELDS:
        field_conditions.append(f"{field} IS NULL")
    
    condition_sql = " OR ".join(field_conditions)
    
    sql = f"""
    SELECT date 
    FROM {TABLE_NAME} 
    WHERE symbol = %s 
      AND date BETWEEN %s AND %s
      AND ({condition_sql})
    ORDER BY date
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, start_date, end_date))
        missing_dates = [row[0] for row in cur.fetchall()]
    
    return missing_dates

def get_missing_date_ranges(missing_dates, threshold_days=20):
    """
    将连续的缺失日期合并为日期范围，提高处理效率
    
    优化策略：如果缺失天数超过阈值，直接进行完整同步
    
    Args:
        missing_dates: 缺失日期列表
        threshold_days: 阈值天数，超过此天数则进行完整同步
        
    Returns:
        list: 日期范围列表，每个元素为(start_date, end_date)
    """
    if not missing_dates:
        return []
    
    missing_dates.sort()
    
    # 如果总缺失天数超过阈值，直接返回完整范围
    total_missing_days = (missing_dates[-1] - missing_dates[0]).days + 1
    if total_missing_days > threshold_days:
        # 直接返回完整范围，避免按小范围逐个处理
        return [(missing_dates[0], missing_dates[-1])]
    
    ranges = []
    current_start = missing_dates[0]
    current_end = missing_dates[0]
    
    for i in range(1, len(missing_dates)):
        # 检查日期是否连续（相差1天）
        if (missing_dates[i] - current_end).days == 1:
            current_end = missing_dates[i]
        else:
            # 检查当前范围的天数是否超过阈值
            current_range_days = (current_end - current_start).days + 1
            if current_range_days > threshold_days:
                # 当前范围超过阈值，直接使用完整范围
                ranges.append((current_start, current_end))
            else:
                ranges.append((current_start, current_end))
            current_start = missing_dates[i]
            current_end = missing_dates[i]
    
    # 处理最后一个范围
    current_range_days = (current_end - current_start).days + 1
    if current_range_days > threshold_days:
        ranges.append((current_start, current_end))
    else:
        ranges.append((current_start, current_end))
    
    return ranges

def update_adjusted_prices(conn, symbol, df):
    """
    更新复权价格数据，支持增量更新
    
    Args:
        conn: 数据库连接
        symbol: 股票代码
        df: 包含复权价格数据的DataFrame
        
    Returns:
        int: 成功更新的记录数
    """
    updated_count = 0
    for _, row in df.iterrows():
        # 检查该日期是否已经有数据，避免重复更新
        check_sql = f"""
        SELECT COUNT(*) FROM {TABLE_NAME} 
        WHERE symbol=%s AND date=%s 
          AND open_qfq IS NOT NULL AND close_qfq IS NOT NULL
          AND high_qfq IS NOT NULL AND low_qfq IS NOT NULL
          AND open_hfq IS NOT NULL AND close_hfq IS NOT NULL
          AND high_hfq IS NOT NULL AND low_hfq IS NOT NULL
        """
        
        with conn.cursor() as cur:
            cur.execute(check_sql, (symbol, row['date'].strftime('%Y-%m-%d')))
            has_data = cur.fetchone()[0] > 0
        
        # 如果已经有完整数据，跳过更新
        if has_data:
            continue
            
        # 更新缺失的数据
        sql = f"""
        UPDATE {TABLE_NAME}
        SET open_qfq=%s, close_qfq=%s, high_qfq=%s, low_qfq=%s,
            open_hfq=%s, close_hfq=%s, high_hfq=%s, low_hfq=%s
        WHERE symbol=%s AND date=%s
        """
        params = (
            row.get('open_qfq'), row.get('close_qfq'), row.get('high_qfq'), row.get('low_qfq'),
            row.get('open_hfq'), row.get('close_hfq'), row.get('high_hfq'), row.get('low_hfq'),
            symbol, row['date'].strftime('%Y-%m-%d')
        )
        with conn.cursor() as cur:
            cur.execute(sql, params)
            updated_count += 1
    
    conn.commit()
    return updated_count

def check_adjust_fields_complete(conn, symbol, start_date, end_date):
    """
    检查复权价字段是否完整更新
    
    Args:
        conn: 数据库连接
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        bool: 如果所有复权价字段都已更新返回True，否则返回False
    """
    # 构建检查复权价字段的SQL条件
    field_conditions = []
    for field in ADJUST_FIELDS:
        field_conditions.append(f"{field} IS NULL")
    
    condition_sql = " OR ".join(field_conditions)
    
    sql = f"""
    SELECT COUNT(*) 
    FROM {TABLE_NAME} 
    WHERE symbol = %s 
      AND date BETWEEN %s AND %s
      AND ({condition_sql})
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, start_date, end_date))
        incomplete_count = cur.fetchone()[0]
    
    return incomplete_count == 0

def record_missing_data(symbol, missing_ranges, reason="数据缺失"):
    """
    记录缺失数据信息到文件
    
    Args:
        symbol: 股票代码
        missing_ranges: 缺失日期范围列表
        reason: 缺失原因
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(MISSING_DATA_FILE, 'a') as f:
        for start_date, end_date in missing_ranges:
            f.write(f"{timestamp}|{symbol}|{start_date}|{end_date}|{reason}\n")

def get_retry_symbols_from_missing_data():
    """
    从missing_data.txt中读取需要重试的股票列表
    
    Returns:
        set: 需要重试的股票代码集合
    """
    retry_symbols = set()
    
    if not MISSING_DATA_FILE.exists():
        return retry_symbols
    
    try:
        with open(MISSING_DATA_FILE, 'r') as f:
            lines = f.readlines()
            # 跳过标题行
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        symbol = parts[1]
                        retry_symbols.add(symbol)
    except Exception as e:
        print(f"读取missing_data.txt失败: {e}")
    
    return retry_symbols

def process_symbol(conn, provider, symbol, idx, total_symbols, retry_mode=False):
    """
    处理单个股票的数据同步
    
    Args:
        conn: 数据库连接
        provider: 数据提供者
        symbol: 股票代码
        idx: 当前索引
        total_symbols: 总股票数
        retry_mode: 是否为重试模式
        
    Returns:
        tuple: (success, updated_count, missing_count)
    """
    # 获取股票的日期范围
    start_date, end_date = get_date_range(conn, symbol)
    if not start_date or not end_date:
        print(f"[{idx+1}/{total_symbols}] {symbol} 无数据，跳过")
        return False, 0, 0
        
    print(f"[{idx+1}/{total_symbols}] {symbol} {start_date}~{end_date}")
    
    try:
        # 检查数据完整性
        missing_dates = check_data_completeness(conn, symbol, start_date, end_date)
        
        total_updated = 0
        total_missing = len(missing_dates)
        
        if missing_dates:
            # 有缺失数据，进行智能增量更新
            missing_ranges = get_missing_date_ranges(missing_dates, threshold_days=20)
            
            print(f"  -> 发现 {len(missing_dates)} 个缺失数据点，合并为 {len(missing_ranges)} 个日期范围")
            
            # 检查是否有大量缺失数据（超过阈值）
            if len(missing_ranges) == 1 and len(missing_dates) > 20:
                print(f"  -> 检测到大量缺失数据（{len(missing_dates)}天），将进行完整同步")
            
            # 记录缺失数据信息
            record_missing_data(symbol, missing_ranges, "数据完整性检查发现缺失")
            
            # 对每个缺失日期范围进行增量更新
            range_updated = 0
            for range_idx, (range_start, range_end) in enumerate(missing_ranges):
                range_days = (range_end - range_start).days + 1
                
                # 根据范围大小调整处理策略
                if range_days > 20:
                    print(f"    [{range_idx+1}/{len(missing_ranges)}] 处理大范围: {range_start} ~ {range_end} ({range_days}天)")
                else:
                    print(f"    [{range_idx+1}/{len(missing_ranges)}] 处理范围: {range_start} ~ {range_end} ({range_days}天)")
                
                # 获取缺失日期范围的数据
                df = provider.get_stock_data_with_adjust(symbol, str(range_start), str(range_end))
                if df is not None and not df.empty:
                    updated_count = update_adjusted_prices(conn, symbol, df)
                    range_updated += updated_count
                    print(f"      -> 更新了 {updated_count} 条记录")
                else:
                    print(f"      -> 无数据，跳过")
                    # 记录无数据的情况
                    record_missing_data(symbol, [(range_start, range_end)], "获取数据失败")
                
                # 根据范围大小调整间隔时间
                if range_days > 20:
                    # 大范围处理，适当增加间隔
                    time.sleep(SLEEP_BETWEEN_STOCKS)
                else:
                    # 小范围处理，使用较短间隔
                    time.sleep(SLEEP_BETWEEN_STOCKS / 2)
            
            total_updated += range_updated
            print(f"  -> 该股票共更新了 {range_updated} 条记录")
            
        else:
            # 数据完整，进行全量检查更新（确保所有复权字段都有数据）
            print(f"  -> 数据完整，进行全量验证更新")
            df = provider.get_stock_data_with_adjust(symbol, str(start_date), str(end_date))
            if df is not None and not df.empty:
                updated_count = update_adjusted_prices(conn, symbol, df)
                total_updated += updated_count
                print(f"  -> 验证更新了 {updated_count} 条记录")
            else:
                print("  -> 无数据，跳过")
                # 记录无数据的情况
                record_missing_data(symbol, [(start_date, end_date)], "获取数据失败")
        
        # 检查复权价字段是否完整更新
        adjust_complete = check_adjust_fields_complete(conn, symbol, start_date, end_date)
        
        if adjust_complete:
            # 数据同步成功，返回成功状态
            # 注意：即使total_updated为0（数据已经完整），也认为是成功
            return True, total_updated, total_missing
        else:
            # 数据同步不完整或失败
            print(f"  -> 数据同步不完整，复权价字段更新失败")
            return False, total_updated, total_missing
        
    except Exception as e:
        print(f"  -> 处理失败: {e}")
        # 记录处理失败的股票
        record_missing_data(symbol, [(start_date, end_date)], f"处理失败: {str(e)}")
        return False, 0, total_missing

def main():
    """主函数：智能数据完整性检查和增量更新，支持-r参数重新执行"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量填充股票复权价数据')
    parser.add_argument('-r', '--retry', action='store_true', 
                       help='重新执行模式，优先处理缺失数据的股票')
    args = parser.parse_args()
    
    provider = AkshareDataProvider()
    conn = pymysql.connect(**MYSQL_CONFIG)
    
    # 获取需要处理的股票列表
    valid_symbols = get_main_a_symbols()
    symbols = get_all_symbols(conn, valid_symbols)
    
    # 读取已完成处理的股票
    done_set = set()
    if DONE_FILE.exists():
        with open(DONE_FILE, 'r') as f:
            done_set = set(line.strip() for line in f if line.strip())
    
    # 初始化缺失数据记录文件
    if not MISSING_DATA_FILE.exists():
        with open(MISSING_DATA_FILE, 'w') as f:
            f.write("timestamp|symbol|start_date|end_date|reason\n")
    
    # 确定需要处理的股票列表
    if args.retry:
        print("重新执行模式：优先处理缺失数据的股票")
        # 获取需要重试的股票
        retry_symbols = get_retry_symbols_from_missing_data()
        # 过滤掉已经完成的股票
        retry_symbols = retry_symbols - done_set
        # 合并重试股票和未完成股票
        todo_symbols = list(retry_symbols) + [s for s in symbols if s not in done_set and s not in retry_symbols]
    else:
        print("正常执行模式")
        # 只处理未完成的股票
        todo_symbols = [s for s in symbols if s not in done_set]
    
    print(f"共{len(todo_symbols)}只股票待处理")
    print(f"已完成 {len(done_set)} 只，将跳过")
    
    # 检查网络连接
    print("正在检查网络连接...")
    network_ok = False
    try:
        # 测试网络连接
        import requests
        test_response = requests.get('https://www.baidu.com', timeout=10)
        if test_response.status_code == 200:
            network_ok = True
            print("网络连接正常")
        else:
            print("网络连接异常，状态码:", test_response.status_code)
    except Exception as e:
        print(f"网络连接检查失败: {e}")
        print("提示: 请检查网络连接或代理设置")
    
    if not network_ok:
        print("\n警告: 网络连接异常，数据获取可能失败")
        print("建议检查以下内容:")
        print("1. 网络连接是否正常")
        print("2. 代理设置是否正确")
        print("3. 防火墙或安全软件设置")
        print("\n是否继续执行? (y/N): ", end="")
        
        try:
            user_input = input().strip().lower()
            if user_input not in ['y', 'yes']:
                print("用户取消执行")
                conn.close()
                return
        except:
            print("输入异常，继续执行")
    
    total_processed = 0
    total_updated = 0
    total_missing = 0
    success_symbols = []
    
    for idx, symbol in enumerate(todo_symbols):
        success, updated_count, missing_count = process_symbol(conn, provider, symbol, idx, len(todo_symbols), args.retry)
        
        if success:
            # 数据同步成功，记录到成功列表
            success_symbols.append(symbol)
            total_updated += updated_count
            total_missing += missing_count
            total_processed += 1
        else:
            # 数据同步失败，不记录到done.txt
            print(f"  -> {symbol} 数据同步失败，不标记为已完成")
        
        # 股票间间隔
        time.sleep(SLEEP_BETWEEN_STOCKS)
    
    # 将成功处理的股票写入done.txt
    if success_symbols:
        with open(DONE_FILE, 'a') as f:
            for symbol in success_symbols:
                f.write(symbol + '\n')
        print(f"成功处理 {len(success_symbols)} 只股票，已写入done.txt")
    else:
        print("\n警告: 没有成功处理的股票，done.txt文件未生成")
        print("可能的原因:")
        print("1. 网络连接问题导致所有股票数据获取失败")
        print("2. 数据源服务暂时不可用")
        print("3. 代理设置或防火墙阻止了数据获取")
        print("\n建议检查网络连接后重新执行程序")
    
    conn.close()
    
    # 输出统计信息
    print("\n" + "="*50)
    print("处理完成统计:")
    print(f"总处理股票数: {total_processed}")
    print(f"总更新记录数: {total_updated}")
    print(f"总缺失数据点: {total_missing}")
    print(f"成功处理股票数: {len(success_symbols)}")
    print(f"缺失数据记录已保存到: {MISSING_DATA_FILE}")
    
    if not success_symbols:
        print("\n注意: 由于没有成功处理的股票，done.txt文件未生成")
    print("="*50)

if __name__ == '__main__':
    main()
