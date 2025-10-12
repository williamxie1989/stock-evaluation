"""
批量填充prices_daily表的复权价字段（open_qfq, close_qfq, high_qfq, low_qfq, open_hfq, close_hfq, high_hfq, low_hfq）
依赖AkshareDataProvider.get_stock_data_with_adjust

优化版本：增加数据完整性检查和智能增量更新功能
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import time
import pandas as pd
import os
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

def get_missing_date_ranges(missing_dates):
    """
    将连续的缺失日期合并为日期范围，提高处理效率
    
    Args:
        missing_dates: 缺失日期列表
        
    Returns:
        list: 日期范围列表，每个元素为(start_date, end_date)
    """
    if not missing_dates:
        return []
    
    missing_dates.sort()
    ranges = []
    current_start = missing_dates[0]
    current_end = missing_dates[0]
    
    for i in range(1, len(missing_dates)):
        # 检查日期是否连续（相差1天）
        if (missing_dates[i] - current_end).days == 1:
            current_end = missing_dates[i]
        else:
            ranges.append((current_start, current_end))
            current_start = missing_dates[i]
            current_end = missing_dates[i]
    
    ranges.append((current_start, current_end))
    return ranges

def update_adjusted_prices(conn, symbol, df):
    """
    更新复权价格数据，支持增量更新
    
    Args:
        conn: 数据库连接
        symbol: 股票代码
        df: 包含复权价格数据的DataFrame
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

def main():
    """主函数：智能数据完整性检查和增量更新"""
    provider = AkshareDataProvider()
    conn = pymysql.connect(**MYSQL_CONFIG)
    
    # 获取需要处理的股票列表
    valid_symbols = get_main_a_symbols()
    symbols = get_all_symbols(conn, valid_symbols)
    print(f"共{len(symbols)}只主板A股待处理")
    
    # 读取已完成处理的股票
    done_set = set()
    if DONE_FILE.exists():
        with open(DONE_FILE, 'r') as f:
            done_set = set(line.strip() for line in f if line.strip())
    print(f"已完成 {len(done_set)} 只，将跳过")
    
    # 初始化缺失数据记录文件
    if not MISSING_DATA_FILE.exists():
        with open(MISSING_DATA_FILE, 'w') as f:
            f.write("timestamp|symbol|start_date|end_date|reason\n")
    
    total_processed = 0
    total_updated = 0
    total_missing = 0
    
    for idx, symbol in enumerate(symbols):
        # 跳过已处理的股票
        if symbol in done_set:
            continue
            
        # 获取股票的日期范围
        start_date, end_date = get_date_range(conn, symbol)
        if not start_date or not end_date:
            print(f"[{idx+1}/{len(symbols)}] {symbol} 无数据，跳过")
            continue
            
        print(f"[{idx+1}/{len(symbols)}] {symbol} {start_date}~{end_date}")
        
        try:
            # 检查数据完整性
            missing_dates = check_data_completeness(conn, symbol, start_date, end_date)
            
            if missing_dates:
                # 有缺失数据，进行智能增量更新
                missing_ranges = get_missing_date_ranges(missing_dates)
                total_missing += len(missing_dates)
                
                print(f"  -> 发现 {len(missing_dates)} 个缺失数据点，合并为 {len(missing_ranges)} 个日期范围")
                
                # 记录缺失数据信息
                record_missing_data(symbol, missing_ranges, "数据完整性检查发现缺失")
                
                # 对每个缺失日期范围进行增量更新
                range_updated = 0
                for range_idx, (range_start, range_end) in enumerate(missing_ranges):
                    print(f"    [{range_idx+1}/{len(missing_ranges)}] 处理范围: {range_start} ~ {range_end}")
                    
                    # 获取缺失日期范围的数据
                    df = provider.get_stock_data_with_adjust(symbol, str(range_start), str(range_end))
                    if df is not None and not df.empty:
                        updated_count = update_adjusted_prices(conn, symbol, df)
                        range_updated += updated_count
                        print(f"      -> 更新了 {updated_count} 条记录")
                    else:
                        print(f"      -> 无数据，跳过")
                    
                    # 小范围间隔，避免请求过快
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
            
            # 标记该股票为已处理
            with open(DONE_FILE, 'a') as f:
                f.write(symbol+'\n')
            
            total_processed += 1
            
        except Exception as e:
            print(f"  -> 处理失败: {e}")
            # 记录处理失败的股票
            record_missing_data(symbol, [(start_date, end_date)], f"处理失败: {str(e)}")
        
        # 股票间间隔
        time.sleep(SLEEP_BETWEEN_STOCKS)
    
    conn.close()
    
    # 输出统计信息
    print("\n" + "="*50)
    print("处理完成统计:")
    print(f"总处理股票数: {total_processed}")
    print(f"总更新记录数: {total_updated}")
    print(f"总缺失数据点: {total_missing}")
    print(f"缺失数据记录已保存到: {MISSING_DATA_FILE}")
    print("="*50)

if __name__ == '__main__':
    main()
