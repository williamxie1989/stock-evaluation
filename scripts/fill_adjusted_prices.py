"""
批量填充prices_daily表的复权价字段（open_qfq, close_qfq, high_qfq, low_qfq, open_hfq, close_hfq, high_hfq, low_hfq）
依赖AkshareDataProvider.get_stock_data_with_adjust
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import time
import pandas as pd
import os
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

def update_adjusted_prices(conn, symbol, df):
    # 只更新有复权价的行
    for _, row in df.iterrows():
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
    conn.commit()

def main():
    provider = AkshareDataProvider()
    conn = pymysql.connect(**MYSQL_CONFIG)
    valid_symbols = get_main_a_symbols()
    symbols = get_all_symbols(conn, valid_symbols)
    print(f"共{len(symbols)}只主板A股待处理")
    done_set = set()
    if DONE_FILE.exists():
        with open(DONE_FILE, 'r') as f:
            done_set = set(line.strip() for line in f if line.strip())
    print(f"已完成 {len(done_set)} 只，将跳过")
    for idx, symbol in enumerate(symbols):
        if symbol in done_set:
            continue
        start_date, end_date = get_date_range(conn, symbol)
        if not start_date or not end_date:
            continue
        print(f"[{idx+1}/{len(symbols)}] {symbol} {start_date}~{end_date}")
        try:
            df = provider.get_stock_data_with_adjust(symbol, str(start_date), str(end_date))
            if df is not None and not df.empty:
                update_adjusted_prices(conn, symbol, df)
                print(f"  -> 已更新 {len(df)} 条")
            else:
                print("  -> 无数据")
            with open(DONE_FILE, 'a') as f:
                f.write(symbol+'\n')
        except Exception as e:
            print(f"  -> 失败: {e}")
        time.sleep(SLEEP_BETWEEN_STOCKS)
    conn.close()
    print("全部处理完成")

if __name__ == '__main__':
    main()
