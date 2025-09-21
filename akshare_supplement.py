import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import time
from db import DatabaseManager

class AkshareDataSupplement:
    def __init__(self, db_path='stock_data.db'):
        self.db = DatabaseManager(db_path)
        self.db_path = db_path
    
    def get_active_stocks_from_api(self):
        """从akshare API获取当前活跃股票列表"""
        try:
            print('正在获取当前A股股票列表...')
            stock_list = ak.stock_info_a_code_name()
            
            if stock_list is not None and len(stock_list) > 0:
                # 转换格式
                stocks = []
                for _, row in stock_list.iterrows():
                    code = str(row['code']).zfill(6)
                    name = row['name']
                    
                    # 判断市场
                    if code.startswith(('60', '68', '90', '73')):
                        market = 'SH'
                        symbol = f'{code}.SH'
                    elif code.startswith(('00', '30', '20', '83', '87')):
                        market = 'SZ'
                        symbol = f'{code}.SZ'
                    else:
                        continue
                    
                    stocks.append({
                        'symbol': symbol,
                        'code': code,
                        'name': name,
                        'market': market
                    })
                
                print(f'获取到 {len(stocks)} 只当前活跃股票')
                return pd.DataFrame(stocks)
            
        except Exception as e:
            print(f'获取股票列表失败: {e}')
        
        return pd.DataFrame()
    
    def supplement_recent_data(self, symbol, days=90):
        """补充最近的交易数据"""
        try:
            code = symbol.split('.')[0]
            
            # 获取最近的交易数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            # 获取日线数据
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="")
            
            if df is not None and len(df) > 0:
                # 重命名列以匹配数据库格式
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                })
                
                # 添加股票信息
                df['symbol'] = symbol
                df['code'] = code
                df['market'] = symbol.split('.')[1]
                
                # 选择需要的列
                df = df[['symbol', 'code', 'market', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                
                # 转换日期格式
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                
                return df
            
        except Exception as e:
            print(f'补充 {symbol} 的最近数据失败: {e}')
        
        return pd.DataFrame()
    
    def delete_existing_data(self, symbol, start_date, end_date):
        """删除指定日期范围内的现有数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM prices_daily WHERE symbol = ? AND date >= ? AND date <= ?",
                    (symbol, start_date, end_date)
                )
                conn.commit()
                print(f'已删除 {symbol} 从 {start_date} 到 {end_date} 的现有数据 ({cursor.rowcount} 条)')
        except Exception as e:
            print(f'删除现有数据失败: {e}')
    
    def supplement_active_stocks(self):
        """补充活跃股票的最新数据"""
        print('=== 开始补充活跃股票数据 ===')
        
        # 获取当前活跃股票列表
        active_stocks = self.get_active_stocks_from_api()
        
        if active_stocks.empty:
            print('无法获取活跃股票列表')
            return 0, 0
        
        success_count = 0
        total_records = 0
        
        for i, stock in active_stocks.iterrows():
            symbol = stock['symbol']
            name = stock['name']
            
            if i % 100 == 0:
                print(f'[{i+1}/{len(active_stocks)}] 补充: {symbol} - {name}')
            
            # 补充最近90天的数据
            df = self.supplement_recent_data(symbol, days=90)
            
            if not df.empty:
                try:
                    # 先删除这些日期的现有数据，避免重复
                    self.delete_existing_data(symbol, df['date'].min(), df['date'].max())
                    
                    # 保存新数据
                    self.db.upsert_prices_daily(df, source='akshare')
                    success_count += 1
                    total_records += len(df)
                    
                    if i % 100 == 0:
                        print(f'  成功: 补充了 {len(df)} 条记录')
                except Exception as e:
                    print(f'  错误: 保存 {symbol} 失败 - {e}')
            
            # 避免请求过于频繁
            time.sleep(0.2)
        
        print(f'\n活跃股票数据补充完成!')
        print(f'成功补充: {success_count} 只股票')
        print(f'总记录数: {total_records} 条')
        
        return success_count, total_records
    
    def run_supplement(self):
        """运行数据补充流程"""
        print('=== 开始akshare数据补充 ===')
        
        # 补充活跃股票的最新数据
        active_stocks, active_records = self.supplement_active_stocks()
        
        print('\n=== 数据补充完成 ===')
        print(f'活跃股票补充数据: {active_stocks} 只股票, {active_records} 条记录')

if __name__ == '__main__':
    supplement = AkshareDataSupplement()
    supplement.run_supplement()