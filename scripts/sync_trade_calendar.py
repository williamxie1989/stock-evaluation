#!/usr/bin/env python3
"""
交易日数据同步工具
使用 AKShare 的 tool_trade_date_hist_sina 接口获取历史交易日数据
并同步到 MySQL 数据库的 trade_calendar 表中
"""
import os
import sys
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import akshare as ak
import mysql.connector
import pandas as pd
from dotenv import load_dotenv

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class TradeCalendarSync:
    """交易日数据同步器"""
    
    def __init__(self):
        """初始化数据库连接"""
        self.config = {
            'host': os.getenv('MYSQL_HOST', '127.0.0.1'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'user': os.getenv('MYSQL_USER', 'root'),
            'password': os.getenv('MYSQL_PASSWORD', ''),
            'database': 'stock_evaluation'
        }
        self.conn = None
        self.cursor = None
        self._connect()
    
    def _connect(self):
        """连接数据库"""
        try:
            self.conn = mysql.connector.connect(**self.config)
            self.cursor = self.conn.cursor()
            logger.info(f"成功连接到数据库: {self.config['host']}:{self.config['port']}")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def get_existing_dates(self, market: str = None) -> set:
        """
        获取数据库中已存在的交易日数据
        
        Args:
            market: 市场类型(None表示所有市场)
        
        Returns:
            已存在的日期集合
        """
        try:
            if market:
                sql = "SELECT DISTINCT trade_date FROM trade_calendar WHERE market = %s"
                self.cursor.execute(sql, (market,))
            else:
                sql = "SELECT DISTINCT trade_date FROM trade_calendar"
                self.cursor.execute(sql)
            
            existing_dates = {row[0].strftime('%Y-%m-%d') for row in self.cursor.fetchall()}
            logger.info(f"数据库中已有 {len(existing_dates)} 个交易日数据")
            return existing_dates
        except Exception as e:
            logger.error(f"获取已存在日期失败: {e}")
            return set()
    
    def get_trade_calendar_from_akshare(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        从 AKShare 获取交易日数据
        
        Args:
            start_date: 开始日期(格式: YYYY-MM-DD)
            end_date: 结束日期(格式: YYYY-MM-DD)
        
        Returns:
            交易日数据 DataFrame
        """
        try:
            # 如果没有指定日期范围，默认获取最近5年的数据
            if not start_date:
                start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y%m%d')
            else:
                start_date = start_date.replace('-', '')
            
            if not end_date:
                end_date = datetime.now().strftime('%Y%m%d')
            else:
                end_date = end_date.replace('-', '')
            
            logger.info(f"从 AKShare 获取交易日数据: {start_date} 到 {end_date}")
            
            # 调用 AKShare 接口
            df = ak.tool_trade_date_hist_sina()
            
            if df is None or len(df) == 0:
                logger.error("从 AKShare 获取交易日数据失败")
                return pd.DataFrame()
            
            # 数据预处理
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 过滤日期范围
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df['trade_date'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df['trade_date'] <= end_dt]
            
            # 添加市场字段(默认A股市场)
            df['market'] = 'A'
            
            # 重命名列
            df = df.rename(columns={'trade_date': 'trade_date'})
            
            logger.info(f"从 AKShare 获取到 {len(df)} 条交易日数据")
            return df
            
        except Exception as e:
            logger.error(f"获取交易日数据失败: {e}")
            return pd.DataFrame()
    
    def prepare_trade_calendar_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        准备交易日数据用于数据库插入
        
        Args:
            df: 从 AKShare 获取的原始数据
        
        Returns:
            处理后的数据列表
        """
        if df.empty:
            return []
        
        data_list = []
        for _, row in df.iterrows():
            try:
                # 确保日期格式正确
                trade_date = row['trade_date']
                if isinstance(trade_date, pd.Timestamp):
                    trade_date = trade_date.strftime('%Y-%m-%d')
                
                data = {
                    'trade_date': trade_date,
                    'is_trading_day': 1,  # AKShare 返回的都是交易日
                    'market': row.get('market', 'A'),
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                }
                data_list.append(data)
            except Exception as e:
                logger.warning(f"处理交易日数据失败: {row} - {e}")
                continue
        
        return data_list
    
    def upsert_trade_calendar(self, data_list: List[Dict]) -> int:
        """
        批量插入或更新交易日数据
        
        Args:
            data_list: 交易日数据列表
        
        Returns:
            成功插入/更新的记录数
        """
        if not data_list:
            return 0
        
        success_count = 0
        batch_size = 100  # 批量处理大小
        
        try:
            # 分批处理
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                
                # 构建批量插入SQL
                sql = """
                INSERT INTO trade_calendar (trade_date, is_trading_day, market, created_at, updated_at)
                VALUES (%(trade_date)s, %(is_trading_day)s, %(market)s, %(created_at)s, %(updated_at)s)
                ON DUPLICATE KEY UPDATE 
                    is_trading_day = VALUES(is_trading_day),
                    updated_at = VALUES(updated_at)
                """
                
                self.cursor.executemany(sql, batch)
                self.conn.commit()
                
                success_count += len(batch)
                logger.info(f"已处理 {i + len(batch)}/{len(data_list)} 条记录")
            
            logger.info(f"成功插入/更新 {success_count} 条交易日数据")
            return success_count
            
        except Exception as e:
            logger.error(f"批量插入交易日数据失败: {e}")
            self.conn.rollback()
            return 0
    
    def sync_trade_calendar(self, start_date: str = None, end_date: str = None, 
                          market: str = None, force_update: bool = False) -> Dict:
        """
        同步交易日数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            market: 市场类型
            force_update: 是否强制更新已有数据
        
        Returns:
            同步结果统计
        """
        logger.info("开始同步交易日数据...")
        
        # 获取已存在的日期
        existing_dates = self.get_existing_dates(market)
        
        # 从 AKShare 获取数据
        df = self.get_trade_calendar_from_akshare(start_date, end_date)
        
        if df.empty:
            return {'success': False, 'message': '获取交易日数据失败'}
        
        # 准备数据
        data_list = self.prepare_trade_calendar_data(df)
        
        if not data_list:
            return {'success': False, 'message': '数据预处理失败'}
        
        # 过滤已存在的数据(除非强制更新)
        if not force_update and existing_dates:
            filtered_data = []
            for data in data_list:
                if data['trade_date'] not in existing_dates:
                    filtered_data.append(data)
            
            logger.info(f"过滤后剩余 {len(filtered_data)} 条新数据")
            data_list = filtered_data
        
        # 插入数据
        success_count = self.upsert_trade_calendar(data_list)
        
        # 统计结果
        result = {
            'success': True,
            'total_fetched': len(df),
            'total_processed': len(data_list),
            'success_count': success_count,
            'existing_count': len(existing_dates)
        }
        
        logger.info(f"同步完成: 获取 {result['total_fetched']} 条, "
                   f"处理 {result['total_processed']} 条, "
                   f"成功 {result['success_count']} 条")
        
        return result
    
    def validate_sync_result(self) -> Dict:
        """
        验证同步结果
        
        Returns:
            验证结果统计
        """
        try:
            # 统计总记录数
            self.cursor.execute("SELECT COUNT(*) FROM trade_calendar")
            total_count = self.cursor.fetchone()[0]
            
            # 统计不同市场的记录数
            self.cursor.execute("SELECT market, COUNT(*) FROM trade_calendar GROUP BY market")
            market_stats = {row[0]: row[1] for row in self.cursor.fetchall()}
            
            # 统计日期范围
            self.cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM trade_calendar")
            min_date, max_date = self.cursor.fetchone()
            
            # 统计交易日和非交易日
            self.cursor.execute("SELECT is_trading_day, COUNT(*) FROM trade_calendar GROUP BY is_trading_day")
            trading_stats = {row[0]: row[1] for row in self.cursor.fetchall()}
            
            result = {
                'total_count': total_count,
                'market_stats': market_stats,
                'date_range': {'min': min_date, 'max': max_date},
                'trading_stats': trading_stats
            }
            
            logger.info(f"验证结果: 总记录数={total_count}, "
                       f"市场分布={market_stats}, "
                       f"日期范围={min_date} 到 {max_date}")
            
            return result
            
        except Exception as e:
            logger.error(f"验证同步结果失败: {e}")
            return {'error': str(e)}
    
    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("数据库连接已关闭")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='同步交易日数据到数据库')
    parser.add_argument('--start-date', type=str, help='开始日期(格式: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='结束日期(格式: YYYY-MM-DD)')
    parser.add_argument('--market', type=str, default='A', help='市场类型(默认: A)')
    parser.add_argument('--force', action='store_true', help='强制更新已有数据')
    parser.add_argument('--validate', action='store_true', help='同步后验证结果')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info("交易日数据同步工具")
    logger.info("="*60)
    
    sync = None
    try:
        sync = TradeCalendarSync()
        
        # 执行同步
        result = sync.sync_trade_calendar(
            start_date=args.start_date,
            end_date=args.end_date,
            market=args.market,
            force_update=args.force
        )
        
        if result['success']:
            logger.info("✅ 交易日数据同步成功")
        else:
            logger.error("❌ 交易日数据同步失败")
            sys.exit(1)
        
        # 验证结果
        if args.validate:
            logger.info("开始验证同步结果...")
            validation_result = sync.validate_sync_result()
            
            if 'error' not in validation_result:
                logger.info("✅ 同步结果验证成功")
            else:
                logger.warning("⚠️ 同步结果验证失败")
        
    except Exception as e:
        logger.error(f"同步过程出错: {e}")
        sys.exit(1)
    finally:
        if sync:
            sync.close()


if __name__ == '__main__':
    main()