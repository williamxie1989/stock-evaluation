#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通达信历史行情数据导入脚本
将hsjday目录中的.day文件导入到SQLite数据库
"""

import os
import struct
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import argparse
from db import DatabaseManager

class TdxDataImporter:
    """
    通达信数据导入器
    负责读取.day文件并导入到数据库
    """
    
    RECORD_SIZE = 32  # 每条记录32字节
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.import_stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_records': 0,
            'imported_records': 0,
            'skipped_records': 0,
            'errors': []
        }
    
    def read_day_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        读取.day文件并解析为字典列表
        
        Args:
            file_path: .day文件路径
            
        Returns:
            包含股票日线数据的字典列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        data = []
        file_size = os.path.getsize(file_path)
        record_count = file_size // self.RECORD_SIZE
        
        with open(file_path, 'rb') as f:
            for i in range(record_count):
                try:
                    # 读取32字节记录
                    record_bytes = f.read(self.RECORD_SIZE)
                    if len(record_bytes) != self.RECORD_SIZE:
                        break
                    
                    # 解析记录 (小端序，8个4字节整数)
                    values = struct.unpack('<8I', record_bytes)
                    
                    # 转换数据
                    date_int = values[0]
                    open_price = values[1] / 100.0
                    high_price = values[2] / 100.0
                    low_price = values[3] / 100.0
                    close_price = values[4] / 100.0
                    volume = values[5]
                    amount = values[6]
                    reserved = values[7]
                    
                    # 转换通达信日期格式 (YYYYMMDD)
                    try:
                        if date_int > 19000000:  # 通达信YYYYMMDD格式
                            year = date_int // 10000
                            month = (date_int % 10000) // 100
                            day = date_int % 100
                            date = datetime(year, month, day)
                        else:  # 可能是天数格式，使用1900-01-01作为基准
                            date = datetime(1900, 1, 1) + timedelta(days=date_int)
                    except (ValueError, OverflowError) as e:
                        # 如果日期解析失败，跳过这条记录
                        self.import_stats['skipped_records'] += 1
                        continue
                    
                    # 数据验证
                    if not self._validate_record(date, open_price, high_price, low_price, close_price, volume):
                        self.import_stats['skipped_records'] += 1
                        continue
                    
                    record = {
                        'date': date.strftime('%Y-%m-%d'),
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume,
                        'amount': amount
                    }
                    
                    data.append(record)
                    self.import_stats['total_records'] += 1
                    
                except Exception as e:
                    error_msg = f"解析记录失败 {file_path}:{i}, 错误: {e}"
                    self.import_stats['errors'].append(error_msg)
                    self.import_stats['skipped_records'] += 1
                    continue
        
        return data
    
    def _validate_record(self, date: datetime, open_price: float, high_price: float, 
                        low_price: float, close_price: float, volume: int) -> bool:
        """
        验证记录数据的合理性
        
        Args:
            date: 日期
            open_price: 开盘价
            high_price: 最高价
            low_price: 最低价
            close_price: 收盘价
            volume: 成交量
            
        Returns:
            是否通过验证
        """
        # 日期范围检查 (1990-2030)
        if date.year < 1990 or date.year > 2030:
            return False
        
        # 价格合理性检查
        prices = [open_price, high_price, low_price, close_price]
        if any(p <= 0 or p > 10000 for p in prices):  # 价格应在0-10000之间
            return False
        
        # 价格逻辑检查
        if not (low_price <= open_price <= high_price and 
                low_price <= close_price <= high_price):
            return False
        
        # 成交量检查
        if volume < 0 or volume > 1e12:  # 成交量应为正数且不超过1万亿
            return False
        
        return True
    
    def extract_symbol_from_filename(self, file_path: str) -> Tuple[str, str]:
        """
        从文件路径提取股票代码和市场信息
        
        Args:
            file_path: .day文件路径
            
        Returns:
            (股票代码, 市场标识)
        """
        filename = os.path.basename(file_path)
        # 移除.day扩展名
        name_without_ext = filename.replace('.day', '')
        
        # 提取市场前缀和股票代码
        if name_without_ext.startswith('sh'):
            return name_without_ext[2:], 'SH'
        elif name_without_ext.startswith('sz'):
            return name_without_ext[2:], 'SZ'
        elif name_without_ext.startswith('bj'):
            return name_without_ext[2:], 'BJ'
        else:
            return name_without_ext, 'UNKNOWN'
    
    def import_single_file(self, file_path: str) -> bool:
        """
        导入单个.day文件
        
        Args:
            file_path: .day文件路径
            
        Returns:
            是否导入成功
        """
        try:
            # 提取股票代码和市场
            symbol, market = self.extract_symbol_from_filename(file_path)
            
            # 读取数据
            records = self.read_day_file(file_path)
            
            if not records:
                return False
            
            # 转换为DataFrame
            df = pd.DataFrame(records)
            df['symbol'] = symbol
            
            # 导入数据库
            imported_count = self.db_manager.upsert_prices_daily(
                df, 
                symbol_col='symbol',
                date_col='date',
                source='tdx_import'
            )
            
            self.import_stats['imported_records'] += imported_count
            self.import_stats['processed_files'] += 1
            
            return True
            
        except Exception as e:
            error_msg = f"导入文件失败 {file_path}: {e}"
            self.import_stats['errors'].append(error_msg)
            return False
    
    def import_market_data(self, hsjday_path: str, market: str, limit: int = None) -> None:
        """
        导入指定市场的数据
        
        Args:
            hsjday_path: hsjday目录路径
            market: 市场标识 (sh/sz/bj)
            limit: 限制导入文件数量，None表示全部导入
        """
        market_path = os.path.join(hsjday_path, market, 'lday')
        
        if not os.path.exists(market_path):
            print(f"市场目录不存在: {market_path}")
            return
        
        # 获取所有.day文件
        day_files = [f for f in os.listdir(market_path) if f.endswith('.day')]
        
        if limit:
            day_files = day_files[:limit]
        
        self.import_stats['total_files'] += len(day_files)
        
        print(f"\n开始导入{market.upper()}市场数据，共{len(day_files)}个文件...")
        
        # 使用进度条显示导入进度
        for filename in tqdm(day_files, desc=f"导入{market.upper()}市场"):
            file_path = os.path.join(market_path, filename)
            self.import_single_file(file_path)
    
    def import_all_markets(self, hsjday_path: str = './hsjday', limit_per_market: int = None) -> None:
        """
        导入所有市场数据
        
        Args:
            hsjday_path: hsjday目录路径
            limit_per_market: 每个市场限制导入文件数量
        """
        print("=== 通达信历史行情数据导入 ===")
        print(f"数据源目录: {hsjday_path}")
        
        if not os.path.exists(hsjday_path):
            print(f"数据目录不存在: {hsjday_path}")
            return
        
        # 导入各市场数据
        markets = ['sh', 'sz', 'bj']
        
        for market in markets:
            self.import_market_data(hsjday_path, market, limit_per_market)
        
        # 打印导入统计
        self.print_import_stats()
    
    def print_import_stats(self) -> None:
        """
        打印导入统计信息
        """
        print("\n=== 导入统计 ===")
        print(f"总文件数: {self.import_stats['total_files']}")
        print(f"已处理文件数: {self.import_stats['processed_files']}")
        print(f"总记录数: {self.import_stats['total_records']}")
        print(f"已导入记录数: {self.import_stats['imported_records']}")
        print(f"跳过记录数: {self.import_stats['skipped_records']}")
        
        if self.import_stats['errors']:
            print(f"\n错误数: {len(self.import_stats['errors'])}")
            for i, error in enumerate(self.import_stats['errors'][:10]):  # 只显示前10个错误
                print(f"  {i+1}. {error}")
            if len(self.import_stats['errors']) > 10:
                print(f"  ... 还有{len(self.import_stats['errors']) - 10}个错误")
        
        success_rate = (self.import_stats['processed_files'] / self.import_stats['total_files'] * 100) if self.import_stats['total_files'] > 0 else 0
        print(f"\n导入成功率: {success_rate:.1f}%")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='通达信历史行情数据导入工具')
    parser.add_argument('--hsjday-path', default='./hsjday', help='hsjday数据目录路径')
    parser.add_argument('--market', choices=['sh', 'sz', 'bj', 'all'], default='all', help='导入的市场')
    parser.add_argument('--limit', type=int, help='限制每个市场导入的文件数量（用于测试）')
    parser.add_argument('--db-path', default='stock_data.sqlite3', help='数据库文件路径')
    
    args = parser.parse_args()
    
    # 初始化数据库管理器
    db_manager = DatabaseManager(args.db_path)
    
    # 创建导入器
    importer = TdxDataImporter(db_manager)
    
    # 执行导入
    if args.market == 'all':
        importer.import_all_markets(args.hsjday_path, args.limit)
    else:
        importer.import_market_data(args.hsjday_path, args.market, args.limit)

if __name__ == '__main__':
    main()