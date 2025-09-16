#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通达信历史行情数据(.day文件)格式分析工具
用于分析hsjday目录中的.day文件格式和数据结构
"""

import os
import struct
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Any, Tuple

class TdxDayReader:
    """
    通达信日线数据(.day文件)读取器
    .day文件格式：每条记录32字节
    - 日期: 4字节 (通达信日期格式)
    - 开盘价: 4字节 (实际价格*100)
    - 最高价: 4字节 (实际价格*100)
    - 最低价: 4字节 (实际价格*100)
    - 收盘价: 4字节 (实际价格*100)
    - 成交量: 4字节 (股)
    - 成交额: 4字节 (元)
    - 保留字段: 4字节
    """
    
    RECORD_SIZE = 32  # 每条记录32字节
    
    def __init__(self):
        self.data = []
    
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
        
        print(f"文件: {file_path}")
        print(f"文件大小: {file_size} 字节")
        print(f"记录数量: {record_count}")
        
        with open(file_path, 'rb') as f:
            for i in range(record_count):
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
                    print(f"日期解析失败: {date_int}, 错误: {e}")
                    continue
                
                record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'amount': amount,
                    'reserved': reserved
                }
                
                data.append(record)
        
        return data
    
    def analyze_file_sample(self, file_path: str, sample_size: int = 10) -> None:
        """
        分析文件样本数据
        
        Args:
            file_path: .day文件路径
            sample_size: 样本数量
        """
        try:
            data = self.read_day_file(file_path)
            
            if not data:
                print("文件为空或无法读取数据")
                return
            
            print(f"\n=== 数据样本分析 (前{min(sample_size, len(data))}条记录) ===")
            
            for i, record in enumerate(data[:sample_size]):
                print(f"记录 {i+1}: {record}")
            
            if len(data) > sample_size:
                print(f"\n=== 数据样本分析 (后{min(sample_size, len(data))}条记录) ===")
                for i, record in enumerate(data[-sample_size:]):
                    print(f"记录 {len(data)-sample_size+i+1}: {record}")
            
            # 数据统计
            df = pd.DataFrame(data)
            print(f"\n=== 数据统计信息 ===")
            print(f"总记录数: {len(data)}")
            print(f"日期范围: {data[0]['date']} 至 {data[-1]['date']}")
            print(f"价格统计:")
            print(f"  开盘价: 最小={df['open'].min():.2f}, 最大={df['open'].max():.2f}, 平均={df['open'].mean():.2f}")
            print(f"  收盘价: 最小={df['close'].min():.2f}, 最大={df['close'].max():.2f}, 平均={df['close'].mean():.2f}")
            print(f"  成交量: 最小={df['volume'].min()}, 最大={df['volume'].max()}, 平均={df['volume'].mean():.0f}")
            
        except Exception as e:
            print(f"分析文件时出错: {e}")
    
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

def analyze_hsjday_structure(hsjday_path: str = './hsjday'):
    """
    分析hsjday目录结构和数据文件
    
    Args:
        hsjday_path: hsjday目录路径
    """
    print("=== 通达信历史行情数据(hsjday)结构分析 ===")
    
    if not os.path.exists(hsjday_path):
        print(f"目录不存在: {hsjday_path}")
        return
    
    reader = TdxDayReader()
    
    # 遍历各市场目录
    for market_dir in ['sh', 'sz', 'bj']:
        market_path = os.path.join(hsjday_path, market_dir, 'lday')
        
        if not os.path.exists(market_path):
            print(f"市场目录不存在: {market_path}")
            continue
        
        print(f"\n=== {market_dir.upper()}市场数据分析 ===")
        
        # 获取所有.day文件
        day_files = [f for f in os.listdir(market_path) if f.endswith('.day')]
        print(f"文件数量: {len(day_files)}")
        
        if day_files:
            # 分析第一个文件作为样本
            sample_file = os.path.join(market_path, day_files[0])
            symbol, market = reader.extract_symbol_from_filename(sample_file)
            print(f"样本文件: {day_files[0]} (股票代码: {symbol}, 市场: {market})")
            
            # 分析样本数据
            reader.analyze_file_sample(sample_file, 5)
            
            # 显示更多文件信息
            print(f"\n其他文件示例:")
            for i, filename in enumerate(day_files[1:6]):  # 显示接下来5个文件
                symbol, market = reader.extract_symbol_from_filename(filename)
                file_path = os.path.join(market_path, filename)
                file_size = os.path.getsize(file_path)
                record_count = file_size // reader.RECORD_SIZE
                print(f"  {filename}: 股票代码={symbol}, 市场={market}, 记录数={record_count}")

def main():
    """
    主函数：分析通达信数据格式
    """
    print("通达信历史行情数据格式分析工具")
    print("=" * 50)
    
    # 分析hsjday目录结构
    analyze_hsjday_structure()
    
    print("\n=== 分析完成 ===")
    print("\n数据格式总结:")
    print("1. 文件组织: hsjday/{market}/lday/{market}{code}.day")
    print("2. 市场标识: sh(上海), sz(深圳), bj(北京)")
    print("3. 文件格式: 二进制，每条记录32字节")
    print("4. 数据字段: 日期、开高低收、成交量、成交额")
    print("5. 价格单位: 实际价格需除以100")
    print("6. 日期格式: 从1900-01-01开始的天数")

if __name__ == '__main__':
    main()