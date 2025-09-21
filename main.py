import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
import os
import struct
from dotenv import load_dotenv
from typing import Dict, List, Any
from akshare_data_provider import AkshareDataProvider
from signal_generator import SignalGenerator
from backtest_engine import BacktestEngine
from risk_management import RiskManager
import logging
import re

# 创建logger实例
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 加载环境变量
load_dotenv()

class StockAnalyzer:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # 加载本地大模型配置
        self.local_model_url = os.getenv('LOCAL_MODEL_URL')
        self.local_model_name = os.getenv('LOCAL_MODEL_NAME')
        
        # 初始化akshare数据提供者
        self.akshare_provider = AkshareDataProvider()
        
        # 初始化信号生成器、回测引擎和风险管理器
        self.signal_generator = SignalGenerator()
        self.backtest_engine = BacktestEngine()
        self.risk_manager = RiskManager()
    
    def get_free_stock_data(self, stock_symbol):
        """使用免费API获取股票数据"""
        try:
            # 免费API端点（示例使用公开数据源）
            if stock_symbol.endswith('.SS'):
                # 上海证券交易所股票
                code = stock_symbol.replace('.SS', '')
                url = f"http://api.mairui.club/hsrl/ssjy/{code}/b997d4403688d5e66a"
            elif stock_symbol.endswith('.SZ'):
                # 深圳证券交易所股票
                code = stock_symbol.replace('.SZ', '')
                url = f"http://api.mairui.club/hsrl/ssjy/{code}/b997d4403688d5e66a"
            else:
                # 默认使用yfinance作为备选
                return self.get_stock_data_yfinance(stock_symbol)
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # 转换数据为DataFrame格式
                df = self._convert_to_dataframe(data)
                return {
                    'stock_data': df,
                    'market_data': None,  # 免费API可能不提供大盘数据
                    'stock_info': {'symbol': stock_symbol, 'longName': stock_symbol}
                }
            else:
                # 如果免费API失败，回退到yfinance
                return self.get_stock_data_yfinance(stock_symbol)
                
        except Exception as e:
            print(f"免费API获取失败: {e}")
            # 回退到yfinance
            return self.get_stock_data_yfinance(stock_symbol)
    
    def _convert_to_dataframe(self, data):
        """将API返回的数据转换为DataFrame"""
        # 这里需要根据实际API返回格式进行调整
        # 示例实现，需要根据实际API响应调整
        
        # 如果数据是历史数据列表，处理为DataFrame
        if isinstance(data, list) and len(data) > 0:
            # 假设数据是历史数据列表
            records = []
            for item in data:
                records.append({
                    'Date': item.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'Open': item.get('open', 0),
                    'High': item.get('high', 0),
                    'Low': item.get('low', 0),
                    'Close': item.get('close', item.get('price', 0)),
                    'Volume': item.get('volume', 0)
                })
            df = pd.DataFrame(records)
        else:
            # 单条数据的情况
            df = pd.DataFrame([{
                'Date': datetime.now().strftime('%Y-%m-%d'),
                    'Open': data[0] if isinstance(data, list) and len(data) > 0 else 0,
                'High': data[1] if isinstance(data, list) and len(data) > 1 else 0,
                'Low': data[2] if isinstance(data, list) and len(data) > 2 else 0,
                'Close': data[3] if isinstance(data, list) and len(data) > 3 else 0,
                'Volume': data[4] if isinstance(data, list) and len(data) > 4 else 0
            }])
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def get_stock_data_yfinance(self, stock_symbol, period="1y"):
        """使用yfinance获取股票数据（备用方案）"""
        import time
        max_retries = 1
        retry_delay = 10
        
        for attempt in range(max_retries):
            try:
                # 在每次请求前都等待一段时间
                if attempt > 0:
                    print(f"API请求等待 {retry_delay} 秒...")
                    time.sleep(retry_delay)
                
                stock = yf.Ticker(stock_symbol)
                df = stock.history(period=period)
                
                # 如果数据为空，可能是股票代码错误或数据不可用
                if df.empty:
                    raise Exception(f"股票 {stock_symbol} 无数据，可能代码错误或已退市")
                
                # 获取大盘数据（沪深300），忽略错误
                market_df = None
                try:
                    market = yf.Ticker("000300.SS")
                    market_df = market.history(period=period)
                except:
                    pass
                
                # 尝试获取股票信息，但忽略错误
                try:
                    stock_info = stock.info
                except:
                    stock_info = {'symbol': stock_symbol, 'longName': 'N/A'}
                
                return {
                    'stock_data': df,
                    'market_data': market_df,
                    'stock_info': stock_info
                }
                
            except Exception as e:
                error_msg = str(e)
                if ("Rate limited" in error_msg or "Too Many Requests" in error_msg) and attempt < max_retries - 1:
                    retry_delay *= 2
                    continue
                raise Exception(f"yfinance获取失败: {error_msg}")
        
        raise Exception("yfinance数据服务暂时不可用")
    
    def get_stock_data(self, stock_symbol, period="1y"):
        """获取股票数据（主方法，兼容旧代码）"""
        return self.get_stock_data_yfinance(stock_symbol, period)
    
    def calculate_technical_indicators(self, df):
        """计算技术指标"""
        # 移动平均线
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # MACD (12,26,9)
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df
    
    def generate_trading_signals(self, df):
        """生成交易信号，包括短线/日内交易信号"""
        signals = []
        
        # 确保有足够的数据进行分析
        if len(df) < 20:
            return signals
        
        # 金叉死叉信号（基于5日和20日均线）
        for i in range(1, len(df)):
            # 检查索引是否有效
            if i >= len(df) or i-1 < 0:
                continue
                
            # 确保所需列存在且不为空值
            if 'MA5' not in df.columns or 'MA20' not in df.columns:
                continue
                
            if pd.isna(df['MA5'].iloc[i]) or pd.isna(df['MA20'].iloc[i]) or pd.isna(df['MA5'].iloc[i-1]) or pd.isna(df['MA20'].iloc[i-1]):
                continue
            
            if df['MA5'].iloc[i] > df['MA20'].iloc[i] and df['MA5'].iloc[i-1] <= df['MA20'].iloc[i-1]:
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': '5日均线上穿20日均线（金叉）'
                })
            elif df['MA5'].iloc[i] < df['MA20'].iloc[i] and df['MA5'].iloc[i-1] >= df['MA20'].iloc[i-1]:
                signals.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': df['Close'].iloc[i],
                    'reason': '5日均线下穿20日均线（死叉）'
                })
        
        # RSI信号
        for i in range(1, len(df)):
            # 确保所需列存在且不为空值
            if 'RSI' not in df.columns or pd.isna(df['RSI'].iloc[i]) or pd.isna(df['RSI'].iloc[i-1]):
                continue
            
            # RSI超买超卖信号
            if df['RSI'].iloc[i] < 30 and df['RSI'].iloc[i-1] >= 30:
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': 'RSI从超卖区回升，可能是买入机会'
                })
            elif df['RSI'].iloc[i] > 70 and df['RSI'].iloc[i-1] <= 70:
                signals.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': df['Close'].iloc[i],
                    'reason': 'RSI进入超买区，可能是卖出机会'
                })
        
        # MACD信号
        for i in range(1, len(df)):
            # 确保所需列存在且不为空值
            if 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
                continue
            if pd.isna(df['MACD'].iloc[i]) or pd.isna(df['MACD_Signal'].iloc[i]) or pd.isna(df['MACD'].iloc[i-1]) or pd.isna(df['MACD_Signal'].iloc[i-1]):
                continue
            
            # MACD金叉死叉信号
            if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]:
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': 'MACD金叉，市场可能转强'
                })
            elif df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]:
                signals.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': df['Close'].iloc[i],
                    'reason': 'MACD死叉，市场可能转弱'
                })
        
        # 布林带信号
        for i in range(1, len(df)):
            # 确保所需列存在且不为空值
            if 'Close' not in df.columns or 'BB_Upper' not in df.columns or 'BB_Lower' not in df.columns:
                continue
            if pd.isna(df['Close'].iloc[i]) or pd.isna(df['BB_Upper'].iloc[i]) or pd.isna(df['BB_Lower'].iloc[i]):
                continue
            
            # 价格触及布林带边界信号
            if df['Close'].iloc[i] <= df['BB_Lower'].iloc[i] and df['Close'].iloc[i-1] > df['BB_Lower'].iloc[i-1]:
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': '价格触及布林带下轨，可能出现反弹'
                })
            elif df['Close'].iloc[i] >= df['BB_Upper'].iloc[i] and df['Close'].iloc[i-1] < df['BB_Upper'].iloc[i-1]:
                signals.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': df['Close'].iloc[i],
                    'reason': '价格触及布林带上轨，可能出现回调'
                })
        
        # 短线/日内交易信号
        # 基于价格波动和成交量的短线信号
        for i in range(1, len(df)):
            # 确保所需列存在且不为空值
            if 'Close' not in df.columns or 'Open' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns:
                continue
            
            # 计算当日价格波动幅度
            price_change = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            price_range = df['High'].iloc[i] - df['Low'].iloc[i]
            
            # 如果价格波动幅度大于等于价格区间的70%，且收盘价高于开盘价，则为买入信号
            if price_range > 0 and price_change / price_range >= 0.7 and df['Close'].iloc[i] > df['Open'].iloc[i]:
                signals.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['Close'].iloc[i],
                    'reason': '日内价格波动剧烈且收涨，可能是短线买入机会'
                })
            # 如果价格波动幅度大于等于价格区间的70%，且收盘价低于开盘价，则为卖出信号
            elif price_range > 0 and price_change / price_range >= 0.7 and df['Close'].iloc[i] < df['Open'].iloc[i]:
                signals.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': df['Close'].iloc[i],
                    'reason': '日内价格波动剧烈且收跌，可能是短线卖出机会'
                })
        
        # 按日期排序信号
        signals.sort(key=lambda x: x['date'])
        
        # 过滤掉过于接近的信号（避免频繁交易）
        filtered_signals = []
        for i, signal in enumerate(signals):
            # 如果是第一个信号，直接添加
            if i == 0:
                filtered_signals.append(signal)
                continue
            
            # 检查当前信号与上一个信号是否同类型
            if signal['type'] == filtered_signals[-1]['type']:
                # 同类型信号，跳过
                continue
            
            # 检查距离上一个信号的时间间隔
            days_diff = (signal['date'] - filtered_signals[-1]['date']).days
            if days_diff >= 3:  # 至少间隔3天才添加新信号
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def _extract_signals_from_ai_analysis(self, ai_analysis, technical_data=None):
        """从AI分析结果中提取交易信号（已注释掉）"""
        return []  # 返回空列表，禁用AI分析信号提取
    
    def _extract_signals_from_text(self, text, technical_data=None):
        """从文本中提取交易信号"""
        signals = []
        
        # 使用更精确的模式匹配信号，包括日期、类型和价格
        signal_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)\s*-\s*(买入|卖出|持有|BUY|SELL|HOLD)\s*[@￥]\s*(\d+\.\d+)'
        
        # 查找所有匹配的信号
        matches = re.findall(signal_pattern, text)
        for date_str, signal_type, price_str in matches:
            try:
                # 解析日期
                date = datetime.strptime(date_str, '%Y年%m月%d日')
                
                # 解析价格
                price = float(price_str)
                
                # 标准化信号类型
                signal_type_mapping = {
                    '买入': 'BUY',
                    'BUY': 'BUY',
                    '卖出': 'SELL',
                    'SELL': 'SELL',
                    '持有': 'HOLD',
                    '观望': 'HOLD',
                    'HOLD': 'HOLD'
                }
                normalized_type = signal_type_mapping.get(signal_type, 'HOLD')
                
                signals.append({
                    'date': date,
                    'type': normalized_type,
                    'price': price,
                    'reason': f'AI分析建议{normalized_type.lower()}'
                })
            except ValueError:
                # 如果日期或价格解析失败，跳过这个信号
                continue
        
        # 如果没有找到任何信号，尝试使用旧的方法
        if not signals:
            # 定义关键词模式
            buy_patterns = [r'买入', r'购入', r'增持', r'买进', r'投资建议.*买入']
            sell_patterns = [r'卖出', r'抛售', r'减持', r'卖出', r'投资建议.*卖出']
            hold_patterns = [r'持有', r'观望', r'持有观望', r'投资建议.*持有']
            
            # 查找买入信号
            for pattern in buy_patterns:
                if re.search(pattern, text):
                    # 尝试从文本中提取日期，如果找不到则使用技术指标数据中的历史日期
                    date = None
                    date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)'
                    date_match = re.search(date_pattern, text)
                    if date_match:
                        try:
                            date_str = date_match.group(1)
                            date = datetime.strptime(date_str, '%Y年%m月%d日')
                        except ValueError:
                            pass  # 如果日期解析失败，使用技术指标数据中的历史日期
                    
                    # 如果没有找到日期或者日期解析失败，使用技术指标数据中的历史日期
                    if date is None and technical_data is not None and len(technical_data) > 0:
                        # 随机选择技术指标数据中的历史日期，而不是总是使用最新日期
                        import random
                        if len(technical_data) > 10:
                            # 从最近30天的数据中随机选择日期
                            recent_data = technical_data.tail(30)
                            date = random.choice(recent_data.index)
                        else:
                            date = random.choice(technical_data.index)
                    
                    # 尝试提取买入价格
                    price = 0
                    # 查找"买入价"或"目标价"后的价格
                    buy_price_patterns = [r'买入价[为是]\s*(\d+\.\d+)', r'目标价[为是]\s*(\d+\.\d+)', r'买入[目标][价为]\s*(\d+\.\d+)']
                    for price_pattern in buy_price_patterns:
                        match = re.search(price_pattern, text)
                        if match:
                            price = float(match.group(1))
                            break
                    
                    # 如果没找到特定买入价，则根据日期从技术指标数据中获取对应价格
                    if price == 0 and technical_data is not None and len(technical_data) > 0:
                        if date is not None and date in technical_data.index:
                            price = float(technical_data.loc[date, 'Close'])
                        else:
                            # 如果找不到对应日期的价格，使用最新收盘价
                            price = float(technical_data['Close'].iloc[-1])
                    
                    # 尝试从文本中提取日期，如果找不到则使用技术指标数据中的历史日期
                    date = None
                    date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)'
                    date_match = re.search(date_pattern, text)
                    if date_match:
                        try:
                            date_str = date_match.group(1)
                            date = datetime.strptime(date_str, '%Y年%m月%d日')
                        except ValueError:
                            pass  # 如果日期解析失败，使用技术指标数据中的历史日期
                    
                    # 如果没有找到日期或者日期解析失败，使用技术指标数据中的历史日期
                    if date is None and technical_data is not None and len(technical_data) > 0:
                        # 随机选择技术指标数据中的历史日期，而不是总是使用最新日期
                        import random
                        if len(technical_data) > 10:
                            # 从最近30天的数据中随机选择日期
                            recent_data = technical_data.tail(30)
                            date = random.choice(recent_data.index)
                        else:
                            date = random.choice(technical_data.index)
                    
                    signals.append({
                        'date': date,
                        'type': 'BUY',
                        'price': price,
                        'reason': f'AI分析建议{pattern}'
                    })
            
            # 查找卖出信号
            for pattern in sell_patterns:
                if re.search(pattern, text):
                    # 尝试从文本中提取日期，如果找不到则使用技术指标数据中的历史日期
                    date = None
                    date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)'
                    date_match = re.search(date_pattern, text)
                    if date_match:
                        try:
                            date_str = date_match.group(1)
                            date = datetime.strptime(date_str, '%Y年%m月%d日')
                        except ValueError:
                            pass  # 如果日期解析失败，使用技术指标数据中的历史日期
                    
                    # 如果没有找到日期或者日期解析失败，使用技术指标数据中的历史日期
                    if date is None and technical_data is not None and len(technical_data) > 0:
                        # 随机选择技术指标数据中的历史日期，而不是总是使用最新日期
                        import random
                        if len(technical_data) > 10:
                            # 从最近30天的数据中随机选择日期
                            recent_data = technical_data.tail(30)
                            date = random.choice(recent_data.index)
                        else:
                            date = random.choice(technical_data.index)
                    
                    # 尝试提取卖出价格
                    price = 0
                    # 查找"卖出价"或"目标价"后的价格
                    sell_price_patterns = [r'卖出价[为是]\s*(\d+\.\d+)', r'目标价[为是]\s*(\d+\.\d+)', r'卖出[目标][价为]\s*(\d+\.\d+)']
                    for price_pattern in sell_price_patterns:
                        match = re.search(price_pattern, text)
                        if match:
                            price = float(match.group(1))
                            break
                    
                    # 如果没找到特定卖出价，则根据日期从技术指标数据中获取对应价格
                    if price == 0 and technical_data is not None and len(technical_data) > 0:
                        if date is not None and date in technical_data.index:
                            price = float(technical_data.loc[date, 'Close'])
                        else:
                            # 如果找不到对应日期的价格，使用最新收盘价
                            price = float(technical_data['Close'].iloc[-1])
                    
                    # 尝试从文本中提取日期，如果找不到则使用技术指标数据中的历史日期
                    date = None
                    date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)'
                    date_match = re.search(date_pattern, text)
                    if date_match:
                        try:
                            date_str = date_match.group(1)
                            date = datetime.strptime(date_str, '%Y年%m月%d日')
                        except ValueError:
                            pass  # 如果日期解析失败，使用技术指标数据中的历史日期
                    
                    # 如果没有找到日期或者日期解析失败，使用技术指标数据中的历史日期
                    if date is None and technical_data is not None and len(technical_data) > 0:
                        # 随机选择技术指标数据中的历史日期，而不是总是使用最新日期
                        import random
                        if len(technical_data) > 10:
                            # 从最近30天的数据中随机选择日期
                            recent_data = technical_data.tail(30)
                            date = random.choice(recent_data.index)
                        else:
                            date = random.choice(technical_data.index)
                    
                    signals.append({
                        'date': date,
                        'type': 'SELL',
                        'price': price,
                        'reason': f'AI分析建议{pattern}'
                    })
            
            # 查找持有信号
            for pattern in hold_patterns:
                if re.search(pattern, text):
                    # 持有信号通常不涉及具体价格，但可以根据日期从技术指标数据中获取对应的价格
                    price = 0
                    
                    # 尝试从文本中提取日期，如果找不到则使用技术指标数据中的历史日期
                    date = None
                    date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)'
                    date_match = re.search(date_pattern, text)
                    if date_match:
                        try:
                            date_str = date_match.group(1)
                            date = datetime.strptime(date_str, '%Y年%m月%d日')
                        except ValueError:
                            pass  # 如果日期解析失败，使用技术指标数据中的历史日期
                    
                    # 如果没有找到日期或者日期解析失败，使用技术指标数据中的历史日期
                    if date is None and technical_data is not None and len(technical_data) > 0:
                        # 随机选择技术指标数据中的历史日期，而不是总是使用最新日期
                        import random
                        if len(technical_data) > 10:
                            # 从最近30天的数据中随机选择日期
                            recent_data = technical_data.tail(30)
                            date = random.choice(recent_data.index)
                        else:
                            date = random.choice(technical_data.index)
                    
                    # 根据日期从技术指标数据中获取对应价格
                    if price == 0 and technical_data is not None and len(technical_data) > 0:
                        if date is not None and date in technical_data.index:
                            price = float(technical_data.loc[date, 'Close'])
                        else:
                            # 如果找不到对应日期的价格，使用最新收盘价
                            price = float(technical_data['Close'].iloc[-1])
                    
                    signals.append({
                        'date': date,
                        'type': 'HOLD',
                        'price': price,
                        'reason': f'AI分析建议{pattern}'
                    })
        
        # 去重：如果存在相同日期和类型的信号，只保留一个
        unique_signals = []
        seen_signals = set()
        for signal in signals:
            signal_key = (signal['date'], signal['type'])
            if signal_key not in seen_signals:
                seen_signals.add(signal_key)
                unique_signals.append(signal)
        
        return unique_signals
    
    def _serialize_dataframe(self, df):
        """将DataFrame转换为可序列化的字典"""
        result = {}
        for column in df.columns:
            # 将numpy类型转换为Python原生类型，处理NaN和无限值
            serialized_values = []
            for x in df[column]:
                try:
                    if hasattr(x, 'item'):
                        value = float(x.item())
                    else:
                        value = float(x)
                    
                    # 检查是否为NaN或无限值
                    if pd.isna(value) or not np.isfinite(value):
                        value = 0.0
                    
                    serialized_values.append(value)
                except (ValueError, TypeError):
                    serialized_values.append(0.0)
            
            result[column] = serialized_values
        return result
    
    def _serialize_signal(self, signal):
        """将交易信号转换为可序列化的字典"""
        return {
            'date': signal['date'].strftime('%Y-%m-%d'),
            'type': signal['type'],
            'price': float(signal['price']),
            'reason': signal['reason']
        }
    
    def analyze_with_ai(self, technical_data, market_data, stock_info, signals):
        """使用大模型API分析股票数据"""
        try:
            # 检查是否配置了大模型API
            if not self.local_model_url or not self.local_model_name:
                # 如果没有配置大模型API，则使用本地逻辑
                print("未配置大模型API，使用本地逻辑进行分析")
                return self._analyze_with_local_logic(technical_data, market_data, stock_info, signals)
            
            # 准备发送给大模型的数据
            # 构建股票分析报告
            report = ""
            
            # 添加股票基本信息
            report += f"股票代码: {stock_info.get('symbol', 'N/A')}, 股票名称: {stock_info.get('longName', 'N/A')}\n\n"
            
            # 分析最近的收盘价趋势
            if len(technical_data) >= 5:
                recent_prices = technical_data['Close'].tail(5)
                report += f"最近5个交易日收盘价: {', '.join([f'{price:.2f}' for price in recent_prices])}\n"
                
                # 简单趋势分析
                if recent_prices.iloc[-1] > recent_prices.iloc[0]:
                    report += "近期价格呈上升趋势。\n"
                elif recent_prices.iloc[-1] < recent_prices.iloc[0]:
                    report += "近期价格呈下降趋势。\n"
                else:
                    report += "近期价格保持稳定。\n"
            
            # 分析交易信号
            if signals:
                latest_signal = signals[-1]
                report += f"\n最新交易信号: {latest_signal.get('signal', 'N/A')} - {latest_signal.get('reason', 'N/A')}\n"
                
                # 统计信号类型
                signal_types = [s.get('signal', 'N/A') for s in signals[-10:]]
                buy_signals = signal_types.count('买入')
                sell_signals = signal_types.count('卖出')
                hold_signals = signal_types.count('持有')
                
                report += f"最近10个信号统计: 买入({buy_signals}), 卖出({sell_signals}), 持有({hold_signals})\n"
            
            # 添加技术指标摘要
            if len(technical_data) > 0:
                latest = technical_data.iloc[-1]
                report += f"\n最新技术指标:\n"
                report += f"  收盘价: {latest['Close']:.2f}\n"
                if 'SMA_20' in latest:
                    report += f"  20日均线: {latest['SMA_20']:.2f}\n"
                if 'RSI' in latest:
                    report += f"  RSI: {latest['RSI']:.2f}\n"
                if 'MACD' in latest:
                    report += f"  MACD: {latest['MACD']:.2f}\n"
            
            # 获取市场环境数据
            market_index = "N/A"
            industry_performance = "N/A"
            if market_data is not None and len(market_data) > 0:
                # 获取最新的大盘指数
                market_index = f"{market_data['Close'].iloc[-1]:.2f}"
                # 计算大盘最近5天的趋势
                if len(market_data) >= 5:
                    market_recent_prices = market_data['Close'].tail(5)
                    if market_recent_prices.iloc[-1] > market_recent_prices.iloc[0]:
                        market_trend = "上升"
                    elif market_recent_prices.iloc[-1] < market_recent_prices.iloc[0]:
                        market_trend = "下降"
                    else:
                        market_trend = "稳定"
                    market_index += f" (最近5天趋势: {market_trend})"
            
            # 构造优化后的提示词
            # 读取优化后的提示词模板
            with open("optimized_prompt_template.txt", "r", encoding="utf-8") as f:
                prompt_template = f.read()
            
            # 准备模板变量
            stock_code = stock_info.get('symbol', 'N/A')
            stock_name = stock_info.get('longName', 'N/A')
            recent_prices = "N/A"
            price_trend = "N/A"
            latest_signal = "N/A"
            signal_reason = "N/A"
            buy_signals = 0
            sell_signals = 0
            hold_signals = 0
            close_price = "N/A"
            rsi = "N/A"
            macd = "N/A"
            ma20 = "N/A"
            ma50 = "N/A"
            
            # 填充模板变量
            if len(technical_data) >= 5:
                recent_prices = ', '.join([f'{price:.2f}' for price in technical_data['Close'].tail(5)])
                # 简单趋势分析
                recent_prices_series = technical_data['Close'].tail(5)
                if recent_prices_series.iloc[-1] > recent_prices_series.iloc[0]:
                    price_trend = "上升"
                elif recent_prices_series.iloc[-1] < recent_prices_series.iloc[0]:
                    price_trend = "下降"
                else:
                    price_trend = "稳定"
            
            if signals:
                latest_signal_data = signals[-1]
                latest_signal = latest_signal_data.get('type', 'N/A')
                signal_reason = latest_signal_data.get('reason', 'N/A')
                
                # 统计信号类型
                signal_types = [s.get('type', 'N/A') for s in signals[-10:]]
                buy_signals = signal_types.count('BUY')
                sell_signals = signal_types.count('SELL')
            
            if len(technical_data) > 0:
                latest = technical_data.iloc[-1]
                close_price = f"{latest['Close']:.2f}"
                if 'RSI' in latest:
                    rsi = f"{latest['RSI']:.2f}"
                if 'MACD' in latest:
                    macd = f"{latest['MACD']:.2f}"
                if 'MA20' in latest:
                    ma20 = f"{latest['MA20']:.2f}"
                if 'MA50' in latest:
                    ma50 = f"{latest['MA50']:.2f}"
            
            # 填充模板
            prompt = prompt_template.format(
                stock_code=stock_code,
                stock_name=stock_name,
                recent_prices=recent_prices,
                price_trend=price_trend,
                latest_signal=latest_signal,
                signal_reason=signal_reason,
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                close_price=close_price,
                rsi=rsi,
                macd=macd,
                ma20=ma20,
                ma50=ma50,
                market_index=market_index,
                industry_performance=industry_performance
            )
            
            # 调用大模型API，添加重试机制和更短的超时时间
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.local_model_name,
                "messages": [
                    {"role": "system", "content": "你是一位专业的股票分析师，擅长根据技术指标、交易信号、市场环境和行业表现提供投资建议。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
            
            # 尝试最多3次，每次超时时间逐渐增加
            max_retries = 3
            timeout_values = [65, 75, 85]  # 逐渐增加的超时时间
            
            for attempt in range(max_retries):
                try:
                    timeout = timeout_values[attempt]
                    print(f"第{attempt + 1}次尝试调用AI模型，超时时间: {timeout}秒")
                    
                    response = requests.post(
                        f"{self.local_model_url}/v1/chat/completions",
                        headers=headers,
                        data=json.dumps(data),
                        timeout=timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        analysis = result['choices'][0]['message']['content']
                        print(f"AI模型调用成功，耗时: {response.elapsed.total_seconds():.2f}秒")
                        return analysis
                    else:
                        print(f"AI模型调用失败: {response.status_code} - {response.text}")
                        
                except requests.exceptions.Timeout:
                    print(f"第{attempt + 1}次尝试超时")
                    if attempt == max_retries - 1:  # 最后一次尝试仍然超时
                        print("AI模型调用多次超时，回退到本地逻辑")
                        return f"AI分析超时，已回退到本地逻辑进行分析。\n\n{self._analyze_with_local_logic(technical_data, market_data, stock_info, signals)}"
                except Exception as e:
                    print(f"AI模型调用出现异常: {str(e)}")
                    if attempt == max_retries - 1:  # 最后一次尝试仍然失败
                        print("AI模型调用多次失败，回退到本地逻辑")
                        return self._analyze_with_local_logic(technical_data, market_data, stock_info, signals)
            
            # 如果所有尝试都失败，使用本地逻辑
            print("所有尝试都失败，使用本地逻辑进行分析")
            return self._analyze_with_local_logic(technical_data, market_data, stock_info, signals)
        except Exception as e:
            # 如果出现异常，使用本地逻辑
            print(f"AI分析出现异常: {str(e)}")
            return self._analyze_with_local_logic(technical_data, market_data, stock_info, signals)
    
    def read_tdx_day_file(self, file_path):
        """读取通达信.day文件"""
        try:
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # 每条记录32字节
            record_size = 32
            records = []
            
            for i in range(0, len(content), record_size):
                record_data = content[i:i+record_size]
                if len(record_data) < record_size:
                    break
                
                # 解析记录
                # 日期(4字节), 开盘价(4字节), 最高价(4字节), 最低价(4字节), 收盘价(4字节), 成交额(4字节), 成交量(4字节), 未使用(4字节)
                date, open_price, high_price, low_price, close_price, amount, volume, _ = struct.unpack('IIIIIfIf', record_data)
                
                # 日期转换
                date_str = str(date)
                if len(date_str) == 8:
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                else:
                    continue
                
                # 价格和成交量需要除以100得到实际值
                records.append({
                    'Date': date_obj,
                    'Open': open_price / 100.0,
                    'High': high_price / 100.0,
                    'Low': low_price / 100.0,
                    'Close': close_price / 100.0,
                    'Amount': amount,
                    'Volume': volume / 100.0
                })
            
            # 转换为DataFrame
            if records:
                df = pd.DataFrame(records)
                df.set_index('Date', inplace=True)
                return df
            
            return None
        except Exception as e:
            print(f"读取通达信数据文件失败: {e}")
            return None
    
    def get_stock_data_from_tdx(self, stock_symbol):
        """从通达信数据文件获取股票数据"""
        try:
            # 根据股票代码确定市场和文件路径
            if stock_symbol.endswith('.SS'):
                # 上海证券交易所
                code = stock_symbol.replace('.SS', '')
                file_path = f"data/sh/lday/sh{code}.day"
            elif stock_symbol.endswith('.SZ'):
                # 深圳证券交易所
                code = stock_symbol.replace('.SZ', '')
                file_path = f"data/sz/lday/sz{code}.day"
            elif stock_symbol.endswith('.BJ'):
                # 北京证券交易所
                code = stock_symbol.replace('.BJ', '')
                file_path = f"data/bj/lday/bj{code}.day"
            else:
                # 默认尝试上海
                file_path = f"data/sh/lday/sh{stock_symbol}.day"
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return None
            
            # 读取数据
            df = self.read_tdx_day_file(file_path)
            if df is not None and not df.empty:
                return {
                    'stock_data': df,
                    'market_data': None,
                    'stock_info': {'symbol': stock_symbol, 'longName': stock_symbol},
                    'data_source': 'tdx_file'
                }
            
            return None
        except Exception as e:
            print(f"从通达信数据文件获取股票数据失败: {e}")
            return None
    
    def analyze_stock(self, stock_symbol):
        """分析股票"""
        try:
            print(f"开始分析股票: {stock_symbol}")
            
            # 优先使用akshare获取数据
            data_result = self.akshare_provider.get_stock_data(stock_symbol)
            
            # 如果akshare获取失败，则从通达信数据文件获取数据
            if data_result is None:
                print(f"akshare获取数据失败，尝试从通达信数据文件获取数据")
                data_result = self.get_stock_data_from_tdx(stock_symbol)
            
            # 如果从通达信数据文件获取失败，则使用免费API
            if data_result is None:
                print(f"从通达信数据文件获取数据失败，尝试使用免费API")
                data_result = self.get_free_stock_data(stock_symbol)
            
            df = data_result['stock_data']
            stock_info = data_result['stock_info']
            market_data = data_result.get('market_data')
            print(f"获取到股票数据，数据长度: {len(df)}")
            
            # 如果数据量太少，尝试使用yfinance获取更多历史数据
            if len(df) < 10:
                try:
                    yfinance_data = self.get_stock_data_yfinance(stock_symbol)
                    if len(yfinance_data['stock_data']) > len(df):
                        data_result = yfinance_data
                        df = data_result['stock_data']
                        market_data = data_result.get('market_data')
                except:
                    pass  # 忽略yfinance错误，继续使用已有数据
            
            # 计算技术指标
            technical_data = self.calculate_technical_indicators(df)
            print("技术指标计算完成")
            
            # 生成交易信号
            signals = self.generate_trading_signals(technical_data)
            print(f"生成交易信号，信号数量: {len(signals)}")
            
            # AI分析（重新启用AI分析，但信号提取功能保持禁用）
            ai_analysis = self.analyze_with_ai(technical_data, market_data, stock_info, signals)
            print(f"AI分析完成，分析结果类型: {type(ai_analysis)}")
            
            # 使用技术指标生成的信号，但保留AI分析结果，并重新启用K线图数据
            # 先执行回测并将回测结果传入图表生成器，以便在K线图上叠加回测数据
            backtest_result = self.run_backtest(technical_data, signals)

            result = {
                'success': True,
                'stock_info': stock_info,
                'technical_data': self._serialize_dataframe(technical_data.tail(10)),
                'signals': [self._serialize_signal(signal) for signal in signals[-10:]],  # 最近10个信号
                'ai_analysis': ai_analysis,
                'latest_price': float(technical_data['Close'].iloc[-1]) if len(technical_data) > 0 else 0,
                'data_source': 'akshare' if 'akshare' in str(type(self.akshare_provider)) else ('tdx_file' if data_result.get('data_source') == 'tdx_file' else ('free_api' if len(df) <= 1 else 'yfinance')),
                'backtest': backtest_result,
                'risk_report': self.get_risk_report(technical_data, signals),
                'chart_data': self._generate_chart_data(technical_data, backtest=backtest_result)
            }
            
            # 确保交易信号正确传递到最终结果中
            if 'signals' not in result or not result['signals']:
                result['signals'] = [self._serialize_signal(signal) for signal in signals[-10:]]
            
            print("股票分析完成")
            return result
            
        except Exception as e:
            print(f"股票分析异常: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_with_local_logic(self, technical_data, market_data, stock_info, signals):
        """使用本地逻辑分析股票数据（已注释掉）"""
        return "本地逻辑分析已禁用，仅使用技术指标信号"

    def run_backtest(self, technical_data=None, signals=None):
        """运行回测并返回结果"""
        if technical_data is None or signals is None or technical_data.empty or not signals:
            return {
                "performance_metrics": {
                    "total_return": 0.0,
                    "annualized_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "profit_factor": 0.0,
                    "commission_total": 0.0,
                    "initial_value": 0.0,
                    "final_value": 0.0
                }
            }
        
        try:
            # 使用BacktestEngine进行回测
            backtest_result = self.backtest_engine.run_backtest(technical_data, signals)
            # 返回既包含已存在的 performance_metrics（兼容前端/调用方），
            # 也保留完整的引擎结果以便图表和进一步分析使用
            return {
                "performance_metrics": backtest_result['performance'],
                "engine_result": backtest_result
            }
        except Exception as e:
            print(f"回测执行失败: {str(e)}")
            return {
                "performance_metrics": {
                    "total_return": 0.0,
                    "annualized_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "profit_factor": 0.0,
                    "commission_total": 0.0,
                    "initial_value": 0.0,
                    "final_value": 0.0
                }
            }

    def get_risk_report(self, technical_data=None, signals=None, portfolio_value=100000):
        """生成风险报告"""
        if technical_data is None or technical_data.empty:
            return "数据不足，无法生成风险报告"
        
        try:
            # 使用最新的信号（如果存在）进行风险评估
            latest_signal = signals[-1] if signals else {'type': 'HOLD'}
            
            # 评估信号风险
            risk_assessment = self.risk_manager.assess_signal_risk(technical_data, latest_signal)
            
            # 获取当前价格
            current_price = technical_data['Close'].iloc[-1]
            
            # 计算仓位大小和止损止盈价格
            _, stop_loss_price, take_profit_price = self.risk_manager.calculate_position_size(
                portfolio_value, 
                risk_assessment['risk_score'], 
                current_price
            )
            
            # 构造前端需要的报告格式
            report = {
                "risk_level": risk_assessment['risk_level'],
                "risk_score": risk_assessment['risk_score'] * 100,  # 转换为百分比显示（0-100）
                "position_size": risk_assessment['suggested_position_pct'],
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "details": risk_assessment 
            }
            return report
            
        except Exception as e:
            print(f"生成风险报告失败: {str(e)}")
            return "生成风险报告时发生错误"

    def generate_trading_signals_with_risk(self, df):
        """生成带风险控制的交易信号"""
        try:
            # 调用实际的信号生成逻辑
            return self.generate_trading_signals(df)
        except Exception as e:
            print(f"交易信号生成失败: {str(e)}")
            return []

    def _generate_chart_data(self, technical_data=None, backtest=None):
        """生成K线图数据并可选叠加回测数据（组合曲线、交易标记）"""
        if technical_data is None or len(technical_data) == 0:
            return {"kline": [], "indicators": {}, "signals": []}
        
        try:
            # 获取最近60天的数据用于图表显示
            chart_data = technical_data.tail(60).copy()
            
            # 准备K线数据
            kline_data = []
            for idx, row in chart_data.iterrows():
                kline_data.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
                })
            
            # 准备技术指标数据
            indicators = {}
            
            # 移动平均线
            if 'MA5' in chart_data.columns:
                indicators['ma5'] = [{'date': idx.strftime('%Y-%m-%d'), 'value': float(row['MA5'])} 
                                   for idx, row in chart_data.iterrows() if not pd.isna(row['MA5'])]
            if 'MA10' in chart_data.columns:
                indicators['ma10'] = [{'date': idx.strftime('%Y-%m-%d'), 'value': float(row['MA10'])} 
                                    for idx, row in chart_data.iterrows() if not pd.isna(row['MA10'])]
            if 'MA20' in chart_data.columns:
                indicators['ma20'] = [{'date': idx.strftime('%Y-%m-%d'), 'value': float(row['MA20'])} 
                                    for idx, row in chart_data.iterrows() if not pd.isna(row['MA20'])]
            
            # MACD
            if 'MACD' in chart_data.columns:
                indicators['macd'] = [{'date': idx.strftime('%Y-%m-%d'), 'value': float(row['MACD'])} 
                                   for idx, row in chart_data.iterrows() if not pd.isna(row['MACD'])]
            if 'MACD_Signal' in chart_data.columns:
                indicators['macd_signal'] = [{'date': idx.strftime('%Y-%m-%d'), 'value': float(row['MACD_Signal'])} 
                                           for idx, row in chart_data.iterrows() if not pd.isna(row['MACD_Signal'])]
            if 'MACD_Hist' in chart_data.columns:
                indicators['macd_hist'] = [{'date': idx.strftime('%Y-%m-%d'), 'value': float(row['MACD_Hist'])} 
                                          for idx, row in chart_data.iterrows() if not pd.isna(row['MACD_Hist'])]
            
            # RSI
            if 'RSI' in chart_data.columns:
                indicators['rsi'] = [{'date': idx.strftime('%Y-%m-%d'), 'value': float(row['RSI'])} 
                                   for idx, row in chart_data.iterrows() if not pd.isna(row['RSI'])]
            
            # 布林带
            if 'BB_Upper' in chart_data.columns:
                indicators['bb_upper'] = [{'date': idx.strftime('%Y-%m-%d'), 'value': float(row['BB_Upper'])} 
                                        for idx, row in chart_data.iterrows() if not pd.isna(row['BB_Upper'])]
            if 'BB_Middle' in chart_data.columns:
                indicators['bb_middle'] = [{'date': idx.strftime('%Y-%m-%d'), 'value': float(row['BB_Middle'])} 
                                         for idx, row in chart_data.iterrows() if not pd.isna(row['BB_Middle'])]
            if 'BB_Lower' in chart_data.columns:
                indicators['bb_lower'] = [{'date': idx.strftime('%Y-%m-%d'), 'value': float(row['BB_Lower'])} 
                                        for idx, row in chart_data.iterrows() if not pd.isna(row['BB_Lower'])]
            
            # 交易信号
            signals_data = []
            if hasattr(self, 'generate_trading_signals'):
                try:
                    signals = self.generate_trading_signals(technical_data)
                    for signal in signals[-20:]:  # 最近20个信号
                        signals_data.append({
                            'date': signal['date'].strftime('%Y-%m-%d'),
                            'type': signal['type'],
                            'price': float(signal['price']),
                            'reason': signal.get('reason', '')
                        })
                except Exception as e:
                    print(f"生成交易信号数据失败: {str(e)}")

            # 回测叠加数据：将组合价值曲线重采样到当前K线图的日期并生成交易标记
            portfolio_series = []
            trade_markers = []
            try:
                if backtest and isinstance(backtest, dict) and 'engine_result' in backtest:
                    engine_res = backtest.get('engine_result', {})
                    ph = engine_res.get('portfolio_history', [])
                    trades = engine_res.get('trades', [])

                    # 将 portfolio_history（按顺序）映射到 technical_data 的索引位置，构建一个 Series
                    full_idxs = list(technical_data.index)
                    values = []
                    for i, p in enumerate(ph):
                        if i >= len(full_idxs):
                            break
                        # 提取 value（支持 dataclass 或 dict）
                        value = None
                        if p is not None:
                            value = getattr(p, 'value', None) if hasattr(p, '__dict__') or hasattr(p, 'value') else None
                            if value is None and isinstance(p, dict):
                                value = p.get('value')
                        if value is None:
                            # 跳过无法解析的记录
                            values.append(float('nan'))
                        else:
                            try:
                                values.append(float(value))
                            except Exception:
                                values.append(float('nan'))

                    if values:
                        # 构建按完整数据索引的Series，并重采样到用于展示的 chart_data 索引（通常是 tail(60)）
                        import pandas as _pd
                        port_series = _pd.Series(data=values, index=full_idxs[:len(values)])
                        kline_index = chart_data.index

                        # 重索引到 kline_index，向前填充最近的组合价值，若前端范围早于首个可用值则向后填充，再用0替换缺失
                        port_resampled = port_series.reindex(kline_index, method='ffill')
                        port_resampled = port_resampled.fillna(method='bfill').fillna(0.0)

                        for d, v in port_resampled.items():
                            portfolio_series.append({'date': d.strftime('%Y-%m-%d'), 'value': float(v)})

                    # 交易标记：仅保留在当前图表时间窗（chart_data）内的标记，避免图上显示过多无关点
                    chart_dates = {idx.strftime('%Y-%m-%d') for idx in chart_data.index}
                    for t in trades:
                        trade_date = getattr(t, 'date', None)
                        trade_type = getattr(t, 'type', None)
                        trade_price = getattr(t, 'price', None)
                        # 支持 dict 形式
                        if trade_date is None and isinstance(t, dict):
                            trade_date = t.get('date')
                            trade_type = t.get('type')
                            trade_price = t.get('price')

                        # 标准化日期为字符串
                        if isinstance(trade_date, str):
                            date_str = trade_date
                        elif hasattr(trade_date, 'strftime'):
                            date_str = trade_date.strftime('%Y-%m-%d')
                        else:
                            continue

                        # 只添加在当前图表日期范围内的交易标记
                        if date_str in chart_dates:
                            trade_markers.append({
                                'date': date_str,
                                'type': trade_type,
                                'price': float(trade_price) if trade_price is not None else None
                            })
            except Exception as e:
                print(f"叠加回测数据到图表时出错: {e}")
            
            return {
                "kline": kline_data,
                "indicators": indicators,
                "signals": signals_data,
                "portfolio": portfolio_series,
                "trade_markers": trade_markers
            }
            
        except Exception as e:
            print(f"生成K线图数据失败: {str(e)}")
            return {"kline": [], "indicators": {}, "signals": []}

# 示例使用
if __name__ == "__main__":
    import sys
    stock_code = sys.argv[1] if len(sys.argv) > 1 else "TEST.STOCK"
    analyzer = StockAnalyzer()
    result = analyzer.analyze_stock(stock_code)
    
    # 如果分析成功，打印完整的AI分析报告
    if result.get('success', False):
        ai_analysis = result.get('ai_analysis', '')
        if isinstance(ai_analysis, str):
            print(ai_analysis)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))