"""
增强版实时行情数据提供器
支持多数据源互补获取实时行情数据，解决连接失败问题
"""

import os
import time
import random
import logging
import requests
import pandas as pd
import akshare as ak
from typing import Optional, Dict, Any
from datetime import datetime

class EnhancedRealtimeProvider:
    """增强版实时行情数据提供器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 配置重试参数
        self.max_retries = int(os.getenv("ENH_REALTIME_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("ENH_REALTIME_RETRY_DELAY", "2.0"))
        
        # 配置数据源优先级
        self.data_sources = self._init_data_sources()
        
        # 缓存最近成功的请求
        self.cache = {}
        self.cache_ttl = 60  # 缓存60秒
        
        self.logger.info(f"EnhancedRealtimeProvider 初始化完成: {len(self.data_sources)} 个数据源")
    
    def _init_data_sources(self) -> list:
        """初始化数据源列表"""
        sources = [
            {
                "name": "sina_realtime",
                "func": self._get_sina_realtime,
                "priority": 1,
                "description": "新浪财经实时行情"
            },
            {
                "name": "sina_batch_realtime",
                "func": self._get_sina_batch_realtime,
                "priority": 2,
                "description": "新浪财经批量实时行情"
            },
            {
                "name": "tencent_realtime",
                "func": self._get_tencent_realtime,
                "priority": 3,
                "description": "腾讯财经实时行情"
            },
            {
                "name": "eastmoney_realtime",
                "func": self._get_eastmoney_realtime,
                "priority": 4,
                "description": "东方财富实时行情"
            },
            {
                "name": "akshare_ah_spot",
                "func": self._get_akshare_ah_spot,
                "priority": 5,
                "description": "AkShare A股实时行情"
            },
            {
                "name": "xueqiu_realtime",
                "func": self._get_xueqiu_realtime,
                "priority": 6,
                "description": "雪球实时行情"
            }
        ]
        
        # 按优先级排序
        sources.sort(key=lambda x: x["priority"])
        return sources
    
    def get_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取股票实时行情数据
        
        Args:
            symbol: 股票代码，如 "000001.SZ"
            
        Returns:
            实时行情数据字典，包含价格、涨跌幅等信息
        """
        # 检查缓存
        cache_key = f"realtime_{symbol}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                self.logger.debug(f"使用缓存数据: {symbol}")
                return cached_data
        
        # 多数据源尝试
        for source in self.data_sources:
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"尝试数据源 {source['name']} 获取 {symbol} 实时行情 (尝试 {attempt + 1}/{self.max_retries})")
                    
                    result = source["func"](symbol)
                    
                    if result and self._validate_realtime_data(result):
                        # 缓存成功结果
                        self.cache[cache_key] = (time.time(), result)
                        self.logger.info(f"成功从 {source['name']} 获取 {symbol} 实时行情")
                        return result
                    
                    # 如果数据无效，等待后重试
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt) * (0.8 + 0.4 * random.random())
                        time.sleep(delay)
                        
                except Exception as e:
                    self.logger.warning(f"数据源 {source['name']} 获取 {symbol} 实时行情失败 (尝试 {attempt + 1}): {e}")
                    
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt) * (0.8 + 0.4 * random.random())
                        time.sleep(delay)
        
        self.logger.error(f"所有数据源均无法获取 {symbol} 实时行情")
        return None
    
    def _get_akshare_ah_spot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """使用AkShare获取A股实时行情"""
        try:
            # 获取A股实时行情
            df = ak.stock_zh_ah_spot()
            
            if df is None or df.empty:
                return None
            
            # 标准化列名
            df = self._standardize_akshare_columns(df)
            
            # 查找目标股票
            symbol_clean = symbol.replace('.SZ', '').replace('.SS', '')
            
            # 尝试匹配A股代码
            if 'code_a' in df.columns:
                match = df[df['code_a'] == symbol_clean]
                if not match.empty:
                    return self._extract_realtime_data(match.iloc[0], 'a')
            
            # 尝试匹配A股代码
            if 'code_h' in df.columns:
                match = df[df['code_h'] == symbol_clean]
                if not match.empty:
                    return self._extract_realtime_data(match.iloc[0], 'h')
            
            return None
            
        except Exception as e:
            self.logger.warning(f"AkShare A+H实时行情获取失败: {e}")
            return None
    
    def _get_sina_realtime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """使用新浪财经获取实时行情"""
        try:
            # 新浪财经实时行情API
            if symbol.endswith('.SZ'):
                code = f'sz{symbol.replace(".SZ", "")}'
            elif symbol.endswith('.SS'):
                code = f'sh{symbol.replace(".SS", "")}'
            else:
                return None
            
            url = f"http://hq.sinajs.cn/list={code}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'http://finance.sina.com.cn'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                content = response.text
                # 解析新浪财经数据格式
                data = self._parse_sina_data(content)
                if data:
                    return data
            
            return None
            
        except Exception as e:
            self.logger.warning(f"新浪财经实时行情获取失败: {e}")
            return None
    
    def _get_sina_batch_realtime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """使用新浪财经批量获取实时行情（单股票调用时使用批量接口）"""
        try:
            # 新浪财经批量实时行情API
            if symbol.endswith('.SZ'):
                code = f'sz{symbol.replace(".SZ", "")}'
            elif symbol.endswith('.SS'):
                code = f'sh{symbol.replace(".SS", "")}'
            else:
                return None
            
            # 使用批量接口，即使只有一个股票也使用批量接口格式
            url = f"http://hq.sinajs.cn/list={code}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'http://finance.sina.com.cn'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                content = response.text
                # 解析新浪财经数据格式
                data = self._parse_sina_data(content)
                if data:
                    return data
            
            return None
            
        except Exception as e:
            self.logger.warning(f"新浪财经批量实时行情获取失败: {e}")
            return None
    
    def _get_tencent_realtime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """使用腾讯财经获取实时行情"""
        try:
            # 腾讯财经实时行情API
            if symbol.endswith('.SZ'):
                code = f'sz{symbol.replace(".SZ", "")}'
            elif symbol.endswith('.SS'):
                code = f'sh{symbol.replace(".SS", "")}'
            else:
                return None
            
            url = f"http://qt.gtimg.cn/q={code}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'http://gu.qq.com'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                content = response.text
                # 解析腾讯财经数据格式
                data = self._parse_tencent_data(content)
                if data:
                    return data
            
            return None
            
        except Exception as e:
            self.logger.warning(f"腾讯财经实时行情获取失败: {e}")
            return None
    
    def _get_eastmoney_realtime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """使用东方财富获取实时行情"""
        try:
            # 东方财富实时行情API
            if symbol.endswith('.SZ'):
                code = f'0.{symbol.replace(".SZ", "")}'
            elif symbol.endswith('.SS'):
                code = f'1.{symbol.replace(".SS", "")}'
            else:
                return None
            
            url = f"http://push2.eastmoney.com/api/qt/stock/get"
            params = {
                'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
                'invt': '2',
                'fltt': '2',
                'fields': 'f43,f57,f58,f169,f170,f46,f44,f51,f168,f47,f164,f163,f116,f60,f45,f52,f50,f48,f167,f117,f71,f161,f49,f530,f135,f136,f137,f138,f139,f141,f142,f144,f145,f147,f148,f140,f143,f146,f149,f55,f62,f162,f92,f173,f104,f105,f84,f85,f183,f184,f185,f186,f187,f188,f189,f190,f191,f192,f107,f111,f86,f177,f78,f110,f262,f263,f264,f267,f268,f250,f251,f252,f253,f254,f255,f256,f257,f258,f266,f269,f270,f271,f273,f274,f275,f127,f199,f128,f193,f196,f194,f195,f197,f80,f280,f281,f282,f284,f285,f286,f287,f292',
                'secid': code
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'http://quote.eastmoney.com'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('rc') == 0:
                    return self._parse_eastmoney_data(data)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"东方财富实时行情获取失败: {e}")
            return None
    
    def _get_xueqiu_realtime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """使用雪球获取实时行情"""
        try:
            # 雪球实时行情API
            if symbol.endswith('.SZ'):
                code = f'SZ{symbol.replace(".SZ", "")}'
            elif symbol.endswith('.SS'):
                code = f'SH{symbol.replace(".SS", "")}'
            else:
                return None
            
            url = f"https://stock.xueqiu.com/v5/stock/quote.json"
            params = {
                'symbol': code,
                'extend': 'detail'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://xueqiu.com'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('error_code') == 0:
                    return self._parse_xueqiu_data(data)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"雪球实时行情获取失败: {e}")
            return None
    
    def _standardize_akshare_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化AkShare列名"""
        column_mapping = {
            '名称': 'name',
            'A股代码': 'code_a',
            'A股最新价': 'price_a',
            'A股涨跌幅': 'pct_chg_a',
            'A股成交量': 'volume_a'
        }
        
        df = df.rename(columns=column_mapping)
        return df
    
    def _extract_realtime_data(self, row: pd.Series, market_type: str) -> Dict[str, Any]:
        """从AkShare数据行提取实时行情数据"""
        prefix = market_type
        
        return {
            'symbol': row.get(f'code_{prefix}', ''),
            'price': row.get(f'price_{prefix}', 0),
            'change_pct': row.get(f'pct_chg_{prefix}', 0),
            'volume': row.get(f'volume_{prefix}', 0),
            'timestamp': datetime.now().isoformat(),
            'source': 'akshare'
        }
    
    def _parse_sina_data(self, content: str) -> Optional[Dict[str, Any]]:
        """解析新浪财经数据"""
        try:
            # 新浪财经数据格式: var hq_str_sh000001="上证指数,3269.32,3275.93,3269.32,3275.93,3269.32,0,0";
            if '"' in content:
                data_str = content.split('"')[1]
                fields = data_str.split(',')
                
                if len(fields) >= 3:
                    return {
                        'price': float(fields[3]) if fields[3] != '' else 0,
                        'change': float(fields[2]) - float(fields[1]) if fields[1] != '' and fields[2] != '' else 0,
                        'change_pct': ((float(fields[3]) - float(fields[2])) / float(fields[2]) * 100) if fields[2] != '' and float(fields[2]) != 0 else 0,
                        'volume': float(fields[8]) if len(fields) > 8 and fields[8] != '' else 0,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'sina'
                    }
        except Exception as e:
            self.logger.warning(f"解析新浪财经数据失败: {e}")
        
        return None
    
    def _parse_tencent_data(self, content: str) -> Optional[Dict[str, Any]]:
        """解析腾讯财经数据"""
        try:
            # 腾讯财经数据格式: v_sh000001="1~上证指数~000001~3269.32~3275.93~3269.32~...";
            if '="' in content:
                data_str = content.split('="')[1].rstrip('"')
                fields = data_str.split('~')
                
                if len(fields) >= 4:
                    return {
                        'price': float(fields[3]) if fields[3] != '' else 0,
                        'change': float(fields[4]) if fields[4] != '' else 0,
                        'change_pct': float(fields[5]) if fields[5] != '' else 0,
                        'volume': float(fields[6]) if len(fields) > 6 and fields[6] != '' else 0,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'tencent'
                    }
        except Exception as e:
            self.logger.warning(f"解析腾讯财经数据失败: {e}")
        
        return None
    
    def _parse_eastmoney_data(self, data: dict) -> Optional[Dict[str, Any]]:
        """解析东方财富数据"""
        try:
            stock_data = data.get('data', {})
            
            return {
                'price': stock_data.get('f43', 0) / 100,  # 东方财富价格需要除以100
                'change': stock_data.get('f170', 0) / 100,
                'change_pct': stock_data.get('f171', 0),
                'volume': stock_data.get('f47', 0),
                'timestamp': datetime.now().isoformat(),
                'source': 'eastmoney'
            }
        except Exception as e:
            self.logger.warning(f"解析东方财富数据失败: {e}")
        
        return None
    
    def _parse_xueqiu_data(self, data: dict) -> Optional[Dict[str, Any]]:
        """解析雪球数据"""
        try:
            stock_data = data.get('data', {}).get('quote', {})
            
            return {
                'price': stock_data.get('current', 0),
                'change': stock_data.get('chg', 0),
                'change_pct': stock_data.get('percent', 0),
                'volume': stock_data.get('volume', 0),
                'timestamp': datetime.now().isoformat(),
                'source': 'xueqiu'
            }
        except Exception as e:
            self.logger.warning(f"解析雪球数据失败: {e}")
        
        return None
    
    def _validate_realtime_data(self, data: Dict[str, Any]) -> bool:
        """验证实时行情数据有效性"""
        required_fields = ['price', 'change_pct']
        
        for field in required_fields:
            if field not in data or data[field] is None:
                return False
        
        # 检查价格是否合理
        if data['price'] <= 0 or data['price'] > 100000:  # 假设股票价格在0-100000之间
            return False
        
        # 检查涨跌幅是否合理
        if abs(data['change_pct']) > 100:  # 假设单日涨跌幅不超过100%
            return False
        
        return True
    
    def get_batch_realtime_quotes(self, symbols: list) -> Dict[str, Optional[Dict[str, Any]]]:
        """批量获取实时行情数据（优先使用新浪财经批量接口）"""
        results = {}
        
        # 首先尝试使用新浪财经批量接口
        sina_batch_results = self._get_sina_batch_realtime_quotes(symbols)
        if sina_batch_results:
            return sina_batch_results
        
        # 如果批量接口失败，则回退到逐个获取
        for symbol in symbols:
            results[symbol] = self.get_realtime_quote(symbol)
        
        return results
    
    def _get_sina_batch_realtime_quotes(self, symbols: list) -> Optional[Dict[str, Optional[Dict[str, Any]]]]:
        """使用新浪财经批量接口获取实时行情"""
        try:
            # 构建股票代码列表
            codes = []
            for symbol in symbols:
                if symbol.endswith('.SZ'):
                    code = f'sz{symbol.replace(".SZ", "")}'
                elif symbol.endswith('.SS'):
                    code = f'sh{symbol.replace(".SS", "")}'
                else:
                    continue
                codes.append(code)
            
            if not codes:
                return None
            
            # 新浪财经批量接口，最多支持50个股票
            batch_size = 50
            all_results = {}
            
            for i in range(0, len(codes), batch_size):
                batch_codes = codes[i:i + batch_size]
                code_list = ','.join(batch_codes)
                
                url = f"http://hq.sinajs.cn/list={code_list}"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': 'http://finance.sina.com.cn'
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    content = response.text
                    # 解析批量数据
                    batch_results = self._parse_sina_batch_data(content, batch_codes, symbols[i:i + batch_size])
                    all_results.update(batch_results)
                else:
                    # 如果批量请求失败，返回None让调用方回退到逐个获取
                    return None
            
            return all_results
            
        except Exception as e:
            self.logger.warning(f"新浪财经批量实时行情获取失败: {e}")
            return None
    
    def _parse_sina_batch_data(self, content: str, codes: list, symbols: list) -> Dict[str, Optional[Dict[str, Any]]]:
        """解析新浪财经批量数据"""
        results = {}
        
        try:
            # 新浪财经批量数据格式：每行一个股票数据
            lines = content.strip().split(';')
            
            for i, line in enumerate(lines):
                if i >= len(symbols):
                    break
                    
                symbol = symbols[i]
                
                if '"' in line:
                    data_str = line.split('"')[1]
                    fields = data_str.split(',')
                    
                    if len(fields) >= 3:
                        results[symbol] = {
                            'price': float(fields[3]) if fields[3] != '' else 0,
                            'change': float(fields[2]) - float(fields[1]) if fields[1] != '' and fields[2] != '' else 0,
                            'change_pct': ((float(fields[3]) - float(fields[2])) / float(fields[2]) * 100) if fields[2] != '' and float(fields[2]) != 0 else 0,
                            'volume': float(fields[8]) if len(fields) > 8 and fields[8] != '' else 0,
                            'timestamp': datetime.now().isoformat(),
                            'source': 'sina_batch'
                        }
                    else:
                        results[symbol] = None
                else:
                    results[symbol] = None
            
            # 确保所有symbol都有结果
            for symbol in symbols:
                if symbol not in results:
                    results[symbol] = None
                    
        except Exception as e:
            self.logger.warning(f"解析新浪财经批量数据失败: {e}")
            # 如果解析失败，为所有symbol返回None
            for symbol in symbols:
                results[symbol] = None
        
        return results
    
    def set_data_source_priority(self, sources: list):
        """设置数据源优先级"""
        available_sources = {s['name']: s for s in self.data_sources}
        
        new_order = []
        for source_name in sources:
            if source_name in available_sources:
                new_order.append(available_sources[source_name])
        
        # 添加未指定的数据源
        for source in self.data_sources:
            if source not in new_order:
                new_order.append(source)
        
        self.data_sources = new_order
        self.logger.info(f"数据源优先级已更新: {[s['name'] for s in self.data_sources]}")
    


# 全局实例
realtime_provider = EnhancedRealtimeProvider()