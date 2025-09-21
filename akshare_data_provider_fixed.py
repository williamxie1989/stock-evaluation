import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import re
import time
import logging
import os
import random


class AkshareDataProvider:
    def __init__(self):
        # 从环境变量读取重试参数，提供安全的默认值
        try:
            self.max_retries = max(1, int(os.getenv("AK_MAX_RETRIES", "3")))
        except Exception:
            self.max_retries = 3
        try:
            self.retry_delay = max(0.1, float(os.getenv("AK_RETRY_DELAY", "2.0")))
        except Exception:
            self.retry_delay = 2.0
        self.logger = logging.getLogger(__name__)
        try:
            self.logger.info(f"AkshareDataProvider retries configured: max_retries={self.max_retries}, retry_delay={self.retry_delay}s")
        except Exception:
            pass
    
    def get_all_stock_list(self):
        """获取全市场股票列表，包含板块分类信息"""
        all_stocks = []
        
        try:
            from stock_status_filter import StockStatusFilter
            stock_filter = StockStatusFilter()
            
            # 1. 获取沪深A股所有股票
            self.logger.info("正在获取沪深A股股票列表...")
            a_stocks = self._retry_with_backoff(ak.stock_info_a_code_name)
            if a_stocks is not None and not a_stocks.empty:
                for _, row in a_stocks.iterrows():
                    code = row['code']
                    name = row['name']
                    
                    # 过滤无效股票
                    filter_check = stock_filter.should_filter_stock(name, code, exclude_star_market=True)
                    if filter_check['should_filter']:
                        self.logger.debug(f"跳过无效股票: {code} - {name} ({filter_check['reason']})")
                        continue
                    
                    # 根据代码判断板块
                    market_type = self._classify_stock_market(code)
                    
                    all_stocks.append({
                        'symbol': code,
                        'name': name,
                        'market': market_type['market'],
                        'board_type': market_type['board_type'],
                        'exchange': market_type['exchange'],
                        'ah_pair': None
                    })
            
            # 2. 获取上海证券交易所股票（包含科创板）
            self.logger.info("正在获取上海证券交易所股票列表...")
            try:
                # 主板A股
                sh_main = self._retry_with_backoff(ak.stock_info_sh_name_code, symbol="主板A股")
                if sh_main is not None and not sh_main.empty:
                    for _, row in sh_main.iterrows():
                        code = row['证券代码']
                        name = row['证券简称']
                        
                        # 过滤无效股票
                        filter_check = stock_filter.should_filter_stock(name, code, exclude_star_market=True)
                        if filter_check['should_filter']:
                            self.logger.debug(f"跳过无效股票: {code} - {name} ({filter_check['reason']})")
                            continue
                        
                        if not any(s['symbol'] == code for s in all_stocks):
                            all_stocks.append({
                                'symbol': code,
                                'name': name,
                                'market': 'SH',
                                'board_type': '主板',
                                'exchange': '上海证券交易所',
                                'ah_pair': None
                            })
            except Exception as e:
                self.logger.warning(f"获取上海证券交易所股票列表失败: {e}")
            
            # 3. 获取深圳证券交易所股票（包含创业板）
            self.logger.info("正在获取深圳证券交易所股票列表...")
            try:
                # A股列表（包含主板和创业板）
                sz_stocks = self._retry_with_backoff(ak.stock_info_sz_name_code, symbol="A股列表")
                if sz_stocks is not None and not sz_stocks.empty:
                    # 兼容不同版本列名
                    cols = list(sz_stocks.columns)
                    code_col = '证券代码' if '证券代码' in cols else ('A股代码' if 'A股代码' in cols else None)
                    name_col = '证券简称' if '证券简称' in cols else ('A股简称' if 'A股简称' in cols else None)
                    if code_col is None or name_col is None:
                        raise KeyError(f"意外的深交所列名: {cols}")
                    for _, row in sz_stocks.iterrows():
                        code = row[code_col]
                        name = row[name_col]
                        
                        # 过滤无效股票
                        filter_check = stock_filter.should_filter_stock(name, code, exclude_star_market=True)
                        if filter_check['should_filter']:
                            self.logger.debug(f"跳过无效股票: {code} - {name} ({filter_check['reason']})")
                            continue
                        
                        if not any(s['symbol'] == code for s in all_stocks):
                            market_type = self._classify_stock_market(code)
                            all_stocks.append({
                                'symbol': code,
                                'name': name,
                                'market': 'SZ',
                                'board_type': market_type['board_type'],
                                'exchange': '深圳证券交易所',
                                'ah_pair': None
                            })
            except Exception as e:
                self.logger.warning(f"获取深圳证券交易所股票列表失败: {e}")
            
            # 4. 获取北京证券交易所股票 - BJ股票已移除，跳过处理
            self.logger.info("北京证券交易所股票已移除，跳过获取...")
            pass
            
            # 5. 获取A+H股列表并整合
            self.logger.info("正在获取A+H股列表...")
            try:
                ah_stocks = self.get_ah_stock_list()
                if ah_stocks is not None and not ah_stocks.empty:
                    for _, row in ah_stocks.iterrows():
                        # 处理A股代码
                        if 'code_a' in row and pd.notna(row['code_a']):
                            code_a = str(row['code_a']).zfill(6)
                            ah_pair_value = str(row['code_h']).zfill(5) if 'code_h' in row and pd.notna(row['code_h']) else None
                            
                            # 查找是否已存在该A股记录
                            existing_stock = None
                            for i, s in enumerate(all_stocks):
                                if s['symbol'] == code_a:
                                    existing_stock = i
                                    break
                            
                            if existing_stock is not None:
                                # 更新现有记录的ah_pair和board_type
                                all_stocks[existing_stock]['ah_pair'] = ah_pair_value
                                if '+H' not in all_stocks[existing_stock]['board_type']:
                                    all_stocks[existing_stock]['board_type'] += '+H'
                            else:
                                # 过滤无效股票
                                filter_check = stock_filter.should_filter_stock(row.get('name', ''), code_a)
                                if not filter_check['should_filter']:
                                    # 添加新的A股记录
                                    market_type = self._classify_stock_market(code_a)
                                    all_stocks.append({
                                        'symbol': code_a,
                                        'name': row.get('name', ''),
                                        'market': market_type['market'],
                                        'board_type': market_type['board_type'] + '+H',
                                        'exchange': market_type['exchange'],
                                        'ah_pair': ah_pair_value
                                    })
                                else:
                                    self.logger.debug(f"跳过无效A+H股票: {code_a} - {row.get('name', '')} ({filter_check['reason']})")
                        
                        # 处理H股代码
                        if 'code_h' in row and pd.notna(row['code_h']):
                            code_h = str(row['code_h']).zfill(5)
                            if not any(s['symbol'] == code_h for s in all_stocks):
                                # 过滤无效股票
                                filter_check = stock_filter.should_filter_stock(row.get('name', ''), code_h, exclude_star_market=True)
                                if not filter_check['should_filter']:
                                    all_stocks.append({
                                        'symbol': code_h,
                                        'name': row.get('name', ''),
                                        'market': 'HK',
                                        'board_type': 'H股',
                                        'exchange': '香港证券交易所',
                                        'ah_pair': code_a if 'code_a' in row and pd.notna(row['code_a']) else None
                                    })
                                else:
                                    self.logger.debug(f"跳过无效H股: {code_h} - {row.get('name', '')} ({filter_check['reason']})")
                    self.logger.info(f"成功整合 {len(ah_stocks)} 只A+H股")
            except Exception as e:
                self.logger.warning(f"获取A+H股列表失败: {e}")
            
            self.logger.info(f"成功获取 {len(all_stocks)} 只股票信息")
            return pd.DataFrame(all_stocks)
            
        except ImportError as e:
            self.logger.warning(f"无法导入股票过滤器，将不进行股票过滤: {e}")
            # 如果无法导入过滤器，返回原始逻辑的结果
            return self._get_all_stock_list_without_filter()
        except Exception as e:
            self.logger.error(f"获取全市场股票列表失败: {e}")
            return pd.DataFrame(columns=['symbol', 'name', 'market', 'board_type', 'exchange', 'ah_pair'])
    
    def _get_all_stock_list_without_filter(self):
        """获取全市场股票列表（不进行过滤的原始版本）"""
        try:
            all_stocks = []
            
            # A股列表
            a_stocks = self._retry_with_backoff(ak.stock_info_a_code_name)
            if a_stocks is not None and not a_stocks.empty:
                for _, row in a_stocks.iterrows():
                    code = str(row['code']).zfill(6)
                    name = str(row['name']).strip()
                    market_type = self._classify_stock_market(code)
                    
                    all_stocks.append({
                        'symbol': f"{code}.{market_type['exchange']}",
                        'name': name,
                        'market': market_type['market'],
                        'board_type': market_type['board_type'],
                        'exchange': market_type['exchange']
                    })
            
            self.logger.info(f"成功获取 {len(all_stocks)} 只股票信息")
            return pd.DataFrame(all_stocks)
            
        except Exception as e:
            self.logger.error(f"获取全市场股票列表失败: {e}")
            return pd.DataFrame()
    
    def _classify_stock_market(self, code):
        """根据股票代码分类市场和板块"""
        code = str(code).zfill(6)  # 补齐6位
        
        # 首先检查是否为上证所指数（000开头的指数）
        if self.is_shanghai_index_symbol(code):
            return {'market': 'SH_INDEX', 'board_type': '指数', 'exchange': '上海证券交易所指数'}
        
        # 上海证券交易所（股票）
        if code.startswith('60'):
            return {'market': 'SH', 'board_type': '主板', 'exchange': '上海证券交易所'}
        elif code.startswith('688'):
            return {'market': 'SH', 'board_type': '科创板', 'exchange': '上海证券交易所'}
        elif code.startswith('900'):
            return {'market': 'SH', 'board_type': 'B股', 'exchange': '上海证券交易所'}
        
        # 深圳证券交易所
        elif code.startswith('000'):
            return {'market': 'SZ', 'board_type': '主板', 'exchange': '深圳证券交易所'}
        elif code.startswith('001'):
            return {'market': 'SZ', 'board_type': '主板', 'exchange': '深圳证券交易所'}
        elif code.startswith('002'):
            return {'market': 'SZ', 'board_type': '中小板', 'exchange': '深圳证券交易所'}
        elif code.startswith('003'):
            return {'market': 'SZ', 'board_type': '主板', 'exchange': '深圳证券交易所'}
        elif code.startswith('300'):
            return {'market': 'SZ', 'board_type': '创业板', 'exchange': '深圳证券交易所'}
        elif code.startswith('301'):
            return {'market': 'SZ', 'board_type': '创业板', 'exchange': '深圳证券交易所'}
        elif code.startswith('200'):
            return {'market': 'SZ', 'board_type': 'B股', 'exchange': '深圳证券交易所'}
        
        # 北京证券交易所 - BJ股票已移除
        elif code.startswith('8') or code.startswith('4'):
            return {'market': 'UNKNOWN', 'board_type': '已移除', 'exchange': '北京证券交易所(已移除)'}
        
        # 港股（H股）
        elif len(code) == 5 and code.startswith('0'):
            return {'market': 'HK', 'board_type': 'H股', 'exchange': '香港证券交易所'}
        
        # 默认分类
        else:
            return {'market': 'UNKNOWN', 'board_type': '未知', 'exchange': '未知交易所'}
    
    def _get_stock_code_by_name(self, stock_name):
        """通过股票名称获取股票代码"""
        try:
            # 先尝试在上海股票代码表中查找
            try:
                stock_info_sh = ak.stock_info_sh_name_code()
                # 查找包含股票名称的行
                matched_sh = stock_info_sh[stock_info_sh['证券简称'].str.contains(stock_name, na=False)]
                if not matched_sh.empty:
                    return matched_sh.iloc[0]['证券代码']
            except Exception as e:
                print(f"获取上海股票代码表失败: {e}")
            
            # 再尝试在深圳股票代码表中查找
            try:
                stock_info_sz = ak.stock_info_sz_name_code()
                # 查找包含股票名称的行
                matched_sz = stock_info_sz[stock_info_sz['证券简称'].str.contains(stock_name, na=False)]
                if not matched_sz.empty:
                    return matched_sz.iloc[0]['证券代码']
            except Exception as e:
                print(f"获取深圳股票代码表失败: {e}")
            
            # 最后尝试使用实时行情数据搜索
            try:
                spot_data = ak.stock_zh_a_spot_em()
                # 查找包含股票名称的行
                matched_spot = spot_data[spot_data['名称'].str.contains(stock_name, na=False)]
                if not matched_spot.empty:
                    # 返回第一个匹配结果的代码
                    return matched_spot.iloc[0]['代码']
            except Exception as e:
                print(f"股票搜索失败: {e}")
            
            return None
        except Exception as e:
            print(f"通过股票名称获取代码时出错: {e}")
            return None
    
    def _get_stock_industry_and_market_cap(self, ak_symbol, stock_symbol):
        """获取股票行业和市值信息"""
        stock_info = {
            'symbol': stock_symbol,
            'longName': stock_symbol,
            'industry': 'N/A',
            'marketCap': None
        }
        
        try:
            # 方法1: 使用个股基本信息接口（主要方法）
            try:
                individual_info = ak.stock_individual_info_em(symbol=ak_symbol)
                if individual_info is not None and not individual_info.empty:
                    # 查找行业信息
                    industry_row = individual_info[individual_info['item'] == '行业']
                    if not industry_row.empty:
                        industry = industry_row.iloc[0]['value']
                        stock_info['industry'] = industry if industry and industry != '—' else 'N/A'
                    
                    # 查找市值信息
                    market_cap_row = individual_info[individual_info['item'] == '总市值']
                    if not market_cap_row.empty:
                        market_cap_str = str(market_cap_row.iloc[0]['value'])
                        market_cap_str = market_cap_str.replace(',', '')
                        try:
                            market_cap = float(market_cap_str)
                            stock_info['marketCap'] = market_cap
                        except ValueError:
                            print(f"无法解析市值数据: {market_cap_str}")
                    
                    # 获取股票名称
                    name_row = individual_info[individual_info['item'] == '股票简称']
                    if not name_row.empty:
                        stock_info['longName'] = name_row.iloc[0]['value']
            except Exception as e:
                print(f"获取个股基本信息失败: {e}")
            
            # 方法2: 使用实时行情数据作为备选（主要获取市值和名称）
            try:
                spot_data = ak.stock_zh_a_spot_em()
                if spot_data is not None and not spot_data.empty:
                    # 查找匹配的股票
                    matched_stock = spot_data[spot_data['代码'] == ak_symbol]
                    if not matched_stock.empty:
                        # 获取总市值（转换为亿元）
                        if '总市值' in matched_stock.columns:
                            market_cap_str = str(matched_stock.iloc[0]['总市值'])
                            # 处理市值字符串（可能包含逗号分隔符）
                            market_cap_str = market_cap_str.replace(',', '')
                            try:
                                market_cap = float(market_cap_str)
                                # 如果个股基本信息接口没有获取到市值，使用实时行情数据
                                if stock_info['marketCap'] is None:
                                    stock_info['marketCap'] = market_cap
                            except ValueError:
                                print(f"无法解析市值数据: {market_cap_str}")
                        
                        # 获取股票名称
                        if '名称' in matched_stock.columns:
                            stock_name = matched_stock.iloc[0]['名称']
                            if stock_info['longName'] == stock_symbol:
                                stock_info['longName'] = stock_name
            except Exception as e:
                print(f"获取实时行情数据失败: {e}")
                
        except Exception as e:
            print(f"获取股票行业和市值信息失败: {e}")
        
        return stock_info

    def _retry_with_backoff(self, func, *args, **kwargs):
        """带退避策略的重试机制（带抖动）"""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_err = e
                error_msg = str(e).lower()
                # 检查是否为API限流或网络错误
                if any(keyword in error_msg for keyword in ['rate limit', 'too many requests', 'timeout', 'connection', 'timed out', 'reset', 'proxy']):
                    if attempt < self.max_retries - 1:
                        base_delay = self.retry_delay * (2 ** attempt)  # 指数退避
                        # 抖动系数：0.8x ~ 1.2x，避免雪崩
                        delay = base_delay * (0.8 + 0.4 * random.random())
                        self.logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}, {delay:.2f}秒后重试")
                        time.sleep(delay)
                        continue
                # 非可重试错误或已无剩余重试
                raise e
        # 理论上不会到达这里
        if last_err:
            raise last_err
        return None

    def get_stock_data(self, stock_symbol, period="3y"):
        """使用akshare获取股票数据"""
        try:
            # 转换股票代码格式
            if stock_symbol.endswith('.SS') or stock_symbol.endswith('.SH'):
                ak_symbol = stock_symbol.replace('.SS', '').replace('.SH', '')
            elif stock_symbol.endswith('.SZ'):
                ak_symbol = stock_symbol.replace('.SZ', '')
            else:
                # 尝试通过股票名称查找股票代码
                ak_symbol = self._get_stock_code_by_name(stock_symbol)
                if ak_symbol is None:
                    self.logger.error(f"无法找到股票 {stock_symbol} 的代码")
                    return None
            
            # 获取历史行情数据（使用重试机制）
            stock_df = self._retry_with_backoff(ak.stock_zh_a_hist, symbol=ak_symbol, period="daily", adjust="qfq")
            
            # 检查数据是否为空
            if stock_df is None or stock_df.empty:
                self.logger.warning(f"akshare获取股票数据为空: {stock_symbol}")
                return None
            
            # 转换列名以匹配现有格式
            stock_df.rename(columns={
                '日期': 'Date',
                '开盘': 'Open',
                '最高': 'High',
                '最低': 'Low',
                '收盘': 'Close',
                '成交量': 'Volume'
            }, inplace=True)
            
            # 检查必要的列是否存在
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in stock_df.columns:
                    print(f"akshare返回的数据缺少必要列 {col}")
                    print(f"实际列名: {list(stock_df.columns)}")
                    return None
            
            # 设置日期为索引
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            stock_df.set_index('Date', inplace=True)
            
            # 获取大盘数据（上证指数）
            try:
                market_df = self._retry_with_backoff(ak.stock_zh_a_hist, symbol="000001", period="daily", adjust="qfq")
                
                # 检查数据是否为空
                if market_df is None or market_df.empty:
                    self.logger.warning("akshare获取大盘数据为空")
                    market_df = None
                else:
                    # 转换列名以匹配现有格式
                    market_df.rename(columns={
                        '日期': 'Date',
                        '开盘': 'Open',
                        '最高': 'High',
                        '最低': 'Low',
                        '收盘': 'Close',
                        '成交量': 'Volume'
                    }, inplace=True)
                    
                    # 检查必要的列是否存在
                    for col in required_columns:
                        if col not in market_df.columns:
                            self.logger.error(f"akshare返回的大盘数据缺少必要列 {col}，实际列名: {list(market_df.columns)}")
                            market_df = None
                            break
                    
                    if market_df is not None:
                        # 设置日期为索引
                        market_df['Date'] = pd.to_datetime(market_df['Date'])
                        market_df.set_index('Date', inplace=True)
            except Exception as e:
                self.logger.error(f"获取大盘数据失败: {e}")
                market_df = None
            
            # 获取股票行业和市值信息
            stock_info = self._get_stock_industry_and_market_cap(ak_symbol, stock_symbol)
            
            return {
                'stock_data': stock_df,
                'market_data': market_df,
                'stock_info': stock_info
            }
            
        except Exception as e:
            self.logger.error(f"akshare获取数据失败 {stock_symbol}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    # =========================
    # 以下为 A+H 股相关扩展接口
    # =========================
    def get_ah_stock_list(self):
        """
        获取 A+H 股配对列表，通过股票名称匹配建立A+H股配对关系。
        返回字段标准化为：['name', 'code_a', 'code_h']，额外字段透传。
        """
        try:
            # 使用重试机制获取A+H股数据
            df = self._retry_with_backoff(ak.stock_zh_ah_name)
            
            if df is None or (hasattr(df, 'empty') and df.empty):
                self.logger.warning("A+H股接口返回空数据，尝试备用方案")
                return pd.DataFrame(columns=['name', 'code_a', 'code_h'])
            
            # 复制以免修改外部引用
            df_local = df.copy()
            
            # 处理akshare返回的实际列名：['代码', '名称']
            # 这个接口返回的是H股代码和名称
            if '代码' in df_local.columns and '名称' in df_local.columns:
                df_local = df_local.rename(columns={'代码': 'code_h', '名称': 'name'})
                
                # 通过股票名称匹配A股代码
                df_local['code_a'] = None
                
                # 获取A股列表用于匹配
                try:
                    a_stocks = self._retry_with_backoff(ak.stock_info_a_code_name)
                    if a_stocks is not None and not a_stocks.empty:
                        # 创建名称到A股代码的映射
                        name_to_a_code = {}
                        for _, row in a_stocks.iterrows():
                            name = str(row['name']).strip()
                            code = str(row['code']).strip()
                            name_to_a_code[name] = code
                        
                        # 匹配A股代码
                        for idx, row in df_local.iterrows():
                            h_name = str(row['name']).strip()
                            # 尝试直接匹配
                            if h_name in name_to_a_code:
                                df_local.at[idx, 'code_a'] = name_to_a_code[h_name]
                            else:
                                # 尝试模糊匹配（去除常见后缀）
                                clean_name = h_name.replace('股份', '').replace('有限公司', '').replace('集团', '').strip()
                                for a_name, a_code in name_to_a_code.items():
                                    if clean_name in a_name or a_name in clean_name:
                                        df_local.at[idx, 'code_a'] = a_code
                                        break
                except Exception as e:
                    self.logger.warning(f"获取A股列表用于匹配失败: {e}")
            else:
                # 容错列名映射（保留原有逻辑作为备用）
                col_map_candidates = [
                    {'名称': 'name', 'A股代码': 'code_a', 'H股代码': 'code_h'},
                    {'name': 'name', 'a_code': 'code_a', 'h_code': 'code_h'},
                ]
                for cmap in col_map_candidates:
                    if all(col in df_local.columns for col in cmap.keys()):
                        df_local = df_local.rename(columns=cmap)
                        break
                # 如果仍没有标准列，尝试从可能的列中推断
                if 'name' not in df_local.columns:
                    for cand in ['名称', 'name', '股票名称']:
                        if cand in df_local.columns:
                            df_local = df_local.rename(columns={cand: 'name'})
                            break
                if 'code_a' not in df_local.columns:
                    for cand in ['A股代码', 'a_code', 'A代码', 'A股']:
                        if cand in df_local.columns:
                            df_local = df_local.rename(columns={cand: 'code_a'})
                            break
                if 'code_h' not in df_local.columns:
                    for cand in ['H股代码', 'h_code', 'H代码', 'H股']:
                        if cand in df_local.columns:
                            df_local = df_local.rename(columns={cand: 'code_h'})
                            break
            # 仅保留有至少一侧代码的数据
            df_local = df_local[(df_local.get('code_a').notna() if 'code_a' in df_local.columns else False) |
                                 (df_local.get('code_h').notna() if 'code_h' in df_local.columns else False)]
            return df_local.reset_index(drop=True)
        except Exception as e:
            print(f"处理 A+H 股列表失败: {e}")
            return pd.DataFrame(columns=['name', 'code_a', 'code_h'])

    def get_ah_spot(self):
        """
        获取 A+H 股实时行情，优先使用 ak.stock_zh_ah_spot；
        若失败，则回退到 A 股与港股的实时行情并尝试通过名称或代码合并。
        返回 DataFrame，尽可能包含 ['name', 'code_a', 'code_h', 'price_a', 'price_h', 'pct_chg_a', 'pct_chg_h', 'time']。
        """
        try:
            # 1) 首选 A+H 实时行情
            try:
                df = ak.stock_zh_ah_spot()
                if df is not None and not df.empty:
                    # 不清楚返回列名，进行尽力字段标准化
                    df_local = df.copy()
                    # 可能出现的列名做映射与转换
                    rename_map = {}
                    for col in df_local.columns:
                        if col in ['名称', 'name', '股票名称']:
                            rename_map[col] = 'name'
                        if col in ['A股代码', 'a_code', 'A代码', 'A股']:
                            rename_map[col] = 'code_a'
                        if col in ['H股代码', 'h_code', 'H代码', 'H股']:
                            rename_map[col] = 'code_h'
                        if col in ['A股最新价', 'a_price', 'A最新价']:
                            rename_map[col] = 'price_a'
                        if col in ['H股最新价', 'h_price', 'H最新价']:
                            rename_map[col] = 'price_h'
                        if col in ['A股涨跌幅', 'a_pct_chg', 'A涨跌幅']:
                            rename_map[col] = 'pct_chg_a'
                        if col in ['H股涨跌幅', 'h_pct_chg', 'H涨跌幅']:
                            rename_map[col] = 'pct_chg_h'
                        if re.search(r"时间|time|更新时间", col, re.IGNORECASE):
                            rename_map[col] = 'time'
                    if rename_map:
                        df_local = df_local.rename(columns=rename_map)
                    return df_local
            except Exception as e:
                print(f"获取 A+H 实时行情失败, 回退方案: {e}")
            
            # 2) 回退：分别取 A 股与港股
            df_a, df_h = None, None
            try:
                df_a = ak.stock_zh_a_spot_em()
            except Exception as e:
                print(f"获取 A 股实时行情失败: {e}")
            try:
                df_h = ak.stock_hk_spot()
            except Exception as e:
                print(f"获取港股实时行情失败: {e}")
            if (df_a is None or df_a.empty) and (df_h is None or df_h.empty):
                return pd.DataFrame()
            # 标准化部分列名
            if df_a is not None and not df_a.empty:
                df_a = df_a.rename(columns={'名称': 'name', '代码': 'code_a', '最新价': 'price_a', '涨跌幅': 'pct_chg_a'})
            if df_h is not None and not df_h.empty:
                # 港股接口列名可能不同，尽力统一
                rename_h = {}
                for col in df_h.columns:
                    if col in ['名称', 'name', '股票名称']:
                        rename_h[col] = 'name'
                    if col in ['代码', 'symbol', 'hk_code']:
                        rename_h[col] = 'code_h'
                    if col in ['最新价', '最新', 'price', '当前价']:
                        rename_h[col] = 'price_h'
                    if col in ['涨跌幅', '涨幅', 'pct_chg', '变动%']:
                        rename_h[col] = 'pct_chg_h'
                if rename_h:
                    df_h = df_h.rename(columns=rename_h)
            # 通过名称合并（可能存在歧义，但作为兜底）
            if df_a is not None and df_h is not None and 'name' in df_a.columns and 'name' in df_h.columns:
                merged = pd.merge(df_a[['name', 'code_a', 'price_a', 'pct_chg_a']],
                                  df_h[['name', 'code_h', 'price_h', 'pct_chg_h']],
                                  on='name', how='outer')
                return merged
            # 仅返回一侧数据
            if df_a is not None and not df_a.empty:
                return df_a
            if df_h is not None and not df_h.empty:
                return df_h
            return pd.DataFrame()
        except Exception as e:
            print(f"处理 A+H 实时行情失败: {e}")
            return pd.DataFrame()

    def get_ah_daily(self, symbol: str, market: str = 'A', start_date: str = None, end_date: str = None):
        """
        获取单侧(A 或 H)的日频历史行情，优先尝试 ak.stock_zh_ah_daily；若失败则分别按 A 股或港股历史接口回退。
        - symbol: 代码，如 A 股为 600000/000001 等，港股可能为 00700 等；
        - market: 'A' 或 'H'；
        - start_date / end_date: 可选，形如 '20200101' 或 '2020-01-01'，若 None 则返回全量。
        返回标准化列：['date','open','high','low','close','volume']。
        """
        def _standardize(df_raw: pd.DataFrame) -> pd.DataFrame:
            if df_raw is None or df_raw.empty:
                return pd.DataFrame(columns=['date','open','high','low','close','volume'])
            df_local = df_raw.copy()
            rename_cands = [
                {'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'},
                {'date': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'},
                {'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'},
                {'交易日期': 'date', '开盘价': 'open', '最高价': 'high', '最低价': 'low', '收盘价': 'close', '成交量': 'volume'},
                {'交易日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'},
            ]
            for cmap in rename_cands:
                if all(col in df_local.columns for col in cmap.keys()):
                    df_local = df_local.rename(columns=cmap)
                    break
            # 仅保留需要的列
            cols = [c for c in ['date','open','high','low','close','volume'] if c in df_local.columns]
            df_local = df_local[cols]
            # 日期转为 datetime
            if 'date' in df_local.columns:
                try:
                    df_local['date'] = pd.to_datetime(df_local['date'])
                except Exception:
                    pass
            return df_local.sort_values('date')

        def _apply_date_filter(df_local: pd.DataFrame) -> pd.DataFrame:
            if df_local is None or df_local.empty:
                return df_local
            s = None
            e = None
            try:
                if start_date:
                    s = pd.to_datetime(start_date)
            except Exception:
                pass
            try:
                if end_date:
                    e = pd.to_datetime(end_date)
            except Exception:
                pass
            if s is not None:
                df_local = df_local[df_local['date'] >= s]
            if e is not None:
                df_local = df_local[df_local['date'] <= e]
            return df_local.sort_values('date')

        try:
            # 由于akshare的stock_zh_ah_daily接口存在bug，直接使用分市场接口
            print(f"获取{market}市场股票{symbol}的历史数据")
            # 回退逻辑
            if market.upper() == 'A':
                # A 股历史：使用统一 A 股日线接口（增加重试机制）
                try:
                    df_a = self._retry_with_backoff(ak.stock_zh_a_hist, symbol=symbol, period="daily", adjust="qfq")
                    std = _standardize(df_a)
                    return _apply_date_filter(std)
                except Exception as e:
                    print(f"A 股历史行情获取失败: {e}")
                    import traceback
                    print(f"详细错误信息: {traceback.format_exc()}")
                    return pd.DataFrame(columns=['date','open','high','low','close','volume'])
            else:
                # H 股历史：港股日线（增加重试机制）
                try:
                    df_h = self._retry_with_backoff(ak.stock_hk_daily, symbol=symbol)
                    std = _standardize(df_h)
                    return _apply_date_filter(std)
                except Exception as e:
                    print(f"H 股历史行情获取失败: {e}")
                    return pd.DataFrame(columns=['date','open','high','low','close','volume'])
        except Exception as e:
            print(f"处理 A+H 历史行情失败: {e}")
            return pd.DataFrame(columns=['date','open','high','low','close','volume'])