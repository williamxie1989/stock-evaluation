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
    
    def is_shanghai_index_symbol(self, code: str) -> bool:
        """
        判断是否为上证所指数代码
        上证所指数代码特征：000xxx.XSHG 或 000xxx（无后缀）
        """
        # 首先检查是否为指数后缀
        if code.endswith('.XSHG'):
            code_prefix = code.replace('.XSHG', '')
            return code_prefix.startswith('000')
        
        # 检查是否为深市股票（带.SZ后缀）
        if code.endswith('.SZ'):
            return False
        
        # 提取纯数字代码
        code_prefix = code.split('.')[0] if '.' in code else code
        
        # 如果不是000开头，直接返回False
        if not code_prefix.startswith('000'):
            return False
        
        # 上证所指数代码列表（000开头）
        shanghai_index_codes = {
            '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008',
            '000009', '000010', '000011', '000012', '000013', '000015', '000016', '000017',
            '000018', '000019', '000020', '000021', '000022', '000023', '000025', '000026',
            '000027', '000028', '000029', '000030', '000031', '000032', '000033', '000034',
            '000035', '000036', '000037', '000038', '000039', '000040', '000041', '000042',
            '000043', '000044', '000045', '000046', '000047', '000048', '000049', '000050',
            '000051', '000052', '000053', '000054', '000055', '000056', '000057', '000058',
            '000059', '000060', '000061', '000062', '000063', '000064', '000065', '000066',
            '000067', '000068', '000069', '000070', '000071', '000072', '000073', '000074',
            '000075', '000076', '000077', '000078', '000079', '000090', '000091', '000092',
            '000093', '000094', '000095', '000096', '000097', '000098', '000099', '000100',
            '000101', '000102', '000103', '000104', '000105', '000106', '000107', '000108',
            '000109', '000110', '000111', '000112', '000113', '000114', '000115', '000116',
            '000117', '000118', '000119', '000120', '000121', '000122', '000123', '000124',
            '000125', '000126', '000128', '000129', '000130', '000131', '000132', '000133',
            '000134', '000135', '000136', '000137', '000138', '000139', '000140', '000141',
            '000142', '000145', '000146', '000147', '000148', '000149', '000150', '000151',
            '000152', '000153', '000155', '000158', '000159', '000160', '000161', '000162',
            '000170', '000171', '000188', '000300', '000500', '000501', '000510', '000682',
            '000683', '000685', '000687', '000688', '000689', '000690', '000692', '000693',
            '000695', '000697', '000698', '000801', '000802', '000803', '000804', '000805',
            '000806', '000807', '000808', '000809', '000810', '000811', '000812', '000813',
            '000814', '000815', '000816', '000817', '000818', '000819', '000820', '000821',
            '000822', '000823', '000824', '000825', '000826', '000827', '000828', '000829',
            '000830', '000831', '000832', '000833', '000834', '000835', '000836', '000837',
            '000838', '000839', '000840', '000841', '000842', '000843', '000844', '000845',
            '000846', '000847', '000849', '000850', '000851', '000852', '000853', '000854',
            '000855', '000856', '000857', '000858', '000905', '000906', '000907', '000908',
            '000909', '000910', '000911', '000912', '000913', '000914', '000915', '000916',
            '000917', '000918', '000919', '000920', '000921', '000922', '000923', '000925',
            '000926', '000927', '000928', '000929', '000930', '000931', '000932', '000933',
            '000934', '000935', '000936', '000937', '000938', '000939', '000940', '000941',
            '000942', '000943', '000944', '000945', '000946', '000947', '000948', '000949',
            '000950', '000951', '000952', '000953', '000954', '000955', '000956', '000957',
            '000958', '000959', '000960', '000961', '000962', '000963', '000964', '000965',
            '000966', '000967', '000968', '000969', '000970', '000971', '000972', '000973',
            '000974', '000975', '000976', '000977', '000978', '000979', '000980', '000981',
            '000982', '000983', '000984', '000985', '000986', '000987', '000988', '000989',
            '000990', '000991', '000992', '000993', '000994', '000995', '000996', '000997',
            '000998', '000999'
        }
        
        return code_prefix in shanghai_index_codes
    
    def filter_out_shanghai_indices(self, stock_list):
        """
        从股票列表中过滤掉上证所指数
        """
        return [symbol for symbol in stock_list if not self.is_shanghai_index_symbol(symbol)]
    
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
            
            # 4. 获取北京证券交易所股票（已移除，不再获取BJ股票）
            self.logger.info("跳过北京证券交易所股票列表（已移除BJ股票）...")
            pass
            
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