import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import re


class AkshareDataProvider:
    def __init__(self):
        pass
    
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

    def get_stock_data(self, stock_symbol, period="1y"):
        """使用akshare获取股票数据"""
        try:
            # 转换股票代码格式
            if stock_symbol.endswith('.SS'):
                ak_symbol = stock_symbol.replace('.SS', '')
            elif stock_symbol.endswith('.SZ'):
                ak_symbol = stock_symbol.replace('.SZ', '')
            else:
                # 尝试通过股票名称查找股票代码
                ak_symbol = self._get_stock_code_by_name(stock_symbol)
                if ak_symbol is None:
                    print(f"无法找到股票 {stock_symbol} 的代码")
                    return None
            
            # 获取历史行情数据
            stock_df = ak.stock_zh_a_hist(symbol=ak_symbol, period="daily", adjust="qfq")
            
            # 检查数据是否为空
            if stock_df is None or stock_df.empty:
                print(f"akshare获取股票数据为空: {stock_symbol}")
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
                market_df = ak.stock_zh_a_hist(symbol="000001", period="daily", adjust="qfq")
                
                # 检查数据是否为空
                if market_df is None or market_df.empty:
                    print("akshare获取大盘数据为空")
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
                            print(f"akshare返回的大盘数据缺少必要列 {col}")
                            print(f"实际列名: {list(market_df.columns)}")
                            market_df = None
                            break
                    
                    if market_df is not None:
                        # 设置日期为索引
                        market_df['Date'] = pd.to_datetime(market_df['Date'])
                        market_df.set_index('Date', inplace=True)
            except Exception as e:
                print(f"获取大盘数据失败: {e}")
                market_df = None
            
            # 获取股票行业和市值信息
            stock_info = self._get_stock_industry_and_market_cap(ak_symbol, stock_symbol)
            
            return {
                'stock_data': stock_df,
                'market_data': market_df,
                'stock_info': stock_info
            }
            
        except Exception as e:
            print(f"akshare获取数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None