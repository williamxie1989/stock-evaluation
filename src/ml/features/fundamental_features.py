# -*- coding: utf-8 -*-
"""
基本面特征生成器
提供估值、盈利、成长、财务质量4大类特征

数据源: AKShare (免费)
- stock_financial_abstract: 财务摘要（多季度）
- stock_financial_report_sina: 财务报表详细数据
- stock_individual_info_em: 实时市值、PE、PB
- stock_financial_abstract_ths: 同花顺财务指标

预期效果: R² +0.20~0.30
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging
import akshare as ak
from functools import lru_cache

try:
    from src.data.db.symbol_standardizer import get_symbol_standardizer
except Exception:  # pragma: no cover
    get_symbol_standardizer = None

logger = logging.getLogger(__name__)


class FundamentalDataCache:
    """基本面数据缓存（避免重复请求）"""
    
    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 3600  # 缓存1小时
    
    def get(self, key: str):
        if key in self._cache:
            if (datetime.now() - self._cache_time[key]).seconds < self._cache_ttl:
                return self._cache[key]
        return None
    
    def set(self, key: str, value):
        self._cache[key] = value
        self._cache_time[key] = datetime.now()


class FundamentalFeatureGenerator:
    """
    基本面特征生成器
    
    提供4大类特征:
    1. 估值特征 (12个)
    2. 盈利特征 (15个)
    3. 成长特征 (10个)
    4. 财务质量特征 (12个)
    
    V2优化: 优先从数据库读取，大幅减少API调用
    """
    
    def __init__(self, cache_enabled: bool = True, db_manager=None, use_db_cache: bool = True, publish_delay_days: int = 60):
        """
        Parameters
        ----------
        cache_enabled : bool
            是否启用内存缓存
        db_manager : UnifiedDatabaseManager, optional
            数据库管理器（如提供，将优先从数据库读取）
        use_db_cache : bool
            是否使用数据库缓存（优先级高于API调用）
        publish_delay_days : int
            财报公告延迟天数（当数据源缺publish_date时使用）
        """
        self.cache_enabled = cache_enabled
        self.cache = FundamentalDataCache() if cache_enabled else None
        self.db_manager = db_manager
        self.use_db_cache = use_db_cache
        self.publish_delay_days = publish_delay_days  # 🔧 新增参数
        
        # 初始化数据库持久化管理器
        if db_manager and use_db_cache:
            from src.data.db.fundamental_data_manager import FundamentalDataManager
            self.data_manager = FundamentalDataManager(db_manager)
            logger.info(f"基本面特征生成器初始化完成（启用数据库缓存，财报延迟={publish_delay_days}天）")
        else:
            self.data_manager = None
            logger.info(f"基本面特征生成器初始化完成（无数据库缓存，财报延迟={publish_delay_days}天）")
        
        if get_symbol_standardizer:
            try:
                self.symbol_standardizer = get_symbol_standardizer()
            except Exception:
                self.symbol_standardizer = None
        else:
            self.symbol_standardizer = None
    
    def generate_features(
        self, 
        symbol: str, 
        date: pd.Timestamp,
        lookback_quarters: int = 4
    ) -> Dict[str, float]:
        """
        生成某只股票在某个日期的所有基本面特征
        
        Parameters
        ----------
        symbol : str
            股票代码（不带后缀）
        date : pd.Timestamp
            日期
        lookback_quarters : int
            回溯季度数
            
        Returns
        -------
        features : dict
            特征字典
        """
        features = {}
        
        try:
            # 1. 估值特征
            valuation_features = self._build_valuation_features(symbol, date)
            features.update(valuation_features)
            
            # 2. 盈利特征
            profitability_features = self._build_profitability_features(
                symbol, date, lookback_quarters
            )
            features.update(profitability_features)
            
            # 3. 成长特征
            growth_features = self._build_growth_features(
                symbol, date, lookback_quarters
            )
            features.update(growth_features)
            
            # 4. 财务质量特征
            quality_features = self._build_financial_quality_features(
                symbol, date, lookback_quarters
            )
            features.update(quality_features)
            
            # 🔧 数值验证：过滤inf/nan/complex
            features = self._validate_features(features)
            
            logger.debug(f"生成 {symbol} 基本面特征 {len(features)} 个")
            
        except Exception as e:
            logger.warning(f"生成 {symbol} 基本面特征失败: {e}")
        
        return features
    
    def _standardize_symbol(self, symbol: str) -> str:
        if self.symbol_standardizer is not None:
            try:
                return self.symbol_standardizer.standardize_symbol(symbol)
            except Exception:
                return symbol
        return symbol
    
    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [col for col in df.columns if col not in {'symbol'}]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df
    
    def build_daily_dataframe(self, symbol: str, dates: pd.Series) -> pd.DataFrame:
        """
        将季度/日度基本面数据映射到交易日日频
        """
        if self.data_manager is None:
            return pd.DataFrame(columns=['symbol', 'date'])

        # 防御性处理：dates 可能为标量或非可迭代对象，确保转换为可迭代的日期序列
        if dates is None:
            return pd.DataFrame(columns=['symbol', 'date'])

        # 如果传入单个 Timestamp / datetime / str，则包装为列表
        from datetime import datetime as _dt
        try:
            if isinstance(dates, (_dt, pd.Timestamp, str)) or not hasattr(dates, '__iter__'):
                dates = [dates]
        except Exception:
            # 保守回退：若检测失败，尝试直接将其封装为列表
            dates = [dates]

        try:
            date_index = pd.DatetimeIndex(pd.to_datetime(dates)).dropna().sort_values().unique()
        except Exception as e:
            logger.warning(f"{symbol}: build_daily_dataframe 无法解析 dates 参数 - {e}")
            return pd.DataFrame(columns=['symbol', 'date'])
        if len(date_index) == 0:
            return pd.DataFrame(columns=['symbol', 'date'])
        
        start_buffer = (date_index.min() - pd.Timedelta(days=200)).to_pydatetime()
        end_date = date_index.max().to_pydatetime()
        std_symbol = self._standardize_symbol(symbol)
        std_symbol_with_suffix = std_symbol
        if '.' not in std_symbol_with_suffix:
            if std_symbol.startswith('6'):
                std_symbol_with_suffix = f"{std_symbol}.SH"
            else:
                std_symbol_with_suffix = f"{std_symbol}.SZ"
        
        quarterly_df = self.data_manager.get_quarterly_history(std_symbol_with_suffix, start_buffer, end_date)
        daily_valuation_df = self.data_manager.get_daily_valuation_history(std_symbol_with_suffix, start_buffer, end_date)
        if daily_valuation_df.empty and std_symbol_with_suffix != std_symbol:
            daily_valuation_df = self.data_manager.get_daily_valuation_history(std_symbol, start_buffer, end_date)

        if self.data_manager and (quarterly_df.empty or daily_valuation_df.empty):
            try:
                logger.info("%s: 基本面缓存缺失，触发数据刷新 (quarterly_empty=%s, daily_empty=%s)",
                            symbol, quarterly_df.empty, daily_valuation_df.empty)
                refresh_symbol = std_symbol_with_suffix
                self.data_manager.bulk_update_from_akshare(
                    symbols=[refresh_symbol],
                    start_date=start_buffer.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    update_quarterly=quarterly_df.empty,
                    update_daily=daily_valuation_df.empty,
                    max_retries=2,
                    retry_delay=1.0,
                    per_symbol_sleep=0
                )
                if quarterly_df.empty:
                    quarterly_df = self.data_manager.get_quarterly_history(refresh_symbol, start_buffer, end_date)
                if daily_valuation_df.empty:
                    daily_valuation_df = self.data_manager.get_daily_valuation_history(refresh_symbol, start_buffer, end_date)
            except Exception as refresh_exc:
                logger.warning("%s: 基本面数据刷新失败 - %s", symbol, refresh_exc)

        feature_df = pd.DataFrame(index=date_index)
        
        if not daily_valuation_df.empty:
            daily_valuation_df['trade_date'] = pd.to_datetime(daily_valuation_df['trade_date'])
            daily_valuation_df.sort_values('trade_date', inplace=True)
            daily_valuation_df.set_index('trade_date', inplace=True)
            daily_valuation_df = daily_valuation_df.reindex(date_index, method='ffill')
            daily_valuation_df = daily_valuation_df.apply(pd.to_numeric, errors='coerce')
            
            feature_df['pe_ttm'] = daily_valuation_df.get('pe_ttm')
            feature_df['pe_ttm_valid'] = ((feature_df['pe_ttm'] > 0) & (feature_df['pe_ttm'] < 200)).astype(float)
            feature_df['pb'] = daily_valuation_df.get('pb')
            feature_df['pb_valid'] = ((feature_df['pb'] > 0) & (feature_df['pb'] < 20)).astype(float)
        
        if not quarterly_df.empty:
            quarterly_df['report_date'] = pd.to_datetime(quarterly_df['report_date'])
            quarterly_df['publish_date'] = pd.to_datetime(quarterly_df['publish_date'])
            
            # 🔧 关键修复：数据源缺少publish_date时使用保守延迟假设
            # 使用配置的延迟天数（默认60天），避免前视偏差
            missing_publish = quarterly_df['publish_date'].isna()
            if missing_publish.any():
                logger.debug(f"{symbol}: {missing_publish.sum()}条记录缺少publish_date，使用report_date+{self.publish_delay_days}天")
                quarterly_df.loc[missing_publish, 'effective_date'] = (
                    quarterly_df.loc[missing_publish, 'report_date'] + pd.Timedelta(days=self.publish_delay_days)
                )
                quarterly_df.loc[~missing_publish, 'effective_date'] = quarterly_df.loc[~missing_publish, 'publish_date']
            else:
                quarterly_df['effective_date'] = quarterly_df['publish_date']
            
            quarterly_df.sort_values('effective_date', inplace=True)
            quarterly_df.set_index('effective_date', inplace=True)
            quarterly_df = quarterly_df.reindex(date_index, method='ffill')
            quarterly_df = quarterly_df.bfill()
            quarterly_df = quarterly_df.apply(pd.to_numeric, errors='coerce')
            
            feature_df['revenue'] = quarterly_df.get('revenue')
            feature_df['net_profit'] = quarterly_df.get('net_profit')
            feature_df['net_profit_margin'] = quarterly_df.get('net_profit_margin')
            feature_df['roe'] = quarterly_df.get('roe')
            feature_df['eps'] = quarterly_df.get('eps')
            feature_df['revenue_yoy'] = quarterly_df.get('revenue_yoy')
            feature_df['net_profit_yoy'] = quarterly_df.get('net_profit_yoy')
            feature_df['debt_to_asset'] = quarterly_df.get('debt_to_asset')
            feature_df['current_ratio'] = quarterly_df.get('current_ratio')
            feature_df['quick_ratio'] = quarterly_df.get('quick_ratio')
        
        feature_df = self._sanitize_dataframe(feature_df)
        empty_cols = [col for col in feature_df.columns if feature_df[col].isna().all()]
        if empty_cols:
            feature_df.drop(columns=empty_cols, inplace=True)
        feature_df['symbol'] = symbol
        feature_df = feature_df.reset_index().rename(columns={'index': 'date'})

        if len(feature_df.columns) <= 2:
            logger.warning("%s: 基本面特征刷新后仍为空，返回占位列", symbol)
        return feature_df
    
    def _validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """验证并清理特征值，移除无效值"""
        validated = {}
        for key, value in features.items():
            try:
                # 检查是否为数值类型
                if not isinstance(value, (int, float, np.number)):
                    continue
                
                # 转换为float
                val = float(value)
                
                # 检查是否为有效数值（排除nan, inf, complex）
                if np.isnan(val) or np.isinf(val):
                    continue
                
                # 检查是否为实数（排除复数）
                if isinstance(val, complex) or np.iscomplex(val):
                    logger.debug(f"特征 {key} 为复数，已过滤: {val}")
                    continue
                
                validated[key] = val
                
            except (ValueError, TypeError) as e:
                logger.debug(f"特征 {key} 验证失败: {e}")
                continue
        
        return validated
    
    def _build_valuation_features(self, symbol: str, date: pd.Timestamp) -> Dict:
        """
        估值特征 (12个)
        
        包括:
        - PE、PB、PS (静态)
        - PE历史分位数
        - 市值
        """
        features = {}
        
        try:
            # 获取实时估值数据
            info_df = self._get_stock_info(symbol)
            
            if info_df is not None and len(info_df) > 0:
                info_dict = dict(zip(info_df['item'], info_df['value']))
                
                # 市值 (亿元)
                total_mv = info_dict.get('总市值', np.nan)
                if pd.notna(total_mv):
                    features['market_cap'] = float(total_mv) / 1e8
                    features['log_market_cap'] = np.log(features['market_cap'] + 1)
                
                circ_mv = info_dict.get('流通市值', np.nan)
                if pd.notna(circ_mv):
                    features['circ_market_cap'] = float(circ_mv) / 1e8
                
                # PE (市盈率-动态)
                pe = info_dict.get('市盈率-动态', info_dict.get('市盈率', np.nan))
                if pd.notna(pe) and pe != '-':
                    try:
                        features['pe_ttm'] = float(pe)
                        # PE合理性（防止极端值）
                        features['pe_ttm_valid'] = 1 if 0 < features['pe_ttm'] < 200 else 0
                    except:
                        pass
                
                # PB (市净率)
                pb = info_dict.get('市净率', np.nan)
                if pd.notna(pb) and pb != '-':
                    try:
                        features['pb'] = float(pb)
                        features['pb_valid'] = 1 if 0 < features['pb'] < 20 else 0
                    except:
                        pass
            
            # 获取历史财务指标（计算PE历史分位数）
            financial_df = self._get_financial_abstract_ths(symbol)
            
            if financial_df is not None and len(financial_df) > 0:
                # 尝试提取ROE、净利率等
                try:
                    # 报告期作为索引
                    df = financial_df.copy()
                    df['报告期'] = pd.to_datetime(df['报告期'], errors='coerce')
                    df = df.set_index('报告期').sort_index()
                    
                    # 获取最近的数据
                    recent_data = df[df.index <= date].iloc[-1] if len(df[df.index <= date]) > 0 else None
                    
                    if recent_data is not None:
                        # 净资产收益率 (ROE)
                        if '净资产收益率' in recent_data:
                            roe_str = str(recent_data['净资产收益率'])
                            if roe_str and roe_str != 'nan' and roe_str != '--' and roe_str != 'False':
                                try:
                                    roe_val = float(roe_str.replace('%', ''))
                                    features['roe'] = roe_val
                                except:
                                    pass
                        
                        # 销售净利率
                        if '销售净利率' in recent_data:
                            npm_str = str(recent_data['销售净利率'])
                            if npm_str and npm_str != 'nan' and npm_str != '--' and npm_str != 'False':
                                try:
                                    npm_val = float(npm_str.replace('%', ''))
                                    features['net_profit_margin'] = npm_val
                                except:
                                    pass
                
                except Exception as e:
                    logger.debug(f"解析财务指标失败: {e}")
            
        except Exception as e:
            logger.warning(f"估值特征生成失败 {symbol}: {e}")
        
        return features
    
    def _build_profitability_features(
        self, 
        symbol: str, 
        date: pd.Timestamp,
        lookback_quarters: int
    ) -> Dict:
        """
        盈利特征 (15个)
        
        包括:
        - ROE、ROA、净利率、毛利率
        - 盈利增长率（同比）
        - 盈利质量（现金流/净利润）
        """
        features = {}
        
        try:
            # 获取财务摘要数据
            financial_df = self._get_financial_abstract_ths(symbol)
            
            if financial_df is not None and len(financial_df) > 0:
                # 报告期作为索引
                df = financial_df.copy()
                df['报告期'] = pd.to_datetime(df['报告期'], errors='coerce')
                df = df.set_index('报告期').sort_index()
                
                # 获取最近的数据
                recent_dates = df[df.index <= date].tail(lookback_quarters)
                
                if len(recent_dates) > 0:
                    latest = recent_dates.iloc[-1]
                    
                    # ROE (净资产收益率)
                    if '净资产收益率' in latest:
                        roe_str = str(latest['净资产收益率'])
                        if roe_str and roe_str != 'nan' and roe_str != '--':
                            try:
                                features['roe_latest'] = float(roe_str.replace('%', ''))
                            except:
                                pass
                    
                    # ROA (总资产净利率)
                    if '总资产净利率ROA' in latest or '总资产收益率' in latest:
                        roa_col = '总资产净利率ROA' if '总资产净利率ROA' in latest else '总资产收益率'
                        roa_str = str(latest[roa_col])
                        if roa_str and roa_str != 'nan' and roa_str != '--':
                            try:
                                features['roa'] = float(roa_str.replace('%', ''))
                            except:
                                pass
                    
                    # 销售毛利率
                    if '销售毛利率' in latest:
                        gpm_str = str(latest['销售毛利率'])
                        if gpm_str and gpm_str != 'nan' and gpm_str != '--':
                            try:
                                features['gross_profit_margin'] = float(gpm_str.replace('%', ''))
                            except:
                                pass
                    
                    # 净利润
                    if '净利润' in latest:
                        np_str = str(latest['净利润'])
                        if np_str and np_str != 'nan' and np_str != '--':
                            try:
                                # 提取数值（可能有单位"万元"等）
                                np_val = self._parse_financial_value(np_str)
                                if np_val is not None:
                                    features['net_profit'] = np_val
                            except:
                                pass
                    
                    # 营业总收入
                    if '营业总收入' in latest:
                        rev_str = str(latest['营业总收入'])
                        if rev_str and rev_str != 'nan' and rev_str != '--':
                            try:
                                rev_val = self._parse_financial_value(rev_str)
                                if rev_val is not None:
                                    features['total_revenue'] = rev_val
                            except:
                                pass
                    
                    # 计算盈利增长率（同比）
                    if len(recent_dates) >= 5:  # 至少需要5个季度（同比）
                        # ROE同比增长
                        if '净资产收益率' in latest:
                            try:
                                roe_now = self._parse_percentage(str(latest['净资产收益率']))
                                roe_yoy = self._parse_percentage(str(recent_dates.iloc[-5]['净资产收益率']))
                                if roe_now is not None and roe_yoy is not None and roe_yoy != 0:
                                    features['roe_yoy_growth'] = (roe_now - roe_yoy) / abs(roe_yoy) * 100
                            except:
                                pass
                        
                        # 净利润同比增长率
                        if '净利润同比增长率' in latest:
                            npg_str = str(latest['净利润同比增长率'])
                            if npg_str and npg_str != 'nan' and npg_str != 'False' and npg_str != '--':
                                try:
                                    features['net_profit_yoy_growth'] = float(npg_str.replace('%', ''))
                                except:
                                    pass
        
        except Exception as e:
            logger.warning(f"盈利特征生成失败 {symbol}: {e}")
        
        return features
    
    def _build_growth_features(
        self,
        symbol: str,
        date: pd.Timestamp,
        lookback_quarters: int
    ) -> Dict:
        """
        成长特征 (10个)
        
        包括:
        - 营收增长率、净利润增长率
        - EPS增长率
        - CAGR (复合年增长率)
        """
        features = {}
        
        try:
            financial_df = self._get_financial_abstract_ths(symbol)
            
            if financial_df is not None and len(financial_df) > 0:
                df = financial_df.copy()
                df['报告期'] = pd.to_datetime(df['报告期'], errors='coerce')
                df = df.set_index('报告期').sort_index()
                
                recent_dates = df[df.index <= date].tail(lookback_quarters * 3)  # 3年数据
                
                if len(recent_dates) > 0:
                    latest = recent_dates.iloc[-1]
                    
                    # 营业总收入同比增长率
                    if '营业总收入同比增长率' in latest:
                        rg_str = str(latest['营业总收入同比增长率'])
                        if rg_str and rg_str != 'nan' and rg_str != 'False' and rg_str != '--':
                            try:
                                features['revenue_yoy_growth'] = float(rg_str.replace('%', ''))
                            except:
                                pass
                    
                    # 净利润同比增长率 (如果盈利特征没有)
                    if 'net_profit_yoy_growth' not in features and '净利润同比增长率' in latest:
                        npg_str = str(latest['净利润同比增长率'])
                        if npg_str and npg_str != 'nan' and npg_str != 'False' and npg_str != '--':
                            try:
                                features['profit_yoy_growth'] = float(npg_str.replace('%', ''))
                            except:
                                pass
                    
                    # 扣非净利润同比增长率
                    if '扣非净利润同比增长率' in latest:
                        dnpg_str = str(latest['扣非净利润同比增长率'])
                        if dnpg_str and dnpg_str != 'nan' and dnpg_str != 'False' and dnpg_str != '--':
                            try:
                                features['deducted_profit_yoy_growth'] = float(dnpg_str.replace('%', ''))
                            except:
                                pass
                    
                    # 计算CAGR（如果有3年数据）
                    if len(recent_dates) >= 12:
                        try:
                            # 营收CAGR - 修复：避免负数开方产生复数
                            if '营业总收入' in latest:
                                rev_now = self._parse_financial_value(str(latest['营业总收入']))
                                rev_3y = self._parse_financial_value(str(recent_dates.iloc[-12]['营业总收入']))
                                if rev_now and rev_3y and rev_3y != 0:
                                    ratio = rev_now / rev_3y
                                    # 只有当ratio > 0时才计算CAGR（避免负数开方）
                                    if ratio > 0:
                                        features['revenue_cagr_3y'] = (pow(ratio, 1/3) - 1) * 100
                                    else:
                                        # 负增长用简单年化
                                        features['revenue_cagr_3y'] = (ratio - 1) * 100 / 3
                            
                            # 净利润CAGR - 修复：避免负数开方产生复数
                            if '净利润' in latest:
                                np_now = self._parse_financial_value(str(latest['净利润']))
                                np_3y = self._parse_financial_value(str(recent_dates.iloc[-12]['净利润']))
                                if np_now and np_3y and np_3y != 0:
                                    ratio = np_now / np_3y
                                    # 只有当ratio > 0时才计算CAGR（避免负数开方）
                                    if ratio > 0:
                                        features['profit_cagr_3y'] = (pow(ratio, 1/3) - 1) * 100
                                    else:
                                        # 负增长用简单年化
                                        features['profit_cagr_3y'] = (ratio - 1) * 100 / 3
                        except Exception as e:
                            logger.debug(f"CAGR计算失败: {e}")
                            pass
        
        except Exception as e:
            logger.warning(f"成长特征生成失败 {symbol}: {e}")
        
        return features
    
    def _build_financial_quality_features(
        self,
        symbol: str,
        date: pd.Timestamp,
        lookback_quarters: int
    ) -> Dict:
        """
        财务质量特征 (12个)
        
        包括:
        - 资产负债率、流动比率、速动比率
        - 应收账款周转天数
        - 经营现金流
        """
        features = {}
        
        try:
            financial_df = self._get_financial_abstract_ths(symbol)
            
            if financial_df is not None and len(financial_df) > 0:
                df = financial_df.copy()
                df['报告期'] = pd.to_datetime(df['报告期'], errors='coerce')
                df = df.set_index('报告期').sort_index()
                
                recent_dates = df[df.index <= date].tail(lookback_quarters)
                
                if len(recent_dates) > 0:
                    latest = recent_dates.iloc[-1]
                    
                    # 资产负债率
                    if '资产负债率' in latest:
                        dar_str = str(latest['资产负债率'])
                        if dar_str and dar_str != 'nan' and dar_str != '--':
                            try:
                                features['debt_to_asset_ratio'] = float(dar_str.replace('%', ''))
                            except:
                                pass
                    
                    # 流动比率
                    if '流动比率' in latest:
                        cr_str = str(latest['流动比率'])
                        if cr_str and cr_str != 'nan' and cr_str != '--':
                            try:
                                features['current_ratio'] = float(cr_str)
                            except:
                                pass
                    
                    # 速动比率
                    if '速动比率' in latest:
                        qr_str = str(latest['速动比率'])
                        if qr_str and qr_str != 'nan' and qr_str != '--':
                            try:
                                features['quick_ratio'] = float(qr_str)
                            except:
                                pass
                    
                    # 应收账款周转天数
                    if '应收账款周转天数' in latest:
                        art_str = str(latest['应收账款周转天数'])
                        if art_str and art_str != 'nan' and art_str != '--':
                            try:
                                features['receivable_turnover_days'] = float(art_str)
                            except:
                                pass
                    
                    # 每股经营现金流
                    if '每股经营现金流' in latest:
                        ocf_str = str(latest['每股经营现金流'])
                        if ocf_str and ocf_str != 'nan' and ocf_str != '--':
                            try:
                                features['operating_cashflow_per_share'] = float(ocf_str)
                            except:
                                pass
                    
                    # 每股净资产
                    if '每股净资产' in latest:
                        bvps_str = str(latest['每股净资产'])
                        if bvps_str and bvps_str != 'nan' and bvps_str != '--':
                            try:
                                features['book_value_per_share'] = float(bvps_str)
                            except:
                                pass
        
        except Exception as e:
            logger.warning(f"财务质量特征生成失败 {symbol}: {e}")
        
        return features
    
    # ========== 辅助方法 ==========
    
    def _get_stock_info(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取个股基本信息"""
        cache_key = f"info_{symbol}"
        
        if self.cache_enabled:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
            if self.cache_enabled:
                self.cache.set(cache_key, df)
            return df
        except Exception as e:
            logger.debug(f"获取 {symbol} 个股信息失败: {e}")
            return None
    
    def _get_financial_abstract_ths(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取同花顺财务指标"""
        cache_key = f"ths_{symbol}"
        
        if self.cache_enabled:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            df = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期")
            if self.cache_enabled:
                self.cache.set(cache_key, df)
            return df
        except Exception as e:
            logger.debug(f"获取 {symbol} 同花顺财务指标失败: {e}")
            return None
    
    @staticmethod
    def _parse_percentage(s: str) -> Optional[float]:
        """解析百分比字符串"""
        if not s or s == 'nan' or s == '--' or s == 'False':
            return None
        try:
            return float(s.replace('%', ''))
        except:
            return None
    
    @staticmethod
    def _parse_financial_value(s: str) -> Optional[float]:
        """
        解析财务数值（可能包含单位）
        例如: "1234.56万元" -> 12345600
        """
        if not s or s == 'nan' or s == '--' or s == 'False':
            return None
        
        try:
            # 移除中文字符和空格
            s_clean = s.replace('元', '').replace(' ', '').replace(',', '')
            
            # 处理单位
            multiplier = 1
            if '亿' in s:
                multiplier = 1e8
                s_clean = s_clean.replace('亿', '')
            elif '万' in s:
                multiplier = 1e4
                s_clean = s_clean.replace('万', '')
            
            value = float(s_clean) * multiplier
            return value
        except:
            return None


# ========== 便捷函数 ==========

_global_generator = None

def get_fundamental_features(
    symbol: str,
    date: pd.Timestamp,
    lookback_quarters: int = 4
) -> Dict[str, float]:
    """
    便捷函数：获取基本面特征
    
    使用全局单例生成器（带缓存）
    """
    global _global_generator
    
    if _global_generator is None:
        _global_generator = FundamentalFeatureGenerator(cache_enabled=True)
    
    return _global_generator.generate_features(symbol, date, lookback_quarters)
