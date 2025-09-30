"""
应用脚本功能模块 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import json
import os

# 导出FeatureGenerator类以保持向后兼容性
class FeatureGenerator:
    """特征生成器（向后兼容）"""
    
    def __init__(self):
        self.features = {}
        self.config = {}
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成基础特征"""
        if data.empty:
            return pd.DataFrame()
        
        # 基础特征生成逻辑
        features_df = data.copy()
        
        # 添加一些基础技术指标
        if 'close' in features_df.columns:
            # 简单移动平均线
            features_df['ma_5'] = features_df['close'].rolling(window=5).mean()
            features_df['ma_20'] = features_df['close'].rolling(window=20).mean()
            
            # 价格变化率
            features_df['price_change'] = features_df['close'].pct_change()
            
            # 波动率
            features_df['volatility'] = features_df['price_change'].rolling(window=20).std()
        
        return features_df
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return ['ma_5', 'ma_20', 'price_change', 'volatility']

logger = logging.getLogger(__name__)

class AppScriptsFeatures:
    """应用脚本功能模块"""
    
    def __init__(self, config_path: str = "config/app_scripts.json"):
        self.config_path = config_path
        self.feature_configs = {}
        self.script_cache = {}
        
        # 加载配置
        self._load_config()
        logger.info(f"AppScriptsFeatures initialized with config_path: {config_path}")
    
    def _load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.feature_configs = json.load(f)
            else:
                # 默认配置
                self.feature_configs = {
                    "enabled_features": {
                        "data_analysis": True,
                        "signal_generation": True,
                        "backtesting": True,
                        "portfolio_optimization": True,
                        "risk_analysis": True,
                        "market_analysis": True
                    },
                    "feature_settings": {
                        "data_analysis": {
                            "auto_refresh": True,
                            "refresh_interval": 300,  # 5分钟
                            "cache_duration": 3600    # 1小时
                        },
                        "signal_generation": {
                            "real_time": True,
                            "signal_types": ["moving_average", "rsi", "macd", "bollinger_bands"],
                            "confidence_threshold": 0.7
                        },
                        "backtesting": {
                            "default_period": "1y",
                            "commission_rate": 0.001,
                            "slippage": 0.0005
                        }
                    }
                }
                
                # 创建配置目录
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                
                # 保存默认配置
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.feature_configs, f, ensure_ascii=False, indent=2)
                
                logger.info(f"创建默认配置文件: {self.config_path}")
            
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            self.feature_configs = {}
    
    def _save_config(self):
        """保存配置"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_configs, f, ensure_ascii=False, indent=2)
            logger.info(f"配置已保存: {self.config_path}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get_enabled_features(self) -> Dict[str, bool]:
        """获取启用的功能"""
        return self.feature_configs.get("enabled_features", {})
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """检查功能是否启用"""
        enabled_features = self.get_enabled_features()
        return enabled_features.get(feature_name, False)
    
    def enable_feature(self, feature_name: str):
        """启用功能"""
        if "enabled_features" not in self.feature_configs:
            self.feature_configs["enabled_features"] = {}
        
        self.feature_configs["enabled_features"][feature_name] = True
        self._save_config()
        logger.info(f"功能已启用: {feature_name}")
    
    def disable_feature(self, feature_name: str):
        """禁用功能"""
        if "enabled_features" in self.feature_configs:
            self.feature_configs["enabled_features"][feature_name] = False
            self._save_config()
            logger.info(f"功能已禁用: {feature_name}")
    
    def get_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """获取功能配置"""
        feature_settings = self.feature_configs.get("feature_settings", {})
        return feature_settings.get(feature_name, {})
    
    def update_feature_config(self, feature_name: str, config: Dict[str, Any]):
        """更新功能配置"""
        if "feature_settings" not in self.feature_configs:
            self.feature_configs["feature_settings"] = {}
        
        self.feature_configs["feature_settings"][feature_name] = config
        self._save_config()
        logger.info(f"功能配置已更新: {feature_name}")
    
    def run_data_analysis_script(self, data: pd.DataFrame, symbol: str = "Unknown") -> Dict[str, Any]:
        """运行数据分析脚本"""
        try:
            if not self.is_feature_enabled("data_analysis"):
                return {"error": "数据分析功能未启用"}
            
            if data.empty:
                return {"error": "数据为空"}
            
            config = self.get_feature_config("data_analysis")
            
            # 基本统计分析
            analysis_result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "basic_stats": {},
                "price_analysis": {},
                "volume_analysis": {},
                "volatility_analysis": {},
                "trend_analysis": {}
            }
            
            # 基本统计
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                analysis_result["basic_stats"][column] = {
                    "count": int(data[column].count()),
                    "mean": float(data[column].mean()),
                    "std": float(data[column].std()),
                    "min": float(data[column].min()),
                    "max": float(data[column].max()),
                    "median": float(data[column].median())
                }
            
            # 价格分析
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # 价格变化
                price_changes = data['close'].pct_change().dropna()
                analysis_result["price_analysis"] = {
                    "total_return": float((data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100),
                    "daily_return_mean": float(price_changes.mean() * 100),
                    "daily_return_std": float(price_changes.std() * 100),
                    "max_daily_gain": float(price_changes.max() * 100),
                    "max_daily_loss": float(price_changes.min() * 100),
                    "price_range": float(data['high'].max() - data['low'].min()),
                    "avg_daily_range": float((data['high'] - data['low']).mean())
                }
            
            # 成交量分析
            if 'volume' in data.columns:
                volume_changes = data['volume'].pct_change().dropna()
                analysis_result["volume_analysis"] = {
                    "avg_volume": float(data['volume'].mean()),
                    "volume_std": float(data['volume'].std()),
                    "max_volume": float(data['volume'].max()),
                    "min_volume": float(data['volume'].min()),
                    "volume_trend": float(np.polyfit(range(len(data)), data['volume'], 1)[0])
                }
            
            # 波动率分析
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                rolling_volatility = returns.rolling(window=20).std() * np.sqrt(252) * 100
                
                analysis_result["volatility_analysis"] = {
                    "current_volatility": float(rolling_volatility.iloc[-1]) if not rolling_volatility.empty else 0,
                    "avg_volatility": float(rolling_volatility.mean()) if not rolling_volatility.empty else 0,
                    "max_volatility": float(rolling_volatility.max()) if not rolling_volatility.empty else 0,
                    "min_volatility": float(rolling_volatility.min()) if not rolling_volatility.empty else 0
                }
            
            # 趋势分析
            if 'close' in data.columns:
                # 简单移动平均线趋势
                ma_5 = data['close'].rolling(window=5).mean()
                ma_20 = data['close'].rolling(window=20).mean()
                
                current_price = data['close'].iloc[-1]
                current_ma5 = ma_5.iloc[-1] if not ma_5.empty else current_price
                current_ma20 = ma_20.iloc[-1] if not ma_20.empty else current_price
                
                analysis_result["trend_analysis"] = {
                    "current_price": float(current_price),
                    "ma5": float(current_ma5),
                    "ma20": float(current_ma20),
                    "above_ma5": bool(current_price > current_ma5),
                    "above_ma20": bool(current_price > current_ma20),
                    "ma5_above_ma20": bool(current_ma5 > current_ma20),
                    "short_trend": "bullish" if current_price > current_ma5 else "bearish",
                    "long_trend": "bullish" if current_price > current_ma20 else "bearish"
                }
            
            logger.info(f"数据分析脚本运行完成: {symbol}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"数据分析脚本运行失败: {e}")
            return {"error": str(e)}
    
    def run_signal_generation_script(self, data: pd.DataFrame, symbol: str = "Unknown") -> Dict[str, Any]:
        """运行信号生成脚本"""
        try:
            if not self.is_feature_enabled("signal_generation"):
                return {"error": "信号生成功能未启用"}
            
            if data.empty:
                return {"error": "数据为空"}
            
            config = self.get_feature_config("signal_generation")
            signal_types = config.get("signal_types", ["moving_average", "rsi", "macd"])
            confidence_threshold = config.get("confidence_threshold", 0.7)
            
            signals_result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "signals": {},
                "overall_signal": "neutral",
                "confidence": 0.0,
                "signal_strength": 0.0
            }
            
            total_signals = 0
            bullish_signals = 0
            bearish_signals = 0
            
            # 移动平均线信号
            if "moving_average" in signal_types and 'close' in data.columns:
                try:
                    ma_5 = data['close'].rolling(window=5).mean()
                    ma_20 = data['close'].rolling(window=20).mean()
                    
                    if not ma_5.empty and not ma_20.empty:
                        current_price = data['close'].iloc[-1]
                        current_ma5 = ma_5.iloc[-1]
                        current_ma20 = ma_20.iloc[-1]
                        
                        if current_ma5 > current_ma20 and current_price > current_ma5:
                            signals_result["signals"]["moving_average"] = "buy"
                            bullish_signals += 1
                        elif current_ma5 < current_ma20 and current_price < current_ma5:
                            signals_result["signals"]["moving_average"] = "sell"
                            bearish_signals += 1
                        else:
                            signals_result["signals"]["moving_average"] = "hold"
                        
                        total_signals += 1
                        
                except Exception as e:
                    logger.warning(f"移动平均线信号生成失败: {e}")
            
            # RSI信号
            if "rsi" in signal_types and 'close' in data.columns:
                try:
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    
                    if not loss.empty and loss.iloc[-1] != 0:
                        rs = gain.iloc[-1] / loss.iloc[-1]
                        rsi = 100 - (100 / (1 + rs))
                        
                        if rsi < 30:
                            signals_result["signals"]["rsi"] = "buy"
                            bullish_signals += 1
                        elif rsi > 70:
                            signals_result["signals"]["rsi"] = "sell"
                            bearish_signals += 1
                        else:
                            signals_result["signals"]["rsi"] = "hold"
                        
                        total_signals += 1
                        
                except Exception as e:
                    logger.warning(f"RSI信号生成失败: {e}")
            
            # MACD信号
            if "macd" in signal_types and 'close' in data.columns:
                try:
                    ema_12 = data['close'].ewm(span=12).mean()
                    ema_26 = data['close'].ewm(span=26).mean()
                    macd_line = ema_12 - ema_26
                    signal_line = macd_line.ewm(span=9).mean()
                    
                    if not macd_line.empty and not signal_line.empty:
                        current_macd = macd_line.iloc[-1]
                        current_signal = signal_line.iloc[-1]
                        
                        if current_macd > current_signal and current_macd > 0:
                            signals_result["signals"]["macd"] = "buy"
                            bullish_signals += 1
                        elif current_macd < current_signal and current_macd < 0:
                            signals_result["signals"]["macd"] = "sell"
                            bearish_signals += 1
                        else:
                            signals_result["signals"]["macd"] = "hold"
                        
                        total_signals += 1
                        
                except Exception as e:
                    logger.warning(f"MACD信号生成失败: {e}")
            
            # 布林带信号
            if "bollinger_bands" in signal_types and 'close' in data.columns:
                try:
                    middle_band = data['close'].rolling(window=20).mean()
                    std = data['close'].rolling(window=20).std()
                    upper_band = middle_band + (std * 2)
                    lower_band = middle_band - (std * 2)
                    
                    if not upper_band.empty and not lower_band.empty:
                        current_price = data['close'].iloc[-1]
                        current_upper = upper_band.iloc[-1]
                        current_lower = lower_band.iloc[-1]
                        
                        if current_price > current_upper:
                            signals_result["signals"]["bollinger_bands"] = "sell"
                            bearish_signals += 1
                        elif current_price < current_lower:
                            signals_result["signals"]["bollinger_bands"] = "buy"
                            bullish_signals += 1
                        else:
                            signals_result["signals"]["bollinger_bands"] = "hold"
                        
                        total_signals += 1
                        
                except Exception as e:
                    logger.warning(f"布林带信号生成失败: {e}")
            
            # 计算整体信号
            if total_signals > 0:
                confidence = max(bullish_signals, bearish_signals) / total_signals
                signals_result["confidence"] = float(confidence)
                signals_result["signal_strength"] = float(max(bullish_signals, bearish_signals) / total_signals)
                
                if bullish_signals > bearish_signals and confidence >= confidence_threshold:
                    signals_result["overall_signal"] = "buy"
                elif bearish_signals > bullish_signals and confidence >= confidence_threshold:
                    signals_result["overall_signal"] = "sell"
                else:
                    signals_result["overall_signal"] = "hold"
            
            logger.info(f"信号生成脚本运行完成: {symbol}, 整体信号: {signals_result['overall_signal']}")
            return signals_result
            
        except Exception as e:
            logger.error(f"信号生成脚本运行失败: {e}")
            return {"error": str(e)}
    
    def run_backtesting_script(self, data: pd.DataFrame, signals: Dict[str, Any], 
                              symbol: str = "Unknown") -> Dict[str, Any]:
        """运行回测脚本"""
        try:
            if not self.is_feature_enabled("backtesting"):
                return {"error": "回测功能未启用"}
            
            if data.empty:
                return {"error": "数据为空"}
            
            config = self.get_feature_config("backtesting")
            commission_rate = config.get("commission_rate", 0.001)
            slippage = config.get("slippage", 0.0005)
            
            backtest_result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "initial_capital": 100000,  # 初始资金
                "final_capital": 100000,
                "total_return": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "trades": []
            }
            
            # 简化的回测逻辑
            capital = backtest_result["initial_capital"]
            position = 0  # 持仓数量
            entry_price = 0
            max_capital = capital
            drawdown_periods = []
            
            # 获取信号数据
            if "signals" in signals and "overall_signal" in signals:
                signal_data = signals["signals"]
                overall_signal = signals["overall_signal"]
                
                # 简化的交易逻辑
                for i in range(len(data)):
                    current_price = data['close'].iloc[i]
                    current_date = data.index[i] if hasattr(data, 'index') else i
                    
                    # 买入信号且未持仓
                    if overall_signal == "buy" and position == 0:
                        position = int(capital / current_price)  # 全仓买入
                        entry_price = current_price * (1 + slippage)  # 考虑滑点
                        capital -= position * entry_price * (1 + commission_rate)  # 扣除佣金
                        
                        backtest_result["trades"].append({
                            "date": str(current_date),
                            "action": "buy",
                            "price": float(entry_price),
                            "quantity": position,
                            "capital": float(capital)
                        })
                        backtest_result["total_trades"] += 1
                    
                    # 卖出信号且持仓
                    elif overall_signal == "sell" and position > 0:
                        exit_price = current_price * (1 - slippage)  # 考虑滑点
                        capital += position * exit_price * (1 - commission_rate)  # 扣除佣金
                        
                        # 计算盈亏
                        profit_loss = (exit_price - entry_price) * position
                        if profit_loss > 0:
                            backtest_result["winning_trades"] += 1
                            backtest_result["average_win"] += profit_loss
                        else:
                            backtest_result["losing_trades"] += 1
                            backtest_result["average_loss"] += abs(profit_loss)
                        
                        backtest_result["trades"].append({
                            "date": str(current_date),
                            "action": "sell",
                            "price": float(exit_price),
                            "quantity": position,
                            "capital": float(capital),
                            "profit_loss": float(profit_loss)
                        })
                        
                        position = 0
                        entry_price = 0
                        backtest_result["total_trades"] += 1
                    
                    # 更新最大资本和回撤
                    if capital > max_capital:
                        max_capital = capital
                    
                    current_drawdown = (max_capital - capital) / max_capital * 100
                    if current_drawdown > backtest_result["max_drawdown"]:
                        backtest_result["max_drawdown"] = current_drawdown
            
            # 最终平仓
            if position > 0 and len(data) > 0:
                final_price = data['close'].iloc[-1]
                exit_price = final_price * (1 - slippage)
                capital += position * exit_price * (1 - commission_rate)
                
                profit_loss = (exit_price - entry_price) * position
                if profit_loss > 0:
                    backtest_result["winning_trades"] += 1
                    backtest_result["average_win"] += profit_loss
                else:
                    backtest_result["losing_trades"] += 1
                    backtest_result["average_loss"] += abs(profit_loss)
                
                backtest_result["trades"].append({
                    "date": str(data.index[-1] if hasattr(data, 'index') else len(data)-1),
                    "action": "sell",
                    "price": float(exit_price),
                    "quantity": position,
                    "capital": float(capital),
                    "profit_loss": float(profit_loss)
                })
            
            # 计算最终指标
            backtest_result["final_capital"] = float(capital)
            backtest_result["total_return"] = float((capital - backtest_result["initial_capital"]) / backtest_result["initial_capital"] * 100)
            
            if backtest_result["total_trades"] > 0:
                backtest_result["win_rate"] = float(backtest_result["winning_trades"] / backtest_result["total_trades"] * 100)
                
                if backtest_result["winning_trades"] > 0:
                    backtest_result["average_win"] = float(backtest_result["average_win"] / backtest_result["winning_trades"])
                
                if backtest_result["losing_trades"] > 0:
                    backtest_result["average_loss"] = float(backtest_result["average_loss"] / backtest_result["losing_trades"])
            
            logger.info(f"回测脚本运行完成: {symbol}, 总收益: {backtest_result['total_return']:.2f}%")
            return backtest_result
            
        except Exception as e:
            logger.error(f"回测脚本运行失败: {e}")
            return {"error": str(e)}
    
    def run_portfolio_optimization_script(self, stock_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """运行投资组合优化脚本"""
        try:
            if not self.is_feature_enabled("portfolio_optimization"):
                return {"error": "投资组合优化功能未启用"}
            
            if not stock_data_dict:
                return {"error": "股票数据字典为空"}
            
            optimization_result = {
                "timestamp": datetime.now(),
                "num_stocks": len(stock_data_dict),
                "optimal_weights": {},
                "expected_return": 0.0,
                "portfolio_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "individual_stats": {}
            }
            
            # 计算收益率
            returns_dict = {}
            for symbol, data in stock_data_dict.items():
                if 'close' in data.columns and len(data) > 1:
                    returns = data['close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_dict[symbol] = returns
                        
                        # 个股统计
                        optimization_result["individual_stats"][symbol] = {
                            "mean_return": float(returns.mean() * 252),  # 年化收益率
                            "volatility": float(returns.std() * np.sqrt(252)),  # 年化波动率
                            "sharpe_ratio": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                        }
            
            if len(returns_dict) < 2:
                return {"error": "股票数量不足，无法进行组合优化"}
            
            # 创建收益率矩阵
            returns_df = pd.DataFrame(returns_dict).dropna()
            
            if returns_df.empty or len(returns_df) < 10:
                return {"error": "收益率数据不足"}
            
            # 计算预期收益率和协方差矩阵
            expected_returns = returns_df.mean() * 252
            cov_matrix = returns_df.cov() * 252
            
            # 简单等权重组合（作为基准）
            n_assets = len(expected_returns)
            equal_weights = np.array([1/n_assets] * n_assets)
            
            # 计算等权重组合指标
            portfolio_return = np.dot(equal_weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # 设置结果
            for i, symbol in enumerate(expected_returns.index):
                optimization_result["optimal_weights"][symbol] = float(equal_weights[i])
            
            optimization_result["expected_return"] = float(portfolio_return)
            optimization_result["portfolio_volatility"] = float(portfolio_volatility)
            optimization_result["sharpe_ratio"] = float(sharpe_ratio)
            
            logger.info(f"投资组合优化脚本运行完成: {optimization_result['num_stocks']} 支股票, 夏普比率: {sharpe_ratio:.2f}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"投资组合优化脚本运行失败: {e}")
            return {"error": str(e)}
    
    def run_risk_analysis_script(self, data: pd.DataFrame, symbol: str = "Unknown") -> Dict[str, Any]:
        """运行风险分析脚本"""
        try:
            if not self.is_feature_enabled("risk_analysis"):
                return {"error": "风险分析功能未启用"}
            
            if data.empty or 'close' not in data.columns:
                return {"error": "数据为空或缺少价格数据"}
            
            risk_result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "var_analysis": {},
                "drawdown_analysis": {},
                "volatility_analysis": {},
                "risk_metrics": {}
            }
            
            # 计算收益率
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return {"error": "收益率数据不足"}
            
            # VaR分析 (Value at Risk)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            risk_result["var_analysis"] = {
                "var_95": float(var_95 * 100),  # 转换为百分比
                "var_99": float(var_99 * 100),
                "var_95_annual": float(var_95 * np.sqrt(252) * 100),  # 年化
                "var_99_annual": float(var_99 * np.sqrt(252) * 100)
            }
            
            # 回撤分析
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            risk_result["drawdown_analysis"] = {
                "max_drawdown": float(drawdown.min() * 100),
                "avg_drawdown": float(drawdown.mean() * 100),
                "current_drawdown": float(drawdown.iloc[-1] * 100),
                "drawdown_periods": int((drawdown < 0).sum()),
                "max_drawdown_duration": int(self._calculate_max_drawdown_duration(drawdown))
            }
            
            # 波动率分析
            rolling_volatility_20 = returns.rolling(window=20).std() * np.sqrt(252)
            rolling_volatility_60 = returns.rolling(window=60).std() * np.sqrt(252)
            
            risk_result["volatility_analysis"] = {
                "current_volatility_20d": float(rolling_volatility_20.iloc[-1] * 100) if not rolling_volatility_20.empty else 0,
                "current_volatility_60d": float(rolling_volatility_60.iloc[-1] * 100) if not rolling_volatility_60.empty else 0,
                "avg_volatility": float(returns.std() * np.sqrt(252) * 100),
                "volatility_trend": float(np.polyfit(range(len(rolling_volatility_20.dropna())), rolling_volatility_20.dropna(), 1)[0]) if len(rolling_volatility_20.dropna()) > 1 else 0
            }
            
            # 综合风险指标
            risk_score = 0
            risk_score += abs(var_95) * 10  # VaR权重
            risk_score += abs(risk_result["drawdown_analysis"]["max_drawdown"]) / 100 * 20  # 最大回撤权重
            risk_score += risk_result["volatility_analysis"]["avg_volatility"] / 100 * 15  # 波动率权重
            
            # 风险等级
            if risk_score < 2:
                risk_level = "low"
            elif risk_score < 5:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            risk_result["risk_metrics"] = {
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "risk_adjusted_return": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                "calmar_ratio": float(returns.mean() * 252 / abs(risk_result["drawdown_analysis"]["max_drawdown"])) if risk_result["drawdown_analysis"]["max_drawdown"] != 0 else 0
            }
            
            logger.info(f"风险分析脚本运行完成: {symbol}, 风险等级: {risk_level}, 风险评分: {risk_score:.2f}")
            return risk_result
            
        except Exception as e:
            logger.error(f"风险分析脚本运行失败: {e}")
            return {"error": str(e)}
    
    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """计算最大回撤持续时间"""
        try:
            max_duration = 0
            current_duration = 0
            
            for value in drawdown_series:
                if value < 0:  # 处于回撤状态
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
            
            return max_duration
            
        except Exception:
            return 0
    
    def run_market_analysis_script(self, market_data: pd.DataFrame, 
                                  market_symbol: str = "Market") -> Dict[str, Any]:
        """运行市场分析脚本"""
        try:
            if not self.is_feature_enabled("market_analysis"):
                return {"error": "市场分析功能未启用"}
            
            if market_data.empty:
                return {"error": "市场数据为空"}
            
            market_result = {
                "market_symbol": market_symbol,
                "timestamp": datetime.now(),
                "market_overview": {},
                "sector_analysis": {},
                "market_sentiment": {},
                "market_breadth": {}
            }
            
            # 市场概览分析
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                
                market_result["market_overview"] = {
                    "total_return": float((market_data['close'].iloc[-1] / market_data['close'].iloc[0] - 1) * 100),
                    "volatility": float(returns.std() * np.sqrt(252) * 100),
                    "current_price": float(market_data['close'].iloc[-1]),
                    "avg_price": float(market_data['close'].mean()),
                    "price_trend": float(np.polyfit(range(len(market_data)), market_data['close'], 1)[0])
                }
            
            # 成交量分析
            if 'volume' in market_data.columns:
                volume_returns = market_data['volume'].pct_change().dropna()
                
                market_result["market_breadth"] = {
                    "avg_volume": float(market_data['volume'].mean()),
                    "volume_trend": float(np.polyfit(range(len(market_data)), market_data['volume'], 1)[0]),
                    "volume_volatility": float(volume_returns.std() * 100)
                }
            
            # 市场情绪和广度指标
            if len(market_data) > 20:
                # 计算一些简单的市场情绪指标
                recent_data = market_data.tail(20)
                earlier_data = market_data.head(len(market_data) - 20)
                
                if 'close' in recent_data.columns and 'close' in earlier_data.columns:
                    recent_avg = recent_data['close'].mean()
                    earlier_avg = earlier_data['close'].mean()
                    
                    market_result["market_sentiment"] = {
                        "short_term_trend": "bullish" if recent_avg > earlier_avg else "bearish",
                        "momentum_strength": float(abs(recent_avg - earlier_avg) / earlier_avg * 100)
                    }
            
            logger.info(f"市场分析脚本运行完成: {market_symbol}")
            return market_result
            
        except Exception as e:
            logger.error(f"市场分析脚本运行失败: {e}")
            return {"error": str(e)}
    
    def get_all_features_status(self) -> Dict[str, Any]:
        """获取所有功能状态"""
        try:
            enabled_features = self.get_enabled_features()
            feature_configs = self.feature_configs.get("feature_settings", {})
            
            status = {
                "enabled_features": enabled_features,
                "feature_configs": feature_configs,
                "total_features": len(enabled_features),
                "active_features": sum(1 for enabled in enabled_features.values() if enabled)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"获取功能状态失败: {e}")
            return {"error": str(e)}
    
    def reset(self):
        """重置应用脚本功能模块"""
        self.feature_configs.clear()
        self.script_cache.clear()
        self._load_config()
        logger.info("应用脚本功能模块已重置")