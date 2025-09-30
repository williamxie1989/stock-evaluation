"""
自适应交易系统 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MarketState(Enum):
    """市场状态"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class TradingParams:
    """交易参数"""
    position_size: float = 0.1
    stop_loss: float = 0.05
    take_profit: float = 0.15
    max_positions: int = 5
    leverage: float = 1.0

class AdaptiveTradingSystem:
    """自适应交易系统 - 根据市场状态调整交易策略"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.market_state = MarketState.SIDEWAYS
        self.risk_level = RiskLevel.MEDIUM
        self.current_params = TradingParams()
        self.performance_metrics = {}
        
        # 市场状态历史
        self.market_state_history = []
        
        logger.info(f"AdaptiveTradingSystem initialized with capital: {initial_capital}")
    
    def analyze_market_state(self, market_data: pd.DataFrame) -> MarketState:
        """分析市场状态"""
        try:
            if market_data.empty or len(market_data) < 20:
                return MarketState.SIDEWAYS
            
            # 计算关键指标
            recent_data = market_data.tail(30)  # 最近30天
            
            # 1. 趋势强度
            returns = recent_data['close'].pct_change().dropna()
            trend_strength = abs(returns.mean()) / returns.std() if returns.std() != 0 else 0
            
            # 2. 波动率
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            
            # 3. 价格趋势
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            # 4. 成交量趋势
            volume_data = recent_data.get('volume', pd.Series([1] * len(recent_data)))
            volume_trend = (volume_data.iloc[-5:].mean() - volume_data.iloc[:-5].mean()) / volume_data.iloc[:-5].mean() if volume_data.iloc[:-5].mean() != 0 else 0
            
            # 确定市场状态
            if trend_strength > 1.5 and price_change > 0.05:
                market_state = MarketState.TRENDING_UP
            elif trend_strength > 1.5 and price_change < -0.05:
                market_state = MarketState.TRENDING_DOWN
            elif volatility > 0.3:
                market_state = MarketState.VOLATILE
            else:
                market_state = MarketState.SIDEWAYS
            
            # 更新市场状态历史
            self.market_state = market_state
            self.market_state_history.append({
                'timestamp': datetime.now(),
                'market_state': market_state,
                'trend_strength': trend_strength,
                'volatility': volatility,
                'price_change': price_change,
                'volume_trend': volume_trend
            })
            
            # 保持历史记录在合理范围内
            if len(self.market_state_history) > 100:
                self.market_state_history = self.market_state_history[-100:]
            
            logger.info(f"市场状态分析: {market_state.value}, 趋势强度: {trend_strength:.2f}, 波动率: {volatility:.2f}")
            return market_state
            
        except Exception as e:
            logger.error(f"市场状态分析失败: {e}")
            return MarketState.SIDEWAYS
    
    def assess_risk_level(self, market_data: pd.DataFrame, portfolio_data: Dict[str, Any] = None) -> RiskLevel:
        """评估风险等级"""
        try:
            if market_data.empty:
                return RiskLevel.MEDIUM
            
            risk_factors = []
            
            # 1. 市场波动率风险
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            if volatility > 0.4:
                risk_factors.append(0.8)
            elif volatility > 0.2:
                risk_factors.append(0.5)
            else:
                risk_factors.append(0.2)
            
            # 2. 组合集中度风险
            if portfolio_data and 'positions' in portfolio_data:
                positions = portfolio_data['positions']
                if len(positions) > 0:
                    position_sizes = [pos.get('size', 0) for pos in positions.values()]
                    max_position = max(position_sizes) if position_sizes else 0
                    total_capital = sum(position_sizes)
                    concentration_ratio = max_position / total_capital if total_capital > 0 else 0
                    
                    if concentration_ratio > 0.5:
                        risk_factors.append(0.8)
                    elif concentration_ratio > 0.3:
                        risk_factors.append(0.5)
                    else:
                        risk_factors.append(0.2)
            
            # 3. 最大回撤风险
            if len(self.trade_history) > 10:
                recent_trades = self.trade_history[-20:]
                if recent_trades:
                    returns = [trade.get('return', 0) for trade in recent_trades]
                    cumulative_returns = np.cumsum(returns)
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdowns = cumulative_returns - running_max
                    max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
                    
                    if max_drawdown > 0.2:
                        risk_factors.append(0.8)
                    elif max_drawdown > 0.1:
                        risk_factors.append(0.5)
                    else:
                        risk_factors.append(0.2)
            
            # 计算综合风险等级
            avg_risk = np.mean(risk_factors) if risk_factors else 0.5
            
            if avg_risk > 0.7:
                risk_level = RiskLevel.HIGH
            elif avg_risk > 0.4:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            self.risk_level = risk_level
            logger.info(f"风险等级评估: {risk_level.value}, 综合风险分数: {avg_risk:.2f}")
            return risk_level
            
        except Exception as e:
            logger.error(f"风险等级评估失败: {e}")
            return RiskLevel.MEDIUM
    
    def adapt_trading_params(self, market_state: MarketState, risk_level: RiskLevel) -> TradingParams:
        """根据市场状态和风险等级调整交易参数"""
        try:
            params = TradingParams()
            
            # 根据市场状态调整
            if market_state == MarketState.TRENDING_UP:
                params.position_size = 0.15  # 增加仓位
                params.take_profit = 0.20    # 提高止盈
                params.stop_loss = 0.08     # 放宽止损
                params.max_positions = 8    # 增加最大持仓数
            elif market_state == MarketState.TRENDING_DOWN:
                params.position_size = 0.05  # 减少仓位
                params.take_profit = 0.08    # 降低止盈
                params.stop_loss = 0.03     # 收紧止损
                params.max_positions = 3    # 减少最大持仓数
            elif market_state == MarketState.VOLATILE:
                params.position_size = 0.08  # 中等仓位
                params.take_profit = 0.10    # 中等止盈
                params.stop_loss = 0.05     # 中等止损
                params.max_positions = 5    # 中等持仓数
            else:  # SIDEWAYS
                params.position_size = 0.10  # 默认仓位
                params.take_profit = 0.12    # 默认止盈
                params.stop_loss = 0.06     # 默认止损
                params.max_positions = 5    # 默认持仓数
            
            # 根据风险等级进一步调整
            if risk_level == RiskLevel.HIGH:
                params.position_size *= 0.5  # 高风险时减半仓位
                params.take_profit *= 0.7    # 降低止盈目标
                params.stop_loss *= 0.8     # 收紧止损
                params.max_positions = max(2, params.max_positions // 2)  # 减少持仓数
            elif risk_level == RiskLevel.LOW:
                params.position_size *= 1.3  # 低风险时增加仓位
                params.take_profit *= 1.2    # 提高止盈目标
                params.stop_loss *= 1.1     # 放宽止损
                params.max_positions = min(10, int(params.max_positions * 1.2))  # 增加持仓数
            
            # 确保参数在合理范围内
            params.position_size = max(0.01, min(0.5, params.position_size))
            params.take_profit = max(0.02, min(0.5, params.take_profit))
            params.stop_loss = max(0.01, min(0.2, params.stop_loss))
            params.max_positions = max(1, min(20, params.max_positions))
            
            self.current_params = params
            
            logger.info(f"交易参数调整完成: 仓位{params.position_size:.2f}, 止盈{params.take_profit:.2f}, 止损{params.stop_loss:.2f}")
            return params
            
        except Exception as e:
            logger.error(f"交易参数调整失败: {e}")
            return TradingParams()
    
    def execute_trade(self, symbol: str, signal: str, price: float, volume: int = None) -> Dict[str, Any]:
        """执行交易"""
        try:
            # 检查是否可以交易
            if len(self.positions) >= self.current_params.max_positions:
                return {'success': False, 'error': '达到最大持仓限制'}
            
            if symbol in self.positions:
                return {'success': False, 'error': '该标的已持仓'}
            
            # 计算交易量
            if volume is None:
                position_value = self.current_capital * self.current_params.position_size
                volume = int(position_value / price)
            
            if volume <= 0:
                return {'success': False, 'error': '交易量无效'}
            
            # 检查资金
            total_cost = volume * price
            if total_cost > self.current_capital:
                return {'success': False, 'error': '资金不足'}
            
            # 创建持仓
            position = {
                'symbol': symbol,
                'entry_price': price,
                'volume': volume,
                'entry_time': datetime.now(),
                'stop_loss': price * (1 - self.current_params.stop_loss),
                'take_profit': price * (1 + self.current_params.take_profit),
                'market_state': self.market_state.value,
                'risk_level': self.risk_level.value
            }
            
            # 更新资金
            self.current_capital -= total_cost
            self.positions[symbol] = position
            
            # 记录交易
            trade_record = {
                'symbol': symbol,
                'action': 'BUY',
                'price': price,
                'volume': volume,
                'total_cost': total_cost,
                'timestamp': datetime.now(),
                'market_state': self.market_state.value,
                'risk_level': self.risk_level.value
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"交易执行成功: {symbol}, 价格: {price}, 数量: {volume}")
            
            return {
                'success': True,
                'position': position,
                'remaining_capital': self.current_capital
            }
            
        except Exception as e:
            logger.error(f"交易执行失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def close_position(self, symbol: str, price: float) -> Dict[str, Any]:
        """平仓"""
        try:
            if symbol not in self.positions:
                return {'success': False, 'error': '该标的未持仓'}
            
            position = self.positions[symbol]
            
            # 计算收益
            exit_value = position['volume'] * price
            entry_value = position['volume'] * position['entry_price']
            profit = exit_value - entry_value
            return_pct = profit / entry_value
            
            # 更新资金
            self.current_capital += exit_value
            
            # 记录交易
            trade_record = {
                'symbol': symbol,
                'action': 'SELL',
                'entry_price': position['entry_price'],
                'exit_price': price,
                'volume': position['volume'],
                'profit': profit,
                'return_pct': return_pct,
                'holding_days': (datetime.now() - position['entry_time']).days,
                'timestamp': datetime.now(),
                'market_state': self.market_state.value,
                'risk_level': self.risk_level.value
            }
            self.trade_history.append(trade_record)
            
            # 移除持仓
            del self.positions[symbol]
            
            logger.info(f"平仓成功: {symbol}, 收益: {profit:.2f}, 收益率: {return_pct:.2%}")
            
            return {
                'success': True,
                'profit': profit,
                'return_pct': return_pct,
                'remaining_capital': self.current_capital
            }
            
        except Exception as e:
            logger.error(f"平仓失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取组合摘要"""
        try:
            total_value = self.current_capital
            position_values = []
            
            for symbol, position in self.positions.items():
                # 这里假设使用最新价格，实际应用中需要获取实时价格
                position_value = position['volume'] * position['entry_price']  # 简化处理
                position_values.append({
                    'symbol': symbol,
                    'volume': position['volume'],
                    'entry_price': position['entry_price'],
                    'current_value': position_value,
                    'unrealized_pnl': 0  # 简化处理
                })
                total_value += position_value
            
            # 计算收益
            total_return = (total_value - self.initial_capital) / self.initial_capital
            
            summary = {
                'total_value': total_value,
                'cash': self.current_capital,
                'positions_value': total_value - self.current_capital,
                'total_return': total_return,
                'position_count': len(self.positions),
                'current_market_state': self.market_state.value,
                'current_risk_level': self.risk_level.value,
                'trading_params': {
                    'position_size': self.current_params.position_size,
                    'stop_loss': self.current_params.stop_loss,
                    'take_profit': self.current_params.take_profit,
                    'max_positions': self.current_params.max_positions
                },
                'positions': position_values
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取组合摘要失败: {e}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        try:
            if not self.trade_history:
                return {'error': '没有交易记录'}
            
            # 计算各种指标
            total_trades = len(self.trade_history)
            winning_trades = [t for t in self.trade_history if t.get('profit', 0) > 0 or t.get('return_pct', 0) > 0]
            losing_trades = [t for t in self.trade_history if t.get('profit', 0) <= 0 and t.get('return_pct', 0) <= 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            avg_win = np.mean([t.get('return_pct', 0) for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.get('return_pct', 0) for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # 计算夏普比率（简化版）
            returns = [t.get('return_pct', 0) for t in self.trade_history]
            if len(returns) > 1:
                excess_returns = np.array(returns) - 0.02  # 假设无风险利率2%
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
            else:
                sharpe_ratio = 0
            
            # 计算最大回撤
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
            
            metrics = {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': (self.current_capital + sum([p['volume'] * p['entry_price'] for p in self.positions.values()]) - self.initial_capital) / self.initial_capital
            }
            
            self.performance_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"获取性能指标失败: {e}")
            return {'error': str(e)}
    
    def reset(self):
        """重置交易系统"""
        self.current_capital = self.initial_capital
        self.positions.clear()
        self.trade_history.clear()
        self.market_state = MarketState.SIDEWAYS
        self.risk_level = RiskLevel.MEDIUM
        self.current_params = TradingParams()
        self.market_state_history.clear()
        self.performance_metrics.clear()
        logger.info("自适应交易系统已重置")