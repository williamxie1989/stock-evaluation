"""
交易系统 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    profit_loss: float
    profit_loss_pct: float

@dataclass
class Order:
    """订单信息"""
    symbol: str
    order_type: str  # 'buy' or 'sell'
    quantity: int
    price: float
    timestamp: datetime
    status: str  # 'pending', 'filled', 'cancelled'

class TradingSystem:
    """交易系统"""
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.is_trading = False
        logger.info(f"TradingSystem initialized: initial_capital={initial_capital}, commission={commission}")
    
    def start_trading(self):
        """开始交易"""
        self.is_trading = True
        logger.info("交易系统启动")
    
    def stop_trading(self):
        """停止交易"""
        self.is_trading = False
        logger.info("交易系统停止")
    
    def place_order(self, symbol: str, order_type: str, quantity: int, price: float) -> bool:
        """下单"""
        if not self.is_trading:
            logger.warning("交易系统未启动，无法下单")
            return False
        
        # 计算订单成本
        order_value = quantity * price
        commission_cost = order_value * self.commission
        total_cost = order_value + commission_cost
        
        # 检查资金
        if order_type == 'buy' and total_cost > self.cash:
            logger.warning(f"资金不足，无法买入 {symbol}: 需要{total_cost:.2f}, 可用{self.cash:.2f}")
            return False
        
        # 检查持仓
        if order_type == 'sell' and symbol not in self.positions:
            logger.warning(f"没有持仓，无法卖出 {symbol}")
            return False
        
        if order_type == 'sell' and self.positions[symbol].quantity < quantity:
            logger.warning(f"持仓不足，无法卖出 {symbol}: 需要{quantity}, 持仓{self.positions[symbol].quantity}")
            return False
        
        # 创建订单
        order = Order(
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            status='pending'
        )
        
        # 执行订单
        if self._execute_order(order):
            self.orders.append(order)
            logger.info(f"订单执行成功: {order_type} {quantity}股 {symbol} @ {price}")
            return True
        else:
            logger.error(f"订单执行失败: {order_type} {quantity}股 {symbol} @ {price}")
            return False
    
    def _execute_order(self, order: Order) -> bool:
        """执行订单"""
        try:
            if order.order_type == 'buy':
                return self._execute_buy(order)
            elif order.order_type == 'sell':
                return self._execute_sell(order)
            else:
                logger.error(f"未知订单类型: {order.order_type}")
                return False
        except Exception as e:
            logger.error(f"执行订单失败: {e}")
            return False
    
    def _execute_buy(self, order: Order) -> bool:
        """执行买入"""
        total_cost = order.quantity * order.price * (1 + self.commission)
        
        # 更新现金
        self.cash -= total_cost
        
        # 更新持仓
        if order.symbol in self.positions:
            # 已有持仓，更新平均价格
            position = self.positions[order.symbol]
            total_quantity = position.quantity + order.quantity
            total_value = position.quantity * position.avg_price + order.quantity * order.price
            position.avg_price = total_value / total_quantity
            position.quantity = total_quantity
        else:
            # 新建持仓
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=order.quantity,
                avg_price=order.price,
                current_price=order.price,
                market_value=order.quantity * order.price,
                profit_loss=0,
                profit_loss_pct=0
            )
        
        # 更新订单状态
        order.status = 'filled'
        
        # 记录交易历史
        self.trade_history.append({
            'symbol': order.symbol,
            'action': 'buy',
            'quantity': order.quantity,
            'price': order.price,
            'value': order.quantity * order.price,
            'commission': order.quantity * order.price * self.commission,
            'timestamp': order.timestamp,
            'cash_after': self.cash
        })
        
        return True
    
    def _execute_sell(self, order: Order) -> bool:
        """执行卖出"""
        if order.symbol not in self.positions:
            return False
        
        position = self.positions[order.symbol]
        
        # 计算收入
        revenue = order.quantity * order.price * (1 - self.commission)
        
        # 更新现金
        self.cash += revenue
        
        # 计算盈亏
        cost_basis = order.quantity * position.avg_price
        profit_loss = revenue - cost_basis
        profit_loss_pct = profit_loss / cost_basis if cost_basis > 0 else 0
        
        # 更新持仓
        position.quantity -= order.quantity
        if position.quantity == 0:
            del self.positions[order.symbol]
        
        # 更新订单状态
        order.status = 'filled'
        
        # 记录交易历史
        self.trade_history.append({
            'symbol': order.symbol,
            'action': 'sell',
            'quantity': order.quantity,
            'price': order.price,
            'value': order.quantity * order.price,
            'commission': order.quantity * order.price * self.commission,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'timestamp': order.timestamp,
            'cash_after': self.cash
        })
        
        return True
    
    def update_positions(self, current_prices: Dict[str, float]):
        """更新持仓价格"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.profit_loss = position.quantity * (current_price - position.avg_price)
                position.profit_loss_pct = (current_price - position.avg_price) / position.avg_price if position.avg_price > 0 else 0
    
    def get_portfolio_value(self) -> float:
        """获取组合总价值"""
        positions_value = sum(position.market_value for position in self.positions.values())
        return self.cash + positions_value
    
    def get_total_return(self) -> float:
        """获取总收益率"""
        current_value = self.get_portfolio_value()
        return (current_value - self.initial_capital) / self.initial_capital
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取组合摘要"""
        total_value = self.get_portfolio_value()
        total_return = self.get_total_return()
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'number_of_positions': len(self.positions),
            'number_of_trades': len(self.trade_history),
            'timestamp': datetime.now()
        }
    
    def get_position_details(self) -> List[Dict[str, Any]]:
        """获取持仓详情"""
        positions = []
        for position in self.positions.values():
            positions.append({
                'symbol': position.symbol,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'profit_loss': position.profit_loss,
                'profit_loss_pct': position.profit_loss_pct * 100,
                'weight_in_portfolio': position.market_value / self.get_portfolio_value() if self.get_portfolio_value() > 0 else 0
            })
        return positions
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """获取交易统计"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_loss': 0,
                'profit_factor': 0,
                'average_profit': 0,
                'average_loss': 0,
                'largest_profit': 0,
                'largest_loss': 0
            }
        
        # 分离买卖交易
        buy_trades = [t for t in self.trade_history if t['action'] == 'buy']
        sell_trades = [t for t in self.trade_history if t['action'] == 'sell']
        
        # 计算盈亏
        profitable_trades = [t for t in sell_trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('profit_loss', 0) < 0]
        
        total_profit = sum(t.get('profit_loss', 0) for t in profitable_trades)
        total_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
        
        return {
            'total_trades': len(self.trade_history),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(profitable_trades) / len(sell_trades) if sell_trades else 0,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
            'average_profit': np.mean([t.get('profit_loss', 0) for t in profitable_trades]) if profitable_trades else 0,
            'average_loss': np.mean([abs(t.get('profit_loss', 0)) for t in losing_trades]) if losing_trades else 0,
            'largest_profit': max([t.get('profit_loss', 0) for t in profitable_trades]) if profitable_trades else 0,
            'largest_loss': min([t.get('profit_loss', 0) for t in losing_trades]) if losing_trades else 0
        }
    
    def reset(self):
        """重置交易系统"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trade_history.clear()
        self.is_trading = False
        logger.info("交易系统已重置")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        portfolio_summary = self.get_portfolio_summary()
        position_details = self.get_position_details()
        trade_statistics = self.get_trade_statistics()
        
        return {
            'portfolio': portfolio_summary,
            'positions': position_details,
            'trades': trade_statistics,
            'timestamp': datetime.now()
        }