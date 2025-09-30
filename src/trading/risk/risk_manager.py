"""
风险管理器 - 精简版
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """风险管理器"""
    
    def __init__(self, max_position_size: float = 0.1, max_sector_exposure: float = 0.3,
                 stop_loss_pct: float = 0.05, take_profit_pct: float = 0.2,
                 max_drawdown_pct: float = 0.15):
        self.max_position_size = max_position_size  # 最大仓位比例
        self.max_sector_exposure = max_sector_exposure  # 最大行业暴露
        self.stop_loss_pct = stop_loss_pct  # 止损比例
        self.take_profit_pct = take_profit_pct  # 止盈比例
        self.max_drawdown_pct = max_drawdown_pct  # 最大回撤比例
        
        self.portfolio_high = 0  # 组合最高值
        self.current_drawdown = 0  # 当前回撤
        self.risk_metrics = {}  # 风险指标
        
        logger.info(f"RiskManager initialized: max_position_size={max_position_size}, "
                   f"stop_loss_pct={stop_loss_pct}, take_profit_pct={take_profit_pct}")
    
    def check_position_size(self, symbol: str, quantity: int, price: float, 
                           current_portfolio_value: float) -> bool:
        """检查仓位大小"""
        position_value = quantity * price
        position_ratio = position_value / current_portfolio_value
        
        if position_ratio > self.max_position_size:
            logger.warning(f"仓位过大: {symbol} 仓位比例 {position_ratio:.2%} 超过最大限制 {self.max_position_size:.2%}")
            return False
        
        return True
    
    def check_sector_exposure(self, sector_positions: Dict[str, float], 
                             sector: str, additional_exposure: float,
                             current_portfolio_value: float) -> bool:
        """检查行业暴露"""
        current_sector_exposure = sector_positions.get(sector, 0)
        new_sector_exposure = (current_sector_exposure + additional_exposure) / current_portfolio_value
        
        if new_sector_exposure > self.max_sector_exposure:
            logger.warning(f"行业暴露过高: {sector} 暴露比例 {new_sector_exposure:.2%} 超过最大限制 {self.max_sector_exposure:.2%}")
            return False
        
        return True
    
    def check_stop_loss(self, current_price: float, avg_price: float) -> bool:
        """检查止损"""
        if avg_price <= 0:
            return False
        
        loss_pct = (current_price - avg_price) / avg_price
        
        if loss_pct <= -self.stop_loss_pct:
            logger.warning(f"触发止损: 当前价格 {current_price:.2f}, 平均成本 {avg_price:.2f}, 亏损 {loss_pct:.2%}")
            return True
        
        return False
    
    def check_take_profit(self, current_price: float, avg_price: float) -> bool:
        """检查止盈"""
        if avg_price <= 0:
            return False
        
        profit_pct = (current_price - avg_price) / avg_price
        
        if profit_pct >= self.take_profit_pct:
            logger.warning(f"触发止盈: 当前价格 {current_price:.2f}, 平均成本 {avg_price:.2f}, 盈利 {profit_pct:.2%}")
            return True
        
        return False
    
    def check_drawdown(self, current_portfolio_value: float) -> bool:
        """检查回撤"""
        # 更新组合最高值
        if current_portfolio_value > self.portfolio_high:
            self.portfolio_high = current_portfolio_value
        
        # 计算当前回撤
        self.current_drawdown = (self.portfolio_high - current_portfolio_value) / self.portfolio_high
        
        if self.current_drawdown > self.max_drawdown_pct:
            logger.warning(f"回撤过大: 当前回撤 {self.current_drawdown:.2%} 超过最大限制 {self.max_drawdown_pct:.2%}")
            return False
        
        return True
    
    def calculate_portfolio_risk(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """计算组合风险指标"""
        if len(portfolio_returns) < 2:
            return {}
        
        # 计算各种风险指标
        volatility = portfolio_returns.std() * np.sqrt(252)  # 年化波动率
        var_95 = np.percentile(portfolio_returns, 5)  # 95% VaR
        var_99 = np.percentile(portfolio_returns, 1)  # 99% VaR
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        
        self.risk_metrics = {
            'volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'current_drawdown': self.current_drawdown
        }
        
        return self.risk_metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """计算夏普比率"""
        if returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def check_risk_limits(self, symbol: str, quantity: int, price: float,
                         current_portfolio_value: float, sector: str = None,
                         sector_positions: Dict[str, float] = None,
                         portfolio_returns: pd.Series = None) -> Dict[str, Any]:
        """综合风险检查"""
        checks = {
            'position_size': self.check_position_size(symbol, quantity, price, current_portfolio_value),
            'stop_loss': True,  # 默认通过，需要在持仓后检查
            'take_profit': True,  # 默认通过，需要在持仓后检查
            'drawdown': self.check_drawdown(current_portfolio_value),
            'overall': True
        }
        
        # 检查行业暴露
        if sector and sector_positions:
            additional_exposure = quantity * price
            checks['sector_exposure'] = self.check_sector_exposure(
                sector_positions, sector, additional_exposure, current_portfolio_value
            )
        else:
            checks['sector_exposure'] = True
        
        # 计算组合风险指标
        if portfolio_returns is not None:
            risk_metrics = self.calculate_portfolio_risk(portfolio_returns)
            checks['risk_metrics'] = risk_metrics
        
        # 综合判断
        checks['overall'] = all([
            checks['position_size'],
            checks['sector_exposure'],
            checks['drawdown']
        ])
        
        if not checks['overall']:
            logger.warning(f"风险检查未通过: {symbol} - {checks}")
        
        return checks
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        return {
            'risk_limits': {
                'max_position_size': self.max_position_size,
                'max_sector_exposure': self.max_sector_exposure,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'max_drawdown_pct': self.max_drawdown_pct
            },
            'current_risk': {
                'current_drawdown': self.current_drawdown,
                'portfolio_high': self.portfolio_high
            },
            'risk_metrics': self.risk_metrics,
            'timestamp': datetime.now()
        }
    
    def reset(self):
        """重置风险管理器"""
        self.portfolio_high = 0
        self.current_drawdown = 0
        self.risk_metrics.clear()
        logger.info("风险管理器已重置")