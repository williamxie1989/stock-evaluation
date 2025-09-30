import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """风险等级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_date: pd.Timestamp
    stop_loss: float
    take_profit: float
    risk_score: float

@dataclass
class RiskMetrics:
    """风险指标"""
    volatility: float  # 波动率
    beta: float  # Beta系数
    value_at_risk: float  # VaR值
    expected_shortfall: float  # 预期损失
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提诺比率

class RiskManager:
    """风险管理器"""
    
    def __init__(self, max_position_size: float = 0.1, 
                 max_portfolio_risk: float = 0.2,
                 stop_loss_pct: float = 0.08,
                 take_profit_pct: float = 0.15):
        """
        初始化风险管理器
        
        Args:
            max_position_size: 单只股票最大仓位比例（0-1）
            max_portfolio_risk: 组合最大风险敞口
            stop_loss_pct: 止损比例
            take_profit_pct: 止盈比例
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.positions: Dict[str, Position] = {}
        
    def calculate_position_size(self, portfolio_value: float, 
                               risk_score: float, 
                               current_price: float) -> Tuple[int, float, float]:
        """
        计算仓位大小
        
        Args:
            portfolio_value: 投资组合总价值
            risk_score: 风险评分（0-1）
            current_price: 当前价格
            
        Returns:
            Tuple[仓位数量, 止损价格, 止盈价格]
        """
        # 根据风险评分调整仓位比例
        position_pct = self.max_position_size * (1 - risk_score)
        
        # 计算最大可投资金额
        max_investment = portfolio_value * position_pct
        
        # 计算可购买数量
        quantity = int(max_investment / current_price)
        
        # 计算止损和止盈价格
        stop_loss_price = current_price * (1 - self.stop_loss_pct)
        take_profit_price = current_price * (1 + self.take_profit_pct)
        
        return quantity, stop_loss_price, take_profit_price
    
    def assess_signal_risk(self, df: pd.DataFrame, signal: Dict[str, object]) -> Dict[str, object]:
        """
        评估信号风险
        
        Args:
            df: 股票数据DataFrame
            signal: 交易信号
            
        Returns:
            包含风险评估的信号字典
        """
        # 计算技术指标风险
        tech_risk = self._calculate_technical_risk(df, signal)
        
        # 计算波动率风险
        vol_risk = self._calculate_volatility_risk(df)
        
        # 计算市场环境风险
        market_risk = self._calculate_market_risk(df)
        
        # 综合风险评分（0-1，越高风险越大）
        overall_risk = (tech_risk * 0.4 + vol_risk * 0.3 + market_risk * 0.3)
        
        # 确定风险等级
        risk_level = self._determine_risk_level(overall_risk)
        
        # 计算建议仓位比例
        suggested_position_pct = self.max_position_size * (1 - overall_risk)
        
        return {
            'risk_score': overall_risk,
            'risk_level': risk_level.name,
            'suggested_position_pct': suggested_position_pct,
            'technical_risk': tech_risk,
            'volatility_risk': vol_risk,
            'market_risk': market_risk,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }
    
    def _calculate_technical_risk(self, df: pd.DataFrame, signal: Dict[str, object]) -> float:
        """计算技术指标风险"""
        risk_factors = []
        
        # 1. 价格位置风险（相对于历史高低点）
        if len(df) >= 20:
            current_price = df['Close'].iloc[-1]
            high_20 = df['High'].rolling(20).max().iloc[-1]
            low_20 = df['Low'].rolling(20).min().iloc[-1]
            
            if high_20 != low_20:  # 避免除零错误
                price_position = (current_price - low_20) / (high_20 - low_20)
                # 价格在顶部区域风险较高
                position_risk = abs(price_position - 0.5) * 2  # 0-1范围
                risk_factors.append(position_risk)
        
        # 2. RSI超买超卖风险
        if 'RSI' in df.columns and len(df) >= 14:
            rsi = df['RSI'].iloc[-1]
            rsi_risk = 0
            if rsi > 70:
                rsi_risk = (rsi - 70) / 30  # 70-100映射到0-1
            elif rsi < 30:
                rsi_risk = (30 - rsi) / 30  # 0-30映射到0-1
            risk_factors.append(rsi_risk)
        
        # 3. MACD背离风险
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns and len(df) >= 26:
            macd = df['MACD'].iloc[-1]
            signal_line = df['MACD_Signal'].iloc[-1]
            price_trend = df['Close'].iloc[-1] > df['Close'].iloc[-5]  # 短期趋势
            
            macd_risk = 0
            if (macd > signal_line and not price_trend) or (macd < signal_line and price_trend):
                macd_risk = 0.7  # 存在背离
            risk_factors.append(macd_risk)
        
        # 如果没有风险因素，返回中等风险
        if not risk_factors:
            return 0.5
        
        return float(min(1.0, max(0.0, np.mean(risk_factors))))
    
    def _calculate_volatility_risk(self, df: pd.DataFrame, lookback_period: int = 20) -> float:
        """计算波动率风险"""
        if len(df) < lookback_period + 1:
            return 0.5  # 数据不足，返回中等风险
        
        # 计算日收益率
        returns = df['Close'].pct_change().dropna()
        
        if len(returns) < lookback_period:
            return 0.5
        
        # 计算近期波动率
        recent_volatility = returns.tail(lookback_period).std()
        
        # 计算历史波动率
        historical_volatility = returns.std()
        
        if historical_volatility == 0:
            return 0.5
        
        # 波动率风险：近期波动率相对于历史波动率的倍数
        vol_ratio = recent_volatility / historical_volatility
        
        # 映射到0-1范围
        volatility_risk = min(1.0, max(0.0, (vol_ratio - 0.5) * 2))
        
        return volatility_risk
    
    def _calculate_market_risk(self, df: pd.DataFrame) -> float:
        """计算市场环境风险"""
        if len(df) < 50:
            return 0.5
        
        risk_factors = []
        
        # 1. 市场趋势风险（下跌趋势风险高）
        short_term_trend = df['Close'].iloc[-1] > df['Close'].iloc[-20]
        long_term_trend = df['Close'].iloc[-1] > df['Close'].iloc[-50]
        
        trend_risk = 0
        if not short_term_trend and not long_term_trend:
            trend_risk = 0.8  # 双线下行
        elif not short_term_trend:
            trend_risk = 0.6  # 短期下行
        risk_factors.append(trend_risk)
        
        # 2. 成交量风险（缩量下跌风险高）
        if 'Volume' in df.columns:
            volume_avg_20 = float(df['Volume'].rolling(20).mean().values[-1])
            current_volume = df['Volume'].iloc[-1]
            
            volume_risk = 0
            if current_volume < volume_avg_20 * 0.7 and df['Close'].iloc[-1] < df['Close'].iloc[-5]:
                volume_risk = 0.7  # 缩量下跌
            risk_factors.append(volume_risk)
        
        # 3. 波动率聚集风险
        returns = df['Close'].pct_change().dropna()
        if len(returns) >= 20:
            recent_vol = returns.tail(10).std()
            historical_vol = returns.std()
            
            if historical_vol > 0:
                vol_cluster_risk = min(1.0, recent_vol / historical_vol * 0.8)
                risk_factors.append(vol_cluster_risk)
        
        if not risk_factors:
            return 0.3  # 市场环境较好
        
        return min(1.0, max(0.0, np.mean(risk_factors)))
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """确定风险等级"""
        if risk_score < 0.25:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif risk_score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        检查止损止盈条件
        
        Returns:
            None 或 'STOP_LOSS' 或 'TAKE_PROFIT'
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # 检查止损
        if current_price <= position.stop_loss:
            return 'STOP_LOSS'
        
        # 检查止盈
        if current_price >= position.take_profit:
            return 'TAKE_PROFIT'
        
        return None
    
    def update_position(self, symbol: str, current_price: float):
        """更新持仓价格"""
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price
    
    def add_position(self, symbol: str, quantity: int, entry_price: float, 
                    entry_date: pd.Timestamp, stop_loss: float, 
                    take_profit: float, risk_score: float):
        """添加新持仓"""
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            entry_date=entry_date,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_score=risk_score
        )
    
    def remove_position(self, symbol: str):
        """移除持仓"""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def get_portfolio_risk_metrics(self, portfolio_value: float) -> RiskMetrics:
        """获取组合风险指标"""
        # 这里实现组合级别的风险计算
        # 简化实现，实际中需要更复杂的计算
        return RiskMetrics(
            volatility=0.15,
            beta=1.0,
            value_at_risk=portfolio_value * 0.1,
            expected_shortfall=portfolio_value * 0.12,
            max_drawdown=0.08,
            sharpe_ratio=1.2,
            sortino_ratio=1.5
        )
    
    def generate_risk_report(self, portfolio_value: float, positions: Dict[str, Position]) -> str:
        """生成风险报告"""
        total_risk_exposure = sum(
            pos.quantity * pos.current_price * pos.risk_score 
            for pos in positions.values()
        )
        
        risk_pct = total_risk_exposure / portfolio_value if portfolio_value > 0 else 0
        
        report = f"""
风险管理报告
===========

组合风险概况
-----------
- 组合总价值: ¥{portfolio_value:,.2f}
- 总风险敞口: ¥{total_risk_exposure:,.2f}
- 风险敞口比例: {risk_pct:.1%}
- 最大允许风险: {self.max_portfolio_risk:.1%}

持仓风险分析
-----------
"""
        
        for symbol, position in positions.items():
            position_value = position.quantity * position.current_price
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
            
            report += f"\n{symbol}:"
            report += f"\n  持仓价值: ¥{position_value:,.2f} ({position_pct:.1%})"
            report += f"\n  风险评分: {position.risk_score:.3f}"
            report += f"\n  止损价格: ¥{position.stop_loss:.2f} ({((position.entry_price - position.stop_loss) / position.entry_price * 100):.1f}%)"
            report += f"\n  止盈价格: ¥{position.take_profit:.2f} ({((position.take_profit - position.entry_price) / position.entry_price * 100):.1f}%)"
            
            # 当前盈亏
            current_pnl = (position.current_price - position.entry_price) * position.quantity
            pnl_pct = (position.current_price - position.entry_price) / position.entry_price * 100
            report += f"\n  当前盈亏: ¥{current_pnl:,.2f} ({pnl_pct:.1f}%)"
        
        return report

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    data = {
        'Open': np.random.normal(100, 5, 100).cumsum(),
        'High': np.random.normal(105, 5, 100).cumsum(),
        'Low': np.random.normal(95, 5, 100).cumsum(),
        'Close': np.random.normal(100, 5, 100).cumsum(),
        'Volume': np.random.randint(1000, 10000, 100),
        'RSI': np.random.uniform(30, 70, 100),
        'MACD': np.random.normal(0, 1, 100),
        'MACD_Signal': np.random.normal(0, 1, 100)
    }
    df = pd.DataFrame(data, index=dates)
    
    # 创建风险管理器
    risk_manager = RiskManager()
    
    # 评估信号风险
    signal = {'type': 'BUY', 'price': df['Close'].iloc[-1]}
    risk_assessment = risk_manager.assess_signal_risk(df, signal)
    
    print("信号风险评估:")
    for key, value in risk_assessment.items():
        print(f"  {key}: {value}")
    
    # 计算仓位大小
    portfolio_value = 100000
    quantity, stop_loss, take_profit = risk_manager.calculate_position_size(
        portfolio_value, risk_assessment['risk_score'], df['Close'].iloc[-1]
    )
    
    print(f"\n建议仓位: {quantity}股")
    print(f"止损价格: ¥{stop_loss:.2f}")
    print(f"止盈价格: ¥{take_profit:.2f}")