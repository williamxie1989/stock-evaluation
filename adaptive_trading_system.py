import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from advanced_signal_generator import Signal

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """市场环境枚举"""
    TRENDING_BULL = "trending_bull"  # 趋势牛市
    TRENDING_BEAR = "trending_bear"  # 趋势熊市
    RANGING = "ranging"             # 震荡市
    HIGH_VOLATILITY = "high_vol"    # 高波动率
    LOW_VOLATILITY = "low_vol"      # 低波动率

@dataclass
class Position:
    """仓位信息数据类"""
    symbol: str
    entry_price: float
    quantity: int
    entry_date: pd.Timestamp
    stop_loss: float
    take_profit: float
    position_type: str  # LONG/SHORT
    risk_percentage: float
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0

@dataclass
class TradingDecision:
    """交易决策数据类"""
    signal: Signal
    position_size: int
    risk_amount: float
    market_regime: MarketRegime
    volatility_adjustment: float
    confidence_score: float

class AdaptiveTradingSystem:
    """自适应交易系统 - 实现智能仓位管理和风险控制"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: List[Position] = []
        self.trading_history: List[Dict] = []
        
        # 风险控制参数
        self.max_risk_per_trade = 0.02  # 单笔交易最大风险2%
        self.max_portfolio_risk = 0.10  # 组合最大风险10%
        self.max_drawdown_limit = 0.15  # 最大回撤限制15%
        
        # 仓位管理参数
        self.kelly_fraction = 0.5  # 凯利系数
        self.volatility_scaling = True
        self.regime_aware = True
        
        # 性能指标
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        
    def make_trading_decision(self, 
                           signal: Signal, 
                           market_data: pd.DataFrame,
                           current_portfolio: List[Position] = None) -> TradingDecision:
        """
        基于信号和市场环境做出交易决策
        
        Args:
            signal: 交易信号
            market_data: 市场数据
            current_portfolio: 当前持仓组合
            
        Returns:
            TradingDecision: 交易决策
        """
        
        # 1. 分析市场环境
        market_regime = self._analyze_market_regime(market_data)
        
        # 2. 计算波动率调整
        volatility_adj = self._calculate_volatility_adjustment(market_data, market_regime)
        
        # 3. 计算仓位大小
        position_size = self._calculate_position_size(signal, market_data, market_regime, volatility_adj)
        
        # 4. 计算风险金额
        risk_amount = self._calculate_risk_amount(position_size, signal)
        
        # 5. 计算置信度评分
        confidence_score = self._calculate_confidence_score(signal, market_regime, volatility_adj)
        
        # 6. 检查风险限制
        if not self._check_risk_limits(risk_amount, current_portfolio):
            position_size = 0
            risk_amount = 0
            logger.warning("交易风险超过限制，拒绝交易")
        
        return TradingDecision(
            signal=signal,
            position_size=position_size,
            risk_amount=risk_amount,
            market_regime=market_regime,
            volatility_adjustment=volatility_adj,
            confidence_score=confidence_score
        )
    
    def execute_trade(self, decision: TradingDecision, symbol: str) -> Optional[Position]:
        """
        执行交易
        
        Args:
            decision: 交易决策
            symbol: 交易标的
            
        Returns:
            Optional[Position]: 新建仓位，如果交易被拒绝则返回None
        """
        
        if decision.position_size == 0:
            logger.info("仓位大小为0，跳过交易执行")
            return None
        
        # 计算交易成本（简化模型）
        transaction_cost = self._calculate_transaction_cost(decision)
        
        # 检查资金是否充足
        required_capital = (decision.position_size * decision.signal.price) + transaction_cost
        if required_capital > self.current_capital:
            logger.warning("资金不足，无法执行交易")
            return None
        
        # 创建仓位
        position = Position(
            symbol=symbol,
            entry_price=decision.signal.price,
            quantity=decision.position_size,
            entry_date=decision.signal.date,
            stop_loss=decision.signal.stop_loss,
            take_profit=decision.signal.take_profit,
            position_type='LONG' if decision.signal.type == 'BUY' else 'SHORT',
            risk_percentage=decision.risk_amount / self.current_capital,
            current_value=decision.position_size * decision.signal.price
        )
        
        # 更新资金和持仓
        self.current_capital -= required_capital
        self.positions.append(position)
        self.total_trades += 1
        
        # 记录交易历史
        trade_record = {
            'date': decision.signal.date,
            'symbol': symbol,
            'action': decision.signal.type,
            'price': decision.signal.price,
            'quantity': decision.position_size,
            'amount': required_capital,
            'reason': decision.signal.reason,
            'confidence': decision.confidence_score
        }
        self.trading_history.append(trade_record)
        
        logger.info(f"执行交易: {symbol} {decision.signal.type} {decision.position_size}股 "
                   f"@ {decision.signal.price:.2f}, 风险: {decision.risk_amount:.2f}")
        
        return position
    
    def update_positions(self, current_prices: Dict[str, float]):
        """更新持仓市值和盈亏"""
        total_portfolio_value = self.current_capital
        
        for position in self.positions:
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                position.current_value = position.quantity * current_price
                
                if position.position_type == 'LONG':
                    position.unrealized_pnl = position.current_value - (position.quantity * position.entry_price)
                else:  # SHORT
                    position.unrealized_pnl = (position.quantity * position.entry_price) - position.current_value
                
                position.unrealized_pnl_percent = position.unrealized_pnl / (position.quantity * position.entry_price)
                total_portfolio_value += position.current_value
        
        # 计算最大回撤
        self._update_max_drawdown(total_portfolio_value)
    
    def analyze_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        分析市场环境
        
        Args:
            market_data: 市场数据
            
        Returns:
            MarketRegime: 市场环境类型
        """
        return self._analyze_market_regime(market_data)
    
    def generate_trading_decision(self, signal: Signal, market_data: pd.DataFrame) -> TradingDecision:
        """
        生成交易决策（兼容性方法）
        
        Args:
            signal: 交易信号
            market_data: 市场数据
            
        Returns:
            TradingDecision: 交易决策
        """
        return self.make_trading_decision(signal, market_data)
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]) -> List[Position]:
        """检查止损止盈条件"""
        closed_positions = []
        remaining_positions = []
        
        for position in self.positions:
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                
                # 检查止损止盈
                should_close = False
                close_reason = ""
                
                if position.position_type == 'LONG':
                    if current_price <= position.stop_loss:
                        should_close = True
                        close_reason = "止损"
                    elif current_price >= position.take_profit:
                        should_close = True
                        close_reason = "止盈"
                else:  # SHORT
                    if current_price >= position.stop_loss:
                        should_close = True
                        close_reason = "止损"
                    elif current_price <= position.take_profit:
                        should_close = True
                        close_reason = "止盈"
                
                if should_close:
                    # 平仓
                    closed_value = position.quantity * current_price
                    self.current_capital += closed_value
                    
                    # 记录交易结果
                    pnl = closed_value - (position.quantity * position.entry_price)
                    if position.position_type == 'SHORT':
                        pnl = -pnl  # 做空盈亏计算
                    
                    self.total_pnl += pnl
                    if pnl > 0:
                        self.winning_trades += 1
                    
                    logger.info(f"平仓: {position.symbol} {position.position_type} "
                               f"盈亏: {pnl:.2f}, 原因: {close_reason}")
                    
                    closed_positions.append(position)
                else:
                    remaining_positions.append(position)
        
        self.positions = remaining_positions
        return closed_positions
    
    def _analyze_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """分析市场环境"""
        if len(market_data) < 20:
            return MarketRegime.RANGING
        
        # 计算趋势指标
        ma20 = market_data['Close'].rolling(20).mean()
        ma60 = market_data['Close'].rolling(60).mean()
        
        # 计算波动率
        returns = market_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        # 趋势判断
        price_trend = (market_data['Close'].iloc[-1] - market_data['Close'].iloc[-20]) / market_data['Close'].iloc[-20]
        ma_trend = (ma20.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1]
        
        if abs(price_trend) > 0.05:  # 5%以上的趋势
            if price_trend > 0 and ma_trend > 0:
                return MarketRegime.TRENDING_BULL
            elif price_trend < 0 and ma_trend < 0:
                return MarketRegime.TRENDING_BEAR
        
        # 波动率判断
        if volatility > 0.25:  # 25%以上年化波动率
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.15:  # 15%以下年化波动率
            return MarketRegime.LOW_VOLATILITY
        
        return MarketRegime.RANGING
    
    def _calculate_volatility_adjustment(self, market_data: pd.DataFrame, regime: MarketRegime) -> float:
        """计算波动率调整系数"""
        if not self.volatility_scaling:
            return 1.0
        
        # 计算ATR相对值
        atr = self._calculate_atr(market_data).iloc[-1]
        avg_atr = self._calculate_atr(market_data).rolling(20).mean().iloc[-1]
        
        volatility_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        
        # 根据市场环境调整
        adjustment = 1.0
        if regime == MarketRegime.HIGH_VOLATILITY:
            adjustment = max(0.5, 1.0 / volatility_ratio)
        elif regime == MarketRegime.LOW_VOLATILITY:
            adjustment = min(2.0, volatility_ratio)
        
        return adjustment
    
    def _calculate_position_size(self, signal: Signal, market_data: pd.DataFrame, 
                               regime: MarketRegime, volatility_adj: float) -> int:
        """计算仓位大小"""
        
        # 1. 基于凯利公式计算理论仓位
        kelly_position = self._kelly_position_size(signal, market_data)
        
        # 2. 基于风险预算计算仓位
        risk_based_position = self._risk_based_position_size(signal)
        
        # 3. 取较小值
        base_position = min(kelly_position, risk_based_position)
        
        # 4. 市场环境调整
        regime_multiplier = self._get_regime_multiplier(regime)
        
        # 5. 波动率调整
        final_position = int(base_position * regime_multiplier * volatility_adj)
        
        # 确保最小交易单位
        if final_position < 100:  # 最少100股
            return 0
        
        return final_position
    
    def _kelly_position_size(self, signal: Signal, market_data: pd.DataFrame) -> int:
        """基于凯利公式计算仓位大小"""
        # 简化凯利公式: f = (bp - q) / b
        # 这里使用信号置信度作为胜率估计
        win_probability = signal.confidence
        win_loss_ratio = 2.0  # 假设盈亏比2:1
        
        if win_loss_ratio <= 0:
            return 0
        
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        kelly_fraction = max(0, min(self.kelly_fraction, kelly_fraction))  # 限制凯利系数
        
        position_value = self.current_capital * kelly_fraction
        shares = int(position_value / signal.price)
        
        return shares
    
    def _risk_based_position_size(self, signal: Signal) -> int:
        """基于风险预算计算仓位大小"""
        # 检查止损价格是否有效
        if signal.stop_loss is None:
            return 0
        
        # 计算每单位风险
        risk_per_share = abs(signal.price - signal.stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        # 单笔交易最大风险金额
        max_risk_amount = self.current_capital * self.max_risk_per_trade
        
        # 计算最大可买股数
        max_shares = int(max_risk_amount / risk_per_share)
        
        return max_shares
    
    def _calculate_risk_amount(self, position_size: int, signal: Signal) -> float:
        """计算风险金额"""
        risk_per_share = abs(signal.price - signal.stop_loss)
        return position_size * risk_per_share
    
    def _calculate_confidence_score(self, signal: Signal, regime: MarketRegime, volatility_adj: float) -> float:
        """计算综合置信度评分"""
        base_confidence = signal.confidence
        
        # 市场环境加成
        regime_bonus = self._get_regime_confidence_bonus(regime)
        
        # 波动率调整
        volatility_factor = 1.0 + (1.0 - volatility_adj) * 0.1
        
        # 多时间框架确认加成
        timeframe_bonus = 0.1 if signal.multi_timeframe_confirmed else 0.0
        
        # 成交量确认加成
        volume_bonus = 0.05 if signal.volume_confirmation else 0.0
        
        final_confidence = base_confidence * (1.0 + regime_bonus + timeframe_bonus + volume_bonus) * volatility_factor
        
        return min(1.0, max(0.0, final_confidence))
    
    def _check_risk_limits(self, risk_amount: float, current_portfolio: List[Position] = None) -> bool:
        """检查风险限制"""
        # 单笔交易风险检查
        if risk_amount > self.current_capital * self.max_risk_per_trade:
            return False
        
        # 组合风险检查
        if current_portfolio:
            total_risk = risk_amount
            for position in current_portfolio:
                total_risk += position.risk_percentage * self.current_capital
            
            if total_risk > self.current_capital * self.max_portfolio_risk:
                return False
        
        # 最大回撤检查
        if self.max_drawdown > self.max_drawdown_limit:
            return False
        
        return True
    
    def _calculate_transaction_cost(self, decision: TradingDecision) -> float:
        """计算交易成本"""
        # 简化模型：佣金 + 滑点
        commission_rate = 0.0003  # 0.03%
        slippage_rate = 0.0005    # 0.05%
        
        trade_amount = decision.position_size * decision.signal.price
        commission = trade_amount * commission_rate
        slippage = trade_amount * slippage_rate
        
        return commission + slippage
    
    def _update_max_drawdown(self, current_portfolio_value: float):
        """更新最大回撤"""
        peak_value = max(self.initial_capital, current_portfolio_value)
        drawdown = (peak_value - current_portfolio_value) / peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """获取市场环境乘数"""
        multipliers = {
            MarketRegime.TRENDING_BULL: 1.2,
            MarketRegime.TRENDING_BEAR: 0.8,
            MarketRegime.RANGING: 0.6,
            MarketRegime.HIGH_VOLATILITY: 0.4,
            MarketRegime.LOW_VOLATILITY: 1.0
        }
        return multipliers.get(regime, 1.0)
    
    def _get_regime_confidence_bonus(self, regime: MarketRegime) -> float:
        """获取市场环境置信度加成"""
        bonuses = {
            MarketRegime.TRENDING_BULL: 0.15,
            MarketRegime.TRENDING_BEAR: 0.10,
            MarketRegime.RANGING: -0.05,
            MarketRegime.HIGH_VOLATILITY: -0.10,
            MarketRegime.LOW_VOLATILITY: 0.05
        }
        return bonuses.get(regime, 0.0)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR指标"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'active_positions': len(self.positions)
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率（简化版）"""
        if len(self.trading_history) < 2:
            return 0.0
        
        # 这里需要实际的收益率序列来计算夏普比率
        # 简化实现，返回一个估计值
        return self.total_pnl / (self.max_drawdown * self.initial_capital) if self.max_drawdown > 0 else 0.0
    
    def calculate_risk_metrics(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算风险指标
        
        Args:
            market_data: 市场数据
            
        Returns:
            Dict[str, float]: 风险指标字典
        """
        if len(market_data) < 20:
            return {}
        
        # 计算波动率
        returns = market_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        # 计算最大回撤
        peak = market_data['Close'].expanding().max()
        drawdown = (market_data['Close'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # 计算VaR（风险价值）
        var_95 = returns.quantile(0.05)
        
        # 计算Beta（需要基准数据，这里简化）
        beta = 1.0  # 假设与市场同步
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'beta': beta,
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }

# 使用示例
if __name__ == "__main__":
    # 创建自适应交易系统
    trading_system = AdaptiveTradingSystem(initial_capital=100000)
    
    print("自适应交易系统已创建，支持智能仓位管理和风险控制")