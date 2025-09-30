import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """交易记录"""
    date: datetime
    type: str  # BUY or SELL
    price: float
    quantity: int
    reason: str
    commission: float = 0.0

@dataclass
class Portfolio:
    """投资组合"""
    cash: float
    position: int
    current_price: float
    value: float

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, initial_capital: float = 100000.0, commission_rate: float = 0.0003):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.trades: List[Trade] = []
        self.portfolio_history: List[Portfolio] = []
        
    def run_backtest(self, df: pd.DataFrame, signals: List[Dict[str, Any]], 
                     strategy: str = "simple") -> Dict[str, Any]:
        """运行回测"""
        # 重置状态
        self.trades = []
        self.portfolio_history = []
        
        # 初始化投资组合
        cash = self.initial_capital
        position = 0
        
        # 创建信号DataFrame以便快速查找
        signals_df = self._create_signals_dataframe(signals)
        
        # 运行回测
        last_valid_price: Optional[float] = None
        for i, (date, row) in enumerate(df.iterrows()):
            current_price = row.get('Close', None) if hasattr(row, 'get') else row['Close']

            # 如果当前价格无效（NaN、非有限数或<=0），跳过交易信号执行，但仍记录组合价值
            if current_price is None or not np.isfinite(current_price) or current_price <= 0:
                # 使用最后一个有效价格计算持仓市值（如果存在），否则只记录现金
                price_for_valuation = last_valid_price if last_valid_price is not None else 0.0
                portfolio_value = cash + (position * price_for_valuation)
                self.portfolio_history.append(Portfolio(
                    cash=float(cash),
                    position=position,
                    current_price=float(price_for_valuation) if price_for_valuation is not None else 0.0,
                    value=portfolio_value
                ))
                # 不尝试在价格无效时下单
                continue

            # 有效价格时更新最后有效价格
            last_valid_price = float(current_price)
            
            # 检查是否有交易信号
            trade_signal = self._get_trade_signal(date if isinstance(date, pd.Timestamp) else pd.Timestamp(str(date)), signals_df, strategy)
            
            if trade_signal:
                # 执行交易
                if trade_signal['type'] == 'BUY' and cash > 0:
                    # 计算可购买数量（全仓买入）
                    quantity = int(cash / (current_price * (1 + self.commission_rate)))
                    if quantity > 0:
                        cost = quantity * current_price
                        commission = cost * self.commission_rate
                        cash -= (cost + commission)
                        position += quantity
                        
                        self.trades.append(Trade(
                            date=date.to_pydatetime() if isinstance(date, pd.Timestamp) else pd.Timestamp(str(date)).to_pydatetime(),
                            type='BUY',
                            price=float(current_price),
                            quantity=quantity,
                            reason=trade_signal['reason'],
                            commission=float(commission)
                        ))
                
                elif trade_signal['type'] == 'SELL' and position > 0:
                    # 卖出全部持仓
                    revenue = position * current_price
                    commission = revenue * self.commission_rate
                    cash += (revenue - commission)
                    
                    self.trades.append(Trade(
                        date=date.to_pydatetime() if isinstance(date, pd.Timestamp) else pd.Timestamp(str(date)).to_pydatetime(),
                        type='SELL',
                        price=current_price,
                        quantity=position,
                        reason=trade_signal['reason'],
                        commission=commission
                    ))
                    position = 0
            
            # 记录投资组合价值
            portfolio_value = cash + (position * float(current_price))
            # 确保组合价值为数值（防止NaN/inf）
            if not np.isfinite(portfolio_value):
                portfolio_value = float(cash)

            self.portfolio_history.append(Portfolio(
                cash=float(cash),
                position=position,
                current_price=float(current_price),
                value=float(portfolio_value)
            ))
        
        # 计算性能指标
        performance = self._calculate_performance(df)
        
        return {
            'performance': performance,
            'trades': self.trades,
            'portfolio_history': self.portfolio_history,
            'final_portfolio': self.portfolio_history[-1] if self.portfolio_history else None
        }
    
    def _create_signals_dataframe(self, signals: List[Dict[str, Any]]) -> pd.DataFrame:
        """创建信号DataFrame"""
        if not signals:
            return pd.DataFrame()
        
        signals_data = []
        for signal in signals:
            # 统一信号日期为 pd.Timestamp 类型
            date = signal['date']
            if not isinstance(date, pd.Timestamp):
                try:
                    date = pd.Timestamp(date)
                except Exception:
                    continue
            signals_data.append({
                'date': date,
                'type': signal['type'],
                'price': signal['price'],
                'reason': signal['reason'],
                'confidence': signal.get('confidence', 0.5)
            })
        if not signals_data:
            return pd.DataFrame()
        return pd.DataFrame(signals_data).set_index('date')
    
    def _get_trade_signal(self, date: datetime, signals_df: pd.DataFrame, 
                         strategy: str) -> Optional[Dict[str, Any]]:
        """获取交易信号"""
        if signals_df.empty:
            return None
        
        # 查找当天的信号
        if date in signals_df.index:
            signal_row = signals_df.loc[date]
            # 如果当天有多个信号，只取第一个
            if isinstance(signal_row, pd.DataFrame):
                signal_row = signal_row.iloc[0]
            return {
                'type': signal_row['type'],
                'price': signal_row['price'],
                'reason': signal_row['reason'],
                'confidence': signal_row.get('confidence', 0.5)
            }
        
        return None
    
    def _calculate_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算性能指标"""
        # 如果没有足够的历史数据，返回默认值
        if not self.portfolio_history or len(self.portfolio_history) < 2:
            return {
                'initial_value': self.initial_capital,
                'final_value': self.initial_capital,
                'total_return': 0.0,
                'annualized_return': 0.0,
                'max_drawdown': 0.0,
                'drawdown_duration_days': 0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'commission_total': 0.0
            }
            
        # 提取投资组合价值序列，确保所有值都是有效的浮点数
        portfolio_values = [float(p.value) for p in self.portfolio_history]
        # 对应的日期序列（截断以匹配组合记录长度）
        dates = list(df.index[:len(portfolio_values)]) if hasattr(df, 'index') else []

        # 计算基本指标，保护除零和无效数值
        initial_value_raw = portfolio_values[0] if len(portfolio_values) > 0 else self.initial_capital
        initial_value = float(initial_value_raw) if np.isfinite(initial_value_raw) and initial_value_raw > 0 else float(self.initial_capital)
        final_value_raw = portfolio_values[-1]
        final_value = float(final_value_raw) if np.isfinite(final_value_raw) else initial_value

        # 计算总收益率，保护initial_value为0的情况
        if initial_value == 0:
            total_return = 0.0
        else:
            total_return = (final_value - initial_value) / initial_value

        # 计算年化收益率，避免复数结果
        if len(dates) >= 2:
            days = max((dates[-1] - dates[0]).days, 1)
        else:
            days = 1

        annualized_return = 0.0
        try:
            # 当(1+total_return) <= 0时，幂运算可能产生复数；将年化收益设置为-1.0表示完全亏损或无意义
            if (1 + total_return) <= 0 or final_value <= 0:
                annualized_return = -1.0
            else:
                annualized_return = ((1 + total_return) ** (365.0 / days) - 1)
        except Exception:
            annualized_return = 0.0
        
        # 计算最大回撤
        max_drawdown, drawdown_duration = self._calculate_max_drawdown(portfolio_values, dates)
        
        # 计算夏普比率
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_values, dates)
        
        # 计算索提诺比率
        sortino_ratio = self._calculate_sortino_ratio(portfolio_values, dates)
        
        # 计算交易统计
        trade_stats = self._calculate_trade_stats()
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'drawdown_duration_days': drawdown_duration,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_trades': len(self.trades),
            'win_rate': trade_stats['win_rate'],
            'profit_factor': trade_stats['profit_factor'],
            'avg_trade_return': trade_stats['avg_return'],
            'commission_total': sum(trade.commission for trade in self.trades)
        }
    
    def _calculate_max_drawdown(self, values: List[float], dates: List[datetime]) -> tuple:
        """计算最大回撤"""
        try:
            if len(values) < 2:
                return 0.0, 0
                
            # 确保所有值都是有效的正数
            values = [max(v, 0.01) for v in values]
            peak = values[0]
            max_drawdown = 0.0
            drawdown_start = None
            current_drawdown_duration = 0
            max_drawdown_duration = 0
            
            for i, value in enumerate(values):
                if value > peak:
                    peak = value
                    drawdown_start = None
                    current_drawdown_duration = 0
                else:
                    # 防止除零错误
                    if peak > 0:
                        drawdown = (peak - value) / peak
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                        
                        if drawdown_start is None:
                            drawdown_start = dates[i]
                        current_drawdown_duration = (dates[i] - drawdown_start).days
                        max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
            
            # 如果结果无效，返回0
            if np.isnan(max_drawdown) or np.isinf(max_drawdown):
                return 0.0, 0
                
            return float(max_drawdown), max_drawdown_duration
            
        except Exception as e:
            print(f"计算最大回撤时出错: {str(e)}")
            return 0.0, 0
    
    def _calculate_sharpe_ratio(self, values: List[float], dates: List[datetime]) -> float:
        """计算夏普比率"""
        try:
            if len(values) < 2:
                return 0.0
            
            # 确保所有值都大于0，避免除零错误
            values = np.array([max(v, 0.01) for v in values])
            
            # 计算日收益率
            returns = np.diff(values) / values[:-1]
            
            # 如果收益率全是0或者数据无效，返回0
            if len(returns) == 0 or np.all(returns == 0) or np.any(np.isnan(returns)):
                return 0.0
            
            # 年化无风险利率（3%）
            risk_free_rate = 0.03
            daily_risk_free = risk_free_rate / 365
            
            # 计算超额收益
            excess_returns = returns - daily_risk_free
            
            # 计算标准差，如果为0则返回0
            std_dev = np.std(excess_returns)
            if std_dev == 0:
                return 0.0
            
            # 年化夏普比率
            sharpe = np.mean(excess_returns) / std_dev * np.sqrt(252)
            
            # 如果结果无效，返回0
            if np.isnan(sharpe) or np.isinf(sharpe):
                return 0.0
                
            return float(sharpe)
            
        except Exception as e:
            print(f"计算夏普比率时出错: {str(e)}")
            return 0.0
    
    def _calculate_sortino_ratio(self, values: List[float], dates: List[datetime]) -> float:
        """计算索提诺比率"""
        if len(values) < 2:
            return 0.0
        
        # 计算日收益率
        returns = np.diff(values) / values[:-1]
        
        # 年化无风险利率（3%）
        risk_free_rate = 0.03
        daily_risk_free = risk_free_rate / 365
        
        # 计算下行偏差
        downside_returns = returns[returns < daily_risk_free]
        if len(downside_returns) == 0:
            downside_std = 0
        else:
            downside_std = np.std(downside_returns)
        
        # 计算超额收益
        excess_returns = returns - daily_risk_free
        
        if len(excess_returns) == 0 or downside_std == 0:
            return 0.0
        
        # 年化索提诺比率
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        
        # 如果结果无效，返回0
        if np.isnan(sortino) or np.isinf(sortino):
            return 0.0
            
        return float(sortino)
    
    def _calculate_trade_stats(self) -> Dict[str, float]:
        """计算交易统计"""
        if not self.trades:
            return {'win_rate': 0, 'profit_factor': 0, 'avg_return': 0}
        
        # 计算每笔交易的收益
        trade_returns = []
        buy_trades = []
        
        for i in range(0, len(self.trades), 2):
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]
                
                if buy_trade.type == 'BUY' and sell_trade.type == 'SELL':
                    buy_cost = buy_trade.price * buy_trade.quantity + buy_trade.commission
                    sell_revenue = sell_trade.price * sell_trade.quantity - sell_trade.commission
                    trade_return = (sell_revenue - buy_cost) / buy_cost
                    trade_returns.append(trade_return)
                    buy_trades.append(buy_trade)
        
        if not trade_returns:
            return {'win_rate': 0, 'profit_factor': 0, 'avg_return': 0}
        
        # 计算胜率
        winning_trades = sum(1 for r in trade_returns if r > 0)
        win_rate = winning_trades / len(trade_returns)
        
        # 计算盈亏因子
        gross_profit = sum(r for r in trade_returns if r > 0)
        gross_loss = abs(sum(r for r in trade_returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 100.0  # 使用大数值代替无穷大
        
        # 平均收益率
        avg_return = np.mean(trade_returns)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_return': float(avg_return)
        }
    
    def generate_report(self, performance: Dict[str, Any]) -> str:
        """生成回测报告"""
        # 安全获取性能指标，确保没有无穷大或无效值
        initial_value = performance.get('initial_value', 0.0)
        final_value = performance.get('final_value', 0.0)
        total_return = performance.get('total_return', 0.0)
        annualized_return = performance.get('annualized_return', 0.0)
        max_drawdown = performance.get('max_drawdown', 0.0)
        drawdown_duration = performance.get('drawdown_duration_days', 0)
        sharpe_ratio = performance.get('sharpe_ratio', 0.0)
        sortino_ratio = performance.get('sortino_ratio', 0.0)
        total_trades = performance.get('total_trades', 0)
        win_rate = performance.get('win_rate', 0.0)
        profit_factor = performance.get('profit_factor', 0.0)
        avg_trade_return = performance.get('avg_trade_return', 0.0)
        commission_total = performance.get('commission_total', 0.0)
        
        # 确保所有数值都是有效的，避免无穷大和NaN
        def safe_float(value, default=0.0):
            if np.isnan(value) or np.isinf(value):
                return default
            return float(value)
        
        report = f"""
股票策略回测报告
================

基本性能指标
------------
- 初始资金: ¥{safe_float(initial_value):,.2f}
- 最终资金: ¥{safe_float(final_value):,.2f}
- 总收益率: {safe_float(total_return):.2%}
- 年化收益率: {safe_float(annualized_return):.2%}
- 最大回撤: {safe_float(max_drawdown):.2%}
- 回撤持续时间: {int(drawdown_duration)} 天

风险调整收益
------------
- 夏普比率: {safe_float(sharpe_ratio):.2f}
- 索提诺比率: {safe_float(sortino_ratio):.2f}

交易统计
--------
- 总交易次数: {int(total_trades)}
- 胜率: {safe_float(win_rate):.1%}
- 盈亏因子: {safe_float(profit_factor):.2f}
- 平均交易收益率: {safe_float(avg_trade_return):.2%}
- 总佣金费用: ¥{safe_float(commission_total):.2f}

交易明细
--------
"""
        
        for i, trade in enumerate(self.trades):
            report += f"{i+1}. {trade.date.strftime('%Y-%m-%d')} - {trade.type} "
            report += f"{trade.quantity}股 @ ¥{trade.price:.2f}, 佣金: ¥{trade.commission:.2f}\n"
            report += f"   理由: {trade.reason}\n"
        
        return report

# 使用示例
if __name__ == "__main__":
    # 示例数据
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    data = {
        'Open': np.random.normal(100, 5, 100).cumsum(),
        'High': np.random.normal(105, 5, 100).cumsum(),
        'Low': np.random.normal(95, 5, 100).cumsum(),
        'Close': np.random.normal(100, 5, 100).cumsum(),
        'Volume': np.random.randint(1000, 10000, 100)
    }
    df = pd.DataFrame(data, index=dates)
    
    # 示例信号
    signals = [
        {
            'date': dates[10],
            'type': 'BUY',
            'price': df['Close'].iloc[10],
            'reason': '测试买入信号',
            'confidence': 0.8
        },
        {
            'date': dates[50],
            'type': 'SELL',
            'price': df['Close'].iloc[50],
            'reason': '测试卖出信号',
            'confidence': 0.7
        }
    ]
    
    # 运行回测
    engine = BacktestEngine(initial_capital=100000)
    result = engine.run_backtest(df, signals)
    
    # 生成报告
    report = engine.generate_report(result['performance'])
    print(report)