#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段4优化：添加基本面特征 + 自动化特征选择 + Optuna超参数优化

基于V1训练框架（train_unified_models.py），整合：
1. 13个可用基本面特征（从MySQL获取）
2. 自动特征选择（FeatureSelectorOptimizer）
3. Optuna超参数优化（100 trials）
4. 90天预测周期（已验证比30天更好）

可用基本面特征：
- 每日: pe_ttm, pb
- 季度: revenue, net_profit, net_profit_margin, roe, epsm, 
         revenue-yoy, net_profit_yoy, debt_to_asset, current_ratio, quick_ratio
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

from src.data.db.unified_database_manager import UnifiedDatabaseManager
from src.ml.training.train_unified_models import UnifiedModelTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FundamentalFeatureBuilder:
    """基本面特征构建器 - 从MySQL提取13个可用特征"""
    
    def __init__(self):
        self.db_manager = UnifiedDatabaseManager()
        
        # 13个可用基本面特征
        self.daily_features = ['pe_ttm', 'pb']
        self.quarterly_features = [
            'revenue', 'net_profit', 'net_profit_margin', 'roe', 'epsm',
            'revenue_yoy', 'net_profit_yoy', 'debt_to_asset', 
            'current_ratio', 'quick_ratio'
        ]
    
    def fetch_fundamental_features(self, symbols: List[str], 
                                   start_date: str, end_date: str) -> pd.DataFrame:
        """
        提取基本面特征
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
        
        Returns:
            包含基本面特征的DataFrame
        """
        logger.info(f"开始提取基本面特征: {len(symbols)} 只股票, {start_date} ~ {end_date}")
        
        all_data = []
        
        for symbol in symbols:
            try:
                # 获取每日估值特征（pe_ttm, pb）
                daily_query = f"""
                SELECT 
                    trade_date as date,
                    '{symbol}' as symbol,
                    pe_ttm,
                    pb
                FROM stock_daily_{symbol.replace('.', '_')}
                WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY trade_date
                """
                
                daily_df = self.db_manager.execute_query(daily_query)
                
                if daily_df.empty:
                    logger.warning(f"{symbol}: 无每日数据")
                    continue
                
                # 获取季度基本面特征
                quarterly_query = f"""
                SELECT 
                    report_date as date,
                    revenue,
                    net_profit,
                    net_profit_margin,
                    roe,
                    epsm,
                    revenue_yoy,
                    net_profit_yoy,
                    debt_to_asset,
                    current_ratio,
                    quick_ratio
                FROM stock_quarterly_fundamentals_{symbol.replace('.', '_')}
                WHERE report_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY report_date
                """
                
                quarterly_df = self.db_manager.execute_query(quarterly_query)
                
                # 季度数据转为日频（前向填充）
                if not quarterly_df.empty:
                    # 转换日期
                    daily_df['date'] = pd.to_datetime(daily_df['date'])
                    quarterly_df['date'] = pd.to_datetime(quarterly_df['date'])
                    
                    # 合并每日和季度数据
                    merged = pd.merge_asof(
                        daily_df.sort_values('date'),
                        quarterly_df.sort_values('date'),
                        on='date',
                        direction='backward'  # 使用最近公布的季报数据
                    )
                else:
                    merged = daily_df
                    # 添加空的季度特征列
                    for feat in self.quarterly_features:
                        merged[feat] = np.nan
                
                all_data.append(merged)
                
                if len(all_data) % 10 == 0:
                    logger.info(f"已处理 {len(all_data)}/{len(symbols)} 只股票")
                
            except Exception as e:
                logger.error(f"{symbol} 基本面数据提取失败: {e}")
                continue
        
        if not all_data:
            logger.error("未能提取任何基本面数据")
            return pd.DataFrame()
        
        # 合并所有股票数据
        result = pd.concat(all_data, ignore_index=True)
        
        # 数据类型转换
        for col in result.columns:
            if col not in ['date', 'symbol']:
                result[col] = pd.to_numeric(result[col], errors='coerce')
        
        logger.info(f"基本面特征提取完成: {len(result)} 条记录, {len(result.columns)} 列")
        logger.info(f"特征列: {list(result.columns)}")
        
        return result


def build_combined_features(symbols: List[str], start_date: str, end_date: str,
                           prediction_period: int = 90) -> Tuple[pd.DataFrame, Dict]:
    """
    构建组合特征：12个技术特征 + 13个基本面特征 + 超额收益标签
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        prediction_period: 预测周期（天）
    
    Returns:
        (特征DataFrame, 标签Dict)
    """
    logger.info("=" * 80)
    logger.info("开始构建组合特征（技术+基本面+超额收益）")
    logger.info("=" * 80)
    
    # 1. 使用UnifiedModelTrainer准备技术特征和超额收益标签
    logger.info("\n[1/3] 准备技术特征和超额收益标签...")
    trainer = UnifiedModelTrainer(
        enable_feature_selection=False,  # 先不做特征选择，获取全部特征
        reuse_feature_selection=False
    )
    
    # 准备数据（会自动调用超额收益标签生成）
    # 注意：这里复用 train_excess_return_features.py 的逻辑
    from scripts.train_excess_return_features import fetch_and_store_market_index, build_excess_return_features
    
    # 提取市场指数数据
    market_data = fetch_and_store_market_index(start_date, end_date)
    
    # 构建超额收益特征和标签
    X_tech, labels = build_excess_return_features(
        symbols, start_date, end_date, 
        market_data, prediction_period
    )
    
    if X_tech is None or labels is None:
        logger.error("技术特征构建失败")
        return None, None
    
    logger.info(f"技术特征: {X_tech.shape[0]} 样本, {X_tech.shape[1]} 特征")
    logger.info(f"技术特征列: {list(X_tech.columns[:15])}...")  # 显示前15列
    
    # 2. 提取基本面特征
    logger.info("\n[2/3] 提取基本面特征...")
    fundamental_builder = FundamentalFeatureBuilder()
    fundamental_df = fundamental_builder.fetch_fundamental_features(
        symbols, start_date, end_date
    )
    
    if fundamental_df.empty:
        logger.warning("基本面特征为空，仅使用技术特征")
        return X_tech, labels
    
    # 3. 合并技术特征和基本面特征
    logger.info("\n[3/3] 合并技术特征和基本面特征...")
    
    # 重置X_tech索引，确保有 date 和 symbol 列
    if 'date' not in X_tech.columns:
        X_tech = X_tech.reset_index()
    
    # 确保日期列格式一致
    X_tech['date'] = pd.to_datetime(X_tech['date'])
    fundamental_df['date'] = pd.to_datetime(fundamental_df['date'])
    
    # 按 symbol 和 date 合并
    X_combined = pd.merge(
        X_tech,
        fundamental_df,
        on=['symbol', 'date'],
        how='left'
    )
    
    # 基本面特征缺失值处理（前向填充 + 行业中位数）
    fundamental_cols = fundamental_builder.daily_features + fundamental_builder.quarterly_features
    
    for col in fundamental_cols:
        if col in X_combined.columns:
            # 按股票分组前向填充（使用历史最近值）
            X_combined[col] = X_combined.groupby('symbol')[col].ffill()
            # 剩余缺失值用行业中位数填充
            if X_combined[col].isna().sum() > 0:
                median_val = X_combined[col].median()
                X_combined[col].fillna(median_val, inplace=True)
    
    # 移除 symbol 列（训练时不需要）
    if 'symbol' in X_combined.columns:
        X_combined = X_combined.drop(columns=['symbol'])
    
    # 设置 date 为索引
    if 'date' in X_combined.columns:
        X_combined = X_combined.set_index('date')
    
    logger.info(f"\n合并完成:")
    logger.info(f"  - 最终样本数: {X_combined.shape[0]}")
    logger.info(f"  - 技术特征: {X_tech.shape[1] - 2} 个")  # 减去 date, symbol
    logger.info(f"  - 基本面特征: {len(fundamental_cols)} 个")
    logger.info(f"  - 总特征数: {X_combined.shape[1]} 个")
    logger.info(f"  - 特征列: {list(X_combined.columns)}")
    
    return X_combined, labels


def main():
    parser = argparse.ArgumentParser(description="阶段4优化：基本面特征 + 特征选择 + 超参数优化")
    
    # 数据参数
    parser.add_argument("--symbol-file", type=str, required=True,
                       help="股票代码文件路径")
    parser.add_argument("--start-date", type=str, required=True,
                       help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                       help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--prediction-period", type=int, default=90,
                       help="预测周期（天），默认90天")
    parser.add_argument("--excess-threshold", type=float, default=0.02,
                       help="超额收益阈值，默认2%")
    
    # 特征选择参数
    parser.add_argument("--feature-selection-method", type=str, 
                       default='auto',
                       choices=['ensemble', 'importance', 'stability', 'auto'],
                       help="特征选择方法")
    parser.add_argument("--target-features", type=int, default=30,
                       help="目标特征数量，默认30")
    parser.add_argument("--importance-model", type=str, default='xgb',
                       choices=['rf', 'xgb', 'lgb', 'auto'],
                       help="特征重要性计算模型")
    
    # 超参数优化参数
    parser.add_argument("--use-optuna", action='store_true', default=True,
                       help="使用Optuna超参数优化（默认开启）")
    parser.add_argument("--optimization-trials", type=int, default=100,
                       help="Optuna优化迭代次数，默认100")
    parser.add_argument("--optimize-target", type=str, default='ic',
                       choices=['ic', 'r2', 'mse'],
                       help="优化目标指标")
    
    # 其他参数
    parser.add_argument("--output-dir", type=str, 
                       default="results/stage4_fundamental",
                       help="结果输出目录")
    parser.add_argument("--no-feature-selection", action='store_true',
                       help="禁用特征选择（使用全部25个特征）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("阶段4优化：基本面特征 + 自动化特征选择 + Optuna超参数优化")
    logger.info("=" * 80)
    logger.info(f"\n配置:")
    logger.info(f"  - 股票文件: {args.symbol_file}")
    logger.info(f"  - 日期范围: {args.start_date} ~ {args.end_date}")
    logger.info(f"  - 预测周期: {args.prediction_period} 天")
    logger.info(f"  - 超额收益阈值: {args.excess_threshold * 100}%")
    logger.info(f"  - 特征选择: {'禁用' if args.no_feature_selection else args.feature_selection_method}")
    logger.info(f"  - 目标特征数: {args.target_features if not args.no_feature_selection else '全部'}")
    logger.info(f"  - 超参数优化: {'Optuna' if args.use_optuna else '禁用'}")
    logger.info(f"  - 优化目标: {args.optimize_target.upper()}")
    logger.info(f"  - 优化迭代: {args.optimization_trials}")
    
    # 1. 读取股票列表
    logger.info("\n" + "=" * 80)
    logger.info("步骤1: 读取股票列表")
    logger.info("=" * 80)
    
    with open(args.symbol_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    logger.info(f"读取到 {len(symbols)} 只股票")
    logger.info(f"示例: {symbols[:5]}")
    
    # 2. 构建组合特征（技术 + 基本面 + 超额收益标签）
    logger.info("\n" + "=" * 80)
    logger.info("步骤2: 构建组合特征")
    logger.info("=" * 80)
    
    X, labels = build_combined_features(
        symbols, args.start_date, args.end_date, args.prediction_period
    )
    
    if X is None or labels is None:
        logger.error("特征构建失败，退出")
        return
    
    logger.info(f"\n组合特征构建完成:")
    logger.info(f"  - 样本数: {X.shape[0]}")
    logger.info(f"  - 特征数: {X.shape[1]}")
    logger.info(f"  - 分类标签: {labels['classification'].sum()} 正样本 / {len(labels['classification'])} 总样本")
    logger.info(f"  - 回归标签均值: {labels['regression'].mean():.4f}")
    
    # 3. 使用 UnifiedModelTrainer 训练（包含自动特征选择和超参数优化）
    logger.info("\n" + "=" * 80)
    logger.info("步骤3: 模型训练（特征选择 + 超参数优化）")
    logger.info("=" * 80)
    
    # 设置环境变量（优化目标）
    os.environ['OPTIMIZE_TARGET'] = args.optimize_target
    
    trainer = UnifiedModelTrainer(
        enable_feature_selection=(not args.no_feature_selection),
        reuse_feature_selection=False,  # 不复用缓存，每次重新选择
        refresh_feature_selection=True,
        feature_selection_n_jobs=-1,
        importance_model=args.importance_model
    )
    
    # 设置特征选择策略
    if not args.no_feature_selection:
        trainer.feature_selection_strategy = args.feature_selection_method
        trainer.target_n_features = args.target_features
    
    # 训练分类模型
    logger.info("\n" + "-" * 80)
    logger.info("训练分类模型（预测超额收益方向）")
    logger.info("-" * 80)
    
    clf_results = trainer.train_classification_models(
        X, labels['classification'],
        use_grid_search=False,  # 禁用GridSearch，使用Optuna
        use_optuna=args.use_optuna,
        optimization_trials=args.optimization_trials
    )
    
    # 训练回归模型
    logger.info("\n" + "-" * 80)
    logger.info("训练回归模型（预测超额收益大小）")
    logger.info("-" * 80)
    
    reg_results = trainer.train_regression_models(
        X, labels['regression'],
        use_grid_search=False,
        use_optuna=args.use_optuna,
        optimization_trials=args.optimization_trials
    )
    
    # 4. 保存结果
    logger.info("\n" + "=" * 80)
    logger.info("步骤4: 保存结果")
    logger.info("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存分类结果
    clf_output = output_dir / f"classification_results_{timestamp}.txt"
    with open(clf_output, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("阶段4优化 - 分类模型结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"配置:\n")
        f.write(f"  - 预测周期: {args.prediction_period}天\n")
        f.write(f"  - 特征数: {X.shape[1]}\n")
        f.write(f"  - 特征选择: {args.feature_selection_method if not args.no_feature_selection else '禁用'}\n")
        f.write(f"  - 超参数优化: {'Optuna' if args.use_optuna else '禁用'}\n\n")
        f.write(str(clf_results))
    
    # 保存回归结果
    reg_output = output_dir / f"regression_results_{timestamp}.txt"
    with open(reg_output, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("阶段4优化 - 回归模型结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"配置:\n")
        f.write(f"  - 预测周期: {args.prediction_period}天\n")
        f.write(f"  - 特征数: {X.shape[1]}\n")
        f.write(f"  - 特征选择: {args.feature_selection_method if not args.no_feature_selection else '禁用'}\n")
        f.write(f"  - 超参数优化: {'Optuna' if args.use_optuna else '禁用'}\n\n")
        f.write(str(reg_results))
    
    logger.info(f"\n结果已保存:")
    logger.info(f"  - 分类结果: {clf_output}")
    logger.info(f"  - 回归结果: {reg_output}")
    
    logger.info("\n" + "=" * 80)
    logger.info("阶段4优化完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
