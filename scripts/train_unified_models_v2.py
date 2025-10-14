# -*- coding: utf-8 -*-
"""
统一模型训练入口 V2
整合新特征体系和增强训练器
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from collections import OrderedDict

# 添加项目根路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.prediction_config import *
from src.ml.features.unified_feature_builder import UnifiedFeatureBuilder
from src.ml.training.enhanced_trainer_v2 import EnhancedTrainerV2
from src.data.unified_data_access import UnifiedDataAccessLayer

# 🔧 导入修复函数
from src.ml.training.toolkit import (
    add_labels_corrected,
    evaluate_by_month,
    get_conservative_lgbm_params,
    get_conservative_xgb_params,
    improved_time_series_split
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/train_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_training_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    prediction_period: int = PREDICTION_PERIOD_DAYS,
    classification_strategy: str = LABEL_STRATEGY,
    label_quantile: float = LABEL_POSITIVE_QUANTILE,
    label_min_samples: int = LABEL_MIN_SAMPLES_PER_DATE,
    enable_fundamental: bool = False
) -> pd.DataFrame:
    """
    准备训练数据
    
    采用混合价格策略：
    - 特征构建：使用不复权价格（保持技术指标准确性）
    - 标签计算：使用前复权价格（反映真实投资收益）
    
    Parameters
    ----------
    symbols : List[str]
        股票代码列表
    start_date : str
        开始日期
    end_date : str
        结束日期
    prediction_period : int
        预测周期（天）
    
    Returns
    -------
    df : DataFrame
        特征+标签数据
    """
    logger.info("="*80)
    logger.info("准备训练数据（V2增强版）")
    logger.info("="*80)
    logger.info(f"股票数量: {len(symbols)}")
    logger.info(f"日期范围: {start_date} ~ {end_date}")
    logger.info(f"预测周期: {prediction_period}天")
    logger.info(f"价格策略: 特征用不复权 + 标签用前复权")
    logger.info(f"标签策略: {classification_strategy} (quantile={label_quantile:.2f})")
    logger.info(f"基本面特征: {'启用' if enable_fundamental else '禁用'}")

    from src.data.unified_data_access import UnifiedDataAccessLayer, DataAccessConfig
    config = DataAccessConfig()
    config.use_cache = True  # ✅ 缓存系统正常工作（L1 Redis + L2 Parquet都已验证）
    config.auto_sync = False  # ✅ 训练模式禁用外部同步,仅使用数据库数据
    data_access = UnifiedDataAccessLayer(config=config)
    logger.info("✅ 缓存已启用, 外部同步已禁用(仅使用数据库数据)")

    from src.data.db.unified_database_manager import UnifiedDatabaseManager
    db_manager = UnifiedDatabaseManager()

    builder = UnifiedFeatureBuilder(
        data_access=data_access,
        db_manager=db_manager,
        enable_fundamental=enable_fundamental
    )

    market_generator = builder.market_generator
    board_generator = builder.board_generator

    def _standardize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
        """标准化日线数据格式，便于特征与市场因子复用"""
        tmp = df.copy()
        tmp.columns = tmp.columns.str.lower()

        if 'date' in tmp.columns:
            tmp['date'] = pd.to_datetime(tmp['date'])
            tmp.set_index('date', inplace=True)
        elif not isinstance(tmp.index, pd.DatetimeIndex):
            tmp.index = pd.to_datetime(tmp.index, errors='coerce')

        if isinstance(tmp.index, pd.DatetimeIndex):
            tmp.index.name = 'date'

        tmp.sort_index(inplace=True)

        if 'close' in tmp.columns and 'ret' not in tmp.columns:
            tmp['ret'] = tmp['close'].pct_change(fill_method=None)

        return tmp

    all_data: List[pd.DataFrame] = []
    failed_symbols: List[tuple] = []
    quality_stats = {
        'total_processed': 0,
        'data_insufficient': 0,
        'feature_build_failed': 0,
        'qfq_data_failed': 0,
        'qfq_negative_filtered': 0,
        'qfq_extreme_filtered': 0,
        'no_valid_labels': 0,
        'success': 0
    }

    price_frames: Dict[str, pd.DataFrame] = {}
    qfq_frames: Dict[str, pd.DataFrame] = {}

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] 准备原始数据 {symbol}")
        quality_stats['total_processed'] += 1

        try:
            price_df = data_access.get_stock_data(
                symbol,
                start_date,
                end_date,
                adjust_mode='none'
            )

            # 🔧 使用 MIN_TRAINING_DAYS 作为数据足够性检查阈值（而非 LOOKBACK_DAYS）
            if price_df is None or len(price_df) < MIN_TRAINING_DAYS:
                logger.warning(f"  跳过: 不复权数据不足 ({len(price_df) if price_df is not None else 0} < {MIN_TRAINING_DAYS})")
                failed_symbols.append((symbol, 'data_insufficient'))
                quality_stats['data_insufficient'] += 1
                continue

            std_price_df = _standardize_price_frame(price_df)
            if len(std_price_df) < MIN_TRAINING_DAYS:
                logger.warning(f"  跳过: 标准化后可用数据不足 ({len(std_price_df)} < {MIN_TRAINING_DAYS})")
                failed_symbols.append((symbol, 'data_insufficient'))
                quality_stats['data_insufficient'] += 1
                continue

            price_frames[symbol] = std_price_df

            qfq_df = data_access.get_stock_data(
                symbol,
                start_date,
                end_date,
                adjust_mode='qfq'
            )

            if qfq_df is None or len(qfq_df) == 0:
                logger.warning("  跳过: 前复权数据获取失败")
                failed_symbols.append((symbol, 'qfq_data_failed'))
                quality_stats['qfq_data_failed'] += 1
                price_frames.pop(symbol, None)
                continue

            qfq_df = qfq_df.copy()
            if 'date' not in qfq_df.columns:
                qfq_df = qfq_df.reset_index()
            if 'date' not in qfq_df.columns:
                logger.warning("  跳过: 前复权数据缺少日期列")
                failed_symbols.append((symbol, 'qfq_data_failed'))
                quality_stats['qfq_data_failed'] += 1
                price_frames.pop(symbol, None)
                continue

            qfq_df['date'] = pd.to_datetime(qfq_df['date'])
            qfq_df.sort_values('date', inplace=True)

            initial_qfq_count = len(qfq_df)

            qfq_negative_mask = (qfq_df['close'] < 0)
            if qfq_negative_mask.sum() > 0:
                logger.warning(f"  发现 {qfq_negative_mask.sum()} 条负数前复权价格")
                quality_stats['qfq_negative_filtered'] += qfq_negative_mask.sum()
                qfq_df = qfq_df[~qfq_negative_mask]

            qfq_extreme_mask = (qfq_df['close'] > 10000) | (qfq_df['close'] < 0.01)
            if qfq_extreme_mask.sum() > 0:
                logger.warning(f"  发现 {qfq_extreme_mask.sum()} 条极端前复权价格")
                quality_stats['qfq_extreme_filtered'] += qfq_extreme_mask.sum()
                qfq_df = qfq_df[~qfq_extreme_mask]

            filtered_count = initial_qfq_count - len(qfq_df)
            if filtered_count > initial_qfq_count * 0.3:
                logger.warning(f"  跳过: 前复权数据质量差 (过滤{filtered_count}/{initial_qfq_count})")
                failed_symbols.append((symbol, 'qfq_quality_poor'))
                price_frames.pop(symbol, None)
                continue

            # 🔧 前复权数据足够性也使用 MIN_TRAINING_DAYS
            if len(qfq_df) < MIN_TRAINING_DAYS:
                logger.warning(f"  跳过: 前复权数据不足 (过滤后{len(qfq_df)} < {MIN_TRAINING_DAYS})")
                failed_symbols.append((symbol, 'qfq_insufficient'))
                price_frames.pop(symbol, None)
                continue

            qfq_frames[symbol] = qfq_df.reset_index(drop=True)

        except Exception as exc:
            logger.error(f"  原始数据准备失败: {exc}", exc_info=True)
            failed_symbols.append((symbol, 'exception'))
            price_frames.pop(symbol, None)
            qfq_frames.pop(symbol, None)

    active_symbols = [s for s in symbols if s in price_frames and s in qfq_frames]
    
    logger.info(f"准备构建市场因子:")
    logger.info(f"  price_frames 数量: {len(price_frames)}")
    logger.info(f"  qfq_frames 数量: {len(qfq_frames)}")
    logger.info(f"  active_symbols 数量: {len(active_symbols)}")
    logger.info(f"  market_generator: {'已初始化' if market_generator is not None else '未初始化 ❌'}")
    logger.info(f"  MIN_STOCKS_FOR_MARKET: {MIN_STOCKS_FOR_MARKET}")

    market_returns = None
    if market_generator is not None and len(price_frames) > 0:
        try:
            logger.info(f"构建全局市场因子 (输入 {len(price_frames)} 只股票)...")
            # 🔧 关键修复: 传入所有股票的价格数据，而非单只股票
            market_returns = market_generator.build_market_returns(
                price_frames,  # 使用完整的 price_frames 字典
                min_stocks=MIN_STOCKS_FOR_MARKET
            )
            
            if market_returns is not None and len(market_returns) > 0:
                logger.info(f"✅ 市场因子构建成功: {len(market_returns)} 天, 平均 {market_returns.get('count', pd.Series([0])).mean():.0f} 只股票/天")
            else:
                logger.warning(f"⚠️ 市场因子为空 (min_stocks={MIN_STOCKS_FOR_MARKET})，尝试降低阈值...")
                # 动态降低阈值: 取股票数的 10% 或至少 10 只
                fallback_threshold = max(10, len(price_frames) // 10)
                logger.info(f"   重试使用阈值: {fallback_threshold}")
                market_returns = market_generator.build_market_returns(
                    price_frames,
                    min_stocks=fallback_threshold
                )
                if market_returns is not None and len(market_returns) > 0:
                    logger.info(f"✅ 降低阈值后成功: {len(market_returns)} 天")
                else:
                    logger.warning("❌ 即使降低阈值仍无法构建市场因子")
                    market_returns = None
        except Exception as exc:
            logger.error(f"市场因子构建异常: {exc}", exc_info=True)
            market_returns = None

    for j, symbol in enumerate(active_symbols, 1):
        try:
            logger.info(f"[{j}/{len(active_symbols)}] 生成特征 {symbol}")

            features_df = builder.build_features_from_dataframe(price_frames[symbol], symbol)
            if features_df is None or len(features_df) == 0:
                logger.warning("  跳过: 特征构建失败")
                failed_symbols.append((symbol, 'feature_build_failed'))
                quality_stats['feature_build_failed'] += 1
                continue

            if market_generator is not None and market_returns is not None:
                try:
                    market_enriched = market_generator.add_market_features(
                        price_frames[symbol].copy(),
                        symbol,
                        market_returns
                    )
                    candidate_cols = ['MKT'] + market_generator.get_feature_names()
                    available_cols: List[str] = []
                    for col in candidate_cols:
                        if col in market_enriched.columns and col not in available_cols:
                            available_cols.append(col)
                    if available_cols:
                        market_slice = market_enriched.reset_index()[['date'] + available_cols]
                        features_df = features_df.merge(market_slice, on='date', how='left')
                except Exception as market_exc:
                    logger.warning(f"  市场特征添加失败: {market_exc}")

            features_df['symbol'] = symbol
            if board_generator is not None:
                try:
                    features_df = board_generator.add_board_feature(features_df, symbol_col='symbol')
                except Exception as board_exc:
                    logger.warning(f"  板块特征添加失败: {board_exc}")

            # 🔧 关键修复：确保features_df中有date列
            if 'date' not in features_df.columns:
                # 如果date在索引中，转为列
                if features_df.index.name == 'date' or isinstance(features_df.index, pd.DatetimeIndex):
                    features_df = features_df.reset_index()
                    if 'index' in features_df.columns and 'date' not in features_df.columns:
                        features_df.rename(columns={'index': 'date'}, inplace=True)
                    logger.info(f"  ℹ️ date从索引转为列")
                else:
                    logger.error(f"  ❌ 无法找到date列，跳过此股票")
                    logger.error(f"     索引名: {features_df.index.name}, 列: {list(features_df.columns[:10])}")
                    failed_symbols.append((symbol, 'no_date_column'))
                    continue

            # 🔧 使用修正的标签构建函数
            # 注意: 标签采用前复权价格，原始价格用于对照
            try:
                # 准备原始价格数据（不复权）
                price_raw = price_frames[symbol].copy()
                if price_raw.index.name == 'date' or isinstance(price_raw.index, pd.DatetimeIndex):
                    price_raw = price_raw.reset_index()
                    if 'index' in price_raw.columns and 'date' not in price_raw.columns:
                        price_raw.rename(columns={'index': 'date'}, inplace=True)

                if 'close' not in price_raw.columns:
                    logger.error(f"  ❌ price_raw中没有close列: {list(price_raw.columns)}")
                    failed_symbols.append((symbol, 'no_close_column'))
                    continue

                price_raw = price_raw[['date', 'close']].copy()
                price_raw['symbol'] = symbol

                # 准备前复权价格数据
                price_adj = qfq_frames[symbol].copy()
                if 'date' not in price_adj.columns:
                    price_adj = price_adj.reset_index()

                if 'close' not in price_adj.columns:
                    logger.error(f"  ❌ 前复权数据缺少close列: {list(price_adj.columns)}")
                    failed_symbols.append((symbol, 'no_close_column_qfq'))
                    continue

                price_adj['date'] = pd.to_datetime(price_adj['date'])
                price_adj = price_adj[['date', 'close']].copy()
                price_adj['symbol'] = symbol

                # 使用修正的标签构建函数
                features_with_labels = add_labels_corrected(
                    features_df=features_df,
                    price_data=price_adj,
                    prediction_period=prediction_period,
                    threshold=CLS_THRESHOLD,  # absolute 策略兜底
                    price_data_raw=price_raw,
                    classification_strategy=classification_strategy,
                    quantile=label_quantile,
                    min_samples_per_date=label_min_samples,
                    negative_quantile=LABEL_NEGATIVE_QUANTILE,
                    enable_neutral_band=ENABLE_LABEL_NEUTRAL_BAND,
                    neutral_quantile=LABEL_NEUTRAL_QUANTILE,
                    market_returns=market_returns,
                    use_market_baseline=LABEL_USE_MARKET_BASELINE,
                    use_industry_neutral=LABEL_USE_INDUSTRY_NEUTRAL
                )
                features_df = features_with_labels
            except Exception as label_exc:
                logger.warning(f"  标签构建失败: {label_exc}")
                failed_symbols.append((symbol, 'label_build_failed'))
                continue

            if len(features_df) == 0:
                logger.warning("  跳过: 无有效标签")
                failed_symbols.append((symbol, 'no_valid_labels'))
                quality_stats['no_valid_labels'] += 1
                continue

            # 已在add_labels_corrected中处理,这里可选择性二次过滤
            extreme_return_mask = features_df['label_reg'].abs() > 1.0
            if extreme_return_mask.sum() > 0:
                logger.warning(f"  过滤 {extreme_return_mask.sum()} 条极端收益率记录(>100%)")
                features_df = features_df[~extreme_return_mask]

            if len(features_df) == 0:
                logger.warning("  跳过: 过滤后无数据")
                failed_symbols.append((symbol, 'all_filtered'))
                continue

            all_data.append(features_df)
            quality_stats['success'] += 1
            logger.info(f"  ✓ 成功: {len(features_df)} 条记录 (正样本率: {features_df['label_cls'].mean():.1%})")

        except Exception as exc:
            logger.error(f"  特征生成失败: {exc}", exc_info=True)
            failed_symbols.append((symbol, 'exception'))
    
    # 合并数据
    if len(all_data) == 0:
        raise ValueError("没有可用的训练数据，所有股票处理失败")
    
    df = pd.concat(all_data, ignore_index=True)

    # 统一执行行业中性残差计算，确保使用跨股票截面信息
    if LABEL_USE_INDUSTRY_NEUTRAL:
        base_col = None
        if LABEL_USE_MARKET_BASELINE and 'future_excess_return' in df.columns:
            base_col = 'future_excess_return'
        elif 'future_return' in df.columns:
            base_col = 'future_return'

        if base_col is None:
            logger.warning("行业中性处理跳过: 未找到未来收益列")
        elif 'industry' not in df.columns:
            logger.warning("行业中性处理跳过: 数据缺少industry列")
        else:
            grouped = df.groupby(['date', 'industry'])[base_col]
            group_counts = grouped.transform('count')
            min_required = max(min(label_min_samples, 3), 2)
            industry_mean = grouped.transform('mean')
            residual_series = df[base_col] - industry_mean
            sufficient_mask = group_counts >= min_required

            df['future_residual_return'] = np.where(sufficient_mask, residual_series, np.nan)

            updated_rows = int(sufficient_mask.sum())
            if updated_rows > 0:
                df.loc[sufficient_mask, 'label_reg'] = df.loc[sufficient_mask, 'future_residual_return']
                logger.info(
                    "行业中性已应用: %d 条记录 (阈值: >=%d 同日同行业样本)",
                    updated_rows,
                    min_required
                )
            else:
                logger.info(
                    "行业中性未应用: 所有日期同行业样本数不足 %d 条，保留原始收益标签",
                    min_required
                )

    # 截面标准化/排序增强特征
    if ENABLE_CROSS_SECTIONAL_ENRICHMENT and 'date' in df.columns:
        logger.info("构建截面增强特征 (Z-score / Rank)...")
        available_cols = [col for col in CROSS_SECTIONAL_FEATURES if col in df.columns]
        if available_cols:
            grouped = df.groupby('date', group_keys=False)

            def _zscore(series: pd.Series) -> pd.Series:
                mu = series.mean()
                sigma = series.std(ddof=0)
                if np.isnan(mu) or sigma == 0 or np.isnan(sigma):
                    return pd.Series(np.nan, index=series.index)
                return (series - mu) / (sigma + 1e-9)

            for col in available_cols:
                z_col = f'cs_z_{col}'
                rank_col = f'cs_rank_{col}'
                df[z_col] = grouped[col].transform(_zscore)
                df[rank_col] = grouped[col].transform(lambda x: x.rank(pct=True, method='average'))

            logger.info("  截面增强列: %d 个", len(available_cols) * 2)
        else:
            logger.info("  截面增强跳过：无可用基础列")
    
    # ========== 输出数据质量报告 ==========
    logger.info("\n" + "="*80)
    logger.info("数据准备完成 - 质量报告")
    logger.info("="*80)
    logger.info(f"✅ 成功股票: {quality_stats['success']}/{quality_stats['total_processed']} ({quality_stats['success']/quality_stats['total_processed']*100:.1f}%)")
    logger.info(f"📊 总记录数: {len(df):,}")
    logger.info(f"📅 日期范围: {df['date'].min()} ~ {df['date'].max()}")
    logger.info(f"📈 正样本率: {df['label_cls'].mean():.2%}")
    logger.info(f"📉 平均收益: {df['label_reg'].mean():.4f}")
    logger.info(f"📊 收益标准差: {df['label_reg'].std():.4f}")
    
    logger.info("\n失败统计:")
    logger.info(f"  数据不足: {quality_stats['data_insufficient']}")
    logger.info(f"  特征构建失败: {quality_stats['feature_build_failed']}")
    logger.info(f"  前复权数据失败: {quality_stats['qfq_data_failed']}")
    logger.info(f"  无有效标签: {quality_stats['no_valid_labels']}")
    
    logger.info("\n数据清洗统计:")
    logger.info(f"  前复权负数过滤: {quality_stats['qfq_negative_filtered']} 条")
    logger.info(f"  前复权极端值过滤: {quality_stats['qfq_extreme_filtered']} 条")
    
    if failed_symbols:
        logger.info(f"\n失败股票详情 (共{len(failed_symbols)}只):")
        failure_reasons = {}
        for symbol, reason in failed_symbols:
            failure_reasons[reason] = failure_reasons.get(reason, []) + [symbol]
        for reason, symbols in failure_reasons.items():
            logger.info(f"  {reason}: {len(symbols)} 只 - {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
    
    return df


def add_labels(df: pd.DataFrame, prediction_period: int) -> pd.DataFrame:
    """
    添加分类和回归标签（使用不复权价格，已弃用）
    
    注意：此函数已被 add_labels_with_qfq 替代
    保留仅用于向后兼容
    
    Parameters
    ----------
    df : DataFrame
        特征数据
    prediction_period : int
        预测周期
    
    Returns
    -------
    df : DataFrame
        添加标签后的数据
    """
    logger.warning("使用了已弃用的 add_labels 函数，请改用 add_labels_with_qfq")
    
    # 计算未来收益
    df['future_return'] = df['close'].shift(-prediction_period) / df['close'] - 1
    
    # 分类标签: 收益 > CLS_THRESHOLD
    df['label_cls'] = (df['future_return'] > CLS_THRESHOLD).astype(int)
    
    # 回归标签: 收益率
    df['label_reg'] = df['future_return']
    
    return df


def add_labels_with_qfq(
    features_df: pd.DataFrame,
    stock_data_qfq: pd.DataFrame,
    prediction_period: int
) -> pd.DataFrame:
    """
    添加分类和回归标签（使用前复权价格）
    
    将前复权价格数据与特征数据按日期对齐，计算真实投资收益率标签
    
    Parameters
    ----------
    features_df : DataFrame
        特征数据（基于不复权价格构建）
    stock_data_qfq : DataFrame
        前复权价格数据
    prediction_period : int
        预测周期（天）
    
    Returns
    -------
    df : DataFrame
        添加标签后的数据
    """
    # 确保两个DataFrame都有date列且为datetime类型
    if 'date' not in features_df.columns:
        raise ValueError("features_df 缺少 'date' 列")
    if 'date' not in stock_data_qfq.columns:
        raise ValueError("stock_data_qfq 缺少 'date' 列")
    
    features_df = features_df.copy()
    stock_data_qfq = stock_data_qfq.copy()
    
    # 确保date列为datetime类型
    features_df['date'] = pd.to_datetime(features_df['date'])
    stock_data_qfq['date'] = pd.to_datetime(stock_data_qfq['date'])
    
    # 按日期排序并重置索引
    features_df = features_df.sort_values('date').reset_index(drop=True)
    stock_data_qfq = stock_data_qfq.sort_values('date').reset_index(drop=True)
    
    # 方法1: 简单shift方法（如果数据完全对齐）
    # 先尝试通过日期merge对齐
    # 注意: stock_data_qfq可能包含close_qfq列(从UnifiedDataAccessLayer)或只有close列(测试mock数据)
    close_col = 'close_qfq' if 'close_qfq' in stock_data_qfq.columns else 'close'
    
    qfq_for_merge = stock_data_qfq[['date', close_col]].copy()
    if close_col == 'close':
        qfq_for_merge.rename(columns={'close': 'close_qfq'}, inplace=True)
    
    # 删除features_df中可能存在的复权列，避免merge冲突
    cols_to_drop = [c for c in features_df.columns if '_qfq' in c or '_hfq' in c]
    if cols_to_drop:
        features_df = features_df.drop(columns=cols_to_drop)
        logger.debug(f"  从features_df删除复权列: {cols_to_drop}")
    
    merged = features_df.merge(qfq_for_merge, on='date', how='left')
    
    # 计算未来收益率
    merged['future_close_qfq'] = merged['close_qfq'].shift(-prediction_period)
    merged['future_return'] = (merged['future_close_qfq'] - merged['close_qfq']) / merged['close_qfq']
    
    # 分类标签: 收益 > CLS_THRESHOLD
    merged['label_cls'] = (merged['future_return'] > CLS_THRESHOLD).astype(float)
    
    # 回归标签: 收益率
    merged['label_reg'] = merged['future_return']
    
    # 删除临时列
    result = merged.drop(columns=['close_qfq', 'future_close_qfq', 'future_return'])
    
    return result


def train_models(
    df: pd.DataFrame,
    model_save_dir: str = 'models/v2',
    enable_both_tasks: bool = True,
    classification_strategy: str = LABEL_STRATEGY,
    prediction_period: int = PREDICTION_PERIOD_DAYS
):
    """
    训练模型
    
    Parameters
    ----------
    df : DataFrame
        训练数据
    model_save_dir : str
        模型保存目录
    enable_both_tasks : bool
        是否训练分类和回归两个任务
    prediction_period : int
        预测周期（天数），用于模型文件命名
    """
    logger.info("="*80)
    logger.info("开始训练模型")
    logger.info(f"标签策略: {classification_strategy}")
    logger.info("="*80)
    
    # 🔧 关键修复：识别实际存在的特征列（排除标签、元数据和未来信息）
    excluded_cols = {'date', 'symbol', 'label_cls', 'label_reg', 'future_return', 'future_return_raw',
                     'future_excess_return', 'future_residual_return',
                     'open', 'high', 'low', 'close', 'volume', 'amount', 'source',
                     'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq',
                     'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq'}
    
    # 自动检测数值特征和类别特征
    numerical_features = []
    categorical_features = []
    
    for col in df.columns:
        if col in excluded_cols:
            continue
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    logger.info(f"数值特征: {len(numerical_features)} - {numerical_features[:10]}...")
    logger.info(f"类别特征: {len(categorical_features)} - {categorical_features}")
    
    # 准备特征和标签
    feature_cols = numerical_features + categorical_features
    X = df[feature_cols].copy()
    
    # 初始化训练器
    trainer = EnhancedTrainerV2(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        config={
            'use_rolling_cv': True,
            'cv_n_splits': 5
        }
    )
    
    dates_series = pd.to_datetime(df['date']) if 'date' in df.columns else None

    if 'date' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        logger.info("标签诊断: 月度样本统计")
        monthly = df.groupby(df['date'].dt.to_period('M')).agg(
            samples=('label_cls', 'count'),
            pos_rate=('label_cls', 'mean'),
            avg_return=('label_reg', 'mean')
        )
        for period, row in monthly.tail(12).iterrows():
            logger.info(
                "  %s: 样本 %5d, 正样本率 %.2f%%, 平均未来收益 %.4f",
                period.strftime('%Y-%m'),
                int(row['samples']),
                row['pos_rate'] * 100 if not np.isnan(row['pos_rate']) else float('nan'),
                row['avg_return']
            )
        recent_train = df[df['date'] < df['date'].max() - pd.Timedelta(days=PREDICTION_PERIOD_DAYS)]
        if not recent_train.empty:
            logger.info(
                "训练样本总体: %d, 正样本率 %.2f%%, 平均未来收益 %.4f",
                len(recent_train),
                recent_train['label_cls'].mean() * 100,
                recent_train['label_reg'].mean()
            )
        logger.info(
            "全样本: %d, 正样本率 %.2f%%, 平均未来收益 %.4f",
            len(df),
            df['label_cls'].mean() * 100,
            df['label_reg'].mean()
        )

        logger.info("标签对齐诊断: 分类标签与未来收益关系")
        corr = df['label_cls'].corr(df['label_reg']) if df['label_reg'].std() > 0 else float('nan')
        logger.info("  相关系数(label_cls vs future_return): %.4f", corr)
        alignment = df.groupby('label_cls')['label_reg'].agg(['count', 'mean', 'median']).rename(index={0.0: 'neg', 1.0: 'pos'})
        neg_stats = alignment.loc['neg'] if 'neg' in alignment.index else None
        pos_stats = alignment.loc['pos'] if 'pos' in alignment.index else None
        if pos_stats is not None:
            logger.info("  正类: 样本 %d, 平均未来收益 %.4f, 中位数 %.4f", int(pos_stats['count']), pos_stats['mean'], pos_stats['median'])
        if neg_stats is not None:
            logger.info("  负类: 样本 %d, 平均未来收益 %.4f, 中位数 %.4f", int(neg_stats['count']), neg_stats['mean'], neg_stats['median'])
        if classification_strategy == 'absolute':
            inconsistent = int((df['label_cls'] != (df['label_reg'] > CLS_THRESHOLD).astype(float)).sum())
            if inconsistent:
                logger.warning("  警告: 发现 %d 条标签与阈值不一致的记录，需检查对齐逻辑", inconsistent)
            else:
                logger.info("  标签与阈值逻辑一致，未发现异常")
        else:
            logger.info("  当前使用 quantile 策略，跳过绝对阈值一致性检查")

    # 训练分类模型
    y_cls = df['label_cls'].copy()
    class_counts = y_cls.value_counts(dropna=True).to_dict()
    if enable_both_tasks:
        logger.info("分类标签分布: %s", class_counts)
    unique_classes = [cls for cls in y_cls.dropna().unique()]
    train_cls = enable_both_tasks and len(unique_classes) >= 2

    if enable_both_tasks and not train_cls:
        logger.error(
            "分类标签仅包含单一类别 %s (样本数=%d)，跳过分类任务",
            unique_classes[0] if unique_classes else 'N/A',
            int(class_counts.get(unique_classes[0], 0)) if unique_classes else 0
        )

    if train_cls:
        logger.info("\n" + "="*80)
        logger.info("训练分类任务")
        logger.info("="*80)

        # 训练多个模型
        cls_models = {}
        
        # 🔧 LightGBM (使用保守参数)
        logger.info("\n训练 LightGBM 分类器 (保守参数)...")
        lgbm_params = get_conservative_lgbm_params()
        cls_lgb = trainer.train_classification_model(
            X, y_cls,
            model_type='lightgbm',
            dates=dates_series,
            **lgbm_params
        )
        cls_models['lightgbm'] = cls_lgb
        
        # Logistic Regression (基线)
        logger.info("\n训练 Logistic 分类器 (基线)...")
        cls_logistic = trainer.train_classification_model(
            X, y_cls,
            model_type='logistic',
            dates=dates_series,
            max_iter=1000
        )
        cls_models['logistic'] = cls_logistic

        # 🔧 XGBoost (使用保守参数)
        logger.info("\n训练 XGBoost 分类器 (保守参数)...")
        xgb_params = get_conservative_xgb_params()
        cls_xgb = trainer.train_classification_model(
            X, y_cls,
            model_type='xgboost',
            dates=dates_series,
            **xgb_params
        )
        cls_models['xgboost'] = cls_xgb
        
        # 选择最优模型
        best_cls_name = max(cls_models, key=lambda k: cls_models[k]['metrics']['val_auc'])
        best_cls = cls_models[best_cls_name]
        
        logger.info(f"\n最优分类模型: {best_cls_name} (AUC={best_cls['metrics']['val_auc']:.4f})")

        best_val_auc = best_cls['metrics'].get('val_auc', float('nan'))
        if np.isnan(best_val_auc) or best_val_auc < MIN_CLASSIFICATION_AUC:
            raise RuntimeError(
                f"验证AUC {best_val_auc:.4f} 低于阈值 {MIN_CLASSIFICATION_AUC:.2f}, 训练流程已终止"
            )
        
        # 🔧 月度分层评估
        try:
            logger.info("\n" + "="*80)
            logger.info("月度分层评估")
            logger.info("="*80)
            
            # 获取最优模型的预测
            best_pipeline = best_cls['pipeline']
            all_pred = best_pipeline.predict_proba(X)[:, 1]

            production_threshold = trainer.config.get('cls_threshold', CLS_THRESHOLD)
            optimal_threshold = best_cls['metrics'].get('optimal_threshold', production_threshold)
            thresholds = OrderedDict([
                ('prod', production_threshold),
                ('opt', optimal_threshold),
                ('0.5', 0.5)
            ])

            monthly_results = evaluate_by_month(
                y_cls,
                all_pred,
                dates_series,
                thresholds=thresholds
            )
            
            if len(monthly_results) > 0:
                auc_std = monthly_results['auc'].std()
                logger.info(f"\n📊 模型稳定性分析:")
                logger.info(f"  各月份AUC标准差: {auc_std:.4f} {'✅ 稳定' if auc_std < 0.05 else '⚠️ 波动较大'}")
        except Exception as eval_exc:
            logger.warning(f"月度评估失败: {eval_exc}")
        
        # 保存所有分类模型
        for name, model in cls_models.items():
            is_best = (name == best_cls_name)
            filepath = os.path.join(model_save_dir, f'cls_{prediction_period}d_{name}.pkl')
            trainer.save_model(model, filepath, is_best=is_best)
        
        # 额外保存最优模型
        best_filepath = os.path.join(model_save_dir, f'cls_{prediction_period}d_best.pkl')
        trainer.save_model(best_cls, best_filepath, is_best=True)
    
    # 训练回归模型
    if enable_both_tasks:
        logger.info("\n" + "="*80)
        logger.info("训练回归任务")
        logger.info("="*80)
        
        y_reg = df['label_reg'].copy()
        
        # 训练多个模型
        reg_models = {}
        
        # LightGBM
        logger.info("\n训练 LightGBM 回归器...")
        reg_lgb = trainer.train_regression_model(
            X, y_reg,
            model_type='lightgbm',
            dates=dates_series,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05
        )
        reg_models['lightgbm'] = reg_lgb
        
        # XGBoost
        logger.info("\n训练 XGBoost 回归器...")
        reg_xgb = trainer.train_regression_model(
            X, y_reg,
            model_type='xgboost',
            dates=dates_series,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05
        )
        reg_models['xgboost'] = reg_xgb
        
        # 选择最优模型
        best_reg_name = max(reg_models, key=lambda k: reg_models[k]['metrics']['val_r2'])
        best_reg = reg_models[best_reg_name]
        
        logger.info(f"\n最优回归模型: {best_reg_name} (R²={best_reg['metrics']['val_r2']:.4f})")

        best_val_r2 = best_reg['metrics'].get('val_r2', float('-inf'))
        if np.isnan(best_val_r2) or best_val_r2 < MIN_REGRESSION_R2:
            raise RuntimeError(
                f"验证R² {best_val_r2:.4f} 低于阈值 {MIN_REGRESSION_R2:.2f}, 训练流程已终止"
            )
        
        # 保存所有回归模型
        for name, model in reg_models.items():
            is_best = (name == best_reg_name)
            filepath = os.path.join(model_save_dir, f'reg_{prediction_period}d_{name}.pkl')
            trainer.save_model(model, filepath, is_best=is_best)
        
        # 额外保存最优模型
        best_filepath = os.path.join(model_save_dir, f'reg_{prediction_period}d_best.pkl')
        trainer.save_model(best_reg, best_filepath, is_best=True)
    
    logger.info("\n" + "="*80)
    logger.info("✅ 所有模型训练完成")
    logger.info("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练统一预测模型 V2')
    parser.add_argument('--symbols', type=str, nargs='+', help='股票代码列表')
    parser.add_argument('--symbol-file', type=str, help='股票代码文件（每行一个）')
    parser.add_argument('--start-date', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--prediction-period', type=int, default=PREDICTION_PERIOD_DAYS,
                        help=f'预测周期（天），默认{PREDICTION_PERIOD_DAYS}')
    parser.add_argument('--model-dir', type=str, default='models/v2',
                        help='模型保存目录')
    parser.add_argument('--classification-only', action='store_true',
                        help='只训练分类模型')
    parser.add_argument('--regression-only', action='store_true',
                        help='只训练回归模型')
    parser.add_argument('--label-strategy', type=str, choices=['absolute', 'quantile'],
                        default=LABEL_STRATEGY,
                        help='分类标签策略，absolute 或 quantile')
    parser.add_argument('--label-quantile', type=float, default=LABEL_POSITIVE_QUANTILE,
                        help='quantile 策略使用的上分位数（例如 0.7 表示前30% 为正类）')
    parser.add_argument('--label-min-samples', type=int, default=LABEL_MIN_SAMPLES_PER_DATE,
                        help='quantile 策略下每个交易日的最小样本数，低于该值回退 absolute')
    parser.add_argument('--enable-fundamental', action='store_true',
                        help='启用基本面特征（财务数据）')
    
    args = parser.parse_args()
    
    # 获取股票列表
    if args.symbols:
        symbols = args.symbols
    elif args.symbol_file:
        with open(args.symbol_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    else:
        # 默认使用沪深300成分股
        logger.info("未指定股票列表，使用沪深300成分股")
        try:
            import akshare as ak
            df_hs300 = ak.index_stock_cons(symbol="000300")
            symbols = df_hs300['品种代码'].tolist()
        except Exception as e:
            logger.error(f"获取沪深300成分股失败: {e}")
            symbols = ['000001', '600000', '000002']  # Fallback
    
    # 设置日期范围
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    start_date = args.start_date or (
        datetime.now() - timedelta(days=LOOKBACK_DAYS + args.prediction_period + 365)
    ).strftime('%Y-%m-%d')
    
    logger.info("训练配置:")
    logger.info(f"  股票数量: {len(symbols)}")
    logger.info(f"  日期范围: {start_date} ~ {end_date}")
    logger.info(f"  预测周期: {args.prediction_period}天")
    logger.info(f"  模型目录: {args.model_dir}")
    
    # 准备数据
    df = prepare_training_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        prediction_period=args.prediction_period,
        classification_strategy=args.label_strategy,
        label_quantile=args.label_quantile,
        label_min_samples=args.label_min_samples,
        enable_fundamental=args.enable_fundamental
    )
    
    # 训练模型
    enable_both = not (args.classification_only or args.regression_only)
    
    train_models(
        df=df,
        model_save_dir=args.model_dir,
        enable_both_tasks=enable_both,
        classification_strategy=args.label_strategy,
        prediction_period=args.prediction_period
    )
    
    logger.info("\n" + "="*80)
    logger.info("🎉 训练流程全部完成！")
    logger.info("="*80)


if __name__ == '__main__':
    # 确保logs目录存在
    os.makedirs('logs', exist_ok=True)
    
    main()
