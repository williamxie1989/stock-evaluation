# -*- coding: utf-8 -*-
"""
批量训练模块
支持一次性训练多只股票，使用真正的横截面quantile策略
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def prepare_batch_training_data(
    symbols: List[str],
    price_frames: Dict[str, pd.DataFrame],
    qfq_frames: Dict[str, pd.DataFrame],
    builder,
    market_generator,
    board_generator,
    market_returns: pd.DataFrame,
    batch_size: int,
    prediction_period: int,
    classification_strategy: str,
    label_quantile: float,
    label_min_samples: int,
    label_negative_quantile: Optional[float],
    label_threshold: float,
    enable_neutral_band: bool,
    label_neutral_quantile: Optional[float],
    use_market_baseline: bool,
    use_industry_neutral: bool,
    market_column: str = 'MKT',
    **kwargs
) -> Tuple[List[pd.DataFrame], List[Tuple[str, str]]]:
    """
    批量准备训练数据（一次处理batch_size只股票）
    
    Parameters
    ----------
    symbols : List[str]
        股票代码列表
    price_frames : Dict[str, pd.DataFrame]
        不复权价格数据字典
    qfq_frames : Dict[str, pd.DataFrame]
        前复权价格数据字典
    builder : UnifiedFeatureBuilder
        特征构建器
    market_generator : MarketFactorGenerator
        市场因子生成器
    board_generator : BoardFeatureGenerator
        板块特征生成器
    market_returns : pd.DataFrame
        市场收益数据
    batch_size : int
        每批处理的股票数量
    prediction_period : int
        预测周期
    classification_strategy : str
        分类策略
    label_quantile : float
        上分位数
    label_min_samples : int
        最小样本数
    label_negative_quantile : float or None
        下分位数（明确负类或中性区）
    label_threshold : float
        样本不足时回退的绝对收益阈值
    enable_neutral_band : bool
        是否启用中性区
    label_neutral_quantile : float or None
        中性区上界
    use_market_baseline : bool
        是否减去市场基准构建超额收益
    use_industry_neutral : bool
        是否做行业中性
    market_column : str
        市场基准列名，默认 MKT
    
    Returns
    -------
    batch_results : List[pd.DataFrame]
        每批的训练数据
    failed_symbols : List[Tuple[str, str]]
        失败的股票及原因
    """
    from src.ml.training.toolkit import add_labels_corrected
    
    batch_results = []
    failed_symbols = []
    
    # 将股票分批
    num_batches = (len(symbols) + batch_size - 1) // batch_size
    
    logger.info("=" * 80)
    logger.info(f"批量训练模式启动")
    logger.info("=" * 80)
    logger.info(f"总股票数: {len(symbols)}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"批次数量: {num_batches}")
    logger.info(f"标签策略: {classification_strategy} (横截面quantile)")
    logger.info(f"  上分位数: {label_quantile:.2f}")
    if label_negative_quantile is not None:
        logger.info(f"  下分位数: {label_negative_quantile:.2f}")
    logger.info(f"  每日最小样本: {label_min_samples}")
    logger.info(f"  回退阈值: {label_threshold:.3f}")
    logger.info(f"  市场基准: {'启用' if use_market_baseline else '关闭'}")
    logger.info(f"  行业中性: {'启用' if use_industry_neutral else '关闭'}")
    if enable_neutral_band:
        logger.info(f"  中性区启用, 上界: {label_neutral_quantile}")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(symbols))
        batch_symbols = symbols[start_idx:end_idx]
        
        logger.info("")
        logger.info(f"批次 {batch_idx + 1}/{num_batches}: 处理 {len(batch_symbols)} 只股票")
        logger.info(f"  范围: {start_idx + 1}-{end_idx}/{len(symbols)}")
        
        # 收集这批股票的所有特征数据
        batch_features = []
        batch_qfq_prices = []
        batch_raw_prices = []
        
        for symbol in batch_symbols:
            if symbol not in price_frames or symbol not in qfq_frames:
                logger.warning(f"  {symbol}: 跳过 - 价格数据缺失")
                failed_symbols.append((symbol, 'price_data_missing'))
                continue
            
            try:
                # 构建特征
                features_df = builder.build_features_from_dataframe(
                    price_frames[symbol], symbol
                )
                
                if features_df is None or len(features_df) == 0:
                    logger.warning(f"  {symbol}: 跳过 - 特征构建失败")
                    failed_symbols.append((symbol, 'feature_build_failed'))
                    continue
                
                # 添加市场特征
                if market_generator is not None and market_returns is not None:
                    try:
                        market_enriched = market_generator.add_market_features(
                            price_frames[symbol].copy(),
                            symbol,
                            market_returns
                        )
                        candidate_cols = ['MKT'] + market_generator.get_feature_names()
                        available_cols = [
                            col for col in candidate_cols
                            if col in market_enriched.columns
                        ]
                        if available_cols:
                            market_slice = market_enriched.reset_index()[['date'] + available_cols]
                            features_df = features_df.merge(market_slice, on='date', how='left')
                    except Exception as market_exc:
                        logger.warning(f"  {symbol}: 市场特征添加失败 - {market_exc}")
                
                # 添加symbol列
                features_df['symbol'] = symbol
                
                # 添加板块特征
                if board_generator is not None:
                    try:
                        features_df = board_generator.add_board_feature(
                            features_df, symbol_col='symbol'
                        )
                    except Exception as board_exc:
                        logger.warning(f"  {symbol}: 板块特征添加失败 - {board_exc}")
                
                # 确保有date列
                if 'date' not in features_df.columns:
                    if features_df.index.name == 'date' or isinstance(features_df.index, pd.DatetimeIndex):
                        features_df = features_df.reset_index()
                        if 'index' in features_df.columns and 'date' not in features_df.columns:
                            features_df.rename(columns={'index': 'date'}, inplace=True)
                    else:
                        logger.warning(f"  {symbol}: 跳过 - 缺少date列")
                        failed_symbols.append((symbol, 'no_date_column'))
                        continue
                
                # 准备价格数据
                price_raw = price_frames[symbol].copy()
                if price_raw.index.name == 'date' or isinstance(price_raw.index, pd.DatetimeIndex):
                    price_raw = price_raw.reset_index()
                    if 'index' in price_raw.columns and 'date' not in price_raw.columns:
                        price_raw.rename(columns={'index': 'date'}, inplace=True)
                
                price_raw = price_raw[['date', 'close']].copy()
                price_raw['symbol'] = symbol
                
                price_adj = qfq_frames[symbol].copy()
                if 'date' not in price_adj.columns:
                    price_adj = price_adj.reset_index()
                
                price_adj['date'] = pd.to_datetime(price_adj['date'])
                price_adj = price_adj[['date', 'close']].copy()
                price_adj['symbol'] = symbol
                
                # 🔧 关键修复1: 确保所有DataFrame都重置索引，避免合并时的索引冲突
                features_df = features_df.reset_index(drop=True)
                price_adj = price_adj.reset_index(drop=True)
                price_raw = price_raw.reset_index(drop=True)
                
                # 🔧 关键修复2: 去除重复列名，避免concat时的列索引冲突
                if not features_df.columns.is_unique:
                    logger.warning(f"  {symbol}: 检测到重复列名，自动去重（保留第一个）")
                    features_df = features_df.loc[:, ~features_df.columns.duplicated()]
                
                if not price_adj.columns.is_unique:
                    price_adj = price_adj.loc[:, ~price_adj.columns.duplicated()]
                
                if not price_raw.columns.is_unique:
                    price_raw = price_raw.loc[:, ~price_raw.columns.duplicated()]
                
                # 收集到批次列表
                batch_features.append(features_df)
                batch_qfq_prices.append(price_adj)
                batch_raw_prices.append(price_raw)
                
            except Exception as exc:
                logger.error(f"  {symbol}: 处理失败 - {exc}", exc_info=False)
                failed_symbols.append((symbol, f'exception_{type(exc).__name__}'))
        
        if len(batch_features) == 0:
            logger.warning(f"  批次 {batch_idx + 1}: 无有效股票，跳过")
            continue
        
        logger.info(f"  批次 {batch_idx + 1}: 成功准备 {len(batch_features)} 只股票的特征")
        
        # 合并批次数据
        try:
            combined_features = pd.concat(batch_features, ignore_index=True)
            combined_qfq = pd.concat(batch_qfq_prices, ignore_index=True)
            combined_raw = pd.concat(batch_raw_prices, ignore_index=True)
            
            logger.info(f"  批次 {batch_idx + 1}: 合并数据")
            logger.info(f"    特征记录数: {len(combined_features):,}")
            logger.info(f"    股票数/天: {combined_features.groupby('date')['symbol'].nunique().describe()}")
            
            # 批量计算标签（这里会使用横截面quantile）
            logger.info(f"  批次 {batch_idx + 1}: 计算标签（横截面quantile）...")
            
            features_with_labels = add_labels_corrected(
                features_df=combined_features,
                price_data=combined_qfq,
                prediction_period=prediction_period,
                threshold=label_threshold,
                price_data_raw=combined_raw,
                classification_strategy=classification_strategy,
                quantile=label_quantile,
                min_samples_per_date=label_min_samples,
                negative_quantile=label_negative_quantile,
                enable_neutral_band=enable_neutral_band,
                neutral_quantile=label_neutral_quantile,
                market_returns=market_returns,
                use_market_baseline=use_market_baseline,
                market_column=market_column,
                use_industry_neutral=use_industry_neutral
            )
            
            if len(features_with_labels) == 0:
                logger.warning(f"  批次 {batch_idx + 1}: 标签计算后无有效数据")
                continue
            
            # 过滤极端值
            extreme_mask = features_with_labels['label_reg'].abs() > 1.0
            if extreme_mask.sum() > 0:
                logger.info(f"  批次 {batch_idx + 1}: 过滤 {extreme_mask.sum()} 条极端收益率")
                features_with_labels = features_with_labels[~extreme_mask]
            
            if len(features_with_labels) == 0:
                logger.warning(f"  批次 {batch_idx + 1}: 过滤后无数据")
                continue
            
            logger.info(f"  批次 {batch_idx + 1}: ✅ 完成")
            logger.info(f"    最终记录数: {len(features_with_labels):,}")
            logger.info(f"    正样本率: {features_with_labels['label_cls'].mean():.2%}")
            
            batch_results.append(features_with_labels)
            
        except Exception as exc:
            logger.error(f"  批次 {batch_idx + 1}: 标签计算失败 - {exc}", exc_info=True)
            # 将这批所有股票标记为失败
            for symbol in batch_symbols:
                if symbol not in [s for s, _ in failed_symbols]:
                    failed_symbols.append((symbol, 'label_calculation_failed'))
    
    logger.info("")
    logger.info(f"批量训练数据准备完成:")
    logger.info(f"  成功批次: {len(batch_results)}/{num_batches}")
    logger.info(f"  失败股票: {len(failed_symbols)}")
    
    return batch_results, failed_symbols
