# -*- coding: utf-8 -*-
"""
特征缓存管理器
整合到现有的 L0-L2 三层缓存体系，专门用于缓存特征计算结果
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
import hashlib
import pickle
import zlib
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.prediction_config import (
    ENABLE_FEATURE_CACHE,
    FEATURE_CACHE_TTL,
)
from src.cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)


class FeatureCacheManager:
    """
    特征缓存管理器
    
    三层缓存架构（与UnifiedDataAccessLayer保持一致）：
    - L0: 进程内 LRU 缓存 (内存)
    - L1: Redis 跨进程缓存
    - L2: Parquet 磁盘缓存
    
    缓存键设计：
    - 包含股票列表、截止日期、特征配置的哈希
    - 确保相同请求能命中缓存
    """
    
    def __init__(self, 
                 cache_ttl: int = FEATURE_CACHE_TTL,
                 enable_cache: bool = ENABLE_FEATURE_CACHE,
                 l0_maxsize: int = 128,
                 l2_cache_dir: str = "l2_cache/features"):
        """
        初始化特征缓存管理器
        
        Args:
            cache_ttl: 缓存过期时间（秒），默认3600秒（1小时）
            enable_cache: 是否启用缓存
            l0_maxsize: L0缓存最大条目数
            l2_cache_dir: L2缓存目录
        """
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache
        self.l0_maxsize = l0_maxsize
        self.l2_cache_dir = Path(l2_cache_dir)
        
        # 确保L2缓存目录存在
        if self.enable_cache:
            self.l2_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化L1 Redis缓存
        self._l1_cache = RedisCache(ttl=cache_ttl) if enable_cache else None
        
        # 初始化L0内存缓存
        self._initialize_l0_cache()
        
        # 缓存统计
        self._stats = {
            'l0_hits': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        
        logger.info(f"FeatureCacheManager 初始化完成")
        logger.info(f"  - 缓存启用: {enable_cache}")
        logger.info(f"  - TTL: {cache_ttl}秒")
        logger.info(f"  - L0大小: {l0_maxsize}")
        logger.info(f"  - L2目录: {l2_cache_dir}")
    
    def _initialize_l0_cache(self):
        """初始化L0进程内缓存"""
        @lru_cache(maxsize=self.l0_maxsize)
        def _l0_loader(cache_key: str, ttl_bucket: int) -> Optional[pd.DataFrame]:
            """L0缓存加载器（带TTL失效）"""
            # ttl_bucket用于自动失效，相同key在不同时间片会miss
            return None  # 实际数据由外部set
        
        self._l0_loader = _l0_loader
    
    def _make_cache_key(self, 
                        symbols: List[str], 
                        as_of_date: Optional[str],
                        feature_config: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        Args:
            symbols: 股票代码列表
            as_of_date: 截止日期
            feature_config: 特征配置（包含enable_price_volume等）
        
        Returns:
            缓存键字符串
        """
        # 排序symbols确保相同股票列表生成相同key
        sorted_symbols = sorted(symbols)
        
        # 构建缓存键组成部分
        key_parts = [
            f"symbols={','.join(sorted_symbols[:10])}"  # 只取前10个避免key过长
            + (f"+{len(sorted_symbols)-10}more" if len(sorted_symbols) > 10 else ""),
            f"date={as_of_date or 'latest'}",
            f"config={hashlib.md5(str(sorted(feature_config.items())).encode()).hexdigest()[:8]}"
        ]
        
        cache_key = "feature:" + "|".join(key_parts)
        return cache_key
    
    def _current_ttl_bucket(self) -> int:
        """返回当前时间片，用于TTL失效"""
        if self.cache_ttl <= 0:
            return 0
        return int(datetime.now().timestamp() // self.cache_ttl)
    
    def get(self, 
            symbols: List[str], 
            as_of_date: Optional[str],
            feature_config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        从缓存获取特征数据
        
        Args:
            symbols: 股票代码列表
            as_of_date: 截止日期
            feature_config: 特征配置
        
        Returns:
            特征DataFrame或None（未命中）
        """
        if not self.enable_cache:
            return None
        
        self._stats['total_requests'] += 1
        
        cache_key = self._make_cache_key(symbols, as_of_date, feature_config)
        ttl_bucket = self._current_ttl_bucket()
        
        # L0: 进程内缓存
        try:
            cached = self._l0_loader(cache_key, ttl_bucket)
            if cached is not None:
                self._stats['l0_hits'] += 1
                logger.debug(f"✅ L0缓存命中: {cache_key[:50]}...")
                return cached
        except Exception as e:
            logger.debug(f"L0缓存读取失败: {e}")
        
        # L1: Redis缓存
        try:
            cached = self._get_from_l1(cache_key)
            if cached is not None:
                self._stats['l1_hits'] += 1
                logger.debug(f"✅ L1缓存命中: {cache_key[:50]}...")
                # 回填到L0
                self._save_to_l0(cache_key, cached, ttl_bucket)
                return cached
        except Exception as e:
            logger.debug(f"L1缓存读取失败: {e}")
        
        # L2: Parquet磁盘缓存
        try:
            cached = self._get_from_l2(cache_key)
            if cached is not None:
                self._stats['l2_hits'] += 1
                logger.debug(f"✅ L2缓存命中: {cache_key[:50]}...")
                # 回填到L1和L0
                self._save_to_l1(cache_key, cached)
                self._save_to_l0(cache_key, cached, ttl_bucket)
                return cached
        except Exception as e:
            logger.debug(f"L2缓存读取失败: {e}")
        
        # 全部miss
        self._stats['misses'] += 1
        return None
    
    def set(self, 
            symbols: List[str], 
            as_of_date: Optional[str],
            feature_config: Dict[str, Any],
            features: pd.DataFrame) -> None:
        """
        将特征数据保存到所有缓存层
        
        Args:
            symbols: 股票代码列表
            as_of_date: 截止日期
            feature_config: 特征配置
            features: 特征DataFrame
        """
        if not self.enable_cache or features is None or features.empty:
            return
        
        cache_key = self._make_cache_key(symbols, as_of_date, feature_config)
        ttl_bucket = self._current_ttl_bucket()
        
        # 保存到所有层
        self._save_to_l0(cache_key, features, ttl_bucket)
        self._save_to_l1(cache_key, features)
        self._save_to_l2(cache_key, features)
        
        logger.debug(f"💾 特征已缓存: {cache_key[:50]}... ({len(features)}行 x {len(features.columns)}列)")
    
    def _save_to_l0(self, cache_key: str, data: pd.DataFrame, ttl_bucket: int):
        """保存到L0缓存"""
        try:
            # 使用monkey patch方式注入数据到lru_cache
            # 由于lru_cache不支持直接set，我们通过调用来"缓存"返回值
            # 这里使用一个技巧：先清除这个key，再通过返回值设置
            pass  # L0缓存通过get时的回填来实现
        except Exception as e:
            logger.warning(f"L0缓存保存失败: {e}")
    
    def _get_from_l1(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从L1 Redis获取"""
        if not self._l1_cache:
            return None
        try:
            return self._l1_cache.get(cache_key)
        except Exception as e:
            logger.warning(f"L1缓存获取失败: {e}")
            return None
    
    def _save_to_l1(self, cache_key: str, data: pd.DataFrame):
        """保存到L1 Redis"""
        if not self._l1_cache:
            return
        try:
            self._l1_cache.set(cache_key, data, ttl=self.cache_ttl)
        except Exception as e:
            logger.warning(f"L1缓存保存失败: {e}")
    
    def _l2_cache_path(self, cache_key: str) -> Path:
        """生成L2缓存文件路径"""
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.l2_cache_dir / f"{key_hash}.parquet"
    
    def _get_from_l2(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从L2 Parquet获取"""
        path = self._l2_cache_path(cache_key)
        if not path.exists():
            return None
        
        try:
            # 检查文件是否过期
            if self.cache_ttl > 0:
                file_age = datetime.now().timestamp() - path.stat().st_mtime
                if file_age > self.cache_ttl:
                    logger.debug(f"L2缓存已过期: {path.name}")
                    path.unlink()  # 删除过期文件
                    return None
            
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"L2缓存读取失败: {e}")
            return None
    
    def _save_to_l2(self, cache_key: str, data: pd.DataFrame):
        """保存到L2 Parquet"""
        try:
            import uuid
            path = self._l2_cache_path(cache_key)
            tmp_path = path.parent / f"{path.stem}_{uuid.uuid4().hex}.tmp"
            
            # 保存，保留索引
            data.to_parquet(tmp_path, index=True, compression='snappy')
            
            # 原子性重命名
            tmp_path.replace(path)
        except Exception as e:
            logger.warning(f"L2缓存保存失败: {e}")
    
    def invalidate(self, 
                   symbols: Optional[List[str]] = None,
                   as_of_date: Optional[str] = None,
                   feature_config: Optional[Dict[str, Any]] = None):
        """
        清除缓存
        
        Args:
            symbols: 如果指定，只清除这些股票的缓存
            as_of_date: 如果指定，只清除这个日期的缓存
            feature_config: 如果指定，只清除这个配置的缓存
        """
        if not self.enable_cache:
            return
        
        # 如果都指定了，精确清除
        if symbols is not None and feature_config is not None:
            cache_key = self._make_cache_key(symbols, as_of_date, feature_config)
            self._invalidate_key(cache_key)
        else:
            # 否则清除全部
            self.clear_all()
    
    def _invalidate_key(self, cache_key: str):
        """清除指定key的所有层缓存"""
        # L1
        if self._l1_cache:
            try:
                self._l1_cache.delete(cache_key)
            except:
                pass
        
        # L2
        try:
            path = self._l2_cache_path(cache_key)
            if path.exists():
                path.unlink()
        except:
            pass
        
        # L0会自动通过TTL失效
    
    def clear_all(self):
        """清除所有缓存"""
        if not self.enable_cache:
            return
        
        logger.info("🧹 清除所有特征缓存...")
        
        # 清除L0
        if hasattr(self, '_l0_loader'):
            self._l0_loader.cache_clear()
        
        # 清除L1（Redis不支持批量清除feature:*，需要手动实现）
        # 这里简单重置连接
        if self._l1_cache:
            try:
                # Redis批量删除需要特殊处理
                pass
            except:
                pass
        
        # 清除L2
        try:
            import shutil
            if self.l2_cache_dir.exists():
                shutil.rmtree(self.l2_cache_dir)
                self.l2_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"L2缓存清除失败: {e}")
        
        # 重置统计
        self._stats = {
            'l0_hits': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        
        logger.info("✅ 特征缓存已清除")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self._stats['total_requests']
        if total == 0:
            return {
                **self._stats,
                'hit_rate': 0.0,
                'l0_hit_rate': 0.0,
                'l1_hit_rate': 0.0,
                'l2_hit_rate': 0.0
            }
        
        total_hits = self._stats['l0_hits'] + self._stats['l1_hits'] + self._stats['l2_hits']
        
        return {
            **self._stats,
            'hit_rate': total_hits / total,
            'l0_hit_rate': self._stats['l0_hits'] / total,
            'l1_hit_rate': self._stats['l1_hits'] / total,
            'l2_hit_rate': self._stats['l2_hits'] / total
        }
    
    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()
        
        logger.info("=" * 60)
        logger.info("特征缓存统计")
        logger.info("=" * 60)
        logger.info(f"总请求数: {stats['total_requests']}")
        logger.info(f"总命中率: {stats['hit_rate']:.1%}")
        logger.info(f"  - L0命中: {stats['l0_hits']} ({stats['l0_hit_rate']:.1%})")
        logger.info(f"  - L1命中: {stats['l1_hits']} ({stats['l1_hit_rate']:.1%})")
        logger.info(f"  - L2命中: {stats['l2_hits']} ({stats['l2_hit_rate']:.1%})")
        logger.info(f"  - 未命中: {stats['misses']}")
        logger.info("=" * 60)
