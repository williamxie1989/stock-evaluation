"""Redis 缓存封装模块 (L1 跨进程缓存)

该模块提供一个简单的 RedisCache 类，用于跨进程共享缓存数据。
为减少对业务层的侵入，接口保持与简单字典类似的 ``get``/``set``/``delete``/``clear``。

目前仅依赖 ``redis`` 官方客户端 (>=4.x)，在调用方需要提前确保 Redis
服务可用；若连接失败，内部会降级为空操作 (No-Op) 缓存，避免影响主流程。
"""
from __future__ import annotations

import logging
import os
import pickle
import zlib
from typing import Any, Optional

try:
    from redis import Redis, ConnectionError  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "RedisCache 需要安装 `redis` 库，请执行 `pip install redis`"  # noqa: E501
    ) from exc

logger = logging.getLogger(__name__)


class RedisCache:  # noqa: D101
    DEFAULT_TTL = 300  # 秒

    def _connect(self):  # noqa: D401
        """创建并返回 Redis 客户端实例。

        提供独立方法以便在单元测试中通过 ``monkeypatch`` 替换，
        从而使用 ``fakeredis.FakeRedis`` 等内存实现，而无需实际 Redis 服务。
        """
        return Redis(
            host=self._host,
            port=self._port,
            db=self._db,
            password=self._password,
            decode_responses=False,
        )

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        password: str | None = None,
        ttl: int | None = None,
        key_prefix: str = "stock_evaluation:cache:",
        disable: bool | None = None,
    ) -> None:
        # 允许通过环境变量覆盖
        self._host = host or os.getenv("REDIS_HOST", "localhost")
        self._port = port or int(os.getenv("REDIS_PORT", "6379"))
        self._db = db or int(os.getenv("REDIS_DB", "0"))
        self._password = password or os.getenv("REDIS_PASSWORD")
        self._ttl = ttl or int(os.getenv("REDIS_CACHE_TTL", str(self.DEFAULT_TTL)))
        self._key_prefix = key_prefix
        self._disable = (
            disable
            if disable is not None
            else os.getenv("REDIS_CACHE_DISABLE", "false").lower() == "true"
        )

        self._client: Optional[Redis[Any]] = None
        if not self._disable:
            try:
                self._client = self._connect()
                # 简单连通性测试
                self._client.ping()
                logger.info(
                    "RedisCache 连接成功: %s:%s (db=%s)", self._host, self._port, self._db
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("RedisCache 初始化失败，将降级为 No-Op 缓存: %s", exc)
                self._client = None
                self._disable = True
        else:
            logger.info("RedisCache 已被禁用 (DISABLE flag = true)")

    # ──────────────────────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────────────────────
    def _build_key(self, raw_key: str) -> str:  # noqa: D401
        """构造带前缀的 Redis key"""
        return f"{self._key_prefix}{raw_key}"

    @staticmethod
    def _serialize(value: Any) -> bytes:  # noqa: D401
        """使用 pickle + zlib 压缩序列化数据"""
        try:
            return zlib.compress(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as exc:  # pragma: no cover
            logger.error("RedisCache 序列化失败: %s", exc)
            raise

    @staticmethod
    def _deserialize(blob: bytes) -> Any:  # noqa: D401
        """解压并反序列化数据"""
        try:
            return pickle.loads(zlib.decompress(blob))
        except Exception as exc:  # pragma: no cover
            logger.error("RedisCache 反序列化失败: %s", exc)
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────────────────────────────────
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:  # noqa: D401
        """保存数据到 Redis；若 Redis 不可用则忽略"""
        if self._disable or self._client is None:
            return
        try:
            blob = self._serialize(value)
            self._client.set(self._build_key(key), blob, ex=ttl or self._ttl)
        except ConnectionError as exc:  # pragma: no cover
            logger.warning("RedisCache.set 连接异常: %s", exc)
        except Exception as exc:  # pragma: no cover
            logger.error("RedisCache.set 失败: %s", exc)

    def get(self, key: str) -> Any | None:  # noqa: D401
        """从 Redis 获取数据，失败返回 None"""
        if self._disable or self._client is None:
            return None
        try:
            blob = self._client.get(self._build_key(key))
            if blob is None:
                return None
            return self._deserialize(blob)
        except ConnectionError as exc:  # pragma: no cover
            logger.warning("RedisCache.get 连接异常: %s", exc)
            return None
        except Exception as exc:  # pragma: no cover
            logger.error("RedisCache.get 失败: %s", exc)
            return None

    def delete(self, key: str) -> None:  # noqa: D401
        """删除指定 key"""
        if self._disable or self._client is None:
            return
        try:
            self._client.delete(self._build_key(key))
        except Exception as exc:  # pragma: no cover
            logger.error("RedisCache.delete 失败: %s", exc)

    def clear(self) -> None:  # noqa: D401
        """清空当前 key_prefix 下的所有缓存 (谨慎使用)"""
        if self._disable or self._client is None:
            return
        try:
            pattern = self._build_key("*")
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
                logger.info("RedisCache.clear 清理 %s 个键", len(keys))
        except Exception as exc:  # pragma: no cover
            logger.error("RedisCache.clear 失败: %s", exc)