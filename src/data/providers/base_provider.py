"""
通用数据提供者基类 - 提供以下功能:
1. requests.Session 复用
2. 随机 User-Agent/Referer 以降低被封风险
3. 简易速率限制 (QPS) 控制, 线程安全
4. 自动重试与指数退避
5. 代理支持 (可在环境变量或初始化参数传入)

各具体 Provider 只需继承该类并调用 self._request() 完成 HTTP 请求, 专注解析数据.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Sequence

import requests
from requests import Response, Session

logger = logging.getLogger(__name__)


class DataProviderBase:
    """统一 HTTP 访问辅助基类"""

    # 一些常见桌面/移动 UA 片段, 可根据需要补充
    _UA_POOL: Sequence[str] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Linux; Android 13; SM-S9180) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Mobile Safari/537.36",
    )

    # 用于速率限制的全局锁与时间戳
    _rate_lock = threading.Lock()
    _last_request_ts: float = 0.0

    def __init__(
        self,
        timeout: int = 8,
        max_retries: int = 3,
        backoff_factor: float = 0.7,
        rate_limit_qps: float = 5.0,
        proxies: Optional[Dict[str, str]] = None,
    ) -> None:
        """初始化通用 Provider 基类

        Args:
            timeout: 单次请求超时(s)
            max_retries: 最大重试次数
            backoff_factor: 指数退避基数, 下次等待 = backoff_factor * (2 ** retry_count)
            rate_limit_qps: 全局 QPS 限制, 0 表示不限制
            proxies: 代理设置, e.g. {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
        """
        self.session: Session = requests.Session()
        self.session.headers.update({"User-Agent": random.choice(self._UA_POOL)})
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.rate_limit_qps = rate_limit_qps
        self.proxies = proxies or self._load_proxies_from_env()
        logger.debug(
            "DataProviderBase initialized | timeout=%s max_retries=%s rate_limit_qps=%s", timeout, max_retries, rate_limit_qps
        )

    # ------------------------------------------------------------------
    # 公共 HTTP 请求方法
    # ------------------------------------------------------------------
    def _request(self, method: str, url: str, **kwargs) -> Optional[Response]:
        """带速率限制、重试与随机 UA 的请求封装"""
        retry = 0
        while retry <= self.max_retries:
            self._respect_rate_limit()
            # 每次重试更新 UA, 可减少封禁概率
            self.session.headers.update({"User-Agent": random.choice(self._UA_POOL)})
            try:
                resp = self.session.request(
                    method.upper(),
                    url,
                    timeout=self.timeout,
                    proxies=self.proxies,
                    **kwargs,
                )
                resp.raise_for_status()
                return resp
            except Exception as e:
                retry += 1
                if retry > self.max_retries:
                    logger.error("Request failed after %s retries: %s %s", self.max_retries, url, e)
                    return None
                sleep_s = self.backoff_factor * (2 ** (retry - 1))
                logger.warning("Request error, retrying %s/%s after %.2fs: %s", retry, self.max_retries, sleep_s, e)
                time.sleep(sleep_s)
        return None

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _respect_rate_limit(self) -> None:
        if self.rate_limit_qps <= 0:
            return
        min_interval = 1.0 / self.rate_limit_qps
        with self._rate_lock:
            now = time.time()
            diff = now - self._last_request_ts
            if diff < min_interval:
                time.sleep(min_interval - diff)
            self._last_request_ts = time.time()

    @staticmethod
    def _load_proxies_from_env() -> Optional[Dict[str, str]]:
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if http_proxy or https_proxy:
            return {"http": http_proxy, "https": https_proxy}
        return None

    # ------------------------------------------------------------------
    # 供子类扩展的方法
    # ------------------------------------------------------------------
    def refresh_session(self) -> None:
        """刷新 session, 在被封禁时可调用"""
        self.session.close()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": random.choice(self._UA_POOL)})
        logger.info("HTTP session refreshed at %s", datetime.now())