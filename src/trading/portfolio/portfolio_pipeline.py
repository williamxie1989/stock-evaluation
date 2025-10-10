import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

from src.core.unified_data_access_factory import create_unified_data_access
from src.trading.signals.advanced_signal_generator import AdvancedSignalGenerator, Signal
from src.core.risk_management import RiskManager
from src.ml.features.enhanced_features import EnhancedFeatureGenerator
import os
import joblib


logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_LIMIT = 3000


def resolve_candidate_limit(limit: Optional[int]) -> Optional[int]:
    """
    统一解析候选股票数量限制：
    - 正整数 → 限制数量
    - 0 或负数 → 不限制
    - None → 读取环境变量 PORTFOLIO_CANDIDATE_LIMIT，默认 3000
    """
    def _convert(value: Optional[int | str]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            logger.warning("候选股票限制值解析失败：%s", value)
            return None

    resolved = _convert(limit)
    if resolved is not None:
        return resolved if resolved > 0 else None

    env_value = _convert(os.getenv("PORTFOLIO_CANDIDATE_LIMIT"))
    if env_value is not None:
        return env_value if env_value > 0 else None

    return DEFAULT_CANDIDATE_LIMIT


@dataclass
class PickResult:
    symbol: str
    score: float
    reason: str
    signal: Optional[Signal]
    risk_score: float


@dataclass
class Holding:
    symbol: str
    weight: float
    shares: float


class PortfolioPipeline:
    """模型→选股→建仓→定期调仓→收益计算 管线（首版：信号+风险融合，30天等权调仓）"""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.0003,
        lookback_days: int = 120,
        top_n: int = 20,
        rebalance_freq: str = '30D',
        w_model: float = 0.5,
        w_signal: float = 0.3,
        w_risk: float = 0.2,
        model_dir: str | None = None,
        classifier_name: str | None = None,
        regressor_name: str | None = None,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.rebalance_freq = rebalance_freq
        # 打分融合权重（首版仅使用信号+风险，模型占位）
        self.w_model = w_model
        self.w_signal = w_signal
        self.w_risk = w_risk

        self._apply_env_weights()
        self._base_weights = (self.w_model, self.w_signal, self.w_risk)

        # 依赖
        self.signal_gen = AdvancedSignalGenerator()
        self.risk_manager = RiskManager()
        # 允许通过环境变量覆盖模型目录与文件名，默认保持原值
        self.model_dir = model_dir or os.environ.get("PORTFOLIO_MODEL_DIR", "models")
        self.classifier_name = classifier_name or os.environ.get("PORTFOLIO_CLASSIFIER_NAME", "classifier")
        self.regressor_name = regressor_name or os.environ.get("PORTFOLIO_REGRESSOR_NAME", "regressor")
        # 新增：模型加载与特征生成器
        self.feature_gen = EnhancedFeatureGenerator()
        # 使用环境变量或参数解析后的目录/文件名
        self.clf_model = self._load_model_safe(os.path.join(self.model_dir, f"{self.classifier_name}.pkl"))
        self.reg_model = self._load_model_safe(os.path.join(self.model_dir, f"{self.regressor_name}.pkl"))
        self.data_access = self._create_data_access()
        self.candidate_limit = resolve_candidate_limit(None)

    def _load_model_safe(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"模型文件不存在: {path}")
            return None
        try:
            payload = joblib.load(path)
        except Exception as e:
            logger.error(f"加载模型失败 {path}: {e}")
            return None
        model = self._unwrap_model_payload(payload, origin=path)
        if model is None:
            logger.warning(f"模型文件未包含可用模型: {path}")
        return model

    def _unwrap_model_payload(self, payload: Any, *, origin: str) -> Optional[Any]:
        model = payload
        feature_names = None
        metadata = None

        if isinstance(payload, dict):
            for key in ("model", "estimator", "classifier", "regressor", "pipeline"):
                if payload.get(key) is not None:
                    model = payload[key]
                    break
            else:
                logger.warning(f"模型文件 {origin} 未找到可用模型键")
                return None
            feature_names = (
                payload.get("feature_names")
                or payload.get("feature_names_in_")
                or payload.get("features")
                or payload.get("columns")
            )
            metadata = payload.get("metadata")
        elif isinstance(payload, (list, tuple)) and payload:
            model = payload[0]
            if len(payload) > 1 and isinstance(payload[1], (list, tuple, np.ndarray)):
                feature_names = payload[1]

        if model is None:
            return None

        if feature_names is not None:
            try:
                feature_array = np.array(feature_names)
                setattr(model, "feature_names_in_", feature_array)
            except Exception:
                logger.debug("无法设置模型特征名（%s）: %s", origin, feature_names, exc_info=True)

        if metadata is not None:
            try:
                setattr(model, "_metadata", metadata)
            except Exception:
                logger.debug("无法附加模型 metadata（%s）", origin, exc_info=True)

        return model

    def _prepare_features_for_model(self, model: Any, latest_feats: pd.DataFrame) -> Optional[pd.DataFrame]:
        if model is None or latest_feats is None or latest_feats.empty:
            return None

        feature_names_attr = getattr(model, "feature_names_in_", None)
        if feature_names_attr is not None and len(feature_names_attr) > 0:
            feature_names = [str(name) for name in feature_names_attr]
        else:
            feature_names = list(latest_feats.columns)

        expected_dim = getattr(model, "n_features_in_", None)
        if expected_dim is not None:
            try:
                expected_dim = int(expected_dim)
            except Exception:
                expected_dim = None

        available_cols = list(latest_feats.columns)
        available_set = set(available_cols)
        ordered_cols = [name for name in feature_names if name in available_set]

        if expected_dim is not None:
            if len(ordered_cols) > expected_dim:
                ordered_cols = ordered_cols[:expected_dim]
            if len(ordered_cols) < expected_dim:
                remaining = [col for col in available_cols if col not in ordered_cols]
                ordered_cols.extend(remaining[: max(0, expected_dim - len(ordered_cols))])

        if not ordered_cols:
            ordered_cols = available_cols[:expected_dim] if expected_dim is not None else available_cols

        matrix = latest_feats.reindex(columns=ordered_cols, fill_value=0.0).copy()

        if expected_dim is not None:
            if matrix.shape[1] < expected_dim:
                pad_count = expected_dim - matrix.shape[1]
                for idx in range(pad_count):
                    matrix[f"__pad_{idx}"] = 0.0
            if matrix.shape[1] > expected_dim:
                matrix = matrix.iloc[:, :expected_dim]

        return matrix

    def _parse_env_flag(self, name: str) -> Optional[bool]:
        value = os.getenv(name)
        if value is None:
            return None
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
        return None

    def _create_data_access(self):
        config: Dict[str, Any] = {}
        auto_sync_flag = self._parse_env_flag("PORTFOLIO_DATA_AUTO_SYNC")
        if auto_sync_flag is not None:
            config["auto_sync"] = auto_sync_flag
        use_cache_flag = self._parse_env_flag("PORTFOLIO_DATA_USE_CACHE")
        if use_cache_flag is not None:
            config["use_cache"] = use_cache_flag
        cache_ttl_val = os.getenv("PORTFOLIO_DATA_CACHE_TTL")
        if cache_ttl_val and cache_ttl_val.isdigit():
            try:
                config["cache_ttl"] = int(cache_ttl_val)
            except Exception:
                logger.debug("无法解析 PORTFOLIO_DATA_CACHE_TTL=%s", cache_ttl_val, exc_info=True)

        kwargs: Dict[str, Any] = {"validate_sources": False}
        if config:
            kwargs["data_access_config"] = config
        return create_unified_data_access(**kwargs)

    def _parse_env_float(self, name: str) -> Optional[float]:
        value = os.getenv(name)
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            logger.warning("环境变量 %s=%s 解析失败（期望 float）", name, value)
            return None

    def _apply_env_weights(self) -> None:
        env_model = self._parse_env_float("PORTFOLIO_MODEL_WEIGHT")
        env_signal = self._parse_env_float("PORTFOLIO_SIGNAL_WEIGHT")
        env_risk = self._parse_env_float("PORTFOLIO_RISK_WEIGHT")

        if env_model is not None:
            self.w_model = env_model
        if env_signal is not None:
            self.w_signal = env_signal
        if env_risk is not None:
            self.w_risk = env_risk

    def apply_weight_overrides(self, overrides: Dict[str, float]) -> None:
        if overrides is None:
            return
        # 先回到环境/默认权重，避免污染缓存实例
        self.reset_weights()
        model = overrides.get("model")
        signal = overrides.get("signal")
        risk = overrides.get("risk")
        if model is not None:
            self.w_model = float(model)
        if signal is not None:
            self.w_signal = float(signal)
        if risk is not None:
            self.w_risk = float(risk)

    def reset_weights(self) -> None:
        base = getattr(self, "_base_weights", None)
        if base and len(base) == 3:
            self.w_model, self.w_signal, self.w_risk = base
        else:
            self.w_model, self.w_signal, self.w_risk = 0.5, 0.3, 0.2

    def _get_stock_pool(self, limit: Optional[int] = 1000) -> List[str]:
        base_limit = limit if limit is not None else self.candidate_limit
        resolved_limit = resolve_candidate_limit(base_limit) if base_limit is not None else None
        stocks = self.data_access.get_all_stock_list()
        if stocks is None or stocks.empty:
            logger.warning("股票列表为空，返回空池")
            return []
        # 过滤非A股、空symbol
        symbols = []
        for _, r in stocks.iterrows():
            sym = r.get('symbol')
            if not sym or not isinstance(sym, str):
                continue
            if not (sym.endswith('.SH') or sym.endswith('.SZ')):
                continue
            symbols.append(sym)
            if resolved_limit and len(symbols) >= resolved_limit:
                break
        return symbols

    def _fetch_history(self, symbol: str, end_date: Optional[pd.Timestamp] = None, days: int = 120) -> pd.DataFrame:
        if end_date is None:
            end_date = pd.Timestamp(datetime.now().date())
        start_date = end_date - pd.Timedelta(days=days)
        res = self.data_access.get_historical_data(
            symbol=symbol,
            start_date=str(start_date.date()),
            end_date=str(end_date.date()),
            # 不请求数据库中不存在的字段，避免 Unknown column 错误
            fields=["open", "close", "high", "low", "volume", "amount"],
            adjust_mode="qfq",  # 投资组合使用前复权数据
        )
        # 兼容异步/同步实现
        try:
            if asyncio.iscoroutine(res):
                df = asyncio.run(res)
            else:
                df = res
        except RuntimeError:
            # 如果事件循环已运行，使用新事件循环执行
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                df = loop.run_until_complete(res)
            finally:
                loop.close()
        if df is None:
            return pd.DataFrame()
        # 统一列名（lowercase）
        cols = {c.lower(): c for c in df.columns}
        def get_col(name):
            return df[cols[name]] if name in cols else df.get(name)
        out = pd.DataFrame({
            'open': get_col('open'),
            'close': get_col('close'),
            'high': get_col('high'),
            'low': get_col('low'),
            'volume': get_col('volume') if 'volume' in cols or 'volume' in df.columns else df.get('turnover')
        }, index=df.index)
        out = out.dropna()
        # 统一数值列为 float，避免 decimal 与 float 运算冲突
        for col in out.columns:
            if not np.issubdtype(out[col].dtype, np.number) or out[col].dtype == 'object':
                out[col] = pd.to_numeric(out[col], errors='coerce')
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].astype(float)
        # 兼容大小写列名需求
        for lower, upper in [('open','Open'),('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:
            if lower in out.columns and upper not in out.columns:
                out[upper] = out[lower]
            if upper in out.columns and lower not in out.columns:
                out[lower] = out[upper]
        return out

    def _compute_model_score(self, df: pd.DataFrame) -> float:
        """根据模型输出计算得分，若模型缺失则返回0"""
        if df is None or df.empty or (self.clf_model is None and self.reg_model is None):
            return 0.0
        # 生成特征
        feats_full = self.feature_gen.generate_features(df)
        if feats_full.empty:
            return 0.0
        # 只取最后一个时间点（最新）
        latest_feats = feats_full.iloc[-1:]
        latest_feats = latest_feats.fillna(0.0)
        score_parts = []
        # 分类概率
        if self.clf_model is not None:
            try:
                Xc = self._prepare_features_for_model(self.clf_model, latest_feats)
                if Xc is None or Xc.empty:
                    raise ValueError("无法构建分类模型输入特征矩阵")
                proba = self.clf_model.predict_proba(Xc) if hasattr(self.clf_model, 'predict_proba') else self.clf_model.predict(Xc)
                if proba is not None and len(proba) > 0:
                    prob_up = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0])
                    score_parts.append(prob_up)
            except Exception as e:
                logger.warning(f"分类模型推断失败: {e}")
        # 回归预测
        if self.reg_model is not None:
            try:
                Xr = self._prepare_features_for_model(self.reg_model, latest_feats)
                if Xr is None or Xr.empty:
                    raise ValueError("无法构建回归模型输入特征矩阵")
                pred = self.reg_model.predict(Xr)
                if pred is not None and len(pred) > 0:
                    ret_pred = float(pred[0])
                    # 简单缩放到0-1区间（假设合理范围 -0.2 ~ 0.2）
                    ret_scaled = (ret_pred + 0.2) / 0.4
                    ret_scaled = max(0.0, min(1.0, ret_scaled))
                    score_parts.append(ret_scaled)
            except Exception as e:
                logger.warning(f"回归模型推断失败: {e}")
        if not score_parts:
            return 0.0
        return float(np.mean(score_parts))

    def pick_stocks(self, as_of_date: Optional[pd.Timestamp] = None, candidates: Optional[List[str]] = None, top_n: Optional[int] = None) -> List[PickResult]:
        """基于信号与风险评分的选股（首版）"""
        if as_of_date is None:
            as_of_date = pd.Timestamp(datetime.now().date())
        syms = candidates or self._get_stock_pool(limit=3000)
        if not syms:
            return []
        results: List[PickResult] = []
        for sym in syms:
            df = self._fetch_history(sym, end_date=as_of_date, days=self.lookback_days)
            if df is None or df.empty or len(df) < 30:
                continue
            # 模型分数
            model_score = self._compute_model_score(df)
            # 生成信号
            signals = self.signal_gen.generate_signals(df, symbol=sym) or []
            latest_signal = signals[-1] if len(signals) > 0 else None
            # 风险评分
            risk_info = self.risk_manager.assess_signal_risk(df, {'type': latest_signal.signal_type if latest_signal else 'HOLD'})
            risk_score = float(risk_info.get('risk_score', 0.5))
            # 信号强度
            signal_strength = float(latest_signal.strength) if latest_signal else 0.0
            # 综合分数
            score = self.w_model * model_score + self.w_signal * signal_strength - self.w_risk * risk_score
            reason = f"model={model_score:.2f}|sig={signal_strength:.2f}|risk={risk_score:.2f}"
            results.append(PickResult(symbol=sym, score=score, reason=reason, signal=latest_signal, risk_score=risk_score))
        # 排序
        k = top_n or self.top_n
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def _calculate_cost_price(self, weight: float, shares: float, initial_capital: float, commission_rate: float, override_price: Optional[float] = None) -> Optional[float]:
        """计算成本价格
        考虑佣金因素，计算公式为：成本价格 = (initial_capital * weight * (1 - commission_rate)) / shares
        如果提供了override_price，则直接使用该价格
        """
        if override_price is not None:
            return override_price
        if shares > 0 and initial_capital > 0:
            alloc = initial_capital * weight
            cost_price = (alloc * (1 - commission_rate)) / shares
            return cost_price
        return None
    
    def _equal_weight_holdings(self, picks: List[PickResult], as_of_date: pd.Timestamp, capital: float) -> List[Holding]:
        """等权分配，按as_of_date收盘价计算份额（批量获取价格）"""
        if not picks:
            return []
        n = len(picks)
        target_weight = 1.0 / n
        holdings: List[Holding] = []
    
        symbols = [p.symbol for p in picks]
        # 批量获取最近5个交易日的历史数据，避免高频逐只调用
        start_date = (as_of_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = as_of_date.strftime("%Y-%m-%d")
        try:
            batch_data = self.data_access.get_stock_data_batch(symbols, start_date=start_date, end_date=end_date)  # type: ignore
        except AttributeError:
            # 兼容旧版本 data_access，不支持 batch 时回退逐只调用
            batch_data = {sym: self._fetch_history(sym, end_date=as_of_date, days=5) for sym in symbols}
    
        for p in picks:
            df = batch_data.get(p.symbol)
            if df is None or df.empty:
                continue
            price = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
            alloc = capital * target_weight
            shares = (alloc * (1 - self.commission_rate)) / price if price > 0 else 0
            holdings.append(Holding(symbol=p.symbol, weight=target_weight, shares=shares))
        return holdings

    def _portfolio_nav(self, holdings: List[Holding], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
        """计算持仓在区间的净值曲线（不含中途交易，仅按持仓份额估值）- 批量获取价格"""
        if not holdings:
            return pd.Series(dtype=float)
    
        symbols = [h.symbol for h in holdings]
        days_needed = int((end_date - start_date).days) + 5
        fetch_start = (end_date - pd.Timedelta(days=days_needed)).strftime("%Y-%m-%d")
        fetch_end = end_date.strftime("%Y-%m-%d")
        try:
            batch_data = self.data_access.get_stock_data_batch(symbols, start_date=fetch_start, end_date=fetch_end)  # type: ignore
        except AttributeError:
            batch_data = {sym: self._fetch_history(sym, end_date=end_date, days=days_needed) for sym in symbols}
    
        # 对齐索引
        price_frames = {}
        for h in holdings:
            df = batch_data.get(h.symbol)
            if df is None or df.empty:
                continue
            # 确保索引为日期类型，避免RangeIndex导致比较错误
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'trade_date' in df.columns:
                    df = df.set_index(pd.to_datetime(df['trade_date']))
                else:
                    df.index = pd.to_datetime(df.index, errors='coerce')
            price_frames[h.symbol] = df["close"][df.index >= start_date]
        if not price_frames:
            return pd.Series(dtype=float)
    
        idx = None
        for ser in price_frames.values():
            idx = ser.index if idx is None else idx.union(ser.index)
        idx = idx.sort_values()
        nav = pd.Series(index=idx, dtype=float)
        for dt in idx:
            value = 0.0
            for h in holdings:
                ser = price_frames.get(h.symbol)
                if ser is None or dt not in ser.index:
                    continue
                value += float(h.shares) * float(ser.loc[dt])
            nav.loc[dt] = value
        return nav

    def run(self, start_date: str, end_date: str, candidates: Optional[List[str]] = None) -> Dict[str, Any]:
        """执行：初始选股建仓→30天调仓→组合净值与指标"""
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        # 初始选股与建仓（以start_dt为as_of_date）
        picks0 = self.pick_stocks(as_of_date=start_dt, candidates=candidates)
        holdings = self._equal_weight_holdings(picks0, as_of_date=start_dt, capital=self.initial_capital)
        # 生成调仓时间点（月末近似）
        rebal_dates = pd.date_range(start=start_dt, end=end_dt, freq=self.rebalance_freq)
        picks_history: List[Dict[str, Any]] = []
        nav_full = pd.Series(dtype=float)
        last_date = start_dt
        capital = self.initial_capital

        for i, dt in enumerate(rebal_dates):
            # 区间净值
            seg_nav = self._portfolio_nav(holdings, start_date=last_date, end_date=dt)
            if seg_nav is not None and not seg_nav.empty:
                nav_full = pd.concat([nav_full, seg_nav])
                capital = float(seg_nav.iloc[-1])
            # 记录当前持仓的价值与选股结果
            picks_history.append({
                'date': dt.strftime('%Y-%m-%d'),
                'picks': [p.symbol for p in (picks0 if i == 0 else self.pick_stocks(as_of_date=dt, candidates=candidates))]
            })
            # 调仓：以dt为as_of_date的重新等权
            picks_now = self.pick_stocks(as_of_date=dt, candidates=candidates)
            holdings = self._equal_weight_holdings(picks_now, as_of_date=dt, capital=capital)
            last_date = dt

        # 最后区间
        if last_date < end_dt:
            seg_nav = self._portfolio_nav(holdings, start_date=last_date, end_date=end_dt)
            if seg_nav is not None and not seg_nav.empty:
                nav_full = pd.concat([nav_full, seg_nav])

        # 指标计算
        metrics = {}
        if nav_full is not None and not nav_full.empty:
            initial = float(nav_full.iloc[0]) if nav_full.iloc[0] > 0 else self.initial_capital
            final = float(nav_full.iloc[-1])
            total_return = (final - initial) / initial if initial > 0 else 0.0
            days = max((nav_full.index[-1] - nav_full.index[0]).days, 1)
            try:
                annualized = ((1 + total_return) ** (365.0 / days) - 1) if (1 + total_return) > 0 else -1.0
            except Exception:
                annualized = 0.0
            # 最大回撤
            cum = nav_full.values
            running_max = np.maximum.accumulate(cum)
            drawdowns = (cum - running_max) / running_max
            max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            metrics = {
                'initial_value': initial,
                'final_value': final,
                'total_return': total_return,
                'annualized_return': annualized,
                'max_drawdown': abs(max_dd)
            }

        return {
            'nav': nav_full,
            'picks_history': picks_history,
            'metrics': metrics
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pipeline = PortfolioPipeline()
    result = pipeline.run(start_date='2024-01-01', end_date='2024-06-30')
    print("Metrics:", result.get('metrics'))
    print("Picks first:", (result.get('picks_history') or [{}])[0])
