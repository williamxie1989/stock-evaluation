# 股票实战模型优化任务清单与实施过程

本文档用于跟踪从“研究级指标”到“可交易策略”的落地过程，覆盖评估与验证、特征与预处理、模型与目标、交易与回测、工程化与验收。内容按任务清单 → 实施步骤 → 里程碑与验收组织，支持后续回溯与迭代。

## 缓存系统改造进度总览（2025-10-07）

| 序号 | 任务                                                         | 当前状态 | 说明 |
| ---- | ------------------------------------------------------------ | -------- | ---- |
| 1    | 统一梳理组合持仓、推荐建仓、训练/选股的访问路径与高频场景       | ✅ 已完成 | `reports/unified_data_access_usage.md` 总结热点调用路径与优化建议 |
| 2    | 设计 L0/L1/L2 分层缓存策略                                    | ✅ 已完成 | `UnifiedDataAccessLayer` 内已实现内存 + Redis + Parquet 三级缓存 |
| 3    | 基于 `functools.lru_cache` 的进程级缓存（symbol+date_range+fields） | ✅ 已完成 | `UnifiedDataAccessLayer` 采用 `lru_cache`，涵盖 fields 维度并记录性能指标 |
| 4    | 批量读取接口（多股票/多日期）                                 | ✅ 已完成 | `get_bulk_historical_data` 与同步包装器已提供并通过测试 |
| 5    | 缓存一致性/失效/性能测试                                       | ✅ 已完成 | `tests/test_cache_layers.py`、`tests/test_l2_cache_and_bulk.py` 覆盖 L0/L1/L2 |
| 6    | 引入 Redis 跨进程缓存与版本失效策略                            | ✅ 已完成 | `src/cache/redis_cache.py` 提供封装，可通过环境变量禁用/配置 |
| 7    | 预拉取/预计算调度（Celery/APScheduler）                         | ✅ 已完成 | FastAPI `startup` 中启动 `CachePrefetchScheduler`，支持环境变量配置 |
| 8    | 前端组合/推荐建仓流程去重、缓存化                               | 🚧 进行中 | API 侧已有内存缓存，仍需对信号生成/调仓流程做增量读取校验 |
| 9    | 训练/智能选股重度任务的缓存命中检查                             | 🚧 进行中 | 训练流程仅请求核心字段并依赖新缓存，仍需批量接口与命中监控整合 |
| 10   | 性能监控与日志（访问次数、命中率、耗时）                         | 🚧 进行中 | `UnifiedDataAccessLayer.get_data_quality_report` 仅输出静态统计，待补充实时指标 |

## 一、现状速览（来自近期一次训练）
- 训练参数：`--lookback 365 --n-stocks 1000 --prediction-period 30 --mode regression --reg-models stacking --use-optuna --disable-feature-selection --no-feature-cache`
- Optuna 最优值：Weighted MSE ≈ 0.00772。
- 测试集：IC=0.7623，R²=0.5863，MSE=0.007489（30天收益预测）。
- 模型：Stacking 回归，已保存至 `models/stacking_regression.pkl`。

风险提示：该级别指标显著高于常见横截面实证水平，需优先排查时间泄漏、评估口径偏乐观、预处理跨期信息等问题，确保可交易性评估可信。

---

## 二、任务清单（Workstreams）

### A. 评估与验证
1. ✅ 完成：引入 Purged+Embargo 时间序列交叉验证（embargo=预测期）。
2. 搭建 Nested CV（内层调参，外层评估），隔离超参搜索与最终报告。
3. ✅ 完成：新增指标：RankIC（Spearman，按日跨股票）、Top-decile spread、Hit Rate（正向命中率）。
4. 校验基学习器与 Stacking 次级学习器的 OOF 训练流程，排除叠加泄漏。
5. 走前看后滚动评估（训练窗口→前推测试窗口），输出时间切片稳定性。

### B. 特征与预处理
6. ✅ 完成：横截面标准化（按交易日对所有股票做 z-score），提升排序可比性。
7. ✅ 完成：缺失值与缩放管道化（仅用训练折拟合变换器，再用于验证/测试），已在分类模型训练流程中集成 Imputer → Winsorizer → CrossSectionZScore 管道。
8. 行业 One-Hot 与市值处理（log、分桶、winsorize），降低极端值影响。
9. 启用特征选择（可用 LightGBM/XGB 的重要性、互信息、稳定性筛选），产出并缓存特征子集。
10. 扩展稳健特征库（多尺度动量/波动、流动性、财务质量、事件因子）。

### C. 模型与目标
11. 规范 Stacking 的 OOF 预测生成（时间序列 CV），避免使用未来信息。
12. 引入分位数回归（LightGBM/XGB Quantile），输出不确定性区间用于风控。
13. 增加输出校准（IsotonicRegression 或 Platt-like），提升回归到真实收益的贴合度。
14. 多周期联合或多任务（10/30/60天）以增强稳定性与泛化。
15. 正则化与稳健化（学习率衰减、深度约束、子样本/特征子采样）。

### D. 交易与回测
16. 组合构建：按预测收益排序，买入 Top-N（如 50/100），等权或风险平价。
17. 成本与滑点：双边 10–30bp，T+1成交假设，支持情景敏感性分析。
18. 风控约束：单票/行业权重上限、波动/回撤目标、止损与调仓阈值。
19. 降换手机制：预测优势阈值、持仓惯性，不显著变差不卖。
20. 风格/行业中性化：控制对行业或风格因子的暴露，降低宏观噪声。

### E. 工程化与验收
21. 统一评估接口与报告（训练/验证/回测同口径输出），保存到 `evaluation/`。
22. 配置化参数与可重复性（随机种子、数据切片、CV设置），记录到 `README.md` 与 `.env`。
23. 日志与模型版本化（含超参、特征列表、数据范围、指标），保存到 `models/` 子目录。
24. 监控与告警：训练/回测异常中止、数据缺口、分布漂移（PSI）。
25. 验收标准与失败条件：指标阈值、稳健性、成本弹性、回撤上限等。

---

## 三、详细实施步骤（按优先级归并）

### 1) 评估与验证改造
- 目标：消除评估乐观偏差，真实反映可交易能力。
- 步骤：
  - 在 `src/ml/training/train_unified_models.py` 内新增 `PurgedKFoldWithEmbargo`（或复用 `RollingWindowSplit` 增强版），参数：`window`、`horizon=prediction_period`、`embargo=horizon`。
  - 将 Optuna 调参与 Stacking 的 OOF 生成统一改用该 CV，确保不跨越禁区。
  - 在 `src/ml/evaluation/` 新增评估工具：`rank_ic(y_true_by_day, y_pred_by_day)`、`top_decile_spread`、`hit_rate`。
  - 输出训练/验证折级别指标汇总到 `evaluation/reports/<timestamp>.json`，并生成简表 `evaluation/reports/<timestamp>.md`。

### 2) 特征与预处理管道化
- 目标：避免跨期信息泄漏，提升横截面可比性。
- 步骤：
  - 在训练脚本中构建 `sklearn.Pipeline`：`Imputer(median) -> Winsorizer -> CrossSectionZScore -> Model`。
  - 每个 CV 折：仅用训练折 `fit` 这些变换器；验证/测试折只 `transform`。
  - 行业 One-Hot 与市值处理纳入管道（OneHotEncoder(handle_unknown='ignore')，市值做 log/winsorize）。
  - 启用并缓存特征选择：将选择出的列写入 `feature_cache/selected_features.json`；支持 `--refresh-feature-selection` 重算。

### 3) 模型与目标增强
- 目标：提供稳定预测与不确定性估计，利于风控与持仓决策。
- 步骤：
  - 规范 Stacking：基础学习器使用时间序列 CV 生成 OOF 预测，再训练次级学习器；确认不在同一时间窗交叉使用未来信息。
  - 引入分位数回归（LightGBM/XGB）：输出 `q10/q50/q90`；在交易层以区间置信度过滤持仓。
  - 增加校准层：对 `q50` 或点预测使用 `IsotonicRegression` 校准；保存校准映射。
  - 试验多周期联合：分别训练 10/30/60 天模型，组合加权或多任务网络（如共享特征，独立头部）。

### 4) 交易与回测落地
- 目标：将预测转化为净收益，验证成本与风控下的实盘可行性。
- 步骤：
  - 在 `src/core/backtest_engine.py`（或新建 `src/trading/systems/`）实现：每日或每周根据预测排序构建 Top-N 组合；支持等权/风险平价；行业/单票约束。
  - 成本模型：双边成本与滑点（10–30bp），T+1成交；调参支持。
  - 风控模块：波动/回撤目标、止损规则、曝险上限；持仓惯性/优势阈值控制换手。
  - 报告输出：年化收益、波动、Sharpe、最大回撤、月度收益、换手率、成本敏感性。

### 5) 工程化与验收
- 目标：流程稳定、结果可复现、版本可追踪。
- 步骤：
  - 参数配置：将关键参数写入 `.env`/`data_source_config.env` 与脚本参数解析，记录到训练日志。
  - 版本化：模型保存到 `models/<strategy>/<timestamp>/`，包含 `model.pkl`、`features.json`、`metrics.json`、`config.json`。
  - 监控：数据拉取缺口、特征分布漂移（PSI>0.2 报警）、训练失败自动重试与告警。
  - 验收标准：详见下文。

---

## 四、里程碑与时间计划（建议）
- 里程碑 M1（D1–D3）：完成评估与验证改造（Purged+Embargo、Nested CV、指标扩充），出首份稳健报告。
- 里程碑 M2（D4–D7）：完成特征管道化与特征选择，重训对比报告（含 RankIC/TopN）。
- 里程碑 M3（D8–D12）：接入分位数回归与校准，输出不确定性与风控试验结果。
- 里程碑 M4（D13–D16）：完成交易与回测系统，输出 Sharpe/回撤/换手与成本弹性报告。
- 里程碑 M5（D17–D20）：工程化与验收，建立版本化与监控，冻结可实盘 beta 策略。

---

## 五、验收标准与失败条件
- 指标阈值：
  - RankIC（日频，横截面）：≥ 0.05（稳态期）且稳定为正；
  - Top-decile spread：显著为正，t 统计通过 95% 置信；
  - 回测（含 20bp 成本）：年化 Sharpe ≥ 1.2，最大回撤 ≤ 25%，月度收益正占比 ≥ 60%；
  - 换手率：≤ 30%/月（或在成本敏感性下 Sharpe 保持 ≥ 1.0）。
- 稳健性：不同市场阶段（牛/熊/震荡）与行业分组中性能稳定；PSI 漂移可控（≤ 0.2）。
- 失败条件：上述任一核心指标显著低于阈值且缺乏合理解释与修复方案。

---

## 六、参考实现与改造点（代码级）
- `src/ml/training/train_unified_models.py`
  - 替换 CV 为 Purged+Embargo；封装 `get_cv()` 返回统一分割器。
  - 将预处理改为 `Pipeline` 并在 CV 折内 `fit/transform`；避免全样本中位数泄漏。
  - 规范 Stacking 的 OOF 生成；在训练开始前对 `stacking_model.predict(X_train[:5])` 的烟测保留，但避免影响评估流程。
  - 输出评估与模型元数据到 `evaluation/reports` 与 `models/<timestamp>/`。
- `src/ml/evaluation/`
  - 新增 `rank_ic.py`、`portfolio_metrics.py`；统一报告生成器。
- `src/core/backtest_engine.py` 或 `src/trading/systems/`
  - 实现组合构建、成本与风控模块、报告导出。

---

## 七、运行示例（待完成上述改造后）
```
# 训练（启用特征选择与缓存、Purged+Embargo）
python src/ml/training/train_unified_models.py \
  --lookback 730 --n-stocks 1200 --prediction-period 30 \
  --mode regression --reg-models stacking --use-optuna \
  --enable-feature-selection --feature-cache \
  --cv purged_embargo --embargo 30 --nested-cv

# 评估报告汇总
python src/ml/evaluation/evaluation.py --report-dir evaluation/reports

# 回测（Top-50 等权，成本 20bp，T+1）
python src/trading/systems/run_backtest.py \
  --model-path models/stacking_regression.pkl \
  --top-n 50 --rebalance weekly --cost-bps 20 --tplus 1
```

---

## 八、风险与应对
- 时间泄漏：使用禁区与滚动走前看后评估；代码审计未来信息特征。
- 过拟合：简化模型复杂度、正则与子采样；启用特征选择与横截面标准化。
- 成本与滑点：在不同成本场景（10/20/30bp）回测，确保策略韧性。
- 数据漂移：建立 PSI 监控与模型重训节奏（如每月/每季）。

---

## 九、变更记录（文档层）
- v0.1（2025-10-03）：初版任务清单与实施过程文档，基于近期训练结果与改造建议。
