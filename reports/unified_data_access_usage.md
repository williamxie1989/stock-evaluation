# 统一数据访问层热点调用分析（2025-10-07）

本文档针对近期缓存改造任务，对组合持仓、智能推荐建仓、股票训练、智能选股等核心模块的数据访问路径与频次进行梳理，识别高频/大批量读取场景并给出后续优化建议。

## 1. 功能模块与数据访问概览

| 模块 | 代码入口 | 主要用途 | 数据访问特征 |
| ---- | -------- | -------- | ------------ |
| 组合持仓生成 | `src/services/portfolio/portfolio_service.py#L25` `src/trading/portfolio/portfolio_pipeline.py#L92` | 生成模拟组合、估算持仓价格 | 单次生成会触发 `top_n` 只股票的日线抓取 + 特征计算 |
| 智能推荐建仓（前端 `/api/stock-picks`） | `src/apps/api/app.py#L336` | 选股 API，支持限量候选 | 高频调用（前端刷新），需要加载模型 + 批量读取候选股票历史/实时数据 |
| 股票训练流水线 | `src/ml/training/train_unified_models.py#L313` | 大规模批量生成训练集 | 高并发/长时间循环读取（默认 1,000 只股票 * 365 天）|
| 智能选股预测器 | `src/ml/prediction/model_predictor.py#L97` | 单股票预测（批量触发） | 逐票拉取近 90 天数据，命中新模型特征流程 |
| 数据同步服务 | `src/data/unified_data_access.py#L512` | 后台全量/增量同步 | 遍历股票列表逐票同步，常触发自动补数 |

## 2. 高频访问路径细节

### 2.1 组合持仓 / PortfolioPipeline

- `_fetch_history` 每次选股会为候选股票调用 `get_historical_data` （`src/trading/portfolio/portfolio_pipeline.py#L110-L155`），短窗口（默认 120 日）但高频（持仓更新、调仓、信号巡检）。
- 组合生成调用链：`generate_portfolio_holdings` → `PortfolioPipeline.pick_stocks` → `_equal_weight_holdings` → `_fetch_history`（`src/services/portfolio/portfolio_service.py#L25-L114`），前端刷新或后台调度都会触发。
- 建议：
  - 使用批量接口替换循环 `_fetch_history`，减少重复事件循环切换。
  - 将当次选股涉及的符号集传入预拉取任务，提前预热缓存。

### 2.2 智能推荐建仓 API

- `/api/stock-picks` 默认 15 分钟缓存，但在 `force_refresh=1` 或不同参数组合时会重新走 `list_symbols` → `predict_top_n`，内部调用多次行情/特征获取（`src/apps/api/app.py#L336-L417`）。
- 推荐模型加载与候选过滤前均需访问数据库（`UnifiedDatabaseManager.list_symbols`），随后预测阶段对每只候选股票读取最近行情或特征。
- 建议：
  - 将候选股票行情读取改为 `get_bulk_historical_data`，并与模型预测共用一次数据装载。
  - 缓存候选集（标的列表+过滤结果），减少 `list_symbols` 访问频率。

### 2.3 训练流水线

- `prepare_training_data` 先以 `auto_sync=False` 获取本地缓存，数据不足再回退 `auto_sync=True`（`src/ml/training/train_unified_models.py#L313-L404`）。
- 每支股票全部字段读入 + 特征生成后才落盘，缺少对字段选择、缓存命中统计。
- 建议：
  - 在进入循环前批量调用 `get_bulk_stock_data`（或新增支持字段过滤的批量接口）。
  - 记录缓存命中率，若命中率低于阈值提前触发预拉取。

### 2.4 智能选股 / 模型预测

- `StockPredictor.predict` 对单只股票进行 90 日数据加载并经过增强特征/特征选择（`src/ml/prediction/model_predictor.py#L97-L184`）。如果批量调用会连续触发多次独立下载。
- 建议：
  - 在服务层提供批量预测入口，先使用 `get_bulk_historical_data` 拉取所有所需符号的数据，再逐票预测。
  - 结合 L0/L1/L2 缓存命中情况，若数据已在缓存则跳过下载与特征生成。

### 2.5 数据同步 / 自动补数

- `sync_market_data` / `_sync_batch`（`src/data/unified_data_access.py#L488-L556`）和 `sync_market_data_by_date`（同文件 `#L839` 起）属于后台批处理，会遍历全市场符号列表并调用 `get_historical_data`。
- 建议：
  - 调整调度窗口避免与高峰业务时间重叠。
  - 批次内使用批量接口或协程池，结合 Redis 缓存减少重复下载。

## 3. 高频字段与缓存关注点

- 业务层通常只需 `open/high/low/close/volume/turnover`，少数场景访问扩展字段（特征生成时会衍生更多列）。
- 当前缓存键未区分字段，全量缓存导致不同字段组合无法复用。后续改造需将 `fields` 参与 key，并支持按需列选择。
- 训练/预测对数据顺序敏感，应确保缓存返回结果按日期升序并携带索引类型（DatetimeIndex）。

## 4. 后续行动建议

1. **补齐进程级缓存**：用 `functools.lru_cache` 封装 L0 缓存（键含 symbol/date_range/fields），并支持 TTL/逐键失效。
2. **批量接口落地到业务层**：组合、选股、训练模块统一接入 `get_bulk_historical_data`，减少循环调用带来的网络/IO 开销。
3. **缓存预热接入生命周期**：在 FastAPI `startup` 中启动 `CachePrefetchScheduler`，根据常用 symbols + 业务高峰时间预测数据。
4. **性能监控**：统一记录每次访问耗时、命中层级（L0/L1/L2/DB），输出到 `get_data_quality_report` 及日志，便于后续绘制命中率/延迟趋势。
5. **前端流程对接缓存**：对组合与推荐建仓接口增加缓存命中标记、增量更新机制，避免重复触发信号重算。
6. **控制批量抓取并发度**：已在 `get_bulk_historical_data` 内新增 `asyncio.Semaphore` 限流，并支持通过环境变量 `UDA_BULK_CONCURRENCY` 自定义并发阈值（默认从连接池 `pool_size-2` 推断，最低为 1），避免触发 “pool exhausted”。

> 以上分析将作为缓存改造迭代的输入资料，后续在实现 L0 lru_cache、预热调度及性能监控后再迭代评估。
