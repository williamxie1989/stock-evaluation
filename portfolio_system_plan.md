# 模型→选股→建模拟组合→定期调仓→计算组合收益 完整方案

> **阶段B/C 进度更新（2025-09-28）**
>
> - **阶段B – 模型集成与打分融合：已完成核心功能落地**  
>   1. `PortfolioPipeline` 已支持加载分类/回归模型文件（`joblib.pkl`），并通过 `EnhancedFeatureGenerator` 生成特征完成预测。  
>   2. 新增 `_compute_model_score()` 方法：分类概率与回归收益预测经 0-1 归一后求均值，写入综合评分公式 `score = w_model * model + w_signal * signal - w_risk * risk`。  
>   3. CLI 脚本 `scripts/run_portfolio_pipeline.py` 支持指定模型目录、模型文件名、融合权重，运行一次性回测。
>
> - **阶段C – 风控与交易成本加强：初版功能已对接**  
>   1. `AdaptiveTradingSystem` 新增 `evaluate_positions()`：运行中检查止损/止盈条件，返回触发列表供调仓或预警。  
>   2. 回测脚本集成 `AdaptiveTradingSystem`，在每日结束后输出止损/止盈提示，并在定期调仓环节强制平仓触发票。  
>   3. 交易手续费（commission_rate）已在 `execute_trade()` 与 `close_position()` 中计算；调仓逻辑同时扣除交易成本。
>
> 下一步计划：  
> - 继续完善 **阶段C**：实现换手率/最大持仓约束、根据仓位动态调整止损阈值；  
> - **阶段D**：生成组合净值图、持仓走势图和评估报告，并加入 HTML 报告模板。  


## 一、现有能力梳理（已验证模块）
- 回测引擎（`src/core/backtest_engine.py`）
  - 交易执行、组合记录、收益与风控指标计算（总收益、年化、最大回撤、Sharpe、Sortino、胜率、利润因子等）。
  - 支持以信号驱动的买卖逻辑，含佣金处理与报告生成。
- 风险管理（`src/core/risk_management.py` 与 `src/trading/risk/risk_manager.py`）
  - 风险评分（技术、波动、市场）与仓位建议；止损/止盈参数；组合风险评估指标。
- 交易信号（`src/trading/signals/signal_generator.py`、`advanced_signal_generator.py`）
  - 基于均线、RSI、MACD、布林带、成交量、动量等生成买卖/HOLD信号，含强度与过滤。
- 自适应交易系统（`src/trading/systems/adaptive_trading_system.py`）
  - 市场状态识别（趋势/震荡/波动），根据风险等级动态调整仓位、止损止盈、最大持仓数。
- 统一特征与模型训练（`src/ml/features/*`、`src/ml/training/train_unified_models.py`）
  - 增强特征生成、特征选择优化（集成/重要性/稳定性）；统一训练器支持分类/回归、多模型（XGB/LGBM/CatBoost/线性/Stacking），含Optuna与GridSearch、Nested CV与评估。
- 统一数据访问层（`src/core/unified_data_access_factory.py` 及 `src/data/unified_data_access.py`）
  - 获取全市场股票列表、历史数据、实时数据、增量同步与质量报告。
- 评估脚本（`evaluation/evaluation.py`）
  - API分析结果采集、信号收益粗略评估与报告输出。

## 二、目标体系架构
1) 数据层
   - 使用统一数据访问层加载股票池与历史OHLCV，标准化列名为 `open/close/high/low/volume/date`。
   - 提供缓存与限制（股票数/日期范围）以控制训练与回测成本。
2) 特征与标签层
   - 通过增强特征生成器构造多维特征，结合 `FeatureSelectorOptimizer` 自动筛选特征。
   - 标签定义：
     - 分类：未来 `prediction_period` 天收益率是否超过阈值（正例）。
     - 回归：未来 `prediction_period` 天累积收益率预测。
3) 模型层（模型选择）
   - `UnifiedModelTrainer` 统一准备数据与训练，支持 Nested CV；选择指标最佳的模型（分类AUC或回归R²），持久化模型及特征列表。
4) 选股层（打分与过滤）
   - 以最新特征做推断，得到每只股票的预测得分（概率或预期收益）。
   - 融合信号强度（AdvancedSignalGenerator）与风险评分（RiskManager），形成综合选股分数与理由。
   - 产出 Top N 股票及其目标权重建议（可选等权、风险平价占位）。
5) 组合构建与仓位管理
   - 初始建仓：等权或风险权重；应用止损/止盈参数；保留现金比例可配置。
6) 定期调仓
   - 频率：月度/周度；
   - 调仓规则：
     - 更新模型预测与信号；
     - 退出低分股票、引入高分新股票；
     - 控制换手与交易成本；
     - 维持目标权重范围（如±2%带）。
7) 收益计算与评估
   - 组合净值序列、收益曲线与指标（总收益、年化、回撤、Sharpe/Sortino、胜率、利润因子等）。
   - 输出报告与CSV/JSON结果，支持绘图与持仓变动明细。

## 三、实现计划（分步落地）
阶段A：方案与骨架
- 编写并保存完整方案文档（本文件）。
- 新增 `src/trading/portfolio/portfolio_pipeline.py`：
  - `PortfolioPipeline` 类：封装模型选择→选股→建仓→调仓→收益计算全流程的接口。
  - 先实现基于信号+风险的选股与等权组合，月度调仓，组合收益序列计算。

阶段B：模型集成与打分融合
- 接入 `UnifiedModelTrainer`：训练并选择最佳模型，缓存/加载模型；预测股票打分。
- 打分融合：`score = w_model * model_score + w_signal * signal_strength - w_risk * risk_score`。

阶段C：风控与交易成本加强
- 应用止损/止盈触发；
- 佣金与滑点处理；
- 换手率与最大单票权重限制。

阶段D：评估与可视化
- 生成评估报告与可视化图表；
- 接入 `evaluation/evaluation.py` 或新增脚本，输出可读报告。

## 四、接口设计（核心）
- `PortfolioPipeline.run(start_date, end_date, top_n, rebalance_freq)`
  - 返回：`{ nav: pd.Series, picks_history: List, trades: List, metrics: Dict }`
- `PortfolioPipeline.pick_stocks(as_of_date, candidates, top_n)`
  - 返回：`List[Dict]`，包含 `symbol/score/reason/signal/risk`。
- `PortfolioPipeline.rebalance(date, current_holdings)`
  - 返回：新目标持仓与交易指令。

## 五、默认超参数（可在构造器中配置）
- 初始资金：`100_000`；
- 佣金率：`0.0003`；
- 调仓频率：`M`（月度）；
- Top N：`20`；
- 看盘回溯天数：`120`；
- 打分融合权重：`w_model=0.5, w_signal=0.3, w_risk=0.2`（先占位，后续网格/Optuna优化）。

## 六、后续扩展
- 风险平价/最小方差组合构建；
- 交易执行模拟（挂单/成交价偏差）；
- 因子库与Alpha研究接口；
- 在线推理与策略服务化。

—— 以上方案将与代码实现保持一致迭代更新。

## 七、组合管理功能扩展（2025-10-07）

### 后端能力
- `src/services/portfolio/portfolio_management_service.py` 新增组合绩效管道：创建组合时记录风险等级、基准、策略标签，实时计算净值、日收益、累计收益并生成调仓历史。
- 提供 `nav_history`、`rebalance_history`、扩展持仓快照（含名称、最新价、持仓市值、建仓日期、盈亏）等结构，便于前端直接渲染。
- `/api/portfolios` 支持按时间点查询组合列表，返回汇总指标；`/api/portfolios/{pid}` 返回完整详情；`/api/portfolios/{pid}/holdings` 同步提供持仓、指标与净值走势。

### 前端体验
- `static/portfolios.html` 重构列表页，展示组合净值、日收益、累计收益、风险等级与策略标签，提供新建组合弹窗（可输入初始资金）。
- `static/portfolio_detail.html` 新增净值走势图（Chart.js）、持仓收益明细、调仓历史时间线以及关键指标颜色标记，整体界面保持 Tailwind 风格。
- 路由层新增 `/portfolios`、`/portfolios.html`、`/portfolio_detail.html`，前端可直接访问，解决此前 404 问题。

### 最佳实践补充
- 组合返回值中附带风险等级、基准对照、策略标签，利于后续风控与绩效归因。
- 默认持久化在内存，短期内满足页面展示；后续可将 PortfolioDetail 映射到数据库表（组合信息、持仓快照、指标、调仓历史）以实现重启恢复。
- 净值计算目前基于等权持仓 + 最新行情，后续可集成真实交易成本、现金头寸与基准收益比较，并通过定时任务刷新。
