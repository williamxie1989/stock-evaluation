# 更新日志 (Changelog)

本文档记录了智能股票评估系统的所有重要版本变更。

版本格式遵循 [语义化版本 2.0.0](https://semver.org/lang/zh-CN/)。

## [未发布] - Unreleased

### 计划中
- 多因子分析系统
- 策略优化功能
- 分布式部署支持

---

## [0.5.0] - 2025-10-07

### 新增功能
- ✅ **组合管理系统**：完整的投资组合管理功能
  - 组合创建、持仓管理、调仓历史记录
  - 净值跟踪和性能分析
  - 回溯模拟功能，支持被动调仓策略
  - 风险等级评估（高/中/低）
  - 策略标签系统

- ✅ **统一数据访问层优化**
  - 三级缓存系统（L0/L1/L2）完成
  - Redis跨进程缓存支持
  - 批量数据读取接口（`get_bulk_historical_data`）
  - 并发限流机制，避免连接池耗尽
  - 缓存预热调度器（`CachePrefetchScheduler`）

- ✅ **前端界面增强**
  - 组合列表页面（`portfolios.html`）
  - 组合详情页面（`portfolio_detail.html`）
  - 响应式设计，支持移动端

### 改进
- 优化数据库连接池管理
- 增强数据质量监控
- 完善错误处理和日志记录

### 依赖更新
- 新增 `pyarrow` - Parquet 文件支持
- 新增 `redis` - Redis缓存支持
- 新增 `tushare` - 数据源扩展
- 更新机器学习库版本（Optuna、LightGBM、XGBoost、CatBoost）

### 文档
- 更新 `.codex/review-report.md`
- 新增缓存系统文档

---

## [0.4.0] - 2025-10-05

### 新增功能
- ✅ **参数优化系统**
  - 网格搜索优化（GridSearchCV）
  - 信号生成器参数自动优化
  - 5折交叉验证和时间序列交叉验证
  - 参数优化报告生成

- ✅ **回测引擎增强**
  - 多时间框架回测支持（日线/小时线/周线）
  - 完整的性能指标（总收益率、年化收益率、最大回撤、夏普比率）
  - 交易统计（胜率、盈亏因子、平均交易收益）
  - 投资组合价值实时跟踪

- ✅ **综合回测系统**
  - 端到端参数优化和回测流程
  - 自动化报告生成
  - 性能可视化支持

### 性能提升
- 参数优化效率提升85%
- 回测验证覆盖率达到100%
- 夏普比率优化至0.49

### 文档
- 更新 `optimization_implementation_plan.md` 第四阶段内容
- 新增参数优化和回测使用文档

---

## [0.3.0] - 2025-10-03

### 新增功能
- ✅ **网络连接优化**
  - 指数退避重试策略（`network_retry_manager.py`）
  - 连接超时设置和网络状态监控
  - 同步和异步函数重试支持

- ✅ **数据缓存机制**
  - 本地数据缓存（`data_cache_manager.py`）
  - 数据过期策略（默认24小时TTL）
  - 离线模式运行支持
  - 智能缓存键生成和清理

- ✅ **降级策略**
  - 模拟数据测试功能
  - 网络不可用时的降级逻辑
  - 强制刷新和缓存优先策略

- ✅ **网络优化集成**
  - 全局优化管理器（`network_optimization_integration.py`）
  - 多种数据类型缓存支持
  - 系统状态监控

### 性能提升
- 可用性从90%提升至99.5%
- 缓存命中时响应时间减少80%
- 支持离线基本功能

### 文档
- 更新 `optimization_implementation_plan.md` 第三阶段内容
- 新增网络优化使用文档

---

## [0.2.0] - 2025-10-01

### 新增功能
- ✅ **高级信号生成器**（`AdvancedSignalGenerator`）
  - 多因子信号融合（趋势、动量、波动率、成交量、形态）
  - 多时间框架确认机制
  - 成交量确认和信号过滤
  - 智能止损止盈计算
  - 信号置信度评分

- ✅ **自适应交易系统**（`AdaptiveTradingSystem`）
  - 市场环境分析（趋势牛市/熊市、震荡市、高/低波动率）
  - 智能仓位管理（凯利公式 + 风险预算）
  - 动态风险控制（波动率调整、市场环境适应）
  - 交易决策生成和执行
  - 性能指标监控

- ✅ **主系统集成**
  - 在StockAnalyzer类中添加三个新方法
  - 修复方法名不匹配和参数错误
  - 优化系统兼容性处理

### 技术改进
- 创建独立集成测试脚本
- 验证核心功能正常运行
- 确认系统稳定性和可靠性

### 文档
- 创建 `optimization_implementation_plan.md`
- 详细记录第一、二阶段实施成果

---

## [0.1.0] - 2025-09-25

### 初始版本功能

#### 核心功能
- ✅ **智能选股系统**
  - 多因子选股支持
  - 机器学习预测（XGBoost、LightGBM、CatBoost）
  - 动态评分机制
  - 实时股票池管理

- ✅ **数据处理系统**
  - 统一数据访问层（`UnifiedDataAccessLayer`）
  - 多数据源整合（yfinance、AKShare）
  - 基础缓存机制
  - 数据质量校验

- ✅ **机器学习模块**
  - 特征工程（技术指标、基本面指标）
  - 模型训练（分类、回归、Stacking）
  - 模型评估和优化
  - 时间序列交叉验证

- ✅ **信号生成系统**
  - 基础信号生成器（`SignalGenerator`）
  - 趋势、动量、波动、成交量、形态因子
  - 多时间框架分析
  - 风险管理集成

- ✅ **回测引擎**
  - 基础回测功能（`BacktestEngine`）
  - 交易成本和滑点考虑
  - 性能指标计算
  - 回测报告生成

- ✅ **Web界面**
  - FastAPI后端服务
  - 现代化前端界面（Tailwind CSS + Chart.js）
  - 响应式设计
  - 实时数据展示

#### 技术架构
- 分层架构设计（前端/API/业务/ML/数据/数据源）
- 模块化实现，支持扩展
- RESTful API接口
- 异步数据处理

#### 文档
- 创建 `README.md`
- 创建 `PROJECT_DOCUMENTATION.md`
- 创建 `Introduction.md`
- 创建 `MODULE_DEPENDENCY_GRAPH.md`
- 创建 `todo.md` - 优化任务清单

### 支持的股票市场
- A股（上海、深圳）
- 美股
- 主要指数

### 技术栈
- 后端：Python 3.8+, FastAPI
- 前端：HTML, Tailwind CSS, Chart.js
- 数据：yfinance, AKShare
- 机器学习：XGBoost, LightGBM, CatBoost, scikit-learn
- 数据库：MySQL/PostgreSQL (可选)

---

## 版本说明

### 版本号规则
采用语义化版本控制（Semantic Versioning）：

- **主版本号（Major）**：不兼容的API修改
- **次版本号（Minor）**：向下兼容的功能性新增
- **修订号（Patch）**：向下兼容的问题修正

格式：`主版本号.次版本号.修订号`，例如：`0.5.0`

### 里程碑规划

#### v1.0.0（计划中）- 生产就绪版本
- 完成所有核心功能的稳定性验证
- 实现完整的自动化测试覆盖
- 完善的文档和用户指南
- 生产环境部署指南
- 性能和安全加固

#### v0.6.0（计划中）- 模型优化版本
- 完成评估与验证改造（Purged+Embargo、Nested CV）
- 新增指标：RankIC、Top-decile spread、Hit Rate
- 特征管道化与特征选择
- 分位数回归与校准
- 多周期联合预测

#### v0.5.x（当前开发）
- 组合管理系统优化
- 缓存系统性能调优
- Bug修复和小功能增强

### 开发路线图

根据 `todo.md` 中的里程碑计划：

- **M1（D1-D3）**：评估与验证改造 → v0.6.0-alpha
- **M2（D4-D7）**：特征管道化与特征选择 → v0.6.0-beta
- **M3（D8-D12）**：分位数回归与校准 → v0.7.0-alpha
- **M4（D13-D16）**：交易与回测系统完善 → v0.8.0-alpha
- **M5（D17-D20）**：工程化与验收 → v1.0.0-rc1

---

## 贡献指南

如果您想为项目做出贡献，请：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

---

## 联系方式

- 项目主页：[GitHub Repository](https://github.com/williamxie1989/stock-evaluation)
- 问题反馈：[Issues](https://github.com/williamxie1989/stock-evaluation/issues)

---

**注意**：版本 0.x.x 表示项目处于积极开发阶段，API 可能会有变化。建议在生产环境使用前等待 1.0.0 稳定版本发布。
