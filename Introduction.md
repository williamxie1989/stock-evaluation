整体结构速览
仓库根目录包含主程序脚本、核心配置、模型与静态资源等，核心业务代码集中在 src/ 目录；此外还有 static/ 前端页面、models/ 模型文件、scripts/ 与规划文档等辅助资源。

src/ 进一步划分为 apps（运行级应用与接口层）、core（回测与风控等核心引擎）、data（统一数据访问/存储）、trading（信号、组合与交易系统）、services（业务服务包装）、utils 等模块化子目录，便于按职能协作。

项目 README 概述了系统目标、技术栈与运行方式：后端 FastAPI、前端 Tailwind/Chart.js、数据源来自 yfinance 与 AKShare，并强调需要配置 .env 中的大模型与数据访问密钥。

关键模块与协作方式
API 层 (src/apps/api)：app.py 定义 FastAPI 应用、CORS、中台生命周期钩子及静态资源挂载，并在启动时初始化统一数据访问层、智能选股服务、信号生成器、股票列表管理器等，同时触发数据缓存预热与首启数据同步，是前端请求进入系统的主入口。

统一数据访问：UnifiedDataAccessLayer 组合数据库、缓存与多数据源，提供 L0/L1/L2 多级缓存、质量校验、自动同步与性能统计，是数据读写的基础抽象；支持通过配置选择偏好的行情源并自动加载 .env 优先级。

数据提供者：UnifiedDataProvider 维护主备行情源、线程池与数据质量评分，封装缓存键管理与质量评估逻辑，为统一访问层提供底层数据抓取能力。

数据库管理：UnifiedDatabaseManager 兼容 MySQL/SQLite，内置连接池、表结构初始化与通用 CRUD 方法，并允许通过环境变量或配置切换数据库类型。

缓存预取：CachePrefetchScheduler 借助 APScheduler 定时调用统一数据访问层，提前拉取特定股票区间数据，避免接口冷启动延迟。

选股与模型：IntelligentStockSelector 负责加载机器学习模型、增强预处理与特征工程，结合信号生成和股票状态过滤完成候选排序，是 AI 驱动的核心组件。

信号与交易系统：基础 SignalGenerator 覆盖趋势、动量、波动、成交量与形态因子，高级版本 AdvancedSignalGenerator 聚合多种技术指标输出结构化信号，配合 RiskManager 评估风险；PortfolioPipeline 将数据获取、信号、风险与模型打分整合为选股/调仓流程；portfolio_service 再把结果转换成前端所需的持仓结构并做缓存。

遗留/备用脚本：main.py 保留了早期的单体 StockAnalyzer、数据抓取与指标计算逻辑，便于理解系统演进与在非 API 场景快速试验。

