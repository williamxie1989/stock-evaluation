# 股票评估系统 - 模块依赖关系图

## 1. 整体架构依赖图

```mermaid
graph TB
    %% 前端层
    A[前端界面<br/>static/index.html<br/>Tailwind CSS + Chart.js]
    
    %% API层
    B[FastAPI应用<br/>src/apps/api/app.py]
    
    %% 业务服务层
    C1[市场选择服务<br/>MarketSelectorService]
    C2[股票列表管理<br/>StockListManager]
    C3[投资组合服务<br/>PortfolioManagementService]
    C4[数据修复服务<br/>DataRepairService]
    
    %% 交易信号层
    D1[高级信号生成器<br/>AdvancedSignalGenerator]
    D2[基础信号生成器<br/>SignalGenerator]
    D3[自适应交易系统<br/>AdaptiveTradingSystem]
    
    %% 机器学习层
    E1[特征工程生成器<br/>EnhancedFeatureGenerator]
    E2[统一模型验证器<br/>UnifiedModelValidator]
    E3[训练管理器<br/>TrainingManager]
    E4[统一训练器<br/>UnifiedTrainer]
    
    %% 数据访问层
    F1[统一数据访问层<br/>UnifiedDataAccessLayer]
    F2[统一数据库管理器<br/>UnifiedDatabaseManager]
    F3[数据同步服务<br/>DataSyncService]
    F4[并发数据同步<br/>ConcurrentDataSyncService]
    
    %% 数据提供者
    G1[增强实时数据提供者<br/>EnhancedRealtimeProvider]
    G2[优化增强数据提供者<br/>OptimizedEnhancedDataProvider]
    
    %% 数据源
    H1[yfinance数据源]
    H2[AKShare数据源]
    
    %% 依赖关系
    A --> B
    
    B --> C1
    B --> C2
    B --> C3
    B --> C4
    B --> D1
    B --> D2
    B --> F1
    
    C1 --> F1
    C2 --> F1
    C3 --> F1
    C4 --> F1
    
    D1 --> E1
    D1 --> F1
    D2 --> F1
    D3 --> F1
    
    E1 --> F1
    E2 --> F1
    E3 --> F1
    E4 --> F1
    
    F1 --> F2
    F1 --> F3
    F1 --> G1
    F1 --> G2
    
    F3 --> G1
    F3 --> G2
    F4 --> F1
    
    G1 --> H1
    G1 --> H2
    G2 --> H1
    G2 --> H2
```

## 2. 核心模块详细依赖

### 2.1 API服务层依赖
```mermaid
graph LR
    APP[app.py] --> UDL[UnifiedDataAccessLayer]
    APP --> MSS[MarketSelectorService]
    APP --> SLM[StockListManager]
    APP --> ISS[IntelligentStockSelector]
    APP --> SG[SignalGenerator]
    APP --> CDS[ConcurrentDataSyncService]
    APP --> DSS[DataSyncService]
    
    ISS --> UDL
    MSS --> UDL
    SLM --> UDL
    SG --> UDL
    CDS --> UDL
    DSS --> UDL
```

### 2.2 机器学习模块依赖
```mermaid
graph TD
    EF[EnhancedFeatureGenerator] --> UDL
    UT[UnifiedTrainer] --> UDL
    TM[TrainingManager] --> UDL
    UV[UnifiedModelValidator] --> UDL
    
    UT --> EF
    TM --> EF
    TM --> UT
    UV --> UT
```

### 2.3 交易信号模块依赖
```mermaid
graph TD
    ASG[AdvancedSignalGenerator] --> UDL
    ASG --> EF
    SG[SignalGenerator] --> UDL
    ATS[AdaptiveTradingSystem] --> UDL
    
    ASG --> SG
    ATS --> ASG
```

### 2.4 投资组合模块依赖
```mermaid
graph TD
    PP[PortfolioPipeline] --> UDL
    PP --> ASG
    PP --> ISS
    
    PMS[PortfolioManagementService] --> UDL
    PMS --> PP
```

## 3. 数据流依赖链

```mermaid
graph LR
    %% 数据流向
    DS[数据源<br/>yfinance/AKShare] --> EP[增强数据提供者]
    EP --> UDL[统一数据访问层]
    UDL --> EF[特征工程]
    UDL --> ML[机器学习模型]
    UDL --> SG[信号生成]
    
    EF --> ML
    ML --> SG
    SG --> ATS[自适应交易]
    ATS --> PP[投资组合]
    
    PP --> API[API服务]
    API --> UI[前端界面]
```

## 4. 服务初始化依赖顺序

```mermaid
graph TD
    Start[应用启动] --> InitUDL[初始化统一数据访问层]
    InitUDL --> InitDB[初始化数据库管理器]
    InitUDL --> InitServices[初始化业务服务]
    InitServices --> InitSelector[初始化选股服务]
    InitServices --> InitSignal[初始化信号生成器]
    InitServices --> InitSync[初始化数据同步服务]
    
    InitSelector --> LoadModels[加载预训练模型]
    InitSignal --> InitTrading[初始化交易系统]
    InitSync --> PreloadCache[预加载缓存]
    
    LoadModels --> StartScheduler[启动缓存调度器]
    InitTrading --> StartScheduler
    PreloadCache --> StartScheduler
    
    StartScheduler --> Ready[服务就绪]
```

## 5. 关键依赖说明

### 5.1 核心依赖
- **pandas**: 数据处理和分析
- **numpy**: 数值计算
- **scikit-learn**: 机器学习算法
- **xgboost/lightgbm/catboost**: 梯度提升算法
- **fastapi**: Web服务框架
- **sqlalchemy**: 数据库ORM
- **redis**: 缓存服务
- **yfinance/akshare**: 股票数据源

### 5.2 模块间耦合度
- **低耦合**: API层与业务逻辑层通过接口解耦
- **中耦合**: 机器学习模块与数据访问层紧密协作
- **高耦合**: 统一数据访问层是所有模块的基础依赖

### 5.3 循环依赖处理
- 使用延迟导入（lazy import）避免循环依赖
- 通过统一数据访问层作为中介减少直接依赖
- 服务层采用依赖注入模式降低耦合度

## 6. 扩展性设计

### 6.1 横向扩展
- 数据访问层支持多数据源并发访问
- 机器学习模块支持分布式训练
- 交易信号模块支持多策略并行

### 6.2 纵向扩展
- 统一数据访问层支持缓存分层
- 特征工程支持特征插件扩展
- 投资组合管理支持多策略组合

### 6.3 插件化架构
- 数据源插件化，支持新数据源接入
- 机器学习算法插件化，支持新算法集成
- 交易策略插件化，支持自定义策略开发