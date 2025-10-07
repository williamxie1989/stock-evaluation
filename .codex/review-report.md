# 审查报告（2025-10-07）
- **任务**：更新项目依赖并修复 app.py 运行缺失的 Parquet 引擎；限制批量历史数据并发，避免数据库连接池耗尽
- **审查者**：Codex

## 评分
- **技术维度**：90/100 — requirements.txt 新增依赖完整，并发限流通过信号量实现且保留可配置能力；仍缺线上验证。
- **战略维度**：85/100 — 关键链路已补齐（依赖 + 连接池保护），但测试未运行，需要上线前补完。
- **综合评分**：88/100
- **结论**：需改进（待环境安装依赖后补跑测试与冒烟验证）

## 核对要点
- ✅ 缺失依赖（pyarrow、redis、tushare、Optuna、LightGBM、XGBoost、CatBoost 等）已写入 requirements.txt
- ✅ 生成 `.codex/testing.md`、`verification.md` 留存验证范围
- ✅ `get_bulk_historical_data` 增加 `asyncio.Semaphore` 并支持 `UDA_BULK_CONCURRENCY`，默认并发度基于连接池容量自动推断
- ⚠️ 未执行 pytest / FastAPI 冒烟，需在具备依赖的环境完成
- ⚠️ 新增依赖包含多项 C/C++ 扩展，部署需确认编译链或使用官方轮子
- ⚠️ 并发限流未在真实 MySQL/Redis 环境验证，需要观察预热耗时与连接资源占用

## 风险与阻塞
1. **安装风险**：pyarrow、lightgbm、xgboost、catboost 对系统环境要求较高，建议在部署文档明确所需编译组件。
2. **验证缺失**：缺少线上实际运行（uvicorn 启动、训练脚本、缓存预热）与单测结果。

## 建议动作
1. 在具备网络与编译环境的机器安装新依赖，执行 pytest 与关键脚本冒烟。
2. 启动 FastAPI 预热或直接调用 `IntelligentStockSelector.get_latest_features`，确认无 “pool exhausted” 日志并评估批量耗时。
3. 记录安装注意事项（如系统包、CUDA/OMP 依赖）供部署团队参考。
