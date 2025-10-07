# 验证记录（2025-10-07）

- **静态依赖校验**：运行 `python` 脚本比对源码导入与 `requirements.txt`，所有第三方模块均已覆盖。
- **代码编译检查**：执行 `python -m compileall src/data/unified_data_access.py`，确认新增并发限流逻辑通过语法检查。
- **组合模块语法验证**：执行 `python -m compileall src/services/portfolio/portfolio_management_service.py`，确认数据库迁移后代码可正常编译。
- **组合功能单元测试**：尝试运行 `pytest tests/services/test_portfolio_management_service.py`（含 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`NPY_DISABLE_MACOSX_ACCELERATE=1` 等降级方案），均在导入 `numpy` 的 `numpy._mac_os_check()` 阶段触发 Segmentation Fault（MacOS Accelerate 库缺失）。需在部署环境补齐 BLAS/Accelerate 依赖后重试。
- **未执行测试说明**：当前环境未安装新增依赖（pyarrow/catboost/xgboost/lightgbm 等），无法运行 `pytest`、集成测试或 FastAPI 冒烟测试。待依赖安装完成后需补充以下验证：
  1. `pytest`
  2. FastAPI 启动冒烟 (`uvicorn src.apps.api.app:app --reload`)
  3. 关键训练脚本（`src/ml/training/train_unified_models.py`）的最小样本运行
- **额外验证待办**：部署环境需连接真实 MySQL/Redis，触发 FastAPI 启动预热流程，确认 `pool exhausted` 日志不再出现并观察批量预热耗时。
- **风险评估**：高性能依赖多为 C/C++ 扩展包，可能在不同平台需额外系统库；需在部署阶段确认可用性。
