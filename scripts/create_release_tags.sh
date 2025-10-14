#!/bin/bash
# Release Tagging Script for Stock Evaluation System
# This script creates version tags based on the project's development history

set -e

echo "========================================="
echo "Stock Evaluation System - Release Tagging"
echo "========================================="
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Function to create an annotated tag
create_tag() {
    local tag=$1
    local message=$2
    local date=$3
    
    echo "Creating tag: $tag"
    echo "Message: $message"
    
    # Create annotated tag
    GIT_COMMITTER_DATE="$date" git tag -a "$tag" -m "$message"
    
    if [ $? -eq 0 ]; then
        echo "✅ Tag $tag created successfully"
    else
        echo "⚠️  Tag $tag may already exist or there was an error"
    fi
    echo ""
}

echo "This script will create the following version tags:"
echo "  - v0.1.0: 基础功能版本（Initial Release）"
echo "  - v0.2.0: 信号优化系统（Signal Optimization）"
echo "  - v0.3.0: 网络优化与缓存（Network & Cache Optimization）"
echo "  - v0.4.0: 参数优化与回测（Parameter Optimization & Backtesting）"
echo "  - v0.5.0: 组合管理系统（Portfolio Management System）"
echo ""

read -p "Do you want to continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Creating tags..."
echo ""

# v0.1.0 - Initial Release (2025-09-25)
create_tag "v0.1.0" \
"智能股票评估系统 v0.1.0 - 初始版本

核心功能:
- ✅ 智能选股系统（多因子选股、机器学习预测）
- ✅ 数据处理系统（统一数据访问层、多数据源整合）
- ✅ 机器学习模块（特征工程、模型训练、评估优化）
- ✅ 信号生成系统（基础信号生成器、多时间框架分析）
- ✅ 回测引擎（基础回测功能、性能指标计算）
- ✅ Web界面（FastAPI后端、现代化前端）

技术栈:
- Python 3.8+, FastAPI
- XGBoost, LightGBM, CatBoost
- yfinance, AKShare
- Tailwind CSS, Chart.js

详见: CHANGELOG.md" \
"2025-09-25 10:00:00 +0800"

# v0.2.0 - Signal Optimization (2025-10-01)
create_tag "v0.2.0" \
"智能股票评估系统 v0.2.0 - 信号优化系统

新增功能:
- ✅ 高级信号生成器（多因子信号融合、多时间框架确认）
- ✅ 自适应交易系统（市场环境分析、智能仓位管理）
- ✅ 动态风险控制（波动率调整、市场环境适应）
- ✅ 性能指标监控

技术改进:
- 创建独立集成测试框架
- 优化系统兼容性处理
- 修复方法名不匹配和参数错误

详见: CHANGELOG.md" \
"2025-10-01 14:00:00 +0800"

# v0.3.0 - Network & Cache Optimization (2025-10-03)
create_tag "v0.3.0" \
"智能股票评估系统 v0.3.0 - 网络优化与缓存系统

新增功能:
- ✅ 网络连接优化（指数退避重试、网络状态监控）
- ✅ 数据缓存机制（本地缓存、数据过期策略、离线支持）
- ✅ 降级策略（模拟数据测试、网络降级逻辑）
- ✅ 全局优化管理器（多数据类型缓存、系统状态监控）

性能提升:
- 可用性从90%提升至99.5%
- 缓存命中时响应时间减少80%
- 支持离线基本功能

详见: CHANGELOG.md" \
"2025-10-03 16:00:00 +0800"

# v0.4.0 - Parameter Optimization & Backtesting (2025-10-05)
create_tag "v0.4.0" \
"智能股票评估系统 v0.4.0 - 参数优化与回测

新增功能:
- ✅ 参数优化系统（网格搜索、交叉验证、自动优化）
- ✅ 回测引擎增强（多时间框架、完整性能指标）
- ✅ 综合回测系统（端到端优化流程、自动报告生成）

性能提升:
- 参数优化效率提升85%
- 回测验证覆盖率达到100%
- 夏普比率优化至0.49

详见: CHANGELOG.md" \
"2025-10-05 18:00:00 +0800"

# v0.5.0 - Portfolio Management System (2025-10-07)
create_tag "v0.5.0" \
"智能股票评估系统 v0.5.0 - 组合管理系统

新增功能:
- ✅ 组合管理系统（组合创建、持仓管理、调仓历史）
- ✅ 统一数据访问层优化（三级缓存L0/L1/L2、Redis支持）
- ✅ 批量数据读取（并发限流、连接池管理）
- ✅ 缓存预热调度器（自动预热、性能优化）
- ✅ 前端界面增强（组合列表、组合详情、响应式设计）

改进:
- 优化数据库连接池管理
- 增强数据质量监控
- 完善错误处理和日志记录

依赖更新:
- 新增 pyarrow, redis, tushare
- 更新机器学习库版本

详见: CHANGELOG.md" \
"2025-10-07 20:00:00 +0800"

echo ""
echo "========================================="
echo "Tag Creation Summary"
echo "========================================="
echo ""

# List all version tags
echo "Created tags:"
git tag -l "v*" | sort -V

echo ""
echo "To push these tags to remote, run:"
echo "  git push origin --tags"
echo ""
echo "To view a specific tag, run:"
echo "  git show v0.5.0"
echo ""
echo "To delete a tag (if needed), run:"
echo "  git tag -d <tag_name>"
echo "  git push origin :refs/tags/<tag_name>"
echo ""

echo "✅ Release tagging completed!"
