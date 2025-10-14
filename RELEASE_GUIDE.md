# 发布管理指南 (Release Management Guide)

本文档提供了智能股票评估系统的版本管理和发布流程指南。

## 目录

1. [版本管理策略](#版本管理策略)
2. [发布流程](#发布流程)
3. [标签管理](#标签管理)
4. [变更日志维护](#变更日志维护)
5. [版本规划](#版本规划)

---

## 版本管理策略

### 语义化版本控制

本项目采用 [语义化版本 2.0.0](https://semver.org/lang/zh-CN/) 规范：

```
主版本号.次版本号.修订号 (MAJOR.MINOR.PATCH)
```

- **主版本号（MAJOR）**：进行不兼容的 API 修改时递增
- **次版本号（MINOR）**：向下兼容的功能性新增时递增
- **修订号（PATCH）**：向下兼容的问题修正时递增

### 版本号示例

- `v0.1.0` - 初始开发版本
- `v0.2.0` - 添加新功能（向下兼容）
- `v0.2.1` - Bug 修复（向下兼容）
- `v1.0.0` - 第一个稳定版本
- `v2.0.0` - 重大架构调整（可能不兼容）

### 预发布版本

对于未正式发布的版本，可以使用以下标识：

- `v0.6.0-alpha` - Alpha 版本（功能未完成，可能有严重 bug）
- `v0.6.0-beta` - Beta 版本（功能基本完成，可能有 bug）
- `v0.6.0-rc1` - Release Candidate（发布候选版本）

---

## 发布流程

### 1. 准备发布

#### 1.1 确认功能完成

- 所有计划功能已实现
- 所有测试通过
- 文档已更新
- 已知 bug 已修复或记录

#### 1.2 更新版本号

在 `VERSION` 文件中更新版本号：

```bash
echo "0.6.0" > VERSION
```

#### 1.3 更新 CHANGELOG.md

在 `CHANGELOG.md` 中添加新版本的变更记录：

```markdown
## [0.6.0] - 2025-10-15

### 新增功能
- ✅ 评估与验证改造
- ✅ 新增 RankIC 指标

### 改进
- 优化特征工程流程

### 修复
- 修复 XX 问题
```

#### 1.4 更新 README.md

确保 README.md 中的版本标识和文档链接正确。

### 2. 创建版本标签

#### 2.1 手动创建标签

```bash
# 创建带注释的标签
git tag -a v0.6.0 -m "Release v0.6.0 - 评估与验证改造

新增功能:
- ✅ 评估与验证改造（Purged+Embargo、Nested CV）
- ✅ 新增指标：RankIC、Top-decile spread、Hit Rate
- ✅ 特征管道化与特征选择

详见: CHANGELOG.md"

# 查看标签
git show v0.6.0

# 推送标签到远程
git push origin v0.6.0
```

#### 2.2 使用脚本批量创建标签

对于初次设置或批量创建历史标签：

```bash
# 运行标签创建脚本
./scripts/create_release_tags.sh

# 推送所有标签
git push origin --tags
```

### 3. 发布到 GitHub

#### 3.1 创建 GitHub Release

1. 访问 GitHub 仓库的 Releases 页面
2. 点击 "Draft a new release"
3. 选择刚创建的标签（如 `v0.6.0`）
4. 填写发布标题：`v0.6.0 - 评估与验证改造`
5. 在描述中粘贴 CHANGELOG.md 中对应版本的内容
6. 可选：上传编译好的二进制文件或文档
7. 点击 "Publish release"

#### 3.2 自动化发布（可选）

可以使用 GitHub Actions 自动化发布流程：

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Create Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          body_path: CHANGELOG.md
```

### 4. 发布后检查

- ✅ 标签已正确创建
- ✅ GitHub Release 已发布
- ✅ 文档已更新
- ✅ 通知相关人员

---

## 标签管理

### 查看所有标签

```bash
# 列出所有标签
git tag

# 列出所有版本标签并排序
git tag -l "v*" | sort -V

# 查看标签详情
git show v0.5.0
```

### 删除标签

```bash
# 删除本地标签
git tag -d v0.5.0

# 删除远程标签
git push origin :refs/tags/v0.5.0

# 或使用以下命令删除远程标签
git push origin --delete v0.5.0
```

### 标签最佳实践

1. **使用带注释的标签**：提供详细的发布说明
   ```bash
   git tag -a v1.0.0 -m "Release message"
   ```

2. **标签命名规范**：始终使用 `v` 前缀
   - ✅ `v0.5.0`
   - ❌ `0.5.0`

3. **不要随意修改标签**：标签应该是不可变的
   - 如果需要修改，应该删除后重新创建

4. **及时推送标签**：创建标签后及时推送到远程
   ```bash
   git push origin v0.5.0
   ```

---

## 变更日志维护

### CHANGELOG.md 结构

```markdown
# 更新日志 (Changelog)

## [未发布] - Unreleased
### 计划中
- 功能规划

## [版本号] - 发布日期
### 新增功能
- 新功能描述

### 改进
- 改进项描述

### 修复
- Bug 修复描述

### 已知问题
- 已知问题列表

### 依赖更新
- 依赖变更说明
```

### 变更分类

- **新增功能（Added）**：新增的功能
- **改进（Changed）**：现有功能的变更
- **废弃（Deprecated）**：即将移除的功能
- **移除（Removed）**：已移除的功能
- **修复（Fixed）**：Bug 修复
- **安全（Security）**：安全相关的修复

### 维护规范

1. **及时更新**：每次重要提交后更新 CHANGELOG
2. **清晰描述**：使用清晰简洁的语言描述变更
3. **链接引用**：重要变更链接到相关 issue 或 PR
4. **用户视角**：从用户角度描述变更影响

---

## 版本规划

### 当前版本

**v0.5.0** - 组合管理系统（2025-10-07）

### 近期规划

#### v0.6.0（计划 2025-10-15）- 模型优化 Alpha

基于 `todo.md` 中的里程碑 M1-M2：

- ✅ 评估与验证改造（Purged+Embargo、Nested CV）
- ✅ 新增指标：RankIC、Top-decile spread、Hit Rate
- ✅ 特征管道化与特征选择
- ✅ 横截面标准化
- ✅ 规范 Stacking OOF 生成

**对应里程碑**：M1（D1-D3）、M2（D4-D7）

#### v0.7.0（计划 2025-10-20）- 模型优化 Beta

基于 `todo.md` 中的里程碑 M3：

- ✅ 分位数回归（LightGBM/XGB Quantile）
- ✅ 输出校准（IsotonicRegression）
- ✅ 不确定性估计
- ✅ 多周期联合预测

**对应里程碑**：M3（D8-D12）

#### v0.8.0（计划 2025-10-25）- 交易系统增强

基于 `todo.md` 中的里程碑 M4：

- ✅ 组合构建优化（Top-N 选股）
- ✅ 成本与滑点模型
- ✅ 风控约束（单票/行业权重上限）
- ✅ 降换手机制
- ✅ 风格/行业中性化

**对应里程碑**：M4（D13-D16）

#### v0.9.0（计划 2025-10-30）- 工程化与监控

基于 `todo.md` 中的里程碑 M5：

- ✅ 统一评估接口与报告
- ✅ 配置化参数与可重复性
- ✅ 日志与模型版本化
- ✅ 监控与告警
- ✅ 验收标准实现

**对应里程碑**：M5（D17-D20）

### 长期规划

#### v1.0.0（目标 2025-11-15）- 生产就绪版本

- ✅ 所有核心功能完成并稳定
- ✅ 完整的自动化测试覆盖
- ✅ 完善的文档和用户指南
- ✅ 生产环境部署指南
- ✅ 性能和安全加固
- ✅ 通过所有验收标准

#### v2.0.0（展望）- 架构升级

- 微服务架构重构
- 分布式计算支持
- 实时流式处理
- 多市场支持（港股、美股、期货等）
- 高级量化策略

---

## 发布检查清单

使用以下检查清单确保发布质量：

### 代码质量

- [ ] 所有单元测试通过
- [ ] 代码审查完成
- [ ] 无严重代码质量问题
- [ ] 性能测试通过

### 文档

- [ ] README.md 已更新
- [ ] CHANGELOG.md 已更新
- [ ] VERSION 文件已更新
- [ ] API 文档已更新（如有变更）
- [ ] 用户指南已更新（如有变更）

### 版本控制

- [ ] 所有变更已提交
- [ ] 创建版本标签
- [ ] 标签已推送到远程
- [ ] GitHub Release 已创建

### 通知

- [ ] 团队成员已通知
- [ ] 用户已通知（如需要）
- [ ] 相关文档已更新

### 部署

- [ ] 部署文档已更新
- [ ] 依赖项已更新
- [ ] 迁移脚本已准备（如需要）
- [ ] 回滚方案已准备

---

## 常见问题

### Q: 如何选择版本号？

A: 遵循语义化版本规范：
- 不兼容的 API 修改 → 增加主版本号
- 向下兼容的新功能 → 增加次版本号
- 向下兼容的 bug 修复 → 增加修订号

### Q: 何时发布新版本？

A: 建议在以下情况发布：
- 完成了一组相关功能
- 修复了重要 bug
- 达到了里程碑目标
- 定期发布（如每月一次）

### Q: 如何处理紧急 bug 修复？

A: 
1. 创建 hotfix 分支
2. 修复 bug 并测试
3. 更新版本号（修订号 +1）
4. 更新 CHANGELOG
5. 创建标签并发布
6. 合并回主分支和开发分支

### Q: 标签和 Release 有什么区别？

A:
- **标签（Tag）**：Git 中的一个引用，指向特定的提交
- **Release（发布）**：GitHub 特性，基于标签创建，可以包含发布说明、附件等

建议同时使用两者，标签用于版本控制，Release 用于发布管理。

---

## 相关资源

- [语义化版本 2.0.0](https://semver.org/lang/zh-CN/)
- [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)
- [Git 标签文档](https://git-scm.com/book/zh/v2/Git-基础-打标签)
- [GitHub Releases 文档](https://docs.github.com/cn/repositories/releasing-projects-on-github)

---

**最后更新**: 2025-10-14
**维护者**: Stock Evaluation Team
