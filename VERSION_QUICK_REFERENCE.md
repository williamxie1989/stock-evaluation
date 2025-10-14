# 版本发布快速参考 (Quick Reference)

## 📦 已创建的版本标签

```
v0.1.0 ──> v0.2.0 ──> v0.3.0 ──> v0.4.0 ──> v0.5.0 (当前)
  │          │          │          │          │
  │          │          │          │          └─ 组合管理系统
  │          │          │          └─ 参数优化与回测
  │          │          └─ 网络优化与缓存
  │          └─ 信号优化系统
  └─ 初始版本
```

## 🚀 快速操作指南

### 查看所有标签
```bash
git tag -l "v*"
```

### 查看特定标签详情
```bash
git show v0.5.0
```

### 推送标签到远程（需要权限）
```bash
# 方式1: 使用提供的脚本
./scripts/push_release_tags.sh

# 方式2: 手动推送所有标签
git push origin --tags

# 方式3: 单独推送每个标签
git push origin v0.1.0
git push origin v0.2.0
git push origin v0.3.0
git push origin v0.4.0
git push origin v0.5.0
```

### 在GitHub上查看标签
访问: https://github.com/williamxie1989/stock-evaluation/tags

### 创建GitHub Release
访问: https://github.com/williamxie1989/stock-evaluation/releases/new

## 📋 版本信息一览表

| 版本 | 发布日期 | 主要特性 | 关键改进 |
|------|---------|---------|---------|
| v0.1.0 | 2025-09-25 | 初始版本 | 智能选股、ML模块、基础回测 |
| v0.2.0 | 2025-10-01 | 信号优化 | 高级信号生成器、自适应交易 |
| v0.3.0 | 2025-10-03 | 网络优化 | 缓存机制、离线支持、可用性99.5% |
| v0.4.0 | 2025-10-05 | 参数优化 | 网格搜索、回测增强、夏普0.49 |
| v0.5.0 | 2025-10-07 | 组合管理 | 三级缓存、Redis、批量读取 |

## 🎯 未来版本规划

```
当前 v0.5.0 ──> v0.6.0 ──> v0.7.0 ──> v0.8.0 ──> v0.9.0 ──> v1.0.0
                   │          │          │          │          │
                   │          │          │          │          └─ 生产就绪
                   │          │          │          └─ 工程化与监控
                   │          │          └─ 交易系统增强
                   │          └─ 分位数回归
                   └─ 评估验证改造

预计时间线: 2025-10 → 2025-11-15
```

## 📚 相关文档

| 文档 | 说明 | 路径 |
|-----|------|------|
| CHANGELOG.md | 详细变更日志 | [查看](CHANGELOG.md) |
| RELEASE_GUIDE.md | 发布管理指南 | [查看](RELEASE_GUIDE.md) |
| VERSION | 当前版本号 | [查看](VERSION) |
| README.md | 项目说明 | [查看](README.md) |
| RELEASE_TAGS_CREATED.md | 标签创建通知 | [查看](RELEASE_TAGS_CREATED.md) |

## 🛠️ 脚本工具

| 脚本 | 功能 | 使用方法 |
|-----|------|---------|
| create_release_tags.sh | 创建版本标签 | `./scripts/create_release_tags.sh` |
| push_release_tags.sh | 推送标签到远程 | `./scripts/push_release_tags.sh` |

## ⚠️ 重要提示

1. **标签已创建但未推送**: 由于权限限制，标签已在本地创建，需要在有权限的环境手动推送
2. **推送前确认**: 推送标签前请确保版本信息正确，标签一旦推送应避免修改
3. **创建Release**: 标签推送后，建议在GitHub上创建对应的Release，方便用户下载和查看

## ✅ 验证清单

在推送标签前，请确认：

- [ ] 所有标签已正确创建（5个版本标签）
- [ ] CHANGELOG.md 内容准确完整
- [ ] VERSION 文件显示当前版本
- [ ] README.md 版本信息已更新
- [ ] 所有文档链接正确有效

## 🔗 快速链接

- **仓库地址**: https://github.com/williamxie1989/stock-evaluation
- **标签列表**: https://github.com/williamxie1989/stock-evaluation/tags
- **发布页面**: https://github.com/williamxie1989/stock-evaluation/releases
- **问题追踪**: https://github.com/williamxie1989/stock-evaluation/issues

---

**最后更新**: 2025-10-14
**当前版本**: v0.5.0
**标签总数**: 5
