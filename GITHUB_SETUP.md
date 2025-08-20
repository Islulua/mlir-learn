# GitHub 管理指南

## 🎯 概述

本指南将帮助你将 MLIR 学习笔记项目与 GitHub 进行关联，实现代码版本控制、协作开发和自动部署。

## 🚀 快速设置

### 方法 1: 使用自动设置脚本

```bash
# 运行设置脚本
./setup-github.sh
```

脚本会引导你输入 GitHub 信息并自动配置。

### 方法 2: 手动设置

## 📋 详细步骤

### 1. 在 GitHub 上创建仓库

1. 访问 [GitHub](https://github.com)
2. 点击右上角 "+" → "New repository"
3. 填写仓库信息：
   - **Repository name**: `mlir-learn`
   - **Description**: MLIR 学习笔记和心得
   - **Visibility**: Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"
4. 点击 "Create repository"

### 2. 配置本地 Git 仓库

```bash
# 添加远程仓库 (替换 your-username 为你的 GitHub 用户名)
git remote add origin https://github.com/your-username/mlir-learn.git

# 验证配置
git remote -v
```

### 3. 推送代码到 GitHub

```bash
# 推送主分支
git push -u origin main

# 推送所有标签
git push --tags
```

### 4. 启用 GitHub Pages

1. 在仓库页面点击 "Settings"
2. 左侧菜单选择 "Pages"
3. Source 选择 "Deploy from a branch"
4. Branch 选择 "gh-pages"，点击 "Save"

## 🔄 日常使用流程

### 添加新内容

```bash
# 1. 编辑文档
# 2. 查看修改状态
git status

# 3. 添加修改的文件
git add .

# 4. 提交更改
git commit -m "添加新的 MLIR 学习内容"

# 5. 推送到 GitHub
git push origin main
```

### 更新现有内容

```bash
# 1. 拉取最新更改
git pull origin main

# 2. 编辑内容
# 3. 提交并推送
git add .
git commit -m "更新 MLIR 方言相关内容"
git push origin main
```

## 🌐 自动部署

项目已配置 GitHub Actions，每次推送到 `main` 分支都会：

1. **自动构建**: 使用 MkDocs 构建网站
2. **自动测试**: 验证构建结果
3. **自动部署**: 部署到 `gh-pages` 分支
4. **自动更新**: GitHub Pages 网站自动更新

### 查看部署状态

1. 在仓库页面点击 "Actions" 标签
2. 查看最新的工作流运行状态
3. 绿色勾号表示部署成功

## 🔧 高级配置

### 分支管理

```bash
# 创建新功能分支
git checkout -b feature/new-dialect

# 开发完成后合并
git checkout main
git merge feature/new-dialect

# 删除功能分支
git branch -d feature/new-dialect
```

### 标签管理

```bash
# 创建版本标签
git tag -a v1.0.0 -m "第一个正式版本"

# 推送标签
git push origin v1.0.0

# 查看所有标签
git tag
```

### 协作开发

1. **Fork 仓库**: 其他用户 Fork 你的仓库
2. **创建分支**: 在 Fork 的仓库中创建功能分支
3. **提交 PR**: 创建 Pull Request 到原仓库
4. **代码审查**: 审查代码并合并

## 📱 移动端管理

### GitHub 移动应用

- 下载 [GitHub 移动应用](https://github.com/mobile)
- 查看仓库状态和 Issues
- 快速回复评论

### 网页版

- 使用浏览器访问 GitHub
- 支持响应式设计
- 功能完整

## 🚨 常见问题

### 1. 推送失败

```bash
# 错误: remote: Permission to xxx denied
# 解决: 检查仓库权限，确保有推送权限

# 错误: failed to push some refs
# 解决: 先拉取最新代码
git pull origin main
git push origin main
```

### 2. 构建失败

- 检查 GitHub Actions 日志
- 验证 `requirements.txt` 依赖
- 确保 Markdown 语法正确

### 3. 网站不更新

- 等待几分钟（GitHub Pages 更新有延迟）
- 检查 `gh-pages` 分支是否有最新内容
- 验证 GitHub Pages 设置

## 🔐 安全建议

1. **不要**在代码中硬编码敏感信息
2. **使用**环境变量存储 API 密钥
3. **定期**更新依赖包
4. **启用**双因素认证

## 📊 监控和分析

### GitHub Insights

- 查看仓库活跃度
- 分析贡献者活动
- 监控代码变化趋势

### 网站分析

- 集成 Google Analytics
- 查看访问统计
- 分析用户行为

## 🎉 开始使用

现在你已经了解了如何使用 GitHub 管理项目：

1. 运行 `./setup-github.sh` 快速配置
2. 按照指南创建 GitHub 仓库
3. 推送代码并启用 GitHub Pages
4. 开始日常的开发和协作

祝你使用愉快！🚀 