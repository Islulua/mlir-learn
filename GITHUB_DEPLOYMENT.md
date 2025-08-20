# GitHub Pages 部署指南

本指南将帮助你将 MLIR 学习笔记网站部署到 GitHub Pages 上。

## 🚀 部署方式选择

### 方式一：GitHub Actions 自动部署（推荐）
- ✅ 自动化程度高
- ✅ 每次推送代码自动更新网站
- ✅ 无需手动操作
- ✅ 支持分支保护

### 方式二：手动部署
- ✅ 完全控制部署过程
- ✅ 适合一次性部署
- ✅ 可以自定义部署逻辑

## 📋 前置要求

1. **GitHub 账户**
2. **Git 仓库**（可以是公开或私有）
3. **推送权限**（如果是组织仓库）

## 🌐 方式一：GitHub Actions 自动部署

### 1. 推送代码到 GitHub

```bash
# 添加所有文件
git add .

# 提交更改
git commit -m "Add MLIR learning notes website"

# 推送到 GitHub
git push origin main
```

### 2. 检查 GitHub Actions

1. 在 GitHub 仓库页面，点击 **Actions** 标签
2. 查看工作流运行状态
3. 等待部署完成

### 3. 启用 GitHub Pages

1. 在仓库页面，点击 **Settings**
2. 左侧菜单选择 **Pages**
3. **Source** 选择 **Deploy from a branch**
4. **Branch** 选择 **gh-pages**，文件夹选择 **/(root)**
5. 点击 **Save**

### 4. 访问网站

部署完成后，你的网站将在以下地址可用：
```
https://你的用户名.github.io/仓库名/
```

## 🛠️ 方式二：手动部署

### 1. 运行部署脚本

```bash
# 运行自动部署脚本
./deploy-github.sh
```

### 2. 手动步骤（如果脚本失败）

```bash
# 构建网站
export PATH=$PATH:$HOME/.local/bin
mkdocs build

# 进入构建目录
cd site

# 初始化 Git 仓库
git init
git remote add origin https://github.com/你的用户名/仓库名.git

# 添加文件
git add .

# 提交
git commit -m "Deploy MLIR Learning Notes"

# 推送到 gh-pages 分支
git push -f origin HEAD:gh-pages
```

## ⚙️ 配置说明

### GitHub Actions 配置

`.github/workflows/deploy.yml` 文件配置了自动部署流程：

```yaml
name: Deploy MLIR Docs to GitHub Pages

on:
  push:
    branches: [ main, master ]  # 在这些分支推送时触发

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest      # 使用 Ubuntu 最新版本
    
    steps:
    - name: Checkout            # 检出代码
    - name: Set up Python      # 设置 Python 环境
    - name: Install dependencies # 安装依赖
    - name: Build website      # 构建网站
    - name: Deploy to GitHub Pages # 部署到 GitHub Pages
```

### 自定义配置

你可以根据需要修改以下配置：

1. **触发分支**：修改 `branches` 部分
2. **Python 版本**：修改 `python-version`
3. **自定义域名**：在 `cname` 字段填写
4. **发布分支**：修改 `publish_branch`

## 🔧 故障排除

### 常见问题

#### 1. 构建失败

**错误信息**：`Build failed`

**解决方案**：
```bash
# 本地测试构建
mkdocs build

# 检查错误信息
mkdocs serve --verbose
```

#### 2. 部署失败

**错误信息**：`Deploy failed`

**解决方案**：
1. 检查 GitHub Actions 权限
2. 确保仓库设置中启用了 Actions
3. 检查 `GITHUB_TOKEN` 权限

#### 3. 网站无法访问

**问题**：部署成功但网站无法访问

**解决方案**：
1. 检查 GitHub Pages 设置
2. 确认选择了正确的源分支
3. 等待几分钟让更改生效

### 调试步骤

```bash
# 1. 检查本地构建
mkdocs build

# 2. 检查构建结果
ls -la site/
cat site/index.html | head -20

# 3. 检查 Git 状态
git status
git remote -v

# 4. 检查 GitHub Actions 日志
# 在 GitHub 仓库的 Actions 页面查看详细日志
```

## 📱 移动端测试

部署完成后，建议测试移动端效果：

1. 使用手机浏览器访问网站
2. 测试响应式设计
3. 检查触摸操作
4. 验证搜索功能

## 🔄 更新网站

### 自动更新（推荐）
1. 修改文档内容
2. 提交并推送到 GitHub
3. GitHub Actions 自动部署
4. 网站自动更新

### 手动更新
```bash
# 修改内容后
git add .
git commit -m "Update documentation"
git push origin main

# 或者手动重新部署
./deploy-github.sh
```

## 🌍 自定义域名

如果你想使用自定义域名：

1. 在 GitHub 仓库设置中添加自定义域名
2. 在 DNS 提供商处添加 CNAME 记录
3. 在 GitHub Actions 中设置 `cname` 字段

## 📊 监控和统计

### 访问统计
- GitHub 提供基本的访问统计
- 可以集成 Google Analytics
- 支持自定义统计服务

### 性能监控
- 使用 Google PageSpeed Insights 测试
- 监控加载时间和性能指标
- 优化图片和资源

## 🎯 最佳实践

1. **定期更新**：保持文档内容最新
2. **测试部署**：在开发分支测试后再合并到主分支
3. **备份配置**：保存重要的配置文件
4. **监控状态**：定期检查网站状态
5. **用户反馈**：收集用户反馈并改进

## 🎉 部署完成

恭喜！你的 MLIR 学习笔记网站现在已经成功部署到 GitHub Pages 上。

### 下一步建议

1. **分享链接**：将网站链接分享给同事和朋友
2. **收集反馈**：获取用户使用反馈
3. **持续改进**：根据反馈优化网站
4. **扩展功能**：添加更多有用的功能

---

**🌐 你的网站地址：https://你的用户名.github.io/仓库名/**

**📧 如有问题，请检查 GitHub Actions 日志或联系支持。**