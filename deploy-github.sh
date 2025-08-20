#!/bin/bash

# MLIR 学习笔记 GitHub Pages 部署脚本

echo "🚀 开始部署 MLIR 学习笔记到 GitHub Pages..."

# 检查是否在 Git 仓库中
if [ ! -d ".git" ]; then
    echo "❌ 错误：当前目录不是 Git 仓库"
    echo "请先初始化 Git 仓库：git init"
    exit 1
fi

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  警告：有未提交的更改"
    echo "建议先提交更改：git add . && git commit -m 'Update docs'"
    read -p "是否继续？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 设置 PATH
export PATH=$PATH:$HOME/.local/bin

# 构建网站
echo "📚 构建网站..."
if ! mkdocs build; then
    echo "❌ 构建失败"
    exit 1
fi

echo "✅ 网站构建成功"

# 检查构建结果
if [ ! -d "site" ]; then
    echo "❌ 错误：未找到 site 目录"
    exit 1
fi

if [ ! -f "site/index.html" ]; then
    echo "❌ 错误：未找到 index.html"
    exit 1
fi

echo "📁 构建文件检查通过"

# 创建 gh-pages 分支
echo "🌿 创建 gh-pages 分支..."
cd site

# 初始化 Git 仓库（如果还没有）
if [ ! -d ".git" ]; then
    git init
    git remote add origin "$(cd .. && git remote get-url origin)"
fi

# 添加所有文件
git add .

# 提交更改
git commit -m "Deploy MLIR Learning Notes - $(date)"

# 推送到 gh-pages 分支
echo "🚀 推送到 GitHub Pages..."
if git push -f origin HEAD:gh-pages; then
    echo "✅ 部署成功！"
    echo ""
    echo "🌐 你的网站将在几分钟后可用："
    echo "   https://$(cd .. && git remote get-url origin | sed 's/.*github.com[:/]\([^/]*\)\/\([^/]*\)\.git/\1.github.io\/\2/')"
    echo ""
    echo "📝 注意："
    echo "   1. 确保在 GitHub 仓库设置中启用了 GitHub Pages"
    echo "   2. 选择 gh-pages 分支作为源"
    echo "   3. 网站可能需要几分钟才能生效"
else
    echo "❌ 推送失败"
    echo "请检查："
    echo "   1. 是否有推送权限"
    echo "   2. 远程仓库是否正确配置"
    exit 1
fi

cd ..

echo ""
echo "🎉 部署完成！"