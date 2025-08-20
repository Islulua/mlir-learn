#!/bin/bash

# MLIR 学习笔记构建脚本

set -e

echo "🚀 开始构建 MLIR 学习笔记网站..."

# 检查依赖
echo "📦 检查依赖..."
if ! command -v mkdocs &> /dev/null; then
    echo "❌ 未找到 mkdocs，正在安装..."
    pip install -r requirements.txt
fi

# 清理之前的构建
echo "🧹 清理之前的构建..."
rm -rf site/

# 构建网站
echo "🔨 构建网站..."
mkdocs build

# 检查构建结果
if [ -d "site" ]; then
    echo "✅ 网站构建成功！"
    echo "📁 构建文件位于: site/"
    
    # 显示构建统计
    echo "📊 构建统计:"
    echo "   - HTML 文件: $(find site -name '*.html' | wc -l)"
    echo "   - 总文件数: $(find site -type f | wc -l)"
    echo "   - 总大小: $(du -sh site | cut -f1)"
    
    # 本地预览提示
    echo ""
    echo "🌐 本地预览:"
    echo "   mkdocs serve"
    echo "   然后在浏览器中打开: http://127.0.0.1:8000"
    
    # 部署提示
    echo ""
    echo "🚀 部署选项:"
    echo "   1. GitHub Pages: 推送到 gh-pages 分支"
    echo "   2. Netlify: 拖拽 site/ 文件夹到 Netlify"
    echo "   3. Vercel: 连接 GitHub 仓库自动部署"
    
else
    echo "❌ 网站构建失败！"
    exit 1
fi

# 可选的 GitHub Pages 部署
if [ "$1" = "--deploy" ]; then
    echo ""
    echo "🚀 部署到 GitHub Pages..."
    
    # 检查是否在 Git 仓库中
    if [ ! -d ".git" ]; then
        echo "❌ 当前目录不是 Git 仓库"
        exit 1
    fi
    
    # 创建 gh-pages 分支
    git checkout --orphan gh-pages
    
    # 删除所有文件
    git rm -rf .
    
    # 复制构建文件
    cp -r site/* .
    
    # 添加 .nojekyll 文件（避免 GitHub Pages 的 Jekyll 处理）
    touch .nojekyll
    
    # 提交更改
    git add .
    git commit -m "Deploy MLIR learning notes to GitHub Pages"
    
    # 推送到远程仓库
    git push origin gh-pages
    
    # 切换回主分支
    git checkout main
    
    echo "✅ 部署完成！"
    echo "🌐 网站将在几分钟后可用: https://your-username.github.io/mlir-learn"
fi

echo ""
echo "🎉 构建完成！" 