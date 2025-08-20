#!/bin/bash

# GitHub Pages 诊断脚本

echo "🔍 检查 GitHub Pages 配置..."

# 检查 Git 配置
echo ""
echo "📋 Git 配置检查:"
echo "当前分支: $(git branch --show-current)"
echo "远程仓库: $(git remote get-url origin 2>/dev/null || echo '未配置')"

# 检查 GitHub Actions 文件
echo ""
echo "📁 GitHub Actions 配置检查:"
if [ -f ".github/workflows/deploy.yml" ]; then
    echo "✅ 找到 deploy.yml 文件"
    echo "文件大小: $(wc -l < .github/workflows/deploy.yml) 行"
else
    echo "❌ 未找到 deploy.yml 文件"
fi

# 检查 MkDocs 配置
echo ""
echo "📚 MkDocs 配置检查:"
if [ -f "mkdocs.yml" ]; then
    echo "✅ 找到 mkdocs.yml 文件"
    echo "文件大小: $(wc -l < mkdocs.yml) 行"
else
    echo "❌ 未找到 mkdocs.yml 文件"
fi

# 检查依赖
echo ""
echo "📦 依赖检查:"
if [ -f "requirements.txt" ]; then
    echo "✅ 找到 requirements.txt 文件"
    echo "依赖数量: $(wc -l < requirements.txt)"
else
    echo "❌ 未找到 requirements.txt 文件"
fi

# 检查文档目录
echo ""
echo "📖 文档目录检查:"
if [ -d "docs" ]; then
    echo "✅ 找到 docs 目录"
    echo "文档文件数量: $(find docs -name "*.md" | wc -l)"
else
    echo "❌ 未找到 docs 目录"
fi

# 检查本地构建
echo ""
echo "🔨 本地构建测试:"
if command -v mkdocs &> /dev/null; then
    echo "✅ MkDocs 已安装"
    echo "版本: $(mkdocs --version)"
    
    echo ""
    echo "🧪 尝试本地构建..."
    if mkdocs build --quiet 2>/dev/null; then
        echo "✅ 本地构建成功"
        if [ -d "site" ]; then
            echo "构建输出: site/ 目录"
            echo "文件数量: $(find site -type f | wc -l)"
        fi
    else
        echo "❌ 本地构建失败"
    fi
else
    echo "❌ MkDocs 未安装"
fi

# 检查 GitHub Pages 状态
echo ""
echo "🌐 GitHub Pages 状态检查:"
echo "请手动检查以下项目："
echo "1. 访问仓库 Settings > Pages"
echo "2. Source 应该设置为 'GitHub Actions'"
echo "3. 确保仓库是公开的，或者你有 GitHub Pro 账户"

echo ""
echo "🎯 修复建议:"
echo "1. 在 GitHub 上启用 Pages (Settings > Pages > Source: GitHub Actions)"
echo "2. 推送修复后的 Actions 配置"
echo "3. 检查 Actions 运行状态"
echo "4. 验证网站部署"

echo ""
echo "🔧 修复命令:"
echo "git add .github/workflows/deploy.yml"
echo "git commit -m 'Fix GitHub Actions deployment configuration'"
echo "git push origin main" 