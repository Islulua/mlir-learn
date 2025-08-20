#!/bin/bash

# MLIR 学习笔记快速启动脚本

echo "🎯 欢迎使用 MLIR 学习笔记项目！"
echo ""

# 检查 Python 版本
echo "🐍 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "✅ Python 版本: $PYTHON_VERSION"

# 检查 pip
echo ""
echo "📦 检查 pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ 未找到 pip3，请先安装 pip"
    exit 1
fi
echo "✅ pip 已安装"

# 安装依赖
echo ""
echo "📚 安装项目依赖..."
pip3 install -r requirements.txt

# 检查安装结果
if ! command -v mkdocs &> /dev/null; then
    echo "❌ MkDocs 安装失败"
    exit 1
fi
echo "✅ 依赖安装完成"

# 显示项目信息
echo ""
echo "📁 项目结构:"
echo "   docs/           - 文档源文件"
echo "   mkdocs.yml      - 配置文件"
echo "   requirements.txt - Python 依赖"
echo "   build.sh        - 构建脚本"
echo ""

# 显示可用命令
echo "🚀 可用命令:"
echo "   ./start.sh          - 显示此帮助信息"
echo "   mkdocs serve        - 启动本地预览服务器"
echo "   mkdocs build        - 构建静态网站"
echo "   ./build.sh          - 使用构建脚本"
echo "   ./build.sh --deploy - 构建并部署到 GitHub Pages"
echo ""

# 启动本地服务器
echo "🌐 是否启动本地预览服务器？(y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo ""
    echo "🚀 启动本地服务器..."
    echo "📖 在浏览器中打开: http://127.0.0.1:8000"
    echo "⏹️  按 Ctrl+C 停止服务器"
    echo ""
    mkdocs serve
else
    echo ""
    echo "💡 提示:"
    echo "   - 运行 'mkdocs serve' 启动本地预览"
    echo "   - 运行 'mkdocs build' 构建网站"
    echo "   - 查看 README.md 了解更多信息"
    echo ""
    echo "🎉 设置完成！开始你的 MLIR 学习之旅吧！"
fi 