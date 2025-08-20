#!/bin/bash

# 主题切换脚本

echo "🎨 MLIR 学习笔记主题切换器"
echo ""

# 定义可用主题
declare -A themes=(
    ["windmill"]="Windmill - 清爽简约，现代感强"
    ["flatly"]="Flatly - 扁平化设计，蓝色主调"
    ["cosmo"]="Cosmo - 宇宙风格，深色科技感"
    ["united"]="United - 统一风格，橙色活力"
    ["yeti"]="Yeti - 雪人风格，深蓝专业"
    ["darkly"]="Darkly - 深色主题，绿色点缀"
    ["material"]="Material - 谷歌 Material Design"
    ["readthedocs"]="Read the Docs - 传统文档风格"
)

# 显示当前主题
current_theme=$(grep "name:" mkdocs.yml | grep -v "site_name" | head -1 | sed 's/.*name:[[:space:]]*//' | tr -d ' ')
echo "📍 当前主题: $current_theme"
echo ""

# 显示可用主题
echo "📋 可用主题列表:"
echo ""

for i in "${!themes[@]}"; do
    if [ "$i" = "$current_theme" ]; then
        echo "  ✅ $i - ${themes[$i]} (当前使用)"
    else
        echo "  🔄 $i - ${themes[$i]}"
    fi
done

echo ""
echo "💡 主题选择建议:"
echo "  🖥️  技术文档: windmill, material, readthedocs"
echo "  🎨 个人博客: cosmo, united, yeti"
echo "  🏢 企业网站: flatly, yeti"
echo "  🌙 夜间阅读: darkly"
echo "  🚀 创意项目: united, cosmo"
echo ""

# 获取用户选择
read -p "请选择要切换的主题 (输入主题名): " selected_theme

# 验证主题是否有效
if [[ -z "${themes[$selected_theme]}" ]]; then
    echo "❌ 无效的主题名: $selected_theme"
    echo "请从以下主题中选择:"
    for i in "${!themes[@]}"; do
        echo "  - $i"
    done
    exit 1
fi

# 如果选择的是当前主题，询问是否继续
if [ "$selected_theme" = "$current_theme" ]; then
    echo "⚠️  你选择的是当前正在使用的主题"
    read -p "是否继续？(y/n): " continue_switch
    if [[ ! "$continue_switch" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "🔄 取消主题切换"
        exit 0
    fi
fi

echo ""
echo "🔄 正在切换到 $selected_theme 主题..."

# 备份当前配置
cp mkdocs.yml mkdocs.yml.backup
echo "💾 已备份当前配置到 mkdocs.yml.backup"

# 更新配置文件
if [ "$selected_theme" = "material" ]; then
    # Material 主题需要特殊配置
    sed -i.bak "s/name:[[:space:]]*[a-zA-Z0-9_-]*/name: material/" mkdocs.yml
    # 添加 Material 主题的配置
    if ! grep -q "features:" mkdocs.yml; then
        sed -i.bak '/theme:/a\  features:\n    - navigation.tabs\n    - navigation.sections\n    - search.highlight\n  palette:\n    - scheme: default\n      primary: indigo\n      accent: indigo' mkdocs.yml
    fi
elif [ "$selected_theme" = "readthedocs" ]; then
    # Read the Docs 主题
    sed -i.bak "s/name:[[:space:]]*[a-zA-Z0-9_-]*/name: readthedocs/" mkdocs.yml
else
    # 其他主题
    sed -i.bak "s/name:[[:space:]]*[a-zA-Z0-9_-]*/name: $selected_theme/" mkdocs.yml
fi

# 清理临时文件
rm -f mkdocs.yml.bak

echo "✅ 配置文件已更新"

# 检查依赖
echo "📦 检查主题依赖..."
if [ "$selected_theme" = "material" ]; then
    echo "✅ Material 主题已包含在基础依赖中"
elif [ "$selected_theme" = "readthedocs" ]; then
    echo "✅ Read the Docs 主题已包含在基础依赖中"
elif [ "$selected_theme" = "windmill" ]; then
    echo "✅ Windmill 主题已安装"
else
    echo "📥 安装 $selected_theme 主题依赖..."
    pip install mkdocs-bootswatch
fi

# 测试构建
echo "🧪 测试主题构建..."
if mkdocs build --quiet 2>/dev/null; then
    echo "✅ 主题构建成功！"
else
    echo "❌ 主题构建失败，恢复原配置"
    cp mkdocs.yml.backup mkdocs.yml
    exit 1
fi

echo ""
echo "🎉 主题切换成功！"
echo "📍 新主题: $selected_theme"
echo "📝 描述: ${themes[$selected_theme]}"
echo ""

# 询问是否启动本地预览
read -p "是否启动本地预览？(y/n): " start_preview
if [[ "$start_preview" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo ""
    echo "🚀 启动本地预览服务器..."
    echo "📖 在浏览器中打开: http://127.0.0.1:8000"
    echo "⏹️  按 Ctrl+C 停止服务器"
    echo ""
    mkdocs serve
else
    echo ""
    echo "💡 预览命令:"
    echo "  mkdocs serve"
    echo ""
    echo "🚀 部署命令:"
    echo "  git add mkdocs.yml"
    echo "  git commit -m 'theme: switch to $selected_theme'"
    echo "  git push origin main"
fi

echo ""
echo "🎨 主题切换完成！" 