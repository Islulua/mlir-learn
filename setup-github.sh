#!/bin/bash

# GitHub 设置脚本

echo "🚀 设置 GitHub 仓库管理..."

# 检查是否在 Git 仓库中
if [ ! -d ".git" ]; then
    echo "❌ 当前目录不是 Git 仓库"
    exit 1
fi

# 获取用户输入
echo ""
echo "请输入你的 GitHub 信息："
read -p "GitHub 用户名: " github_username
read -p "仓库名称 (默认: mlir-learn): " repo_name
repo_name=${repo_name:-mlir-learn}

# 检查远程仓库是否已配置
if git remote get-url origin &> /dev/null; then
    echo "⚠️  远程仓库已配置为: $(git remote get-url origin)"
    read -p "是否要重新配置？(y/n): " reconfigure
    if [[ "$reconfigure" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        git remote remove origin
    else
        echo "✅ 远程仓库配置完成"
        exit 0
    fi
fi

# 配置远程仓库
echo ""
echo "🔗 配置远程仓库..."
git remote add origin "https://github.com/$github_username/$repo_name.git"

# 验证配置
echo "✅ 远程仓库已配置为: $(git remote get-url origin)"

# 推送代码
echo ""
echo "📤 推送代码到 GitHub..."
echo "注意: 如果这是新仓库，请先在 GitHub 上创建仓库"
echo "仓库地址: https://github.com/$github_username/$repo_name"
echo ""

read -p "是否现在推送代码？(y/n): " push_now
if [[ "$push_now" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "🚀 推送代码..."
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo "✅ 代码推送成功！"
        echo ""
        echo "🌐 下一步操作："
        echo "1. 访问: https://github.com/$github_username/$repo_name"
        echo "2. 进入 Settings > Pages"
        echo "3. Source 选择 'Deploy from a branch'"
        echo "4. Branch 选择 'gh-pages'，点击 Save"
        echo "5. 等待几分钟，网站将自动部署"
        echo ""
        echo "📖 网站地址: https://$github_username.github.io/$repo_name"
    else
        echo "❌ 代码推送失败，请检查："
        echo "1. GitHub 仓库是否已创建"
        echo "2. 用户名和仓库名是否正确"
        echo "3. 是否有推送权限"
    fi
else
    echo ""
    echo "💡 手动推送命令："
    echo "git push -u origin main"
    echo ""
    echo "🌐 仓库地址: https://github.com/$github_username/$repo_name"
fi

echo ""
echo "🎉 GitHub 设置完成！" 