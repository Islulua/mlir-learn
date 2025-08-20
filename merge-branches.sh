#!/bin/bash

# 分支合并脚本

echo "🔄 开始合并分支..."

# 检查当前分支
current_branch=$(git branch --show-current)
echo "📍 当前分支: $current_branch"

# 确保在 main 分支上
if [ "$current_branch" != "main" ]; then
    echo "⚠️  当前不在 main 分支，切换到 main 分支..."
    git checkout main
fi

# 拉取最新更改
echo "📥 拉取最新更改..."
git pull origin main

# 检查 gh-pages 分支状态
echo ""
echo "🔍 检查 gh-pages 分支状态..."
git fetch origin gh-pages

# 显示分支差异
echo "📊 分支差异统计:"
git log --oneline --graph --decorate --all -10

echo ""
echo "🌐 分支状态:"
echo "main 分支: $(git rev-parse --short HEAD)"
echo "gh-pages 分支: $(git rev-parse --short origin/gh-pages)"

# 询问是否同步 gh-pages 分支
echo ""
read -p "是否同步 gh-pages 分支？(y/n): " sync_gh_pages

if [[ "$sync_gh_pages" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "🔄 同步 gh-pages 分支..."
    
    # 切换到 gh-pages 分支
    git checkout gh-pages
    
    # 拉取最新更改
    git pull origin gh-pages
    
    # 合并 main 分支的更改
    echo "🔗 合并 main 分支的更改..."
    git merge origin/main
    
    # 推送到远程
    echo "📤 推送到远程 gh-pages 分支..."
    git push origin gh-pages
    
    # 切换回 main 分支
    git checkout main
    
    echo "✅ gh-pages 分支同步完成！"
else
    echo "⏭️  跳过 gh-pages 分支同步"
fi

# 清理本地分支
echo ""
echo "🧹 清理本地分支..."
git branch --merged | grep -v "main" | grep -v "gh-pages" | xargs -r git branch -d

echo ""
echo "📋 当前分支状态:"
git branch -a

echo ""
echo "🎯 下一步操作建议:"
echo "1. 在 GitHub 上合并 Pull Request #1"
echo "2. 删除已合并的功能分支"
echo "3. 检查 GitHub Actions 部署状态"
echo "4. 验证网站是否正常更新"

echo ""
echo "🎉 分支合并操作完成！" 