# 主题使用指南

本项目支持多种美观的 MkDocs 主题，你可以根据需要自由切换。

## 🎨 可用主题

### 1. **Windmill** (当前使用)
- **风格**: 清爽简约，现代感强
- **特点**: 响应式设计，代码高亮，搜索功能
- **配置**: `theme.name: windmill`

### 2. **Bootswatch 系列** (Bootstrap 风格)
基于 Bootstrap 框架，提供多种配色方案：

#### **Flatly** - 扁平化设计
- **风格**: 现代扁平化，蓝色主调
- **适用**: 技术文档，企业网站
- **配置**: `theme.name: flatly`

#### **Cosmo** - 宇宙风格
- **风格**: 深色科技感，紫色点缀
- **适用**: 科技博客，开发者文档
- **配置**: `theme.name: cosmo`

#### **United** - 统一风格
- **风格**: 橙色活力，现代简洁
- **适用**: 创意项目，个人博客
- **配置**: `theme.name: united`

#### **Yeti** - 雪人风格
- **风格**: 深蓝专业，商务风格
- **适用**: 商务文档，专业报告
- **配置**: `theme.name: yeti`

#### **Darkly** - 深色主题
- **风格**: 深色背景，绿色点缀
- **适用**: 夜间阅读，程序员偏好
- **配置**: `theme.name: darkly`

### 3. **Material** (经典主题)
- **风格**: 谷歌 Material Design
- **特点**: 功能丰富，可定制性强
- **配置**: `theme.name: material`

### 4. **Read the Docs** (文档风格)
- **风格**: 传统文档风格
- **特点**: 简洁清晰，专注内容
- **配置**: `theme.name: readthedocs`

## 🔧 主题切换方法

### 方法 1: 修改配置文件

编辑 `mkdocs.yml` 文件，更改 `theme.name` 值：

```yaml
# 当前配置 (Windmill)
theme:
  name: windmill

# 切换到其他主题
theme:
  name: flatly    # 或 cosmo, united, yeti, darkly, material, readthedocs
```

### 方法 2: 使用快速切换脚本

运行主题切换脚本：

```bash
./switch-theme.sh
```

## 📱 主题预览

### 本地预览
```bash
# 安装依赖
pip install -r requirements.txt

# 本地预览
mkdocs serve

# 在浏览器中打开: http://127.0.0.1:8000
```

### 在线预览
- **Windmill**: https://mkdocs-windmill.readthedocs.io/
- **Bootswatch**: https://bootswatch.com/
- **Material**: https://squidfunk.github.io/mkdocs-material/

## 🎯 主题选择建议

### 根据内容类型选择：

- **技术文档**: Windmill, Material, Read the Docs
- **个人博客**: Cosmo, United, Yeti
- **企业网站**: Flatly, Yeti
- **夜间阅读**: Darkly
- **创意项目**: United, Cosmo

### 根据用户群体选择：

- **开发者**: Material, Windmill, Darkly
- **设计师**: Cosmo, United
- **商务用户**: Flatly, Yeti
- **学生**: Read the Docs, Windmill

## 🔄 快速主题切换脚本

我创建了一个主题切换脚本，可以快速预览不同主题：

```bash
# 运行主题切换器
./switch-theme.sh

# 脚本会：
# 1. 显示可用主题列表
# 2. 让你选择主题
# 3. 自动更新配置文件
# 4. 启动本地预览
```

## 📝 自定义主题

### 修改颜色
```yaml
theme:
  name: flatly
  custom_dir: custom_theme/
```

### 添加自定义 CSS
```yaml
extra_css:
  - stylesheets/extra.css
  - stylesheets/custom-theme.css
```

### 添加自定义 JavaScript
```yaml
extra_javascript:
  - javascripts/extra.js
  - javascripts/custom-theme.js
```

## 🚀 部署主题

切换主题后，推送到 GitHub 即可自动部署：

```bash
# 提交主题更改
git add mkdocs.yml
git commit -m "theme: switch to [主题名]"
git push origin main

# GitHub Actions 会自动构建和部署
```

## 🎉 开始使用

1. **选择主题**: 根据你的喜好和需求选择
2. **修改配置**: 编辑 `mkdocs.yml` 中的 `theme.name`
3. **本地预览**: 使用 `mkdocs serve` 查看效果
4. **部署上线**: 推送到 GitHub 自动部署

祝你找到最适合的主题！🎨 