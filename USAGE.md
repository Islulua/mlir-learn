# MLIR 学习笔记使用指南

## 🎯 项目概述

这是一个专门为 MLIR (Multi-Level Intermediate Representation) 学习设计的笔记仓库，使用 MkDocs 构建成美观的静态网站。

## 🚀 快速开始

### 1. 环境要求

- **Python**: 3.8 或更高版本
- **Git**: 用于版本控制和自动部署
- **浏览器**: 支持现代 Web 标准

### 2. 安装依赖

```bash
# 使用 pip 安装
pip install -r requirements.txt

# 或者使用 conda
conda install -c conda-forge mkdocs-material
```

### 3. 本地预览

```bash
# 启动本地服务器
mkdocs serve

# 在浏览器中打开: http://127.0.0.1:8000
```

### 4. 构建网站

```bash
# 构建静态网站
mkdocs build

# 生成的网站文件在 `site/` 目录中
```

## 📁 项目结构

```
mlir-learn/
├── docs/                    # 文档源文件
│   ├── mlir-basics/        # MLIR 基础知识
│   │   ├── overview.md     # MLIR 概述
│   │   ├── dialects.md     # 方言系统
│   │   └── ops-and-types.md # 操作和类型
│   ├── examples/           # 实践案例
│   │   ├── simple-dialect.md # 简单方言定义
│   │   ├── custom-ops.md   # 自定义操作
│   │   └── transformation-pass.md # 转换Pass
│   ├── advanced/           # 高级主题
│   │   ├── pattern-matching.md # 模式匹配
│   │   ├── type-inference.md # 类型推导
│   │   └── optimization.md # 优化策略
│   ├── reference/          # 参考资料
│   │   ├── common-apis.md  # 常用API
│   │   ├── best-practices.md # 最佳实践
│   │   └── faq.md         # 常见问题
│   ├── stylesheets/        # 自定义样式
│   ├── javascripts/        # 自定义脚本
│   └── index.md           # 首页
├── mkdocs.yml             # MkDocs 配置文件
├── requirements.txt        # Python 依赖
├── build.sh               # 构建脚本
├── start.sh               # 快速启动脚本
└── README.md              # 项目说明
```

## 🔧 配置说明

### MkDocs 配置 (mkdocs.yml)

主要配置项包括：

- **主题**: 使用 Material for MkDocs 主题
- **导航**: 分层级的导航结构
- **插件**: 搜索、Git 版本信息、代码文档等
- **自定义**: 样式和脚本文件

### 主题特性

- 🌙 深色/浅色主题切换
- 🔍 全文搜索功能
- 📱 响应式设计
- 🎨 自定义样式和脚本
- 📊 代码语法高亮
- 📋 代码复制功能

## 📝 内容编写

### Markdown 语法

支持标准 Markdown 语法，以及 MkDocs Material 的扩展功能：

```markdown
# 标题 1
## 标题 2
### 标题 3

**粗体文本**
*斜体文本*
`行内代码`

```python
# 代码块
def hello_world():
    print("Hello, MLIR!")
```

> 引用文本

| 表格 | 列1 | 列2 |
|------|-----|-----|
| 行1  | 数据 | 数据 |
| 行2  | 数据 | 数据 |
```

### 特殊功能

#### 1. 警告框

```markdown
!!! warning "警告"
    这是一个警告信息框
```

#### 2. 信息框

```markdown
!!! info "信息"
    这是一个信息框
```

#### 3. 代码标签

```markdown
```mlir
// MLIR 代码示例
func @main() -> tensor<2x2xf32> {
  %0 = "toy.constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
```
```

## 🚀 部署选项

### 1. GitHub Pages (推荐)

#### 自动部署

项目已配置 GitHub Actions，推送到 `main` 分支会自动部署：

1. 推送代码到 GitHub
2. GitHub Actions 自动构建
3. 部署到 `gh-pages` 分支
4. 网站自动上线

#### 手动部署

```bash
# 构建网站
./build.sh --deploy

# 或者手动步骤
git checkout --orphan gh-pages
git rm -rf .
cp -r site/* .
touch .nojekyll
git add .
git commit -m "Deploy website"
git push origin gh-pages
git checkout main
```

### 2. Netlify

1. 拖拽 `site/` 文件夹到 Netlify
2. 配置自定义域名（可选）
3. 自动部署完成

### 3. Vercel

1. 连接 GitHub 仓库
2. 选择构建命令: `mkdocs build`
3. 选择输出目录: `site`
4. 自动部署完成

## 🔍 自定义配置

### 添加新页面

1. 在 `docs/` 目录下创建新的 Markdown 文件
2. 在 `mkdocs.yml` 的 `nav` 部分添加导航链接
3. 重新构建网站

### 修改主题

编辑 `mkdocs.yml` 文件中的 `theme` 部分：

```yaml
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - search.highlight
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
```

### 自定义样式

- 编辑 `docs/stylesheets/extra.css`
- 编辑 `docs/javascripts/extra.js`
- 重新构建网站

## 🧪 测试和调试

### 本地测试

```bash
# 启动本地服务器
mkdocs serve

# 检查构建
mkdocs build

# 验证配置
mkdocs serve --strict
```

### 常见问题

1. **构建失败**: 检查 `requirements.txt` 中的依赖是否安装
2. **页面不显示**: 检查 `mkdocs.yml` 中的导航配置
3. **样式不生效**: 检查 CSS 和 JS 文件路径
4. **Git 版本信息**: 确保在 Git 仓库中运行

## 📚 学习资源

### MLIR 相关

- [MLIR 官方文档](https://mlir.llvm.org/)
- [LLVM 项目](https://llvm.org/)
- [MLIR 论文](https://mlir.llvm.org/getting_started/Paper/)

### MkDocs 相关

- [MkDocs 文档](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [MkDocs 插件](https://github.com/mkdocs/mkdocs/wiki/MkDocs-Plugins)

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/new-content`
3. 提交更改: `git commit -am 'Add new content'`
4. 推送分支: `git push origin feature/new-content`
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 🎉 开始使用

现在你已经了解了如何使用这个项目！开始编写你的 MLIR 学习笔记吧：

1. 运行 `./start.sh` 快速开始
2. 编辑 `docs/` 目录下的 Markdown 文件
3. 使用 `mkdocs serve` 预览效果
4. 使用 `mkdocs build` 构建网站
5. 部署到 GitHub Pages 或其他平台

祝你学习愉快！🚀 