# MLIR 学习笔记

这是一个用于存放 MLIR (Multi-Level Intermediate Representation) 学习笔记的仓库，可以编译成美观的网页供阅读。

## 🎯 项目特点

- 📚 **结构化内容**: 按主题组织，便于系统学习
- 🌐 **网页展示**: 使用 MkDocs 生成美观的静态网站
- 🔍 **搜索功能**: 支持全文搜索和导航
- 📱 **响应式设计**: 支持各种设备访问
- 🌙 **深色模式**: 提供深色和浅色主题切换

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用 pip 安装
pip install -r requirements.txt

# 或者使用 conda
conda install -c conda-forge mkdocs-material
```

### 2. 本地预览

```bash
# 启动本地服务器
mkdocs serve

# 在浏览器中打开 http://127.0.0.1:8000
```

### 3. 构建网站

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
│   ├── examples/           # 实践案例
│   ├── advanced/           # 高级主题
│   ├── reference/          # 参考资料
│   └── index.md           # 首页
├── mkdocs.yml             # MkDocs 配置文件
├── requirements.txt        # Python 依赖
└── README.md              # 项目说明
```

## 📝 内容组织

### MLIR 基础
- [概述](docs/mlir-basics/overview.md) - MLIR 基本概念和设计理念
- [方言系统](docs/mlir-basics/dialects.md) - 方言的定义、注册和管理
- [操作和类型](docs/mlir-basics/ops-and-types.md) - MLIR 操作和类型系统

### 实践案例
- [简单方言定义](docs/examples/simple-dialect.md) - 从零开始创建方言
- [自定义操作](docs/examples/custom-ops.md) - 实现自定义操作和类型
- [转换Pass](docs/examples/transformation-pass.md) - 编写 MLIR 转换 Pass

### 高级主题
- [模式匹配](docs/advanced/pattern-matching.md) - 使用 TableGen 进行模式匹配
- [类型推导](docs/advanced/type-inference.md) - 实现类型推导和验证
- [优化策略](docs/advanced/optimization.md) - MLIR 优化策略和技巧

## 🔧 自定义配置

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

### 添加新页面

1. 在 `docs/` 目录下创建新的 Markdown 文件
2. 在 `mkdocs.yml` 的 `nav` 部分添加导航链接
3. 重新构建网站

## 📚 学习资源

- [MLIR 官方文档](https://mlir.llvm.org/)
- [LLVM 项目](https://llvm.org/)
- [MLIR 论文](https://mlir.llvm.org/getting_started/Paper/)
- [MkDocs 文档](https://www.mkdocs.org/)

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/new-content`
3. 提交更改: `git commit -am 'Add new content'`
4. 推送分支: `git push origin feature/new-content`
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢 LLVM 社区和 MLIR 项目的贡献者们！

---

*最后更新: {{ git_revision_date_localized }}* 