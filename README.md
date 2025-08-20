# MLIR 学习笔记

欢迎来到我的 MLIR 学习笔记！这里记录了我学习 MLIR (Multi-Level Intermediate Representation) 过程中的心得、代码示例和最佳实践。

## 🎯 学习目标

- 深入理解 MLIR 的核心概念和架构
- 掌握方言系统的设计和实现
- 学会编写自定义操作和转换 Pass
- 实践 MLIR 在实际项目中的应用

## 📚 内容结构

### MLIR 基础
- **概述**: MLIR 的基本概念和设计理念
- **方言系统**: 方言的定义、注册和管理
- **操作和类型**: MLIR 操作和类型系统的深入理解

### 实践案例
- **简单方言定义**: 从零开始创建一个简单的方言
- **自定义操作**: 实现自定义操作和类型
- **转换 Pass**: 编写 MLIR 转换 Pass

### 高级主题
- **模式匹配**: 使用 TableGen 进行模式匹配
- **类型推导**: 实现类型推导和验证
- **优化策略**: MLIR 优化策略和技巧

### 参考资料
- **常用 API**: 常用 MLIR API 参考
- **最佳实践**: 开发中的最佳实践
- **常见问题**: 常见问题和解决方案

## 🌐 网站访问

### 本地访问
网站已经在本地启动，你可以通过以下方式访问：

1. **直接访问**: http://localhost:8080/mlir-learn/
2. **命令行查看**: `curl http://localhost:8080/mlir-learn/`

### 启动网站
如果你想重新启动网站，可以使用以下命令：

```bash
# 安装依赖
pip3 install --break-system-packages -r requirements.txt

# 启动本地服务器
export PATH=$PATH:$HOME/.local/bin
mkdocs serve --dev-addr=0.0.0.0:8080
```

### 构建静态网站
如果你想构建静态网站文件：

```bash
export PATH=$PATH:$HOME/.local/bin
mkdocs build
```

构建完成后，静态文件会保存在 `site/` 目录中。

## 🚀 快速开始

1. 克隆仓库
2. 安装依赖: `pip install mkdocs-material`
3. 本地预览: `mkdocs serve`
4. 构建网站: `mkdocs build`

## 📖 推荐阅读

- [MLIR 官方文档](https://mlir.llvm.org/)
- [LLVM 项目](https://llvm.org/)
- [MLIR 论文](https://mlir.llvm.org/getting_started/Paper/)

## 🛠️ 技术特性

### 网站功能
- **响应式设计**: 支持桌面和移动设备
- **搜索功能**: 全文搜索和智能建议
- **代码高亮**: MLIR 代码语法高亮
- **导航优化**: 面包屑导航和页面目录
- **主题切换**: 支持浅色和深色主题

### 文档特性
- **中文界面**: 完全中文化的用户界面
- **代码复制**: 一键复制代码示例
- **版本控制**: Git 时间戳显示
- **SEO 优化**: 搜索引擎友好的结构

## 📁 项目结构

```
.
├── docs/                    # 文档源文件
│   ├── index.md            # 首页
│   ├── mlir-basics/        # MLIR 基础
│   ├── examples/           # 实践案例
│   ├── advanced/           # 高级主题
│   ├── reference/          # 参考资料
│   ├── stylesheets/        # 自定义样式
│   └── javascripts/        # 自定义脚本
├── mkdocs.yml              # MkDocs 配置
├── requirements.txt         # Python 依赖
├── start.sh                # 快速启动脚本
├── build.sh                # 构建脚本
└── README.md               # 项目说明
```

## 🔧 配置说明

### MkDocs 配置
网站使用 MkDocs Material 主题，主要配置包括：

- **主题**: Material for MkDocs
- **语言**: 中文 (zh)
- **插件**: 搜索、Git 时间戳、代码文档等
- **特性**: 标签页、搜索高亮、代码复制等

### 自定义样式
- 响应式设计
- MLIR 代码高亮
- 自定义组件样式
- 移动端优化

## 📱 移动端支持

网站完全支持移动设备：

- 响应式布局
- 触摸手势支持
- 移动端菜单优化
- 触摸友好的界面

## 🌍 部署选项

### 本地开发
```bash
mkdocs serve --dev-addr=0.0.0.0:8080
```

### GitHub Pages
```bash
mkdocs gh-deploy
```

### 静态托管
```bash
mkdocs build
# 将 site/ 目录部署到任何静态托管服务
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这些笔记！

### 贡献方式
1. 报告错误或问题
2. 提出新功能建议
3. 改进文档内容
4. 优化网站功能

### 开发环境
```bash
# 克隆仓库
git clone <repository-url>
cd mlir-learn

# 安装依赖
pip3 install --break-system-packages -r requirements.txt

# 启动开发服务器
export PATH=$PATH:$HOME/.local/bin
mkdocs serve --dev-addr=0.0.0.0:8080
```

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [MkDocs](https://www.mkdocs.org/) - 静态站点生成器
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) - 主题
- [MLIR](https://mlir.llvm.org/) - 多级中间表示框架
- [LLVM](https://llvm.org/) - 编译器基础设施

---

*最后更新: {{ git_revision_date_localized }}*

**🎉 现在你可以访问 http://localhost:8080/mlir-learn/ 来查看完整的 MLIR 学习笔记网站了！** 