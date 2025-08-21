# MLIR 学习笔记

> 🚀 **深入 MLIR 内部实现，掌握编译器基础设施的核心原理**

欢迎来到我的 MLIR 学习笔记！这里记录了我深入学习 MLIR (Multi-Level Intermediate Representation) 过程中的心得、源码分析和最佳实践。通过深入分析 MLIR 的内部实现，帮助你真正理解编译器基础设施的设计思想。

## 🎯 项目特色

- **🔍 深度源码分析**：深入 MLIR 内部实现，理解设计原理
- **🏗️ 架构设计解析**：分析 MLIR 的内存管理、类型系统等核心架构
- **💡 性能优化实践**：结合实际工程经验，提供性能优化策略
- **📚 中文友好**：完全中文化的技术文档，降低学习门槛

## 📚 内容概览

### 🎯 MLIR 基础深入
- **[MLIR 概述](docs/mlir-basics/overview.md)** - 基本概念和设计理念
- **[操作实现深度解析](docs/mlir-basics/dive-into-operation.md)** - Operation 内部实现机制
- **[属性系统深度解析](docs/mlir-basics/dive-into-attributes.md)** - Attributes 内存管理和性能优化

### 🛠️ 实践案例
- **[简单方言定义](docs/examples/simple-dialect.md)** - 从零开始创建方言
- **[自定义操作](docs/examples/custom-ops.md)** - 实现自定义操作和类型
- **[转换 Pass](docs/examples/transformation-pass.md)** - 编写 MLIR 转换 Pass

### 🚀 高级主题
- **[模式匹配](docs/advanced/pattern-matching.md)** - TableGen 模式匹配
- **[类型推导](docs/advanced/type-inference.md)** - 类型推导和验证
- **[优化策略](docs/advanced/optimization.md)** - MLIR 优化技巧

### 📖 参考资料
- **[常用 API](docs/reference/common-apis.md)** - MLIR API 参考
- **[最佳实践](docs/reference/best-practices.md)** - 开发最佳实践
- **[常见问题](docs/reference/faq.md)** - 问题解决方案

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd mlir-learn
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动本地服务器
```bash
# 使用优化后的启动脚本（自动端口检测）
./start.sh

# 或手动启动
mkdocs serve
```

### 4. 访问网站
打开浏览器访问：http://127.0.0.1:8000

## 🏗️ 项目架构

```
mlir-learn/
├── docs/                           # 文档源文件
│   ├── index.md                    # 首页
│   ├── mlir-basics/               # MLIR 基础深入
│   │   ├── overview.md            # MLIR 概述
│   │   ├── dive-into-operation.md # 操作实现深度解析
│   │   └── dive-into-attributes.md# 属性系统深度解析
│   ├── examples/                   # 实践案例
│   ├── advanced/                   # 高级主题
│   ├── reference/                  # 参考资料
│   ├── stylesheets/               # 自定义样式
│   └── images/                    # 图片资源
├── mkdocs.yml                     # MkDocs 配置
├── requirements.txt                # Python 依赖
├── start.sh                       # 智能启动脚本
├── build.sh                       # 构建脚本
└── README.md                      # 项目说明
```

## 🔧 核心特性

### 📖 文档系统
- **Material 主题**：现代化的文档界面
- **响应式设计**：完美支持桌面和移动设备
- **智能搜索**：全文搜索和智能建议
- **代码高亮**：MLIR 代码语法高亮
- **中文优化**：完全中文化的用户界面

### 🚀 开发体验
- **智能启动**：自动检测端口冲突，智能选择可用端口
- **热重载**：文档修改后自动刷新
- **样式定制**：自定义 CSS 样式，支持图片浮空阴影效果
- **版本控制**：Git 集成，显示最后修改时间

### 🎨 视觉优化
- **图片效果**：支持浮空阴影、圆角、悬停动画等效果
- **主题切换**：支持浅色和深色主题
- **导航优化**：面包屑导航和页面目录
- **移动端适配**：触摸友好的界面设计

## 📱 移动端支持

网站完全支持移动设备：
- ✅ 响应式布局
- ✅ 触摸手势支持
- ✅ 移动端菜单优化
- ✅ 触摸友好的界面

## 🌍 部署选项

### 本地开发
```bash
./start.sh                    # 推荐：智能启动
mkdocs serve                  # 手动启动
```

### 构建静态网站
```bash
./build.sh                    # 使用构建脚本
mkdocs build                  # 手动构建
```

### GitHub Pages 部署
```bash
./deploy-github.sh           # 自动部署到 GitHub Pages
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这些笔记！

### 贡献方式
1. 🐛 报告错误或问题
2. 💡 提出新功能建议
3. 📝 改进文档内容
4. ⚡ 优化网站功能

### 开发环境设置
```bash
# 克隆仓库
git clone <repository-url>
cd mlir-learn

# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
./start.sh
```

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [MkDocs](https://www.mkdocs.org/) - 静态站点生成器
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) - 现代化主题
- [MLIR](https://mlir.llvm.org/) - 多级中间表示框架
- [LLVM](https://llvm.org/) - 编译器基础设施

---

*最后更新: {{ git_revision_date_localized }}*

**🎉 现在你可以访问本地服务器来查看完整的 MLIR 学习笔记网站了！** 