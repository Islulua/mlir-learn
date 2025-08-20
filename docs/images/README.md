# 图片使用说明

这个目录用于存放项目文档中使用的图片文件。

## 支持的图片格式

- **PNG** - 适合截图、图表，支持透明背景
- **JPG/JPEG** - 适合照片，文件较小
- **SVG** - 矢量图，可缩放，适合图标和简单图表
- **GIF** - 适合简单动画
- **WebP** - 现代格式，压缩率高

## 使用方法

### 基本语法
```markdown
![图片描述](图片路径)
```

### 示例
```markdown
![MLIR 架构图](images/mlir-architecture.png)
![优化流程](../images/optimization-flow.svg)
```

### 相对路径说明
- `images/logo.png` - 相对于当前 Markdown 文件
- `../images/logo.png` - 向上一级目录
- `/images/logo.png` - 从 docs 目录开始

## 图片命名规范

- 使用小写字母和连字符
- 描述性名称，如：`mlir-architecture.png`
- 避免空格和特殊字符
- 包含版本号（如需要）：`mlir-v1.0-architecture.png`

## 图片优化建议

1. **选择合适的格式**
   - 照片用 JPG
   - 图标和简单图形用 SVG
   - 需要透明背景用 PNG

2. **文件大小**
   - 网页图片建议小于 500KB
   - 使用适当的压缩工具

3. **尺寸**
   - 考虑不同设备的显示需求
   - 提供适当的分辨率

## 当前包含的图片

- `mlir-example.svg` - MLIR 处理流程示例图

## 添加新图片

1. 将图片文件放入此目录
2. 在相应的 Markdown 文件中引用
3. 更新此 README 文件
4. 确保图片符合命名规范 