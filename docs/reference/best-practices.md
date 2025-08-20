# 最佳实践

本文档总结了 MLIR 开发中的最佳实践，包括代码组织、性能优化、错误处理等方面的建议。

## 代码组织

### 项目结构

推荐的项目结构：

```
project/
├── include/           # 头文件
│   └── Project/
│       ├── Dialect/
│       └── Passes/
├── lib/              # 源文件
│   ├── Dialect/
│   └── Passes/
├── test/             # 测试文件
├── docs/             # 文档
├── CMakeLists.txt    # 构建配置
└── README.md         # 项目说明
```

### 命名规范

```cpp
// 方言命名：使用 PascalCase，以 Dialect 结尾
class MyCustomDialect : public mlir::Dialect { ... };

// 操作命名：使用 PascalCase，以 Op 结尾
class MyCustomOp : public Op<MyCustomOp, ...> { ... };

// 类型命名：使用 PascalCase，以 Type 结尾
class MyCustomType : public Type::TypeBase<MyCustomType, Type> { ... };

// 属性命名：使用 PascalCase，以 Attr 结尾
class MyCustomAttr : public Attribute::AttrBase<MyCustomAttr, ...> { ... };

// 文件命名：使用 snake_case
my_custom_op.h
my_custom_op.cpp
my_custom_op.td
```

## 方言设计

### 方言注册

```cpp
// 正确的方言注册方式
class MyCustomDialect : public mlir::Dialect {
public:
  explicit MyCustomDialect(mlir::MLIRContext *context)
      : mlir::Dialect(getDialectNamespace(), context,
                      mlir::TypeID::get<MyCustomDialect>()) {
    
    // 注册操作
    addOperations<
#define GET_OP_LIST
#include "MyCustomDialect/MyCustomOps.h.inc"
    >();
    
    // 注册类型
    addTypes<
#define GET_TYPE_LIST
#include "MyCustomDialect/MyCustomTypes.h.inc"
    >();
    
    // 注册属性
    addAttributes<
#define GET_ATTR_LIST
#include "MyCustomDialect/MyCustomAttrs.h.inc"
    >();
  }
  
  static constexpr llvm::StringLiteral getDialectNamespace() {
    return llvm::StringLiteral("my");
  }
};
```

### 操作定义

```cpp
// 使用 TableGen 定义操作
def MyCustomOp : MyCustom_Dialect_Op<"custom_op", [
  // 特征
  Pure,
  NoSideEffect,
  Commutative,
  // 接口
  InferTypeOpInterface,
  // 验证
  DeclareOpInterfaceMethods<InferShapedTypeOpInterface>
]> {
  let summary = "自定义操作";
  let description = [{
    详细的操作描述。
  }];
  
  let arguments = (ins
    F32Tensor:$input,
    OptionalAttr<F32Attr>:$scale
  );
  
  let results = (outs F32Tensor:$output);
  
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$output)>
  ];
  
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}
```

## 类型系统

### 类型定义

```cpp
// 自定义类型定义
class MyCustomType : public mlir::Type::TypeBase<MyCustomType, mlir::Type> {
public:
  using Base::Base;
  
  static MyCustomType get(mlir::MLIRContext *context) {
    return Base::get(context);
  }
  
  static bool kindof(unsigned kind) {
    return kind == MyCustomType::getTypeID();
  }
  
  // 类型验证
  static LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::MLIRContext *context) {
    // 验证构造参数
    return success();
  }
};
```

### 类型推导

```cpp
// 实现类型推导接口
class MyCustomOp : public Op<MyCustomOp, ...> {
public:
  // 类型推导
  static LogicalResult inferReturnTypes(
      mlir::MLIRContext *context, std::optional<mlir::Location> location,
      mlir::ValueRange operands, mlir::DictionaryAttr attributes,
      mlir::OpaqueProperties properties, mlir::RegionRange regions,
      mlir::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    
    // 检查操作数
    if (operands.empty()) {
      return failure();
    }
    
    // 推导结果类型
    mlir::Type inputType = operands[0].getType();
    if (auto tensorType = inputType.dyn_cast<mlir::TensorType>()) {
      inferredReturnTypes.push_back(tensorType);
      return success();
    }
    
    return failure();
  }
};
```

## 操作实现

### 操作验证

```cpp
// 实现操作验证
LogicalResult MyCustomOp::verify() {
  // 检查操作数数量
  if (getNumOperands() != 2) {
    return emitOpError("需要恰好两个操作数");
  }
  
  // 检查操作数类型
  mlir::Type lhsType = getLhs().getType();
  mlir::Type rhsType = getRhs().getType();
  
  if (lhsType != rhsType) {
    return emitOpError("操作数类型必须相同")
           << ", 得到 " << lhsType << " 和 " << rhsType;
  }
  
  // 检查结果类型
  mlir::Type resultType = getResult().getType();
  if (resultType != lhsType) {
    return emitOpError("结果类型必须与操作数类型相同");
  }
  
  return success();
}
```

### 操作折叠

```cpp
// 实现操作折叠
mlir::OpFoldResult MyCustomOp::fold(mlir::ArrayRef<mlir::Attribute> operands) {
  // 常量折叠
  if (mlir::Attribute lhsAttr = operands[0]) {
    if (mlir::Attribute rhsAttr = operands[1]) {
      // 执行常量计算
      if (auto result = performConstantFolding(lhsAttr, rhsAttr)) {
        return result;
      }
    }
  }
  
  // 恒等折叠
  if (getLhs() == getRhs()) {
    // 如果操作数是相同的值，返回该值
    return getLhs();
  }
  
  return {};
}
```

## Pass 开发

### Pass 结构

```cpp
// 标准的 Pass 结构
struct MyCustomPass : public PassWrapper<MyCustomPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyCustomPass)

  // Pass 标识
  StringRef getArgument() const final { return "my-custom-pass"; }
  StringRef getDescription() const final { return "我的自定义 Pass"; }
  
  // 选项
  Pass::Option<bool> enableOptimization{
      *this, "enable-optimization",
      llvm::cl::desc("启用优化"),
      llvm::cl::init(true)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // 执行转换
    if (failed(performTransformation(module))) {
      signalPassFailure();
      return;
    }
  }

private:
  LogicalResult performTransformation(ModuleOp module);
};
```

### 模式匹配

```cpp
// 使用模式匹配进行转换
struct MyOpRewritePattern : public OpRewritePattern<MyCustomOp> {
  using OpRewritePattern<MyCustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MyCustomOp op,
                               PatternRewriter &rewriter) const override {
    // 检查匹配条件
    if (!shouldRewrite(op)) {
      return failure();
    }
    
    // 创建替换操作
    mlir::Value newResult = createReplacement(op, rewriter);
    
    // 替换原操作
    rewriter.replaceOp(op, newResult);
    return success();
  }

private:
  bool shouldRewrite(MyCustomOp op) const;
  mlir::Value createReplacement(MyCustomOp op, PatternRewriter &rewriter) const;
};
```

## 错误处理

### 错误报告

```cpp
// 提供有意义的错误信息
LogicalResult MyCustomOp::verify() {
  if (getNumOperands() < 2) {
    return emitOpError("操作需要至少两个操作数")
           << ", 但只提供了 " << getNumOperands() << " 个";
  }
  
  // 使用 emitOpError 提供操作特定的错误
  if (failed(validateOperands())) {
    return emitOpError("操作数验证失败");
  }
  
  return success();
}

// 在 Pass 中报告错误
void MyCustomPass::runOnOperation() {
  ModuleOp module = getOperation();
  
  if (failed(processModule(module))) {
    // 使用 signalPassFailure 标记 Pass 失败
    signalPassFailure();
    return;
  }
}
```

### 错误恢复

```cpp
// 实现错误恢复机制
LogicalResult MyCustomPass::processModule(ModuleOp module) {
  // 收集所有错误
  mlir::DiagnosticEngine &diagEngine = module->getContext()->getDiagEngine();
  mlir::ScopedDiagnosticHandler handler(&diagEngine, [&](mlir::Diagnostic &diag) {
    // 记录错误但不中断处理
    errors.push_back(diag.str());
  });
  
  // 继续处理，即使遇到错误
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (failed(processFunction(func))) {
      // 记录错误但继续处理其他函数
      continue;
    }
  }
  
  // 报告收集到的错误
  if (!errors.empty()) {
    llvm::errs() << "处理过程中遇到 " << errors.size() << " 个错误:\n";
    for (const std::string &error : errors) {
      llvm::errs() << "  " << error << "\n";
    }
  }
  
  return success();
}
```

## 性能优化

### 内存管理

```cpp
// 避免不必要的内存分配
class MyCustomPass : public PassWrapper<MyCustomPass, OperationPass<ModuleOp>> {
private:
  // 重用容器，避免重复分配
  mlir::SmallVector<mlir::Value, 8> worklist;
  mlir::DenseSet<mlir::Operation *> visited;
  
  void processOperation(mlir::Operation *op) {
    // 清空容器而不是重新分配
    worklist.clear();
    visited.clear();
    
    // 填充工作列表
    worklist.push_back(op->getResult(0));
    
    // 处理工作列表
    while (!worklist.empty()) {
      mlir::Value value = worklist.pop_back_val();
      if (visited.insert(value.getDefiningOp()).second) {
        // 处理新访问的操作
        processValue(value);
      }
    }
  }
};
```

### 缓存优化

```cpp
// 使用缓存避免重复计算
class AnalysisCache {
private:
  mlir::DenseMap<mlir::Operation *, AnalysisResult> cache;

public:
  AnalysisResult getAnalysis(mlir::Operation *op) {
    auto it = cache.find(op);
    if (it != cache.end()) {
      return it->second;
    }
    
    // 计算分析结果
    AnalysisResult result = computeAnalysis(op);
    cache[op] = result;
    return result;
  }
  
  void clear() { cache.clear(); }
};
```

## 测试

### 单元测试

```cpp
// 使用 MLIR 测试框架
TEST_F(MyCustomOpTest, BasicOperation) {
  // 创建测试上下文
  mlir::MLIRContext context;
  context.getOrLoadDialect<MyCustomDialect>();
  
  // 创建测试操作
  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();
  
  // 创建操作数
  mlir::Type i32Type = builder.getI32Type();
  auto lhs = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(10));
  auto rhs = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(20));
  
  // 创建测试操作
  auto op = builder.create<MyCustomOp>(loc, lhs, rhs);
  
  // 验证操作
  EXPECT_TRUE(op.getLhs() == lhs);
  EXPECT_TRUE(op.getRhs() == rhs);
  EXPECT_TRUE(succeeded(op.verify()));
}
```

### MLIR 测试

```mlir
// RUN: mlir-opt %s -test-my-custom-pass -split-input-file | FileCheck %s

// 测试基本操作
func @test_basic(%arg0: i32, %arg1: i32) -> i32 {
  %0 = my.custom_op %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_basic
// CHECK: %0 = my.custom_op %arg0, %arg1

// 测试类型推导
func @test_type_inference(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = my.custom_op %arg0
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func @test_type_inference
// CHECK: %0 = my.custom_op %arg0 : tensor<2x3xf32>
```

## 文档

### 代码注释

```cpp
/**
 * 自定义操作：执行特定的计算逻辑
 * 
 * 这个操作接受两个操作数并产生一个结果。它支持以下特性：
 * - 类型推导：自动推导结果类型
 * - 常量折叠：在编译时计算常量表达式
 * - 规范化：支持操作规范化
 * 
 * 示例：
 * ```mlir
 * %result = my.custom_op %lhs, %rhs : i32
 * ```
 */
class MyCustomOp : public Op<MyCustomOp, ...> {
  // ... 实现
};
```

### API 文档

```cpp
/**
 * 执行模块转换
 * 
 * @param module 要转换的模块
 * @return 转换是否成功
 * 
 * 这个函数执行以下转换：
 * 1. 规范化操作
 * 2. 应用优化模式
 * 3. 清理死代码
 * 
 * 如果转换失败，函数会返回 failure() 并记录错误信息。
 */
LogicalResult performTransformation(ModuleOp module);
```

## 总结

遵循这些最佳实践可以帮助你：

1. **提高代码质量**: 通过良好的结构和命名规范
2. **减少错误**: 通过完善的验证和错误处理
3. **提升性能**: 通过优化内存使用和算法
4. **增强可维护性**: 通过清晰的文档和测试
5. **促进协作**: 通过一致的代码风格和约定

记住，最佳实践是指导原则，应该根据具体项目的需求进行调整。最重要的是保持代码的清晰性、正确性和可维护性。