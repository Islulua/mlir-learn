# 转换Pass

MLIR 的转换 Pass 是进行代码优化和变换的核心机制。通过编写转换 Pass，可以实现各种优化策略，如常量折叠、死代码消除、循环优化等。

## Pass 概述

Pass 是 MLIR 中表示程序变换的基本单元。每个 Pass 都执行特定的转换，可以组合使用来实现复杂的优化序列。

### Pass 的类型

1. **Function Pass**: 在函数级别进行转换
2. **Module Pass**: 在模块级别进行转换
3. **Operation Pass**: 在特定操作上进行转换

## 基本 Pass 结构

### Function Pass 示例

```cpp
struct MyFunctionPass : public PassWrapper<MyFunctionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyFunctionPass)

  StringRef getArgument() const final { return "my-function-pass"; }
  StringRef getDescription() const final { return "我的函数转换 Pass"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    
    // 获取函数体
    Block &entryBlock = func.getBody().front();
    
    // 遍历所有操作
    for (Operation &op : entryBlock) {
      // 执行转换逻辑
      if (failed(processOperation(&op))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult processOperation(Operation *op);
};
```

### Module Pass 示例

```cpp
struct MyModulePass : public PassWrapper<MyModulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyModulePass)

  StringRef getArgument() const final { return "my-module-pass"; }
  StringRef getDescription() const final { return "我的模块转换 Pass"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // 遍历所有函数
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(processFunction(func))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult processFunction(func::FuncOp func);
};
```

## 操作转换

### 使用 Pattern Rewriter

```cpp
struct MyOpRewritePattern : public OpRewritePattern<MyCustomOp> {
  using OpRewritePattern<MyCustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MyCustomOp op,
                               PatternRewriter &rewriter) const override {
    // 检查是否匹配
    if (!shouldRewrite(op)) {
      return failure();
    }

    // 创建新的操作
    Value newResult = rewriter.create<ArithAddIOp>(
        op.getLoc(), op.getLhs(), op.getRhs());

    // 替换原操作
    rewriter.replaceOp(op, newResult);
    return success();
  }

private:
  bool shouldRewrite(MyCustomOp op) const;
};
```

### 批量应用模式

```cpp
void MyPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  
  // 添加重写模式
  patterns.add<MyOpRewritePattern>(&getContext());
  patterns.add<AnotherRewritePattern>(&getContext>();

  // 应用模式
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                         std::move(patterns)))) {
    signalPassFailure();
  }
}
```

## 类型转换

### 方言转换

```cpp
struct MyDialectConversionPass 
    : public PassWrapper<MyDialectConversionPass, OperationPass<ModuleOp>> {
  
  void runOnOperation() override {
    ConversionTarget target(getContext());
    
    // 设置合法方言
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    
    // 设置非法方言
    target.addIllegalDialect<MyCustomDialect>();

    // 创建重写模式
    RewritePatternSet patterns(&getContext());
    patterns.add<MyOpToArithOpPattern>(&getContext());

    // 执行转换
    if (failed(applyPartialConversion(getOperation(), target,
                                     std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
```

### 类型转换模式

```cpp
struct MyOpToArithOpPattern : public ConversionPattern<MyCustomOp> {
  using ConversionPattern<MyCustomOp>::ConversionPattern;

  LogicalResult matchAndRewrite(MyCustomOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // 转换操作数类型
    Value convertedLhs = rewriter.getRemappedValue(adaptor.getLhs());
    Value convertedRhs = rewriter.getRemappedValue(adaptor.getRhs());

    // 创建新的操作
    Value result = rewriter.create<arith::AddIOp>(
        op.getLoc(), convertedLhs, convertedRhs);

    // 替换原操作
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

## 分析和转换

### 使用分析结果

```cpp
struct MyAnalysisPass : public AnalysisPass<MyAnalysis> {
  MyAnalysis run(Operation *op, AnalysisManager &am) override {
    // 执行分析
    MyAnalysis analysis;
    
    // 收集信息
    op->walk([&](Operation *operation) {
      analysis.addOperation(operation);
    });
    
    return analysis;
  }
};

struct MyTransformPass : public PassWrapper<MyTransformPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // 获取分析结果
    MyAnalysis &analysis = getAnalysis<MyAnalysis>();
    
    // 使用分析结果进行转换
    if (failed(transformBasedOnAnalysis(analysis))) {
      signalPassFailure();
    }
  }
};
```

## Pass 注册

### 注册 Pass

```cpp
void registerMyPasses() {
  PassRegistration<MyFunctionPass>();
  PassRegistration<MyModulePass>();
  PassRegistration<MyDialectConversionPass>();
}

// 在 main 函数中注册
int main(int argc, char **argv) {
  mlir::registerAllPasses();
  registerMyPasses();
  
  // ... 其他代码
}
```

### 命令行使用

```bash
# 运行单个 Pass
mlir-opt input.mlir -my-function-pass -o output.mlir

# 运行多个 Pass
mlir-opt input.mlir -my-function-pass -my-module-pass -o output.mlir

# 查看可用的 Pass
mlir-opt --help
```

## 测试 Pass

### MLIR 测试

```mlir
// RUN: mlir-opt %s -my-function-pass -split-input-file | FileCheck %s

func @test_function(%arg0: i32, %arg1: i32) -> i32 {
  %0 = my.custom_op %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_function
// CHECK: %0 = arith.addi %arg0, %arg1
```

### 单元测试

```cpp
TEST_F(MyPassTest, BasicTransformation) {
  // 创建测试模块
  OwningOpRef<ModuleOp> module = createTestModule();
  
  // 创建 Pass 管理器
  PassManager pm(module->getContext());
  pm.addPass(std::make_unique<MyFunctionPass>());
  
  // 运行 Pass
  LogicalResult result = pm.run(module.get());
  EXPECT_TRUE(succeeded(result));
  
  // 验证结果
  verifyTransformation(module.get());
}
```

## 最佳实践

1. **单一职责**: 每个 Pass 只负责一种转换
2. **幂等性**: Pass 应该可以多次运行而不改变结果
3. **错误处理**: 正确处理转换失败的情况
4. **性能**: 考虑 Pass 的性能影响
5. **测试**: 为每个 Pass 编写全面的测试

## 总结

转换 Pass 是 MLIR 优化系统的核心。通过合理设计和实现 Pass，可以实现各种代码优化和变换，提高程序的性能和可读性。