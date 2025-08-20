# 模式匹配

MLIR 的模式匹配系统是进行代码转换和优化的强大工具。通过定义和应用模式，可以实现复杂的代码重写和规范化。

## 模式匹配概述

模式匹配允许识别特定的代码模式并用新的代码替换它们。这是实现编译器优化、代码规范化和其他转换的基础。

### 模式匹配的类型

1. **重写模式 (Rewrite Patterns)**: 用于代码重写和规范化
2. **转换模式 (Conversion Patterns)**: 用于方言和类型转换
3. **融合模式 (Fusion Patterns)**: 用于操作融合优化

## 重写模式

### 基本重写模式

```cpp
struct MyRewritePattern : public OpRewritePattern<MyCustomOp> {
  using OpRewritePattern<MyCustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MyCustomOp op,
                               PatternRewriter &rewriter) const override {
    // 检查是否匹配
    if (!shouldRewrite(op)) {
      return failure();
    }

    // 创建新的操作
    Value newResult = createReplacement(op, rewriter);

    // 替换原操作
    rewriter.replaceOp(op, newResult);
    return success();
  }

private:
  bool shouldRewrite(MyCustomOp op) const;
  Value createReplacement(MyCustomOp op, PatternRewriter &rewriter) const;
};
```

### 条件匹配

```cpp
struct ConditionalRewritePattern : public OpRewritePattern<MyCustomOp> {
  LogicalResult matchAndRewrite(MyCustomOp op,
                               PatternRewriter &rewriter) const override {
    // 检查操作数类型
    if (!op.getOperand().getType().isa<IntegerType>()) {
      return failure();
    }

    // 检查属性值
    if (op.getFactor() != 1) {
      return failure();
    }

    // 检查操作数是否为常量
    if (auto constOp = op.getOperand().getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
        if (intAttr.getValue() == 0) {
          // 匹配成功，执行重写
          rewriter.replaceOp(op, constOp.getResult());
          return success();
        }
      }
    }

    return failure();
  }
};
```

### 多操作数模式

```cpp
struct MultiOperandPattern : public OpRewritePattern<MyCustomOp> {
  LogicalResult matchAndRewrite(MyCustomOp op,
                               PatternRewriter &rewriter) const override {
    // 检查操作数数量
    if (op.getNumOperands() != 2) {
      return failure();
    }

    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    // 检查操作数类型
    if (!lhs.getType().isa<IntegerType>() || !rhs.getType().isa<IntegerType>()) {
      return failure();
    }

    // 检查是否为相同值
    if (lhs == rhs) {
      // 创建新的操作
      Value result = rewriter.create<arith::MulIOp>(
          op.getLoc(), lhs, rewriter.create<arith::ConstantOp>(
              op.getLoc(), rewriter.getI32IntegerAttr(2)));
      
      rewriter.replaceOp(op, result);
      return success();
    }

    return failure();
  }
};
```

## 转换模式

### 方言转换模式

```cpp
struct MyOpToArithPattern : public ConversionPattern<MyCustomOp> {
  using ConversionPattern<MyCustomOp>::ConversionPattern;

  LogicalResult matchAndRewrite(MyCustomOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // 转换操作数类型
    Value convertedLhs = rewriter.getRemappedValue(adaptor.getLhs());
    Value convertedRhs = rewriter.getRemappedValue(adaptor.getRhs());

    // 检查转换后的类型
    if (!convertedLhs || !convertedRhs) {
      return failure();
    }

    // 创建新的操作
    Value result = rewriter.create<arith::AddIOp>(
        op.getLoc(), convertedLhs, convertedRhs);

    // 替换原操作
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

### 类型转换

```cpp
struct TypeConversionPattern : public ConversionPattern<MyCustomOp> {
  LogicalResult matchAndRewrite(MyCustomOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // 获取目标类型
    Type targetType = getTypeConverter()->convertType(op.getType());
    if (!targetType) {
      return failure();
    }

    // 转换操作数
    SmallVector<Value> convertedOperands;
    if (failed(rewriter.getRemappedValues(adaptor.getOperands(),
                                         convertedOperands))) {
      return failure();
    }

    // 创建新操作
    Operation *newOp = rewriter.create<arith::AddIOp>(
        op.getLoc(), targetType, convertedOperands[0], convertedOperands[1]);

    // 替换原操作
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};
```

## 高级模式匹配

### 递归模式

```cpp
struct RecursivePattern : public OpRewritePattern<MyCustomOp> {
  LogicalResult matchAndRewrite(MyCustomOp op,
                               PatternRewriter &rewriter) const override {
    // 递归处理操作数
    SmallVector<Value> newOperands;
    for (Value operand : op.getOperands()) {
      if (auto definingOp = operand.getDefiningOp<MyCustomOp>()) {
        // 递归重写
        if (failed(rewriter.rewriteOp(definingOp))) {
          return failure();
        }
        newOperands.push_back(definingOp.getResult());
      } else {
        newOperands.push_back(operand);
      }
    }

    // 创建新操作
    Value result = rewriter.create<MyCustomOp>(
        op.getLoc(), op.getType(), newOperands);

    rewriter.replaceOp(op, result);
    return success();
  }
};
```

### 条件重写

```cpp
struct ConditionalRewritePattern : public OpRewritePattern<MyCustomOp> {
  LogicalResult matchAndRewrite(MyCustomOp op,
                               PatternRewriter &rewriter) const override {
    // 分析操作上下文
    if (auto parentOp = op->getParentOfType<func::FuncOp>()) {
      // 检查函数属性
      if (parentOp.hasAttr("optimize")) {
        // 执行优化重写
        return performOptimization(op, rewriter);
      }
    }

    // 默认重写
    return performDefaultRewrite(op, rewriter);
  }

private:
  LogicalResult performOptimization(MyCustomOp op, PatternRewriter &rewriter) const;
  LogicalResult performDefaultRewrite(MyCustomOp op, PatternRewriter &rewriter) const;
};
```

## 模式应用

### 批量应用

```cpp
void applyPatterns(Operation *op) {
  RewritePatternSet patterns(op->getContext());
  
  // 添加模式
  patterns.add<MyRewritePattern>(op->getContext());
  patterns.add<ConditionalRewritePattern>(op->getContext());
  patterns.add<MultiOperandPattern>(op->getContext());

  // 应用模式直到收敛
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
    signalPassFailure();
  }
}
```

### 有序应用

```cpp
void applyOrderedPatterns(Operation *op) {
  // 第一轮：规范化
  {
    RewritePatternSet patterns(op->getContext());
    patterns.add<NormalizationPattern>(op->getContext());
    
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      return;
    }
  }

  // 第二轮：优化
  {
    RewritePatternSet patterns(op->getContext());
    patterns.add<OptimizationPattern>(op->getContext());
    
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      return;
    }
  }
}
```

## 测试模式

### MLIR 测试

```mlir
// RUN: mlir-opt %s -test-pattern-matching -split-input-file | FileCheck %s

func @test_pattern(%arg0: i32, %arg1: i32) -> i32 {
  %0 = my.custom_op %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_pattern
// CHECK: %0 = arith.addi %arg0, %arg1
```

### 单元测试

```cpp
TEST_F(PatternMatchingTest, BasicRewrite) {
  // 创建测试操作
  OwningOpRef<ModuleOp> module = createTestModule();
  MyCustomOp op = findMyCustomOp(module.get());
  
  // 应用模式
  RewritePatternSet patterns(module->getContext());
  patterns.add<MyRewritePattern>(module->getContext());
  
  PatternRewriter rewriter(module->getContext());
  if (succeeded(patterns.matchAndRewrite(op, rewriter))) {
    // 验证重写结果
    verifyRewriteResult(module.get());
  }
}
```

## 最佳实践

1. **模式设计**: 设计清晰、可理解的模式
2. **条件检查**: 在重写前进行充分的匹配检查
3. **错误处理**: 正确处理匹配和重写失败的情况
4. **性能**: 避免在模式中执行昂贵的操作
5. **测试**: 为每个模式编写全面的测试用例

## 总结

模式匹配是 MLIR 代码转换系统的核心。通过合理设计模式，可以实现各种代码优化和规范化，提高代码质量和性能。