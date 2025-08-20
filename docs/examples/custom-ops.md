# 自定义操作

在 MLIR 中，自定义操作是扩展方言功能的核心方式。通过定义自定义操作，可以实现特定领域的计算逻辑。

## 操作定义概述

自定义操作通常使用 TableGen 来定义，这样可以自动生成大量的样板代码，包括：

- 操作类定义
- 操作构建器
- 操作验证逻辑
- 操作打印和解析
- 操作模式匹配支持

## TableGen 操作定义

### 基本操作定义

```tablegen
def MyCustomOp : My_Dialect_Op<"custom_op", []> {
  let summary = "自定义操作示例";
  let description = [{
    这是一个自定义操作的示例，展示了如何定义基本的操作。
  }];
  
  let arguments = (ins F32Tensor:$input);
  let results = (outs F32Tensor:$output);
  
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}
```

### 带属性的操作

```tablegen
def MyOpWithAttr : My_Dialect_Op<"op_with_attr", []> {
  let summary = "带属性的操作";
  
  let arguments = (ins F32Tensor:$input);
  let results = (outs F32Tensor:$output);
  
  let attributes = (ins
    DefaultValuedAttr<I64Attr, "1">:$factor,
    OptionalAttr<F32Attr>:$scale
  );
}
```

### 多结果操作

```tablegen
def MyMultiResultOp : My_Dialect_Op<"multi_result", []> {
  let summary = "多结果操作";
  
  let arguments = (ins F32Tensor:$input);
  let results = (outs F32Tensor:$output1, F32Tensor:$output2);
}
```

## C++ 操作实现

### 操作类定义

```cpp
class MyCustomOp : public Op<MyCustomOp, OpTrait::OneResult, OpTrait::OneOperand> {
public:
  using Op::Op;
  
  static StringRef getOperationName() { return "my.custom_op"; }
  
  // 获取操作数
  Value getInput() { return getOperand(); }
  
  // 获取结果
  Value getOutput() { return getResult(); }
  
  // 操作验证
  LogicalResult verify();
  
  // 操作折叠
  OpFoldResult fold(ArrayRef<Attribute> operands);
};
```

### 操作验证

```cpp
LogicalResult MyCustomOp::verify() {
  // 检查输入类型
  if (!getInput().getType().isa<F32TensorType>()) {
    return emitOpError("输入必须是 F32Tensor 类型");
  }
  
  // 检查结果类型
  if (!getOutput().getType().isa<F32TensorType>()) {
    return emitOpError("输出必须是 F32Tensor 类型");
  }
  
  return success();
}
```

### 操作折叠

```cpp
OpFoldResult MyCustomOp::fold(ArrayRef<Attribute> operands) {
  // 如果输入是常量，尝试常量折叠
  if (Attribute inputAttr = operands[0]) {
    // 实现常量折叠逻辑
    return inputAttr;
  }
  
  return {};
}
```

## 操作构建器

### 默认构建器

```cpp
// 自动生成的构建器
static void build(OpBuilder &builder, OperationState &state,
                  Value input) {
  state.addOperands(input);
  state.addTypes(input.getType());
}
```

### 自定义构建器

```cpp
// 自定义构建器
static void build(OpBuilder &builder, OperationState &state,
                  Value input, Type outputType) {
  state.addOperands(input);
  state.addTypes(outputType);
}
```

## 操作打印和解析

### 自定义打印

```cpp
void print(OpAsmPrinter &p) {
  p << " " << getInput();
  if (getOutput().getType() != getInput().getType()) {
    p << " : " << getOutput().getType();
  }
}
```

### 自定义解析

```cpp
ParseResult parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType input;
  Type inputType, outputType;
  
  if (parser.parseOperand(input) ||
      parser.parseColonType(outputType) ||
      parser.resolveOperand(input, inputType, result.operands)) {
    return failure();
  }
  
  result.addTypes(outputType);
  return success();
}
```

## 操作模式匹配

### 重写模式

```cpp
struct MyOpCanonicalizationPattern
    : public OpRewritePattern<MyCustomOp> {
  using OpRewritePattern<MyCustomOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(MyCustomOp op,
                               PatternRewriter &rewriter) const override {
    // 实现规范化逻辑
    Value input = op.getInput();
    
    // 如果输入是恒等操作，直接替换
    if (auto identity = input.getDefiningOp<IdentityOp>()) {
      rewriter.replaceOp(op, identity.getInput());
      return success();
    }
    
    return failure();
  }
};
```

## 操作注册

### 方言注册

```cpp
void MyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MyDialect/MyOps.h.inc"
  >();
}
```

### 模式注册

```cpp
void MyPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<MyOpCanonicalizationPattern>(&getContext());
  
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                         std::move(patterns)))) {
    signalPassFailure();
  }
}
```

## 测试自定义操作

### MLIR 测试

```mlir
// RUN: mlir-opt %s -test-my-canonicalization -split-input-file | FileCheck %s

func @test_my_op(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = my.custom_op %arg0 : tensor<2x2xf32> -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func @test_my_op
// CHECK: %0 = my.custom_op %arg0
```

## 最佳实践

1. **命名规范**: 使用清晰、描述性的操作名称
2. **类型安全**: 实现严格的类型检查
3. **文档**: 为每个操作提供详细的文档
4. **测试**: 编写全面的测试用例
5. **性能**: 考虑操作的性能影响

## 总结

自定义操作是 MLIR 扩展性的核心。通过合理设计操作，可以实现特定领域的计算逻辑，同时保持与 MLIR 生态系统的兼容性。