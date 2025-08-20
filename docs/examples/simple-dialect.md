# 简单方言定义示例

让我们通过一个简单的例子来学习如何创建自定义 MLIR 方言。

## 🎯 目标

创建一个名为 `Toy` 的简单方言，支持基本的数学运算。

## 📁 项目结构

```
toy-dialect/
├── include/
│   └── Toy/
│       ├── ToyDialect.h
│       ├── ToyOps.h
│       └── ToyTypes.h
├── lib/
│   ├── Dialect/
│   │   ├── ToyDialect.cpp
│   │   ├── ToyOps.cpp
│   │   └── ToyTypes.cpp
├── test/
│   └── dialect.mlir
└── CMakeLists.txt
```

## 🔧 方言定义

### 1. 方言头文件 (ToyDialect.h)

```cpp
#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ToyDialect.h.inc"

#endif // TOY_DIALECT_H
```

### 2. 操作定义 (ToyOps.h)

```cpp
#ifndef TOY_OPS_H
#define TOY_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"

#include "ToyOps.h.inc"

#endif // TOY_OPS_H
```

### 3. 类型定义 (ToyTypes.h)

```cpp
#ifndef TOY_TYPES_H
#define TOY_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace Toy {

class TensorType : public Type::TypeBase<TensorType, Type, TypeStorage> {
public:
  using Base::Base;
  
  static TensorType get(MLIRContext *context, ArrayRef<int64_t> shape, Type elementType);
  
  ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
};

} // namespace Toy
} // namespace mlir

#endif // TOY_TYPES_H
```

## 📝 MLIR 代码示例

### 基本语法

```mlir
// 定义函数
func @main() -> tensor<2x2xf32> {
  // 创建常量
  %0 = "toy.constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  
  // 矩阵加法
  %1 = "toy.add"(%0, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  
  // 返回结果
  return %1 : tensor<2x2xf32>
}
```

### 操作类型

1. **常量操作**: `toy.constant`
2. **加法操作**: `toy.add`
3. **乘法操作**: `toy.mul`
4. **转置操作**: `toy.transpose`

## 🚀 实现步骤

### 1. 注册方言

```cpp
// ToyDialect.cpp
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ToyOps.cpp.inc"
  >();
  
  addTypes<
#define GET_TYPE_LIST
#include "ToyTypes.cpp.inc"
  >();
}
```

### 2. 实现操作

```cpp
// ToyOps.cpp
LogicalResult AddOp::verify() {
  // 验证输入类型
  if (getLhs().getType() != getRhs().getType())
    return emitOpError("operand types must match");
  return success();
}
```

### 3. 类型推导

```cpp
LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  
  // 推导返回类型
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}
```

## 🧪 测试

### 方言测试

```mlir
// RUN: mlir-opt %s --toy-test-dialect | FileCheck %s

// CHECK-LABEL: func @test_add
func @test_add() -> tensor<2x2xf32> {
  %0 = "toy.constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %1 = "toy.add"(%0, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
```

## 🔍 调试技巧

1. **使用 `mlir-opt` 工具**:
   ```bash
   mlir-opt input.mlir --toy-test-dialect -o output.mlir
   ```

2. **启用调试信息**:
   ```bash
   mlir-opt input.mlir --toy-test-dialect --debug-only=toy -o output.mlir
   ```

3. **验证方言注册**:
   ```bash
   mlir-opt input.mlir --print-op-stats
   ```

## 📚 下一步

- [自定义操作详解](custom-ops.md)
- [转换Pass实现](transformation-pass.md)
- [高级模式匹配](../advanced/pattern-matching.md)

## 🎉 总结

通过这个简单的例子，我们学习了：
- 如何定义自定义方言
- 如何实现基本操作
- 如何定义自定义类型
- 如何测试方言功能

这为更复杂的 MLIR 项目奠定了基础！ 