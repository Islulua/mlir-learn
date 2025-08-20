# 操作和类型

MLIR 的操作和类型系统是其表示能力的核心，它们共同定义了计算和数据流的结构。

## 操作（Operations）

操作是 MLIR 中的基本计算单元，表示程序中的各种操作，如算术运算、函数调用、内存访问等。

### 操作的基本结构

每个操作都有以下组成部分：

```mlir
%result = operation_name %operand1, %operand2 : result_type
```

- **操作名称**: 标识操作的类型
- **操作数**: 输入值，以 `%` 开头
- **结果**: 输出值，以 `%` 开头
- **类型**: 指定操作数和结果的类型

### 常见操作示例

#### 算术操作
```mlir
// 整数加法
%sum = arith.addi %a, %b : i32

// 浮点乘法
%product = arith.mulf %x, %y : f32

// 整数除法
%quotient = arith.divsi %dividend, %divisor : i32
```

#### 内存操作
```mlir
// 加载值
%value = memref.load %memref[%index] : memref<10xf32>

// 存储值
memref.store %value, %memref[%index] : memref<10xf32>
```

#### 函数调用
```mlir
// 调用函数
%result = call @function_name(%arg1, %arg2) : (i32, i32) -> i32
```

### 操作属性

操作可以包含属性来配置其行为：

```mlir
// 带有属性的操作
%result = arith.addi %a, %b {overflow = "nsw"} : i32

// 多个属性
%result = operation %input {attr1 = 42, attr2 = "value"} : type
```

## 类型（Types）

类型系统定义了 MLIR 中数据的结构和属性，确保类型安全和正确的操作组合。

### 基本类型

#### 整数类型
```mlir
i1      // 1位整数（布尔值）
i8      // 8位整数
i32     // 32位整数
i64     // 64位整数
```

#### 浮点类型
```mlir
f16     // 16位浮点
f32     // 32位浮点
f64     // 64位浮点
bf16    // 16位脑浮点
```

#### 索引类型
```mlir
index   // 平台相关的索引类型
```

### 复合类型

#### 张量类型
```mlir
tensor<2x3xf32>      // 2x3 的浮点张量
tensor<*xf32>        // 动态形状的浮点张量
tensor<?x?xf32>      // 未知维度的浮点张量
tensor<10x20x30xf64> // 三维双精度张量
```

#### 内存引用类型
```mlir
memref<10xf32>       // 10个浮点的内存引用
memref<*xf32>        // 动态大小的内存引用
memref<10x20xf32>    // 二维内存引用
```

#### 向量类型
```mlir
vector<4xf32>        // 4个浮点的向量
vector<2x3xf32>     // 2x3 的浮点向量
```

### 函数类型
```mlir
(i32, f32) -> f64   // 接受 i32 和 f32，返回 f64 的函数
() -> i32           // 无参数，返回 i32 的函数
```

## 类型推导和验证

MLIR 支持类型推导，可以自动推断某些类型：

```mlir
// 类型推导示例
%result = arith.addi %a, %b : i32  // 结果类型自动推导为 i32
%sum = arith.addi %x, %y           // 如果 %x 和 %y 类型相同，结果类型相同
```

## 自定义类型

可以定义自定义类型来满足特定需求：

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
};
```

## 类型检查

MLIR 在编译时进行类型检查，确保操作的类型兼容性：

```mlir
// 类型不匹配会导致错误
%result = arith.addi %int_value, %float_value : i32  // 错误：类型不匹配
```

## 最佳实践

1. **类型一致性**: 确保操作数和结果类型匹配
2. **类型注解**: 在复杂表达式中明确指定类型
3. **类型安全**: 利用类型系统捕获错误
4. **性能考虑**: 选择合适的类型大小和精度

## 总结

操作和类型系统是 MLIR 表示能力的核心。通过合理设计操作和类型，可以创建清晰、高效、类型安全的中间表示，支持各种编译优化和代码生成需求。