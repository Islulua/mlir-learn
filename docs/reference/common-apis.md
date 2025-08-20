# 常用API

MLIR 提供了丰富的 API 用于操作 IR、执行转换和分析。本文档介绍最常用的 API 和它们的使用方法。

## 核心 API

### MLIRContext

MLIRContext 是 MLIR 系统的核心，管理类型、属性和方言：

```cpp
// 创建 MLIRContext
MLIRContext context;

// 注册方言
context.getOrLoadDialect<arith::ArithDialect>();
context.getOrLoadDialect<func::FuncDialect>();
context.getOrLoadDialect<scf::SCFDialect>();

// 获取方言
auto arithDialect = context.getLoadedDialect<arith::ArithDialect>();
```

### Location

Location 表示操作在源代码中的位置：

```cpp
// 创建位置
Location loc = UnknownLoc::get(&context);
Location fileLoc = FileLineColLoc::get(&context, "file.mlir", 10, 5);
Location nameLoc = NameLoc::get(StringAttr::get(&context, "operation_name"), loc);

// 获取位置信息
if (auto fileLineCol = loc.dyn_cast<FileLineColLoc>()) {
  StringRef filename = fileLineCol.getFilename();
  unsigned line = fileLineCol.getLine();
  unsigned column = fileLineCol.getColumn();
}
```

### Type

Type 表示 MLIR 中的类型：

```cpp
// 基本类型
Type i32Type = IntegerType::get(&context, 32);
Type f32Type = FloatType::getF32(&context);
Type indexType = IndexType::get(&context);

// 张量类型
Type tensorType = RankedTensorType::get({2, 3}, f32Type);
Type dynamicTensorType = RankedTensorType::get({-1, -1}, f32Type);

// 内存引用类型
Type memrefType = MemRefType::get({10, 20}, f32Type);

// 类型检查
if (auto tensor = type.dyn_cast<TensorType>()) {
  // 处理张量类型
  ArrayRef<int64_t> shape = tensor.getShape();
  Type elementType = tensor.getElementType();
}
```

### Attribute

Attribute 表示编译时常量：

```cpp
// 创建属性
Attribute intAttr = IntegerAttr::get(i32Type, 42);
Attribute floatAttr = FloatAttr::get(f32Type, 3.14);
Attribute stringAttr = StringAttr::get(&context, "hello");
Attribute arrayAttr = ArrayAttr::get(&context, {intAttr, floatAttr});

// 获取属性值
if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
  int64_t value = intAttr.getValue().getSExtValue();
}

if (auto arrayAttr = attr.dyn_cast<ArrayAttr>()) {
  for (Attribute element : arrayAttr) {
    // 处理数组元素
  }
}
```

## 操作 API

### Operation

Operation 是 MLIR IR 的基本单元：

```cpp
// 获取操作信息
OperationName name = op->getName();
StringRef opName = name.getStringRef();
Location loc = op->getLoc();

// 操作数
ValueRange operands = op->getOperands();
Value operand = op->getOperand(0);
unsigned numOperands = op->getNumOperands();

// 结果
ValueRange results = op->getResults();
Value result = op->getResult(0);
unsigned numResults = op->getNumResults();

// 属性
DictionaryAttr attrs = op->getAttrDictionary();
Attribute attr = op->getAttr("key");
bool hasAttr = op->hasAttr("key");

// 区域
RegionRange regions = op->getRegions();
Region &region = op->getRegion(0);
```

### OpBuilder

OpBuilder 用于创建和修改操作：

```cpp
// 创建 OpBuilder
OpBuilder builder(&context);
OpBuilder builderAt(op, op->getRegion(0).begin());

// 创建操作
auto addOp = builder.create<arith::AddIOp>(
    loc, lhs, rhs);
auto constOp = builder.create<arith::ConstantOp>(
    loc, builder.getI32IntegerAttr(42));

// 插入操作
builder.setInsertionPoint(op);
builder.setInsertionPointAfter(op);
builder.setInsertionPointToStart(&block);

// 创建块
Block *block = builder.createBlock(&region);
builder.setInsertionPointToStart(block);
```

### PatternRewriter

PatternRewriter 用于在模式匹配中修改 IR：

```cpp
// 替换操作
rewriter.replaceOp(op, newValue);
rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);

// 删除操作
rewriter.eraseOp(op);

// 创建操作
Value newValue = rewriter.create<arith::AddIOp>(
    op->getLoc(), lhs, rhs);

// 修改操作数
rewriter.modifyOpInPlace(op, [&]() {
  op->setOperand(0, newValue);
});
```

## 方言 API

### Dialect

方言是 MLIR 中定义操作和类型的命名空间：

```cpp
// 获取方言
Dialect *dialect = context.getLoadedDialect<arith::ArithDialect>();

// 检查方言
bool isArith = op->getDialect() == dialect;
bool hasDialect = op->getDialect()->getNamespace() == "arith";

// 方言操作
dialect->getOperationNames();
dialect->getRegisteredOperations();
```

### 方言操作

```cpp
// 检查操作是否属于特定方言
bool isArithOp = op->getDialect()->getNamespace() == "arith";

// 创建方言操作
auto addOp = builder.create<arith::AddIOp>(loc, lhs, rhs);
auto subOp = builder.create<arith::SubIOp>(loc, lhs, rhs);
auto mulOp = builder.create<arith::MulIOp>(loc, lhs, rhs);
```

## 函数 API

### FuncOp

```cpp
// 创建函数
auto func = builder.create<func::FuncOp>(
    loc, "function_name",
    FunctionType::get(&context, {i32Type}, {i32Type}));

// 获取函数信息
StringRef funcName = func.getName();
FunctionType funcType = func.getFunctionType();
TypeRange paramTypes = funcType.getInputs();
TypeRange resultTypes = funcType.getResults();

// 函数体
Region &body = func.getBody();
Block &entryBlock = body.front();

// 添加参数
entryBlock.addArgument(i32Type, loc);
```

### CallOp

```cpp
// 创建函数调用
auto callOp = builder.create<func::CallOp>(
    loc, "function_name", resultType, operands);

// 获取调用信息
StringRef callee = callOp.getCallee();
ValueRange args = callOp.getOperands();
ValueRange results = callOp.getResults();
```

## 控制流 API

### SCF 方言

```cpp
// For 循环
auto forOp = builder.create<scf::ForOp>(
    loc, lowerBound, upperBound, step, initArgs);

// 获取循环信息
ValueRange iterArgs = forOp.getInitArgs();
ValueRange results = forOp.getResults();
Region &body = forOp.getBody();

// If 条件
auto ifOp = builder.create<scf::IfOp>(
    loc, TypeRange{i32Type}, condition, true, false);

// 获取分支
Region &thenRegion = ifOp.getThenRegion();
Region &elseRegion = ifOp.getElseRegion();
```

## 张量 API

### Tensor 方言

```cpp
// 张量操作
auto extractOp = builder.create<tensor::ExtractOp>(
    loc, tensor, indices);
auto insertOp = builder.create<tensor::InsertOp>(
    loc, scalar, tensor, indices);
auto reshapeOp = builder.create<tensor::ReshapeOp>(
    loc, tensor, shape);

// 张量类型信息
if (auto tensorType = type.dyn_cast<TensorType>()) {
  ArrayRef<int64_t> shape = tensorType.getShape();
  Type elementType = tensorType.getElementType();
  bool isDynamic = tensorType.hasDynamicShape();
}
```

## 内存 API

### MemRef 方言

```cpp
// 内存操作
auto loadOp = builder.create<memref::LoadOp>(
    loc, memref, indices);
auto storeOp = builder.create<memref::StoreOp>(
    loc, value, memref, indices);
auto allocOp = builder.create<memref::AllocOp>(
    loc, memrefType);

// 内存类型信息
if (auto memrefType = type.dyn_cast<MemRefType>()) {
  ArrayRef<int64_t> shape = memrefType.getShape();
  Type elementType = memrefType.getElementType();
  AffineMap layout = memrefType.getLayout();
}
```

## 工具 API

### 遍历 API

```cpp
// 遍历操作
op->walk([](Operation *op) {
  // 处理每个操作
});

// 遍历特定类型的操作
op->walk([](arith::AddIOp addOp) {
  // 处理加法操作
});

// 遍历函数
module.walk([](func::FuncOp func) {
  // 处理每个函数
});
```

### 分析 API

```cpp
// 数据流分析
DataFlowAnalysis analysis;
analysis.run(module);

// 支配分析
DominanceInfo domInfo(func);

// 循环分析
LoopAnalysis loopAnalysis;
loopAnalysis.analyze(func);
```

## 错误处理

### LogicalResult

```cpp
// 检查操作结果
if (failed(result)) {
  return failure();
}

// 返回成功或失败
return success();
return failure();

// 组合结果
if (failed(operation1()) || failed(operation2())) {
  return failure();
}
```

### emitError

```cpp
// 发出错误
return op->emitError("操作失败");
return op->emitOpError("无效的操作数");

// 带位置的错误
return emitError(loc, "语法错误");
```

## 最佳实践

1. **错误检查**: 始终检查 API 调用的返回值
2. **类型安全**: 使用类型安全的 API 而不是原始指针
3. **资源管理**: 正确管理 MLIRContext 和操作的生命周期
4. **性能**: 避免在热路径中创建临时对象
5. **文档**: 参考官方文档了解最新的 API 变化

## 总结

MLIR 的 API 设计注重类型安全和易用性。通过合理使用这些 API，可以高效地构建、分析和转换 MLIR IR。建议在实际使用中参考官方文档和示例代码。