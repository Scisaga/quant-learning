# Rolling Process Data（滚动窗口数据加工示例）

本示例演示在 **滚动训练（rolling training）** 场景下，如何避免在每个滚动窗口重复生成全量加工数据。

## 背景

在滚动训练中：

- 每个窗口的训练数据会随时间向后移动而变化；
- 一些 Processor（例如标准化所需的均值/方差）具有“可学习状态”，并且该状态会随窗口变化而变化。

如果每个窗口都从头生成一次“完整加工后的数据”，会带来明显的时间与内存开销。

## 核心思路

本示例使用 **DataHandler-based DataLoader**：

1. 先加载与窗口无关的原始特征；
2. 再在窗口内部使用 Processor 生成与窗口相关的加工特征；

从而减少重复工作，并让滚动训练更高效。

## 运行方式

在仓库根目录执行：

```bash
python src/examples/rolling_process_data/workflow.py rolling_process
```

运行产物通常会记录到 `mlruns/`（MLflow），具体以脚本内 recorder 配置为准。
