# GeneralPtNN（统一的 PyTorch NN 基线骨架）

## 背景与目标

GeneralPtNN 的目标是统一/简化 Qlib 中 PyTorch 神经网络模型的基线工程化接口，使其同时更好地支持：

- **时序（time-series）数据**：例如使用 `TSDatasetH` 自动构造时间序列样本；
- **表格（tabular）数据**：例如使用 `DatasetH` 在截面表格上做训练与推理。

在该设计下，你通常只需要替换 PyTorch 网络结构（以及对应 dataset 类），就能用相同的 workflow 运行不同的 NN 模型。

## 本目录示例

本目录通过三个配置文件演示“同一套骨架、不同数据形态”的切换：

- `workflow_config_gru.yaml`：对齐历史 GRU 基线结果（参考 `src/examples/benchmarks/README.md` 中的 GRU 行）。
- `workflow_config_gru2mlp.yaml`：展示如何在 **最小改动** 下把时序配置转换为表格配置（关键是切换 net 与 dataset 类）。
- `workflow_config_mlp.yaml`：展示与 MLP 类似的功能（同样可参考 `src/examples/benchmarks/README.md` 的 MLP 行）。

## 运行方式

在仓库根目录执行（任选其一）：

```bash
cd src/examples/benchmarks/GeneralPtNN
qrun workflow_config_gru.yaml
qrun workflow_config_gru2mlp.yaml
qrun workflow_config_mlp.yaml
```

## 备注（关于结果差异）

`workflow_config_mlp.yaml` 的结果可能与旧版 MLP 基线不同，常见原因是 **训练停止策略不同**：

- GeneralPtNN 通常按 `epochs` 控制训练长度；
- 旧实现可能按 `max_steps` 控制训练长度。

若你希望更严格对齐历史结果，请检查/对齐两侧的 early stop/训练步数设置。

## TODO

- 逐步将更多现有模型对齐到 GeneralPtNN 设计。
