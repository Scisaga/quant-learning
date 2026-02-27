# MLP（多层感知机）

MLP 是最常见的全连接神经网络基线之一，在 Qlib 中常用于验证“深度模型/特征工程”是否能带来相对线性模型与树模型的增量收益。

## 文件说明

- `requirements.txt`：本模型额外依赖（通常包含 PyTorch 等）。
- `workflow_config_mlp_Alpha158.yaml` / `workflow_config_mlp_Alpha360.yaml`：CSI300 基准配置。
- `workflow_config_mlp_Alpha158_csi500.yaml` / `workflow_config_mlp_Alpha360_csi500.yaml`：CSI500 基准配置。

## 运行方式

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd backend/qlib/examples/benchmarks/MLP
pip install -r requirements.txt
qrun workflow_config_mlp_Alpha158.yaml
```

## 输出

- 默认写入 `mlruns/`（MLflow）。
- 指标解释见 `backend/qlib/examples/benchmarks/README.md`。
