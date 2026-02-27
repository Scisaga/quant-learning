# ADD（增强解耦蒸馏）

- 论文：ADD: Augmented Disentanglement Distillation Framework for Improving Stock Trend Forecasting（https://arxiv.org/abs/2012.06289）

本目录提供 ADD 在 Qlib `Alpha360` 数据集上的基准 workflow 配置。

## 文件说明

- `requirements.txt`：本模型额外依赖（通常包含 PyTorch 等）。
- `workflow_config_add_Alpha360.yaml`：`qrun` 配置文件。

## 运行方式

在仓库根目录执行：

```bash
cd backend/qlib/examples/benchmarks/ADD
pip install -r requirements.txt
qrun workflow_config_add_Alpha360.yaml
```

## 输出

- 默认记录到 `mlruns/`（MLflow）。
- 指标解释见 `backend/qlib/examples/benchmarks/README.md`。
