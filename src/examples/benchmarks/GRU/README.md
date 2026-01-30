# GRU（门控循环单元）

- 论文：Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation（https://aclanthology.org/D14-1179.pdf）

本目录提供 GRU 在 Qlib `Alpha158/Alpha360` 数据集上的基准 workflow 配置（并包含部分缓存文件用于快速复现/调试）。

## 文件说明

- `requirements.txt`：本模型额外依赖（通常包含 PyTorch 等）。
- `workflow_config_gru_Alpha158.yaml` / `workflow_config_gru_Alpha360.yaml`：`qrun` 配置文件。
- `csi300_gru_ts.pkl`、`model_gru_csi300.pkl`：用于示例/调试的中间产物（可选）。

## 运行方式

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd src/examples/benchmarks/GRU
pip install -r requirements.txt
qrun workflow_config_gru_Alpha158.yaml
```
