# LSTM（长短期记忆网络）

- 论文：Long Short-Term Memory（https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext）

本目录提供 LSTM 在 Qlib `Alpha158/Alpha360` 数据集上的基准 workflow 配置（并包含部分缓存文件用于快速复现/调试）。

## 文件说明

- `requirements.txt`：本模型额外依赖（通常包含 PyTorch 等）。
- `workflow_config_lstm_Alpha158.yaml` / `workflow_config_lstm_Alpha360.yaml`：`qrun` 配置文件。
- `csi300_lstm_ts.pkl`、`model_lstm_csi300.pkl`：用于示例/调试的中间产物（可选）。

## 运行方式

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd backend/qlib/examples/benchmarks/LSTM
pip install -r requirements.txt
qrun workflow_config_lstm_Alpha158.yaml
```
