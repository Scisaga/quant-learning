# TCN（Temporal Convolutional Network）

- 论文：An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling（https://arxiv.org/abs/1803.01271）
- 参考代码：https://github.com/locuslab/TCN

本目录提供 TCN 在 Qlib `Alpha158/Alpha360` 数据集上的基准 workflow 配置。

## 文件说明

- `requirements.txt`：本模型额外依赖（通常包含 PyTorch 等）。
- `workflow_config_tcn_Alpha158.yaml` / `workflow_config_tcn_Alpha360.yaml`：`qrun` 配置文件。

## 运行方式

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd src/examples/benchmarks/TCN
pip install -r requirements.txt
qrun workflow_config_tcn_Alpha158.yaml
```
