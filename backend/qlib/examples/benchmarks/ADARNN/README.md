# AdaRNN（自适应时序学习）

- 论文：AdaRNN: Adaptive Learning and Forecasting for Time Series（https://arxiv.org/pdf/2108.04443.pdf）
- 参考代码：https://github.com/jindongwang/transferlearning/tree/master/code/deep/adarnn

本目录提供 AdaRNN 在 Qlib `Alpha360` 数据集上的基准 workflow 配置（用于每日截面打分/预测 → 回测评估）。

## 文件说明

- `requirements.txt`：本模型额外依赖（通常包含 PyTorch 等）。
- `workflow_config_adarnn_Alpha360.yaml`：`qrun` 配置文件。

## 运行方式

在仓库根目录执行：

```bash
cd backend/qlib/examples/benchmarks/ADARNN
pip install -r requirements.txt
qrun workflow_config_adarnn_Alpha360.yaml
```

## 输出与结果查看

- 训练/预测/回测等产物默认写入 `mlruns/`（MLflow）。
- 指标含义与对比表见 `backend/qlib/examples/benchmarks/README.md`。
