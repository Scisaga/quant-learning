# SFM（State Frequency Memory）

- 论文：Stock Price Prediction via Discovering Multi-Frequency Trading Patterns（KDD 2017）  
  http://www.eecs.ucf.edu/~gqi/publications/kdd2017_stock.pdf

SFM（State Frequency Memory）是一种循环网络结构：通过离散傅里叶变换（DFT）分解记忆单元隐藏状态，从而捕捉多频交易模式并用于价格/收益预测。

## 文件说明

- `requirements.txt`：本模型额外依赖（通常包含 PyTorch 等）。
- `workflow_config_sfm_Alpha360.yaml`：`qrun` 配置文件。

## 运行方式

在仓库根目录执行：

```bash
cd src/examples/benchmarks/SFM
pip install -r requirements.txt
qrun workflow_config_sfm_Alpha360.yaml
```
