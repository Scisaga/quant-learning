# GATs（图注意力网络）

- 论文：Graph Attention Networks（https://arxiv.org/pdf/1710.10903.pdf）
- 说明：本仓库中的实现为基于 PyTorch 的 Qlib 集成版本（用于基准 workflow）。

GAT（Graph Attention Networks）在图结构数据上引入注意力机制：每个节点可对邻居节点特征做自适应加权聚合，从而在不依赖昂贵矩阵运算（如求逆）的情况下建模“关系/邻接”信息。

## 文件说明

- `requirements.txt`：本模型额外依赖。
- `workflow_config_gats_Alpha158.yaml`：Alpha158 配置（含特征子采样设置）。
- `workflow_config_gats_Alpha360.yaml`：Alpha360 配置。

## 运行方式

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd backend/qlib/examples/benchmarks/GATs
pip install -r requirements.txt
qrun workflow_config_gats_Alpha158.yaml
```

## 输出

- 默认写入 `mlruns/`（MLflow）。
- 指标解释见 `backend/qlib/examples/benchmarks/README.md`。
