# TabNet（可解释表格深度模型）

- 论文：TabNet: Attentive Interpretable Tabular Learning（https://arxiv.org/pdf/1908.07442.pdf）
- 参考代码：https://github.com/dreamquark-ai/tabnet

本目录提供 TabNet 在 Qlib `Alpha158/Alpha360` 数据集上的基准 workflow 配置。

## 文件说明

- `requirements.txt`：TabNet 相关依赖。
- `workflow_config_TabNet_Alpha158.yaml` / `workflow_config_TabNet_Alpha360.yaml`：`qrun` 配置文件。

## 运行方式

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd backend/qlib/examples/benchmarks/TabNet
pip install -r requirements.txt
qrun workflow_config_TabNet_Alpha158.yaml
```
