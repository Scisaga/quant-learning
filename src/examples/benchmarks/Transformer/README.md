# Transformer（注意力机制基线）

- 论文：Attention is All you Need（NeurIPS 2017）  
  https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
- 参考代码：https://github.com/tensorflow/tensor2tensor

本目录提供 Transformer 在 Qlib `Alpha158/Alpha360` 数据集上的基准 workflow 配置。

## 文件说明

- `requirements.txt`：本模型额外依赖（通常包含 PyTorch 等）。
- `workflow_config_transformer_Alpha158.yaml` / `workflow_config_transformer_Alpha360.yaml`：`qrun` 配置文件。

## 运行方式

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd src/examples/benchmarks/Transformer
pip install -r requirements.txt
qrun workflow_config_transformer_Alpha158.yaml
```
