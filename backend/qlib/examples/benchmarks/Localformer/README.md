# Localformer

Localformer 是 Qlib 基准中的一个深度时序模型变体（名称更偏工程命名；公开论文信息相对较少）。你可以把它理解为一种“更强调局部建模/局部注意力”的 Transformer 风格结构，用于金融时序/截面预测任务的基准对比。

## 文件说明

- `requirements.txt`：本模型额外依赖（通常包含 PyTorch 等）。
- `workflow_config_localformer_Alpha158.yaml`：Alpha158 配置。
- `workflow_config_localformer_Alpha360.yaml`：Alpha360 配置。

## 运行方式

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd backend/qlib/examples/benchmarks/Localformer
pip install -r requirements.txt
qrun workflow_config_localformer_Alpha158.yaml
```

## 输出

- 默认写入 `mlruns/`（MLflow）。
- 指标解释见 `backend/qlib/examples/benchmarks/README.md`。
