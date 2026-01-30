# DoubleEnsemble（样本重加权 + 特征选择的集成框架）

- 论文：DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis  
  https://arxiv.org/pdf/2010.01265.pdf
- 说明：本仓库中的 Qlib 版本实现为自研实现（用于基准 workflow）。

DoubleEnsemble 通过两条主线缓解量化预测中“信噪比低 + 特征维度上升”带来的过拟合与不稳定：

1. 基于训练动态（learning trajectory）的 **样本重加权**：识别关键样本并提升其训练权重；
2. 基于打乱（shuffling）消融影响的 **特征选择**：识别关键特征并抑制噪声特征。

## 文件说明

- `requirements.txt`：本模型额外依赖。
- `workflow_config_doubleensemble_Alpha158.yaml` / `workflow_config_doubleensemble_Alpha360.yaml`：CSI300 基准。
- `workflow_config_doubleensemble_Alpha158_csi500.yaml` / `workflow_config_doubleensemble_Alpha360_csi500.yaml`：CSI500 基准。
- `workflow_config_doubleensemble_early_stop_Alpha158.yaml`：带 early stop 的 Alpha158 示例配置。

## 运行方式

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd src/examples/benchmarks/DoubleEnsemble
pip install -r requirements.txt
qrun workflow_config_doubleensemble_Alpha158.yaml
```

## 输出

- 默认写入 `mlruns/`（MLflow）。
- 指标解释见 `src/examples/benchmarks/README.md`。
