# LightGBM（梯度提升树）

- 论文：LightGBM: A Highly Efficient Gradient Boosting Decision Tree（NeurIPS 2017）  
  https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
- 项目主页：https://github.com/microsoft/LightGBM

本目录提供 LightGBM 在 Qlib `Alpha158/Alpha360` 数据集上的基准 workflow 配置，并包含 CSI300/CSI500 两组指数基准示例。

## 文件说明

- `requirements.txt`：LightGBM 相关依赖。
- `workflow_config_lightgbm_Alpha158.yaml` / `workflow_config_lightgbm_Alpha360.yaml`：CSI300 基准。
- `workflow_config_lightgbm_Alpha158_csi500.yaml` / `workflow_config_lightgbm_Alpha360_csi500.yaml`：CSI500 基准。
- `workflow_config_lightgbm_configurable_dataset.yaml`：演示如何用可配置 dataset（便于替换数据段/特征集/标签等）。
- 多频数据（Multi-Frequency）：
  - `workflow_config_lightgbm_multi_freq.yaml`
  - `workflow_config_lightgbm_Alpha158_multi_freq.yaml`
  - 说明：这类配置会混合不同频率的数据源（例如日频 + 更高频特征）用于日频预测；具体数据源/handler 见 YAML 中的 `dataset/handler` 设置，以及目录内的 `multi_freq_handler.py`。
- 辅助脚本：
  - `features_sample.py` / `features_resample_N.py`：用于特征采样/重采样的示例脚本（按需使用）。

## 运行方式

在仓库根目录执行（以 Alpha158 + CSI300 为例）：

```bash
cd src/examples/benchmarks/LightGBM
pip install -r requirements.txt
qrun workflow_config_lightgbm_Alpha158.yaml
```

切换为多频配置：

```bash
qrun workflow_config_lightgbm_multi_freq.yaml
```

## 输出

- 默认写入 `mlruns/`（MLflow）。
- 指标解释与对比表见 `src/examples/benchmarks/README.md`。
