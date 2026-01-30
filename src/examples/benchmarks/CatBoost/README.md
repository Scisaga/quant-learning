# CatBoost（梯度提升树）

- 论文：CatBoost: unbiased boosting with categorical features（NeurIPS 2018）  
  https://proceedings.neurips.cc/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf
- 项目主页：https://github.com/catboost/catboost

本目录提供 CatBoost 在 Qlib `Alpha158/Alpha360` 数据集上的基准配置，并包含 `CSI300/CSI500` 两组指数基准的示例 YAML。

## 文件说明

- `requirements.txt`：CatBoost 相关依赖。
- `workflow_config_catboost_Alpha158.yaml` / `workflow_config_catboost_Alpha360.yaml`：CSI300 基准。
- `workflow_config_catboost_Alpha158_csi500.yaml` / `workflow_config_catboost_Alpha360_csi500.yaml`：CSI500 基准。

## 运行方式

在仓库根目录执行（以 Alpha158 + CSI300 为例）：

```bash
cd src/examples/benchmarks/CatBoost
pip install -r requirements.txt
qrun workflow_config_catboost_Alpha158.yaml
```

## 输出

- 默认写入 `mlruns/`（MLflow）。
- 指标解释见 `src/examples/benchmarks/README.md`。
