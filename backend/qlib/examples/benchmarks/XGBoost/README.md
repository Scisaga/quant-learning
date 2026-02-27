# XGBoost（梯度提升树）

- 论文：XGBoost: A Scalable Tree Boosting System（KDD 2016）  
  https://dl.acm.org/doi/pdf/10.1145/2939672.2939785
- 项目主页：https://github.com/dmlc/xgboost

本目录提供 XGBoost 在 Qlib `Alpha158/Alpha360` 数据集上的基准 workflow 配置。

## 文件说明

- `requirements.txt`：XGBoost 相关依赖。
- `workflow_config_xgboost_Alpha158.yaml` / `workflow_config_xgboost_Alpha360.yaml`：`qrun` 配置文件。

## 运行方式

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd backend/qlib/examples/benchmarks/XGBoost
pip install -r requirements.txt
qrun workflow_config_xgboost_Alpha158.yaml
```
